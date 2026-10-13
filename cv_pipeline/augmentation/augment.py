"""Augment stage: trusted-labelled TRAIN items -> immutable augmentation run.

Guarantees
----------
* Inputs are the validated dataset state, the latest split revision and labels from the
  Phase 4.5 :class:`LabelResolver`; the split and label snapshots must be the same dataset
  state or nothing is written.
* Only items whose current split is ``train`` are ever augmented (hard check).
* Default label policy is trusted human labels only; items without a usable label are
  skipped — never treated as background.
* Positive images that lose boxes are retried deterministically and finally rejected;
  only trusted negatives may stay box-free.
* Outputs are derived records (``aug_*``) with full lineage; they never enter items.csv,
  dedup or split history. Completed runs are immutable.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from cv_pipeline.annotation.schema import AnnotationItemRecord, InferenceStatus
from cv_pipeline.augmentation.report import build_report, render_markdown
from cv_pipeline.augmentation.transforms import build_pipeline, resolve_transforms
from cv_pipeline.config import PipelineConfig
from cv_pipeline.labels import LabelPolicy, LabelResolver, Resolution
from cv_pipeline.labels.schema import ResolutionStatus, UnavailableReason
from cv_pipeline.manifest import ItemRecord, SourceRefRecord, read_records, write_records
from cv_pipeline.splitting.history import SplitItemRecord, load_history
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.utils.hashing import sha256_bytes
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text, decode_image, encode_image
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("augment")

AUG_SCHEMA = "cvp-augmentation-v1"
TRAIN = "train"


class AugmentationError(RuntimeError):
    """Augmentation refused to run (stale split/labels, missing inputs, integrity)."""


class SkipCategory:
    LABEL_UNAVAILABLE = "label_unavailable"
    LABEL_UNTRUSTED = "label_untrusted"
    LABEL_CONFLICT = "label_conflict"
    ANNOTATION_FAILED = "annotation_failed"


_UNTRUSTED_REASONS = {
    UnavailableReason.NOT_REVIEWED,
    UnavailableReason.IMPORTED_UNVERIFIED,
    UnavailableReason.MACHINE_EMPTY,
    UnavailableReason.EMPTY_NOT_CONFIRMED,
    UnavailableReason.IGNORED_ONLY,
    UnavailableReason.REJECTED,
}


@dataclass
class AugmentedItemRecord:
    aug_item_id: str
    parent_item_id: str
    source_id: str
    split_group: str
    split: str
    split_revision: int
    origin: str
    family_key: str
    label_provenance: str
    label_source_ref: str
    parent_label_sha256: str
    parent_negative: bool
    label_notes: str
    run_id: str
    variant: int
    attempt: int
    seed: int
    recipe: str  # JSON: applied transforms and their sampled scalar parameters
    image_path: str  # relative to the run directory
    label_path: str
    input_sha256: str
    output_sha256: str
    width: int
    height: int
    boxes_before: int
    boxes_after: int


@dataclass
class SkippedItemRecord:
    item_id: str
    category: str
    reason: str


@dataclass
class RejectedVariantRecord:
    parent_item_id: str
    variant: int
    attempts: int
    reason: str


@dataclass
class _Parent:
    item: ItemRecord
    split: SplitItemRecord
    resolution: Resolution
    family_key: str


@dataclass
class AugmentResult:
    run_id: str | None
    run_dir: Path | None
    created: bool
    dry_run: bool
    items: list[AugmentedItemRecord] = field(default_factory=list)
    skipped: list[SkippedItemRecord] = field(default_factory=list)
    rejected: list[RejectedVariantRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def label_policy(cfg: PipelineConfig) -> LabelPolicy:
    a = cfg.augmentation
    machine = a.label_policy == "allow_machine_unreviewed"
    return LabelPolicy(
        allow_machine_unreviewed=machine, allow_machine_empty=machine and a.allow_machine_empty
    )


def run_augment_stage(cfg: PipelineConfig, force: bool = False) -> AugmentResult:
    return run_augment(cfg)


def run_augment(cfg: PipelineConfig, dry_run: bool = False) -> AugmentResult:
    t0 = time.perf_counter()
    ws = Workspace(cfg.workspace_dir).ensure()
    a = cfg.augmentation
    policy = label_policy(cfg)

    # 1-3: consistent dataset state, split revision and labels ---------------------------
    resolver = LabelResolver.load(cfg, policy)  # validates state + re-hashes all files
    history = load_history(ws)
    revision = history.last_revision
    if revision is None or not ws.split_items_manifest.is_file():
        raise AugmentationError("No split exists yet — run `split` before `augment`.")
    if revision.get("snapshot_fp") != resolver.snapshot_fp:
        raise AugmentationError(
            f"Split revision {revision['revision']} was made from a different dataset state than "
            "the current labels/data (stale split). Re-run `split` (and `ingest`/`dedup` first "
            "if raw data or imports changed), then `augment`."
        )
    if policy.allow_machine_unreviewed and resolver.annotation.get("status") != "current":
        raise AugmentationError(
            "label_policy allows machine labels, but the active annotation run does not match "
            f"the current dataset state (status: {resolver.annotation.get('status')}). Re-run "
            "`annotate`, then `augment`."
        )

    # 4-5: train parents and eligibility ---------------------------------------------------
    split_rows = {r.item_id: r for r in read_records(ws.split_items_manifest, SplitItemRecord)}
    if set(split_rows) != set(resolver.eligible_ids):
        raise AugmentationError(
            "splits/items.csv does not match the dataset state — re-run `split`."
        )
    items = resolver.state.items
    families = _families(ws)
    failed = _failed_inference(ws, resolver)
    parents: list[_Parent] = []
    skipped: list[SkippedItemRecord] = []
    for iid in sorted(split_rows):
        row = split_rows[iid]
        if row.split != TRAIN:
            continue  # val/test are never augmented
        res = resolver.resolve(iid)
        if res.usable:
            parents.append(_Parent(items[iid], row, res, families.get(items[iid].source_id, "")))
        else:
            skipped.append(SkippedItemRecord(iid, _skip_category(res, iid in failed), res.reason))
    for p in parents:  # hard guard: nothing outside train can reach generation
        if p.split.split != TRAIN:
            raise AssertionError(f"refusing to augment non-train item {p.item.item_id}")

    # 6: identity ---------------------------------------------------------------------------
    transforms = resolve_transforms(a.transforms)
    seed = a.seed if a.seed is not None else cfg.project.seed
    config_payload = {
        "label_policy": policy.as_dict(),
        "variants_per_item": a.variants_per_item,
        "seed": seed,
        "min_visibility": a.min_visibility,
        "min_box_area": a.min_box_area,
        "allow_partial_box_loss": a.allow_partial_box_loss,
        "max_attempts": a.max_attempts,
        "image_format": a.image_format,
        "jpeg_quality": a.jpeg_quality,
        "transforms": transforms,
        "classes": list(cfg.classes),
    }
    versions = {n: _version(n) for n in ("albumentations", "albucore", "opencv-python", "numpy")}
    config_fp = digest({"schema": AUG_SCHEMA, "config": config_payload, "versions": versions})
    labels_digest = digest(
        [
            [
                iid,
                r.status,
                r.label.provenance if r.label else "",
                r.label.label_sha256 if r.label else "",
                r.reason,
            ]
            for iid, r in sorted(
                [(p.item.item_id, p.resolution) for p in parents]
                + [(s.item_id, resolver.resolve(s.item_id)) for s in skipped]
            )
        ]
    )
    identity = {
        "snapshot_fp": resolver.snapshot_fp,
        "split_revision": revision["revision"],
        "split_state_digest": revision["state_digest"],
        "labels_digest": labels_digest,
        "config_fp": config_fp,
    }
    run_id = "augr_" + digest({"schema": AUG_SCHEMA, **identity})[:16]
    run_dir = ws.root / "augmentation" / "runs" / run_id
    split_counts = Counter(r.split for r in split_rows.values())
    context = {
        "cfg": cfg,
        "run_id": run_id,
        "identity": identity,
        "config": config_payload,
        "versions": versions,
        "split_counts": split_counts,
        "parents": parents,
        "skipped": skipped,
        "revision": revision,
        "lineage": resolver.annotation,
    }
    log_event(
        logger,
        "augment_planned",
        run_id=run_id,
        train=split_counts.get(TRAIN, 0),
        eligible=len(parents),
        skipped=len(skipped),
    )

    if dry_run:
        report = build_report(context, [], [], dry_run=True, created=False)
        return AugmentResult(run_id, None, False, True, [], skipped, [], report)

    created = False
    if run_dir.exists():
        _verify_existing(run_dir, run_id, identity)
        aug_items = read_records(run_dir / "items.csv", AugmentedItemRecord)
        rejected = read_records(run_dir / "rejected.csv", RejectedVariantRecord)
        log_event(logger, "augment_run_reused", run_id=run_id)
    else:
        aug_items, rejected = _generate(ws, run_dir, context, transforms, seed)
        created = True

    atomic_write_json(ws.root / "augmentation" / "active_run.json", {"run_id": run_id, **identity})
    report = build_report(context, aug_items, rejected, dry_run=False, created=created)
    atomic_write_json(ws.reports / "augmentation.json", report)
    atomic_write_text(ws.reports / "augmentation.md", render_markdown(report))
    log_event(
        logger,
        "augment_finished",
        run_id=run_id,
        new_run=created,
        generated=len(aug_items),
        rejected=len(rejected),
        seconds=round(time.perf_counter() - t0, 2),
    )
    return AugmentResult(run_id, run_dir, created, False, aug_items, skipped, rejected, report)


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def _generate(
    ws: Workspace,
    run_dir: Path,
    ctx: dict[str, Any],
    transforms: list[dict[str, Any]],
    seed: int,
) -> tuple[list[AugmentedItemRecord], list[RejectedVariantRecord]]:
    cfg: PipelineConfig = ctx["cfg"]
    a = cfg.augmentation
    pipeline = build_pipeline(transforms, a.min_visibility, a.min_box_area)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{ctx['run_id']}-", dir=run_dir.parent))
    out_items: list[AugmentedItemRecord] = []
    rejected: list[RejectedVariantRecord] = []
    ext = f".{a.image_format}"
    try:
        (tmp / "images").mkdir()
        (tmp / "labels").mkdir()
        for p in ctx["parents"]:
            data = (ws.root / p.item.path).read_bytes()
            if sha256_bytes(data) != p.item.sha256:
                raise AugmentationError(
                    f"Workspace image for {p.item.item_id} changed during augmentation — "
                    "re-run `ingest`; nothing was written."
                )
            image = decode_image(data)
            if image is None:
                raise AugmentationError(f"Cannot decode {p.item.path}; nothing was written.")
            label = p.resolution.label
            assert label is not None
            boxes = [(b.cx, b.cy, b.w, b.h) for b in label.boxes]
            class_ids = [b.class_id for b in label.boxes]
            for variant in range(a.variants_per_item):
                outcome = _augment_variant(
                    pipeline, image, boxes, class_ids, seed, p.item.item_id, variant, a
                )
                if isinstance(outcome, str):
                    rejected.append(
                        RejectedVariantRecord(p.item.item_id, variant, a.max_attempts, outcome)
                    )
                    continue
                attempt, attempt_seed, out_img, out_boxes, out_cls, recipe = outcome
                aug_id = (
                    "aug_"
                    + digest(
                        [
                            p.item.item_id,
                            p.item.sha256,
                            label.label_sha256,
                            variant,
                            ctx["identity"]["config_fp"],
                        ]
                    )[:16]
                )
                encoded = encode_image(out_img, ext, a.jpeg_quality)
                (tmp / "images" / f"{aug_id}{ext}").write_bytes(encoded)
                lines = [
                    f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for c, (cx, cy, w, h) in zip(out_cls, out_boxes, strict=True)
                ]
                (tmp / "labels" / f"{aug_id}.txt").write_text(
                    "".join(f"{line}\n" for line in lines), encoding="utf-8"
                )
                h, w = out_img.shape[:2]
                out_items.append(
                    AugmentedItemRecord(
                        aug_item_id=aug_id,
                        parent_item_id=p.item.item_id,
                        source_id=p.item.source_id,
                        split_group=p.split.split_group,
                        split=TRAIN,
                        split_revision=ctx["revision"]["revision"],
                        origin=p.item.origin,
                        family_key=p.family_key,
                        label_provenance=label.provenance,
                        label_source_ref=label.source_ref,
                        parent_label_sha256=label.label_sha256,
                        parent_negative=label.negative,
                        label_notes=";".join(label.notes),
                        run_id=ctx["run_id"],
                        variant=variant,
                        attempt=attempt,
                        seed=attempt_seed,
                        recipe=recipe,
                        image_path=f"images/{aug_id}{ext}",
                        label_path=f"labels/{aug_id}.txt",
                        input_sha256=p.item.sha256,
                        output_sha256=sha256_bytes(encoded),
                        width=int(w),
                        height=int(h),
                        boxes_before=len(boxes),
                        boxes_after=len(out_boxes),
                    )
                )
        out_items.sort(key=lambda r: r.aug_item_id)
        write_records(tmp / "items.csv", out_items, AugmentedItemRecord)
        write_records(tmp / "rejected.csv", rejected, RejectedVariantRecord)
        write_records(tmp / "skipped.csv", ctx["skipped"], SkippedItemRecord)
        atomic_write_json(
            tmp / "run.json",
            {
                "schema": AUG_SCHEMA,
                "run_id": ctx["run_id"],
                "identity": ctx["identity"],
                "config": ctx["config"],
                "versions": ctx["versions"],
                "counts": {
                    "parents": len(ctx["parents"]),
                    "generated": len(out_items),
                    "rejected_variants": len(rejected),
                    "skipped": len(ctx["skipped"]),
                },
            },
        )
        os.rename(tmp, run_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_items, rejected


def attempt_seed(seed: int, item_id: str, variant: int, attempt: int) -> int:
    return int(digest(["aug", seed, item_id, variant, attempt])[:8], 16)


def _augment_variant(
    pipeline: Any,
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    class_ids: list[int],
    seed: int,
    item_id: str,
    variant: int,
    a: Any,
) -> tuple[int, int, np.ndarray, list[tuple[float, ...]], list[int], str] | str:
    """First valid attempt, or the reason the variant was rejected."""
    last_reason = ""
    for attempt in range(a.max_attempts):
        s = attempt_seed(seed, item_id, variant, attempt)
        pipeline.set_random_seed(s)
        out = pipeline(image=image, bboxes=boxes, class_ids=class_ids)
        out_boxes = [tuple(float(v) for v in b) for b in out["bboxes"]]
        out_cls = [int(c) for c in out["class_ids"]]
        problem = _box_problem(out_boxes)
        if problem:
            last_reason = problem
        elif boxes and not out_boxes:
            last_reason = "positive image lost all boxes"
        elif len(out_boxes) < len(boxes) and not a.allow_partial_box_loss:
            last_reason = f"lost {len(boxes) - len(out_boxes)} of {len(boxes)} boxes"
        else:
            rounded = [tuple(round(v, 6) for v in b) for b in out_boxes]
            return (
                attempt,
                s,
                out["image"],
                rounded,
                out_cls,
                _recipe(out.get("applied_transforms", [])),
            )
    return f"no valid result in {a.max_attempts} attempts (last: {last_reason})"


def _box_problem(boxes: list[tuple[float, ...]]) -> str:
    for cx, cy, w, h in boxes:
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            return "non-finite box"
        if w <= 0 or h <= 0:
            return "zero-size box"
        t = 1e-6
        if cx - w / 2 < -t or cy - h / 2 < -t or cx + w / 2 > 1 + t or cy + h / 2 > 1 + t:
            return "box outside image"
    return ""


def _recipe(applied: list[Any]) -> str:
    """Applied transforms with scalar parameters (large arrays such as noise maps are
    summarised by shape; reproduction relies on the recorded seed)."""

    def clean(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return [clean(x) for x in v.tolist()] if v.size <= 16 else f"<array {list(v.shape)}>"
        if isinstance(v, float | np.floating):
            return round(float(v), 6)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, list | tuple):
            return [clean(x) for x in v]
        return v if isinstance(v, int | str | bool) or v is None else str(v)

    steps = [{"transform": name, "params": clean(params)} for name, params in applied]
    return json.dumps(steps, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------------------


def _skip_category(res: Resolution, annotation_failed: bool) -> str:
    if res.status == ResolutionStatus.CONFLICT:
        return SkipCategory.LABEL_CONFLICT
    if res.reason in _UNTRUSTED_REASONS:
        return SkipCategory.LABEL_UNTRUSTED
    if annotation_failed:
        return SkipCategory.ANNOTATION_FAILED
    return SkipCategory.LABEL_UNAVAILABLE


def _failed_inference(ws: Workspace, resolver: LabelResolver) -> set[str]:
    lineage = resolver.annotation
    if lineage.get("status") != "current":
        return set()
    path = ws.annotation_runs / lineage["run_id"] / "items.csv"
    return {
        r.item_id
        for r in read_records(path, AnnotationItemRecord)
        if r.inference_status != InferenceStatus.OK
    }


def _families(ws: Workspace) -> dict[str, str]:
    out: dict[str, str] = {}
    if ws.source_refs_manifest.is_file():
        for r in read_records(ws.source_refs_manifest, SourceRefRecord):
            if r.family_key:
                out[r.source_id] = r.family_key
    return out


def _verify_existing(run_dir: Path, run_id: str, identity: dict[str, Any]) -> None:
    try:
        existing = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("run_id") != run_id or existing.get("identity") != identity:
        raise AugmentationError(
            f"Augmentation run directory {run_dir} exists but does not match its identity; "
            "refusing to overwrite it."
        )


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


__all__ = [
    "AugmentResult",
    "AugmentationError",
    "AugmentedItemRecord",
    "RejectedVariantRecord",
    "SkipCategory",
    "SkippedItemRecord",
    "run_augment",
    "run_augment_stage",
]
