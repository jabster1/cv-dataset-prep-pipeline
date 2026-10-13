"""Validated, immutable YOLO dataset export.

validated dataset state ── split revision ── labels/state.json ── augmentation run
      (all four must describe the same dataset state, or nothing is written)
                               │
                plan: originals with usable labels per split,
                      trusted negatives (empty labels), train augmentations,
                      exclusions with reasons, held-out coverage
                               │
         strict held-out policy · class taxonomy · leakage assertions · re-hashing
                               │
                 export/runs/<run_id>/  (atomic, immutable, reused when identical)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cv_pipeline import __version__
from cv_pipeline.augmentation.augment import AUG_SCHEMA, AugmentedItemRecord
from cv_pipeline.config import PipelineConfig
from cv_pipeline.export.report import build_report, render_markdown
from cv_pipeline.imports.yolo import parse_label
from cv_pipeline.labels import LabelBox, LabelResolver
from cv_pipeline.labels.usage import (
    failed_inference_items,
    policy_from_settings,
    resolution_digest,
    skip_category,
)
from cv_pipeline.manifest import ItemRecord, SourceRefRecord, read_records, write_records
from cv_pipeline.splitting.history import SplitItemRecord, load_history
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.utils.hashing import sha256_bytes
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("export")

EXPORT_SCHEMA = "cvp-export-v1"
SPLITS = ("train", "val", "test")
EXIT_EVALUATION_INCOMPLETE = 5


class ExportError(RuntimeError):
    """Export refused (stale or inconsistent state, integrity, taxonomy)."""


class ExportLeakageError(ExportError):
    """A leakage invariant would be violated by the export."""


class EvaluationCoverageError(ExportError):
    """Strict evaluation policy: held-out items lack usable trusted labels."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ExportRecord:
    split: str
    kind: str  # original | augmented
    export_image: str  # relative to the export run dir
    export_label: str
    item_id: str  # canonical item (the parent for augmented rows)
    aug_item_id: str
    aug_run_id: str
    source_id: str
    split_group: str
    origin: str
    source_path: str  # original file name/path (relative to its origin root)
    family_key: str
    label_provenance: str
    label_source_ref: str
    negative: bool
    n_boxes: int
    image_sha256: str
    label_sha256: str


@dataclass
class ExcludedRecord:
    split: str
    item_id: str
    kind: str  # original | augmented
    category: str
    reason: str


@dataclass
class _Row:
    record: ExportRecord
    src: Path
    label_text: str
    boxes: list[LabelBox]


@dataclass
class ExportResult:
    run_id: str
    run_dir: Path | None
    created: bool
    dry_run: bool
    records: list[ExportRecord] = field(default_factory=list)
    excluded: list[ExcludedRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def run_export_stage(cfg: PipelineConfig, force: bool = False) -> ExportResult:
    return run_export(cfg)


def run_export(cfg: PipelineConfig, dry_run: bool = False) -> ExportResult:
    ws = Workspace(cfg.workspace_dir).ensure()
    e = cfg.export
    policy = policy_from_settings(e.label_policy, e.allow_machine_empty)

    # 1. dataset state (+ re-hash of every canonical file) -----------------------------
    resolver = LabelResolver.load(cfg, policy)
    snapshot = resolver.snapshot_fp
    resolutions = {r.item_id: r for r in resolver.resolve_all()}

    # 2. split ---------------------------------------------------------------------------
    revision = load_history(ws).last_revision
    if revision is None or not ws.split_items_manifest.is_file():
        raise ExportError("No split exists yet — run `split` before `export`.")
    if revision.get("snapshot_fp") != snapshot:
        raise ExportError(
            f"Split revision {revision['revision']} was made from a different dataset state — "
            "re-run `split` (then `labels`, `augment`) before `export`."
        )
    split_rows = {r.item_id: r for r in read_records(ws.split_items_manifest, SplitItemRecord)}
    if set(split_rows) != set(resolver.eligible_ids):
        raise ExportError("splits/items.csv does not match the dataset state — re-run `split`.")

    # 3. labels --------------------------------------------------------------------------
    labels_fp = resolution_digest(resolutions.values())
    _check_labels_state(ws, snapshot, policy.as_dict(), labels_fp)
    if policy.allow_machine_unreviewed and resolver.annotation.get("status") != "current":
        raise ExportError(
            "export.label_policy allows machine labels but the annotation run is not current — "
            "re-run `annotate`, `labels`, then `export`."
        )

    # 4. augmentation run ----------------------------------------------------------------
    train_ids = sorted(i for i, r in split_rows.items() if r.split == "train")
    aug = _select_augmentation(
        ws,
        e.augmentation_run,
        snapshot,
        revision,
        policy.as_dict(),
        resolution_digest(resolutions[i] for i in train_ids),
    )

    # 5. taxonomy ------------------------------------------------------------------------
    target = list(e.classes) if e.classes is not None else list(cfg.classes)
    to_export = {i: (target.index(c) if c in target else None) for i, c in enumerate(cfg.classes)}

    # 6. plan ----------------------------------------------------------------------------
    items = resolver.state.items
    families = _families(ws)
    failed = failed_inference_items(ws, resolver)
    rows: list[_Row] = []
    excluded: list[ExcludedRecord] = []
    non_target: Counter[str] = Counter()
    exported_train: set[str] = set()
    for iid in sorted(split_rows, key=lambda i: (SPLITS.index(split_rows[i].split), i)):
        srow, item, res = split_rows[iid], items[iid], resolutions[iid]
        if not res.usable:
            excluded.append(
                ExcludedRecord(
                    srow.split, iid, "original", skip_category(res, iid in failed), res.reason
                )
            )
            continue
        label = res.label
        assert label is not None
        boxes = _map_boxes(label.boxes, to_export, cfg.classes, non_target, e.non_target_classes)
        if not boxes and not label.negative:
            excluded.append(
                ExcludedRecord(
                    srow.split,
                    iid,
                    "original",
                    "non_target_only",
                    "every box is a non-target class (dropped)",
                )
            )
            continue
        ext = Path(item.path).suffix.lower()
        text = _label_text(boxes)
        rows.append(
            _Row(
                ExportRecord(
                    split=srow.split,
                    kind="original",
                    export_image=f"images/{srow.split}/{iid}{ext}",
                    export_label=f"labels/{srow.split}/{iid}.txt",
                    item_id=iid,
                    aug_item_id="",
                    aug_run_id="",
                    source_id=item.source_id,
                    split_group=srow.split_group,
                    origin=item.origin,
                    source_path=item.source_path,
                    family_key=families.get(item.source_id, ""),
                    label_provenance=label.provenance,
                    label_source_ref=label.source_ref,
                    negative=not boxes,
                    n_boxes=len(boxes),
                    image_sha256=item.sha256,
                    label_sha256=sha256_bytes(text.encode()),
                ),
                ws.root / item.path,
                text,
                boxes,
            )
        )
        if srow.split == "train":
            exported_train.add(iid)

    if aug is not None:
        rows += _augmented_rows(
            aug,
            split_rows,
            items,
            families,
            exported_train,
            to_export,
            cfg.classes,
            non_target,
            e.non_target_classes,
            excluded,
        )

    if non_target and e.non_target_classes == "error":
        raise ExportError(
            f"Labels contain project classes outside export.classes {target}: "
            f"{dict(sorted(non_target.items()))} boxes. Add them to export.classes or set "
            "export.non_target_classes: drop_boxes."
        )

    # 7. held-out coverage / strict policy ----------------------------------------------
    coverage = _coverage(split_rows, rows, excluded)
    incomplete = any(coverage[s]["missing"] for s in ("val", "test"))
    if not any(r.record.split == "train" for r in rows):
        raise ExportError("No train item has a usable label — nothing to train on.")

    # 8. leakage assertions (fail, never warn) -------------------------------------------
    _assert_no_leakage(rows, split_rows, set(resolver.eligible_ids))

    # 9. identity ------------------------------------------------------------------------
    identity = {
        "snapshot_fp": snapshot,
        "split_revision": revision["revision"],
        "split_state_digest": revision["state_digest"],
        "labels_fp": labels_fp,
        "augmentation": {"run_id": aug["run_id"], "identity": aug["identity"]} if aug else None,
        "config": {
            "label_policy": policy.as_dict(),
            "evaluation_policy": e.evaluation_policy,
            "classes": target,
            "project_classes": list(cfg.classes),
            "class_mapping": {cfg.classes[p]: x for p, x in to_export.items()},
            "non_target_classes": e.non_target_classes,
        },
        "pipeline_version": __version__,
    }
    run_id = "exp_" + digest({"schema": EXPORT_SCHEMA, **identity})[:16]
    records = sorted(
        (r.record for r in rows), key=lambda r: (SPLITS.index(r.split), r.kind, r.export_image)
    )
    ctx = {
        "run_id": run_id,
        "identity": identity,
        "target": target,
        "split_rows": split_rows,
        "records": records,
        "excluded": excluded,
        "coverage": coverage,
        "rows": rows,
        "incomplete": incomplete,
        "aug": aug,
        "evaluation_policy": e.evaluation_policy,
        "lineage": resolver.annotation,
        "non_target_dropped": dict(non_target),
    }
    report = build_report(ctx, dry_run=dry_run, created=False)

    strict_problem = _strict_problem(coverage) if e.evaluation_policy == "strict" else ""
    if strict_problem:
        report["status"] = "refused"
        if not dry_run:
            atomic_write_json(ws.reports / "export.json", report)
            atomic_write_text(ws.reports / "export.md", render_markdown(report))
        raise EvaluationCoverageError(strict_problem, report)

    if dry_run:
        _verify_sources(rows)  # integrity is part of validation
        return ExportResult(run_id, None, False, True, records, excluded, report)

    run_dir = ws.root / "export" / "runs" / run_id
    created = False
    if run_dir.exists():
        _verify_existing(run_dir, run_id, identity)
    else:
        _write_run(run_dir, rows, records, excluded, identity, target, incomplete, run_id, ctx)
        created = True
    report = build_report(ctx, dry_run=False, created=created)
    atomic_write_json(ws.root / "export" / "active_run.json", {"run_id": run_id, **identity})
    atomic_write_json(ws.reports / "export.json", report)
    atomic_write_text(ws.reports / "export.md", render_markdown(report))
    log_event(
        logger,
        "export_finished",
        run_id=run_id,
        new_run=created,
        images=len(records),
        incomplete_evaluation_set=incomplete,
    )
    return ExportResult(run_id, run_dir, created, False, records, excluded, report)


# --------------------------------------------------------------------------------------
# State checks
# --------------------------------------------------------------------------------------


def _check_labels_state(ws: Workspace, snapshot: str, policy: dict[str, bool], fp: str) -> None:
    path = ws.labels_dir / "state.json"
    if not path.is_file():
        raise ExportError(
            "No resolved-label state — run `labels` (with the export's policy flags)."
        )
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("snapshot_fp") != snapshot:
        raise ExportError(
            "Resolved labels were made from a different dataset state — re-run `labels`."
        )
    if state.get("policy") != policy:
        raise ExportError(
            f"Resolved labels used label policy {state.get('policy')} but export uses {policy} — "
            "re-run `labels` with matching flags (e.g. --allow-machine-unreviewed)."
        )
    if state.get("resolution_digest") != fp:
        raise ExportError(
            "Labels changed since `labels` ran (e.g. new review decisions or imports) — "
            "re-run `labels` (and `augment` if train labels changed)."
        )


def _select_augmentation(
    ws: Workspace,
    selector: str,
    snapshot: str,
    revision: dict[str, Any],
    policy: dict[str, bool],
    train_labels_fp: str,
) -> dict[str, Any] | None:
    if selector == "none":
        return None
    base = ws.root / "augmentation"
    if selector == "latest":
        active = base / "active_run.json"
        if not active.is_file():
            raise ExportError(
                "No augmentation run exists — run `augment`, or set export.augmentation_run: none."
            )
        run_id = json.loads(active.read_text(encoding="utf-8"))["run_id"]
    else:
        run_id = selector
    run_dir = base / "runs" / run_id
    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Augmentation run {run_id} not found or unreadable.") from exc
    if run.get("schema") != AUG_SCHEMA:
        raise ExportError(
            f"Augmentation run {run_id} uses schema {run.get('schema')} without per-file label "
            "hashes — re-run `augment`."
        )
    ident = run.get("identity", {})
    if run.get("run_id") != run_id:
        raise ExportError(f"Augmentation run {run_id} is not intact (run.json mismatch).")
    if ident.get("snapshot_fp") != snapshot:
        raise ExportError(
            f"Augmentation run {run_id} belongs to a different dataset state — re-run `augment`."
        )
    if (ident.get("split_revision"), ident.get("split_state_digest")) != (
        revision["revision"],
        revision["state_digest"],
    ):
        raise ExportError(
            f"Augmentation run {run_id} belongs to a different split revision — re-run `augment`."
        )
    if run.get("config", {}).get("label_policy") != policy:
        raise ExportError(
            f"Augmentation run {run_id} used label policy {run['config'].get('label_policy')} but "
            f"export uses {policy} — re-run `augment` with the same policy."
        )
    if ident.get("labels_digest") != train_labels_fp:
        raise ExportError(
            f"Train labels changed since augmentation run {run_id} — re-run `augment`."
        )
    return {
        "run_id": run_id,
        "dir": run_dir,
        "identity": ident,
        "items": read_records(run_dir / "items.csv", AugmentedItemRecord),
    }


def _families(ws: Workspace) -> dict[str, str]:
    out: dict[str, str] = {}
    if ws.source_refs_manifest.is_file():
        for r in read_records(ws.source_refs_manifest, SourceRefRecord):
            if r.family_key:
                out[r.source_id] = r.family_key
    return out


# --------------------------------------------------------------------------------------
# Plan helpers
# --------------------------------------------------------------------------------------


def _map_boxes(
    boxes: tuple[LabelBox, ...] | list[LabelBox],
    to_export: dict[int, int | None],
    project_classes: list[str],
    non_target: Counter[str],
    policy: str,
) -> list[LabelBox]:
    out = []
    for b in boxes:
        if b.class_id not in to_export:
            raise ExportError(f"Label class id {b.class_id} is not a project class id.")
        target = to_export[b.class_id]
        if target is None:
            non_target[project_classes[b.class_id]] += 1
            continue
        out.append(LabelBox.make(target, b.cx, b.cy, b.w, b.h))
    return sorted(out)


def _label_text(boxes: list[LabelBox]) -> str:
    return "".join(f"{b.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}\n" for b in boxes)


def _augmented_rows(
    aug: dict[str, Any],
    split_rows: dict[str, SplitItemRecord],
    items: dict[str, ItemRecord],
    families: dict[str, str],
    exported_train: set[str],
    to_export: dict[int, int | None],
    project_classes: list[str],
    non_target: Counter[str],
    policy: str,
    excluded: list[ExcludedRecord],
) -> list[_Row]:
    rows = []
    for a in sorted(aug["items"], key=lambda r: r.aug_item_id):
        parent = split_rows.get(a.parent_item_id)
        if a.split != "train" or parent is None or parent.split != "train":
            raise ExportLeakageError(
                f"Augmented item {a.aug_item_id} has a non-train parent/split — refusing."
            )
        if a.parent_item_id not in exported_train:
            excluded.append(
                ExcludedRecord(
                    "train",
                    a.aug_item_id,
                    "augmented",
                    "parent_excluded",
                    "parent train item is not exported",
                )
            )
            continue
        label_path = aug["dir"] / a.label_path
        data = label_path.read_bytes() if label_path.is_file() else b""
        if not a.label_sha256 or sha256_bytes(data) != a.label_sha256:
            raise ExportError(
                f"Augmented label {a.label_path} of run {aug['run_id']} is missing or changed — "
                "re-run `augment` (completed runs must not be edited)."
            )
        parsed, err = parse_label(data.decode("utf-8"), len(project_classes))
        if err or len(parsed) != a.boxes_after:
            raise ExportError(f"Augmented label {a.label_path} is invalid: {err or 'box count'}")
        boxes = _map_boxes(
            [LabelBox.make(b.source_class_id, b.cx, b.cy, b.w, b.h) for b in parsed],
            to_export,
            project_classes,
            non_target,
            policy,
        )
        if not boxes and not a.parent_negative:
            excluded.append(
                ExcludedRecord(
                    "train",
                    a.aug_item_id,
                    "augmented",
                    "non_target_only",
                    "every box is a non-target class (dropped)",
                )
            )
            continue
        ext = Path(a.image_path).suffix.lower()
        text = _label_text(boxes)
        item = items[a.parent_item_id]
        rows.append(
            _Row(
                ExportRecord(
                    split="train",
                    kind="augmented",
                    export_image=f"images/train/{a.aug_item_id}{ext}",
                    export_label=f"labels/train/{a.aug_item_id}.txt",
                    item_id=a.parent_item_id,
                    aug_item_id=a.aug_item_id,
                    aug_run_id=aug["run_id"],
                    source_id=a.source_id,
                    split_group=a.split_group,
                    origin=item.origin,
                    source_path=item.source_path,
                    family_key=families.get(a.source_id, ""),
                    label_provenance=a.label_provenance,
                    label_source_ref=a.label_source_ref,
                    negative=not boxes,
                    n_boxes=len(boxes),
                    image_sha256=a.output_sha256,
                    label_sha256=sha256_bytes(text.encode()),
                ),
                aug["dir"] / a.image_path,
                text,
                boxes,
            )
        )
    return rows


def _coverage(
    split_rows: dict[str, SplitItemRecord], rows: list[_Row], excluded: list[ExcludedRecord]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in SPLITS:
        total = sum(1 for r in split_rows.values() if r.split == s)
        originals = [r.record for r in rows if r.record.split == s and r.record.kind == "original"]
        exc = [x for x in excluded if x.split == s and x.kind == "original"]
        label_missing = [x for x in exc if x.category != "non_target_only"]
        cats = Counter(x.category for x in exc)
        out[s] = {
            "total": total,
            "exported": len(originals),
            "positives": sum(1 for r in originals if not r.negative),
            "negatives": sum(1 for r in originals if r.negative),
            "unlabeled": cats.get("label_unavailable", 0) + cats.get("annotation_failed", 0),
            "untrusted": cats.get("label_untrusted", 0),
            "conflicts": cats.get("label_conflict", 0),
            "non_target_only": cats.get("non_target_only", 0),
            "missing": len(label_missing),
            "coverage_pct": round(100.0 * (total - len(label_missing)) / total, 2)
            if total
            else 100.0,
            "excluded_by_reason": dict(sorted(Counter(x.reason for x in exc).items())),
        }
    return out


def _strict_problem(coverage: dict[str, dict[str, Any]]) -> str:
    problems = []
    for s in ("val", "test"):
        c = coverage[s]
        if c["missing"]:
            problems.append(
                f"{s}: {c['total']} items, {c['total'] - c['missing']} with usable trusted labels, "
                f"{c['missing']} missing/untrusted (unlabeled {c['unlabeled']}, untrusted "
                f"{c['untrusted']}, conflicts {c['conflicts']}) — coverage {c['coverage_pct']}%"
            )
    if coverage["val"]["exported"] == 0:
        problems.append("val: no exportable items — training needs a validation set")
    if not problems:
        return ""
    return (
        "Strict evaluation policy: held-out labels are incomplete.\n  "
        + "\n  ".join(problems)
        + "\nLabel/review the missing items (`review-queue`), or set "
        "export.evaluation_policy: non_strict to export an explicitly INCOMPLETE evaluation set."
    )


def manifest_leakage_problems(records: list[ExportRecord]) -> list[str]:
    """Leakage invariants checkable from an export manifest alone (shared with training)."""
    problems = []
    seen: dict[str, dict[str, set[str]]] = {
        k: defaultdict(set) for k in ("item", "group", "source", "family")
    }
    train_originals = {r.item_id for r in records if r.kind == "original" and r.split == "train"}
    for rec in records:
        seen["item"][rec.item_id].add(rec.split)
        seen["group"][rec.split_group].add(rec.split)
        seen["source"][rec.source_id].add(rec.split)
        if rec.family_key:
            seen["family"][rec.family_key].add(rec.split)
        if rec.kind == "augmented":
            if rec.split != "train":
                problems.append(f"augmented {rec.aug_item_id} outside train")
            if rec.item_id not in train_originals:
                problems.append(f"augmented {rec.aug_item_id} has no exported train parent")
            if not rec.export_image.startswith("images/train/") or not rec.export_label.startswith(
                "labels/train/"
            ):
                problems.append(f"augmented {rec.aug_item_id} written outside train/")
        elif rec.kind != "original":
            problems.append(f"unknown row kind {rec.kind!r}")
        if rec.split != "train" and Path(rec.export_image).name.startswith("aug_"):
            problems.append(f"augmentation file in {rec.split}: {rec.export_image}")
        for path, sub in ((rec.export_image, "images"), (rec.export_label, "labels")):
            if not path.startswith(f"{sub}/{rec.split}/"):
                problems.append(f"{path} is not under {sub}/{rec.split}/")
    for kind, mapping in seen.items():
        for key, splits in mapping.items():
            if len(splits) > 1:
                problems.append(f"{kind} {key} appears in several splits {sorted(splits)}")
    names = Counter(r.export_image for r in records)
    problems += [f"file name collision {n}" for n, c in names.items() if c > 1]
    return problems


def _assert_no_leakage(
    rows: list[_Row], split_rows: dict[str, SplitItemRecord], eligible: set[str]
) -> None:
    records = [r.record for r in rows]
    problems = manifest_leakage_problems(records)
    for rec in records:  # checks that need the workspace state
        if rec.kind == "original":
            if rec.item_id not in eligible:
                problems.append(f"{rec.item_id} is not dedup-eligible (excluded duplicate)")
            elif split_rows[rec.item_id].split != rec.split:
                problems.append(
                    f"{rec.item_id} exported to {rec.split}, split history says "
                    f"{split_rows[rec.item_id].split}"
                )
        elif split_rows[rec.item_id].split != "train":
            problems.append(f"augmented {rec.aug_item_id} has a non-train parent")
    if problems:
        raise ExportLeakageError("Export leakage check failed:\n  " + "\n  ".join(problems[:20]))


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def _verify_sources(rows: list[_Row]) -> None:
    """Re-hash every source image (canonical and augmented) against its recorded hash."""
    for r in rows:
        data = r.src.read_bytes() if r.src.is_file() else b""
        if sha256_bytes(data) != r.record.image_sha256:
            raise ExportError(
                f"Source image for {r.record.export_image} changed or is missing ({r.src}) — "
                + ("re-run `ingest`" if r.record.kind == "original" else "re-run `augment`")
                + "; nothing was exported."
            )


def _validate_label(boxes: list[LabelBox], nc: int) -> None:
    for b in boxes:
        if not 0 <= b.class_id < nc:
            raise ExportError(f"Exported class id {b.class_id} outside 0..{nc - 1}")
        vals = (b.cx, b.cy, b.w, b.h)
        if not all(math.isfinite(v) for v in vals) or b.w <= 0 or b.h <= 0:
            raise ExportError(f"Invalid box {vals}")
        t = 1e-6
        if (
            b.cx - b.w / 2 < -t
            or b.cy - b.h / 2 < -t
            or b.cx + b.w / 2 > 1 + t
            or b.cy + b.h / 2 > 1 + t
        ):
            raise ExportError(f"Box outside image {vals}")


def dataset_yaml(target: list[str], has_test: bool, incomplete: bool, run_id: str) -> str:
    data: dict[str, Any] = {"train": "images/train", "val": "images/val"}
    if has_test:
        data["test"] = "images/test"
    data["nc"] = len(target)
    data["names"] = {i: n for i, n in enumerate(target)}
    data["incomplete_evaluation_set"] = incomplete
    data["export_run_id"] = run_id
    header = (
        f"# Generated by cv_pipeline export {run_id}. Paths are relative to this file\n"
        "# (no 'path' key: Ultralytics then uses this file's directory as the dataset root).\n"
    )
    if incomplete:
        header += (
            "# WARNING: INCOMPLETE EVALUATION SET — held-out items without trusted labels "
            "were excluded.\n"
        )
    return header + yaml.safe_dump(data, sort_keys=False)


def _write_run(
    run_dir: Path,
    rows: list[_Row],
    records: list[ExportRecord],
    excluded: list[ExcludedRecord],
    identity: dict[str, Any],
    target: list[str],
    incomplete: bool,
    run_id: str,
    ctx: dict[str, Any],
) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{run_id}-", dir=run_dir.parent))
    try:
        for s in SPLITS:
            (tmp / "images" / s).mkdir(parents=True)
            (tmp / "labels" / s).mkdir(parents=True)
        for r in rows:
            data = r.src.read_bytes() if r.src.is_file() else b""
            if sha256_bytes(data) != r.record.image_sha256:
                raise ExportError(
                    f"Source image for {r.record.export_image} changed during export — nothing "
                    "was exported."
                )
            _validate_label(r.boxes, len(target))
            (tmp / r.record.export_image).write_bytes(data)
            (tmp / r.record.export_label).write_bytes(r.label_text.encode())
        has_test = any(r.split == "test" for r in records)
        (tmp / "dataset.yaml").write_text(dataset_yaml(target, has_test, incomplete, run_id))
        write_records(tmp / "manifest.csv", records, ExportRecord)
        write_records(
            tmp / "excluded.csv",
            sorted(excluded, key=lambda x: (SPLITS.index(x.split), x.kind, x.item_id)),
            ExcludedRecord,
        )
        report = build_report(ctx, dry_run=False, created=True)
        atomic_write_json(tmp / "report.json", report)
        atomic_write_text(tmp / "report.md", render_markdown(report))
        atomic_write_json(
            tmp / "run.json",
            {
                "schema": EXPORT_SCHEMA,
                "run_id": run_id,
                "identity": identity,
                "incomplete_evaluation_set": incomplete,
                "counts": {s: sum(1 for r in records if r.split == s) for s in SPLITS},
            },
        )
        os.rename(tmp, run_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _verify_existing(run_dir: Path, run_id: str, identity: dict[str, Any]) -> None:
    try:
        existing = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = {}
    if existing.get("run_id") != run_id or existing.get("identity") != identity:
        raise ExportError(
            f"Export run directory {run_dir} exists but does not match its identity; refusing "
            "to overwrite it."
        )
