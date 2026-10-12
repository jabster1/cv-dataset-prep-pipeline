"""Annotate stage: eligible items -> backend -> policy -> immutable run snapshot -> review queue.

Flow per invocation
-------------------
1. Eligible items = ``dedup.csv`` rows not excluded, in ``item_id`` order (optionally the
   first ``limit``). Excluded duplicates are never sent to the backend.
2. Each image is decoded exactly once; ``input_pixel_sha256`` of the decoded array keys the
   inference cache together with ``inference_fp``.
3. Cache misses (or every item with ``force``) are inferred in batches. A failing batch is
   retried item by item so one bad image cannot fail its neighbours; per-item failures are
   recorded (and never cached).
4. The policy turns raw detections into candidates/discards and review status.
5. The run is identified by ``run_id`` (see :mod:`cv_pipeline.annotation.identity`). A new
   run directory is built in a temp dir and atomically renamed into place; an existing
   one is verified and reused, never rewritten.
6. ``active_run.json`` points at the run; the review queue is exported from it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cv_pipeline.annotation.backend import (
    AnnotationBackend,
    AnnotationSetupError,
    BackendContext,
    create_backend,
)
from cv_pipeline.annotation.cache import InferenceCache
from cv_pipeline.annotation.identity import (
    combine,
    digest,
    input_pixel_sha256,
    make_run_id,
)
from cv_pipeline.annotation.policy import BoxOutcome, PolicySettings, evaluate_boxes, item_review
from cv_pipeline.annotation.report import build_report, render_markdown
from cv_pipeline.annotation.review_queue import (
    ReviewExportResult,
    export_review_queue,
    label_text,
)
from cv_pipeline.annotation.schema import (
    SCHEMA_VERSION,
    AnnotationItemRecord,
    BackendInfo,
    BoxDecision,
    InferenceStatus,
    PredictionRecord,
    Provenance,
    RawDetection,
    canonical_detections,
)
from cv_pipeline.config import PipelineConfig
from cv_pipeline.dataset_state import load_dataset_state
from cv_pipeline.manifest import ItemRecord, write_records
from cv_pipeline.utils.hashing import sha256_bytes
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text, decode_image
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("annotation")

EXIT_ITEM_FAILURES = 3


@dataclass
class ItemOutcome:
    item: ItemRecord
    split_group: str = ""
    status: str = InferenceStatus.OK
    error: str = ""
    file_sha256: str = ""
    pixel_sha256: str = ""
    width: int = 0
    height: int = 0
    detections: list[RawDetection] = field(default_factory=list)


@dataclass
class AnnotationResult:
    run_id: str
    run_dir: Path
    created: bool
    items: list[AnnotationItemRecord]
    predictions: list[PredictionRecord]
    skipped_excluded: int
    cache_hits: int
    inferred: int
    forced_changes: list[str]
    review: ReviewExportResult
    report: dict[str, Any]

    @property
    def failures(self) -> list[AnnotationItemRecord]:
        return [r for r in self.items if r.inference_status != InferenceStatus.OK]

    @property
    def exit_code(self) -> int:
        """3 = completed and outputs written, but one or more items failed inference."""
        return EXIT_ITEM_FAILURES if self.failures else 0


def run_annotate(
    cfg: PipelineConfig, force: bool = False, limit: int | None = None
) -> AnnotationResult:
    t0 = time.perf_counter()
    ws = Workspace(cfg.workspace_dir).ensure()
    items, split_groups, eligible_ids, n_excluded = _eligible_items(cfg)
    if limit is not None:
        eligible_ids = eligible_ids[:limit]

    settings = PolicySettings.from_config(cfg)
    backend = create_backend(cfg.annotation, BackendContext(cfg.root, ws.model_cache))
    try:
        info = backend.info()
        _check_class_mapping(info, settings)
        inference_fp = digest(info.fingerprint_payload())
        policy_fp = digest({"schema": SCHEMA_VERSION, **settings.fingerprint_payload()})
        spec_fp = combine(inference_fp, policy_fp, digest({"model_id": info.model_id}))
        log_event(
            logger,
            "annotate_started",
            eligible=len(eligible_ids),
            skipped_excluded=n_excluded,
            backend=info.name,
            model=info.model_id,
            force=force,
        )
        cache = InferenceCache(ws.inference_cache, inference_fp)
        runner = _Runner(backend, cache, cfg.annotation.batch_size, force)
        outcomes = runner.run([(items[i], split_groups[i]) for i in eligible_ids], ws)
    finally:
        backend.close()

    # Identity -------------------------------------------------------------------------
    snapshot_fp = digest({"schema": SCHEMA_VERSION, "items": [_snapshot_row(o) for o in outcomes]})
    result_digest = digest(
        {"schema": SCHEMA_VERSION, "results": [_result_row(o) for o in outcomes]}
    )
    run_id = make_run_id(spec_fp, snapshot_fp, result_digest)
    identity = {
        "run_id": run_id,
        "spec_fp": spec_fp,
        "inference_fp": inference_fp,
        "policy_fp": policy_fp,
        "snapshot_fp": snapshot_fp,
        "result_digest": result_digest,
    }

    # Policy + records -----------------------------------------------------------------
    boxes: dict[str, list[BoxOutcome]] = {}
    item_records: list[AnnotationItemRecord] = []
    pred_records: list[PredictionRecord] = []
    for o in outcomes:
        iid = o.item.item_id
        b = evaluate_boxes(o.detections, o.width, o.height, settings) if o.width else []
        boxes[iid] = b
        review_status, reasons = item_review(iid, o.status, b, settings)
        item_records.append(_item_record(o, info, run_id, b, review_status, reasons))
        pred_records.extend(_prediction_records(iid, run_id, b, o, cfg.classes))

    run_dir = ws.annotation_runs / run_id
    run_json = {
        "schema": SCHEMA_VERSION,
        "identity": identity,
        "backend": info.to_json(),
        "policy": settings.fingerprint_payload(),
        "counts": _run_counts(item_records, pred_records),
    }
    created = _write_run(ws, run_dir, run_json, item_records, pred_records, boxes, outcomes)

    previous = _read_json(ws.active_annotation_run)
    previous_run_id = previous.get("run_id") if previous else None
    atomic_write_json(
        ws.active_annotation_run,
        {
            **identity,
            "previous_run_id": previous_run_id
            if previous_run_id != run_id
            else previous.get("previous_run_id"),
        },
    )

    review = export_review_queue(cfg, ws, item_records, boxes)
    result = AnnotationResult(
        run_id=run_id,
        run_dir=run_dir,
        created=created,
        items=item_records,
        predictions=pred_records,
        skipped_excluded=n_excluded,
        cache_hits=runner.cache_hits,
        inferred=runner.inferred,
        forced_changes=runner.changed,
        review=review,
        report={},
    )
    result.report = build_report(
        cfg, result, info, identity, previous_run_id, force, limit, len(eligible_ids)
    )
    atomic_write_json(ws.reports / "annotation.json", result.report)
    atomic_write_text(ws.reports / "annotation.md", render_markdown(result.report))

    for f in result.failures:
        log_event(
            logger,
            "item_failed",
            logging.WARNING,
            item_id=f.item_id,
            status=f.inference_status,
            error=f.error,
        )
    log_event(
        logger,
        "annotate_finished",
        seconds=round(time.perf_counter() - t0, 2),
        run_id=run_id,
        new_run=created,
        items=len(item_records),
        cache_hits=runner.cache_hits,
        inferred=runner.inferred,
        failures=len(result.failures),
        review_queue=review.queued,
    )
    return result


def run_annotate_stage(cfg: PipelineConfig, force: bool = False) -> AnnotationResult:
    return run_annotate(cfg, force=force)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def _eligible_items(
    cfg: PipelineConfig,
) -> tuple[dict[str, ItemRecord], dict[str, str], list[str], int]:
    """Items, their current dedup split groups, eligible ids (sorted), excluded count.

    Uses the same validated dataset state as splitting, so both stages agree on whether
    the current ingest/dedup state is usable.
    """
    state = load_dataset_state(cfg)
    split_groups = {iid: d.split_group for iid, d in state.dedup.items()}
    return state.items, split_groups, list(state.eligible_ids), state.excluded


def _check_class_mapping(info: BackendInfo, settings: PolicySettings) -> None:
    if info.class_names is None:  # open-vocabulary: labels come from prompts
        return
    mapped = {n for n in info.class_names if settings.class_id(n) is not None}
    if not mapped:
        raise AnnotationSetupError(
            f"None of the model's {len(info.class_names)} classes map to the project classes "
            f"{list(settings.classes)}. Set annotation.class_map (model label -> project "
            f"class). Model classes start with: {list(info.class_names[:10])}"
        )
    unknown_keys = sorted(set(dict(settings.class_map)) - set(info.class_names))
    if unknown_keys:
        log_event(logger, "class_map_keys_not_in_model", logging.WARNING, keys=unknown_keys)
    log_event(
        logger,
        "class_mapping",
        mapped_model_classes=len(mapped),
        unmapped_model_classes=len(info.class_names) - len(mapped),
    )


class _Runner:
    """Decodes, hashes, consults the cache and runs batched inference with isolation."""

    def __init__(
        self, backend: AnnotationBackend, cache: InferenceCache, batch_size: int, force: bool
    ) -> None:
        self.backend, self.cache, self.batch_size, self.force = backend, cache, batch_size, force
        self.cache_hits = 0
        self.inferred = 0
        self.changed: list[str] = []
        self._pending: list[tuple[ItemOutcome, np.ndarray]] = []
        self._waiters: dict[str, list[ItemOutcome]] = {}
        self._previous: dict[str, list[RawDetection] | None] = {}

    def run(self, items: Sequence[tuple[ItemRecord, str]], ws: Workspace) -> list[ItemOutcome]:
        outcomes = []
        for item, split_group in items:
            o = ItemOutcome(item, split_group)
            outcomes.append(o)
            image = self._load(o, ws)
            if image is None:
                continue
            if o.pixel_sha256 in self._waiters:  # identical pixels already queued this run
                self._waiters[o.pixel_sha256].append(o)
                continue
            if not self.force:
                cached = self.cache.get(o.pixel_sha256)
                if cached is not None:
                    o.detections = cached
                    self.cache_hits += 1
                    continue
            else:
                self._previous[o.pixel_sha256] = self.cache.get(o.pixel_sha256)
            self._waiters[o.pixel_sha256] = [o]
            self._pending.append((o, image))
            if len(self._pending) >= self.batch_size:
                self._flush()
        self._flush()
        return outcomes

    def _load(self, o: ItemOutcome, ws: Workspace) -> np.ndarray | None:
        path = ws.root / o.item.path
        o.width, o.height = o.item.width, o.item.height
        try:
            data = path.read_bytes()
        except OSError as exc:
            o.status, o.error = InferenceStatus.UNREADABLE_IMAGE, f"cannot read file: {exc}"
            return None
        o.file_sha256 = sha256_bytes(data)
        if o.file_sha256 != o.item.sha256:
            o.status = InferenceStatus.UNREADABLE_IMAGE
            o.error = "workspace file changed since ingest (sha256 mismatch); re-run ingest"
            return None
        image = decode_image(data)
        if image is None:
            o.status, o.error = InferenceStatus.UNREADABLE_IMAGE, "image could not be decoded"
            return None
        o.pixel_sha256 = input_pixel_sha256(image)
        o.height, o.width = image.shape[:2]
        return image

    def _flush(self) -> None:
        batch, self._pending = self._pending, []
        if batch:
            self._infer(batch)

    def _infer(self, batch: list[tuple[ItemOutcome, np.ndarray]]) -> None:
        try:
            results = self.backend.predict([img for _, img in batch])
            if len(results) != len(batch):
                raise RuntimeError(
                    f"backend returned {len(results)} results for {len(batch)} images"
                )
        except AnnotationSetupError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate and record per-item failures
            if len(batch) > 1:
                for single in batch:
                    self._infer([single])
                return
            o = batch[0][0]
            error = f"{type(exc).__name__}: {exc}"
            for w in self._waiters.pop(o.pixel_sha256, [o]):
                w.status, w.error = InferenceStatus.FAILED, error
            log_event(
                logger, "inference_failed", logging.WARNING, item_id=o.item.item_id, error=error
            )
            return

        for (o, _), dets in zip(batch, results, strict=True):
            canon = canonical_detections(list(dets))
            self.cache.put(o.pixel_sha256, canon)
            self.inferred += 1
            for w in self._waiters.pop(o.pixel_sha256, [o]):
                w.detections = canon
                prev = self._previous.get(o.pixel_sha256)
                if self.force and prev is not None and prev != canon:
                    self.changed.append(w.item.item_id)


# --------------------------------------------------------------------------------------
# Identity rows and records
# --------------------------------------------------------------------------------------


def _snapshot_row(o: ItemOutcome) -> dict[str, Any]:
    i = o.item
    return {
        "item_id": i.item_id,
        "source_id": i.source_id,
        "source_type": i.source_type,
        "source_path": i.source_path,
        "frame_index": i.frame_index,
        "split_group": o.split_group,
        "image_path": i.path,
        "file_sha256": o.file_sha256,
        "input_pixel_sha256": o.pixel_sha256,
        "width": o.width,
        "height": o.height,
    }


def _result_row(o: ItemOutcome) -> dict[str, Any]:
    return {
        "item_id": o.item.item_id,
        "status": o.status,
        "error": o.error,
        "detections": [d.to_json() for d in o.detections],
    }


def _item_record(
    o: ItemOutcome,
    info: BackendInfo,
    run_id: str,
    boxes: list[BoxOutcome],
    review_status: str,
    reasons: list[str],
) -> AnnotationItemRecord:
    i = o.item
    n_cand = sum(b.decision == BoxDecision.CANDIDATE for b in boxes)
    return AnnotationItemRecord(
        item_id=i.item_id,
        source_id=i.source_id,
        source_type=i.source_type,
        source_path=i.source_path,
        frame_index=i.frame_index,
        split_group=o.split_group,
        image_path=i.path,
        file_sha256=o.file_sha256,
        input_pixel_sha256=o.pixel_sha256,
        width=o.width,
        height=o.height,
        run_id=run_id,
        backend=info.name,
        backend_version=info.version,
        model_id=info.model_id,
        model_sha256=info.model_sha256 or "",
        inference_status=o.status,
        error=o.error,
        n_raw=len(o.detections),
        n_candidates=n_cand,
        n_discarded=len(boxes) - n_cand,
        n_review_flagged=sum(b.review_flag for b in boxes),
        review_status=review_status,
        review_reasons=";".join(reasons),
        provenance=Provenance.MACHINE,
        label_path=f"labels/{i.item_id}.txt" if o.status == InferenceStatus.OK else "",
    )


def _prediction_records(
    item_id: str, run_id: str, boxes: list[BoxOutcome], o: ItemOutcome, classes: list[str]
) -> list[PredictionRecord]:
    out = []
    for idx, b in enumerate(boxes):
        d = b.detection
        yolo = b.yolo(o.width, o.height)
        xyxy = b.xyxy
        out.append(
            PredictionRecord(
                item_id=item_id,
                run_id=run_id,
                pred_index=idx,
                class_id=b.class_id,
                class_name=classes[b.class_id] if b.class_id is not None else "",
                backend_label=d.label,
                backend_label_id=d.label_id,
                confidence=d.to_json()["confidence"],
                x1=xyxy[0] if xyxy else None,
                y1=xyxy[1] if xyxy else None,
                x2=xyxy[2] if xyxy else None,
                y2=xyxy[3] if xyxy else None,
                cx=yolo[0] if yolo else None,
                cy=yolo[1] if yolo else None,
                w=yolo[2] if yolo else None,
                h=yolo[3] if yolo else None,
                clipped=b.clipped,
                decision=b.decision,
                review_flag=b.review_flag,
                provenance=Provenance.MACHINE,
            )
        )
    return out


def _run_counts(items: list[AnnotationItemRecord], preds: list[PredictionRecord]) -> dict[str, Any]:
    def tally(values: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    return {
        "items": len(items),
        "inference_status": tally([r.inference_status for r in items]),
        "review_status": tally([r.review_status for r in items]),
        "boxes": tally([p.decision for p in preds]),
    }


# --------------------------------------------------------------------------------------
# Immutable run directory
# --------------------------------------------------------------------------------------


def _write_run(
    ws: Workspace,
    run_dir: Path,
    run_json: dict[str, Any],
    items: list[AnnotationItemRecord],
    preds: list[PredictionRecord],
    boxes: dict[str, list[BoxOutcome]],
    outcomes: list[ItemOutcome],
) -> bool:
    """Create the run directory atomically. Returns False if it already existed."""
    if run_dir.exists():
        _verify_existing(run_dir, run_json["identity"])
        log_event(logger, "run_reused", run_id=run_dir.name)
        return False

    ws.annotation_runs.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{run_dir.name}-", dir=ws.annotation_runs))
    try:
        atomic_write_json(tmp / "run.json", run_json)
        write_records(tmp / "items.csv", items, AnnotationItemRecord)
        write_records(tmp / "predictions.csv", preds, PredictionRecord)
        labels = tmp / "labels"
        labels.mkdir()
        for o in outcomes:
            if o.status == InferenceStatus.OK:
                iid = o.item.item_id
                atomic_write_text(labels / f"{iid}.txt", label_text(boxes[iid], o.width, o.height))
        os.rename(tmp, run_dir)
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
        if run_dir.exists():  # created concurrently with identical identity
            _verify_existing(run_dir, run_json["identity"])
            return False
        raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log_event(logger, "run_created", run_id=run_dir.name, items=len(items))
    return True


def _verify_existing(run_dir: Path, identity: dict[str, Any]) -> None:
    existing = _read_json(run_dir / "run.json")
    if not existing or existing.get("identity") != identity:
        raise RuntimeError(
            f"Annotation run directory {run_dir} exists but does not match the expected "
            "identity; it may have been modified. Refusing to overwrite it."
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
