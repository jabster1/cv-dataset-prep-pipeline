"""Deliberate review queue for building a trusted dataset from new field data.

    python -m cv_pipeline review-queue --config C [--mode needs_confirmation] [--split val]

Modes
-----
* ``needs_confirmation`` (default): every active item WITHOUT a trusted label — items with
  unreviewed machine pre-labels (exported as the starting label) and items with no
  annotation at all (exported with a blank label).
* ``unreviewed_machine``: only items that have an unreviewed machine label.

This only writes the existing ``review/`` queue (images, labels, previews, manifest) via
:func:`export_review_queue`, so human edits stay protected. It never trusts anything and
never changes provenance: labels become trusted only through ``review/decisions.csv``
(``accepted`` / ``corrected``; ``based_on_run_id: manual`` for items with no machine run).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cv_pipeline.annotation.policy import BoxOutcome
from cv_pipeline.annotation.review_queue import export_review_queue
from cv_pipeline.annotation.schema import (
    AnnotationItemRecord,
    PredictionRecord,
    Provenance,
    RawDetection,
    ReviewStatus,
)
from cv_pipeline.config import PipelineConfig
from cv_pipeline.labels.resolver import MANUAL_RUN, LabelPolicy, LabelResolver
from cv_pipeline.labels.schema import ResolutionStatus, UnavailableReason
from cv_pipeline.manifest import read_records
from cv_pipeline.splitting.history import SplitItemRecord, load_history
from cv_pipeline.utils.io import atomic_write_json
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("review_queue")

MODES = ("needs_confirmation", "unreviewed_machine")
NOT_ANNOTATED = "not_annotated"


class ReviewQueueError(RuntimeError):
    pass


@dataclass
class ReviewQueueResult:
    queued: list[str] = field(default_factory=list)
    with_machine_label: int = 0
    blank: int = 0
    report: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def run_review_queue(
    cfg: PipelineConfig, mode: str = "needs_confirmation", split: str = "all"
) -> ReviewQueueResult:
    if mode not in MODES:
        raise ReviewQueueError(f"Unknown mode {mode!r}; choose from {list(MODES)}")
    ws = Workspace(cfg.workspace_dir).ensure()
    resolver = LabelResolver.load(cfg, LabelPolicy())  # "trusted" = default policy
    candidates = list(resolver.eligible_ids)
    if split != "all":
        revision = load_history(ws).last_revision
        if revision is None or revision.get("snapshot_fp") != resolver.snapshot_fp:
            raise ReviewQueueError("--split needs a current split — run `split` first.")
        splits = {
            r.item_id: r.split for r in read_records(ws.split_items_manifest, SplitItemRecord)
        }
        candidates = [i for i in candidates if splits.get(i) == split]

    selected = []
    for iid in candidates:
        res = resolver.resolve(iid)
        if res.status == ResolutionStatus.TRUSTED:
            continue
        if mode == "unreviewed_machine" and res.reason != UnavailableReason.NOT_REVIEWED:
            continue
        selected.append((iid, res.reason or res.status))

    run_items, run_boxes = _active_run(ws, resolver)
    records: list[AnnotationItemRecord] = []
    boxes: dict[str, list[BoxOutcome]] = {}
    for iid, why in selected:
        reason = f"queued_{mode}:{why}"
        if iid in run_items:
            rec = AnnotationItemRecord(**{**run_items[iid].__dict__})
            rec.review_status, rec.review_reasons = ReviewStatus.NEEDS_REVIEW, reason
            boxes[iid] = run_boxes.get(iid, [])
        else:
            rec = _blank_record(resolver, iid, reason)
            boxes[iid] = []
        records.append(rec)

    # Write only the selected queue; cfg.review.export "flagged" == every queued record.
    export = export_review_queue(cfg, ws, records, boxes)
    result = ReviewQueueResult(
        queued=[r.item_id for r in records],
        with_machine_label=sum(1 for r in records if r.run_id != MANUAL_RUN),
        blank=sum(1 for r in records if r.run_id == MANUAL_RUN),
    )
    result.report = {
        "mode": mode,
        "split": split,
        "queued": len(records),
        "with_machine_prelabel": result.with_machine_label,
        "blank_for_manual_labelling": result.blank,
        "reasons": dict(sorted(Counter(r.review_reasons for r in records).items())),
        "label_states": export.label_states,
        "next_step": "Review in review/, then record accepted/corrected/rejected rows in "
        "review/decisions.csv (based_on_run_id from review_manifest.csv; 'manual' for blank "
        "items), then run `labels`.",
    }
    atomic_write_json(ws.reports / "review_queue.json", result.report)
    log_event(logger, "review_queue_written", mode=mode, split=split, queued=len(records))
    return result


def _active_run(
    ws: Workspace, resolver: LabelResolver
) -> tuple[dict[str, AnnotationItemRecord], dict[str, list[BoxOutcome]]]:
    lineage = resolver.annotation
    if lineage.get("status") != "current":
        return {}, {}
    run_dir = ws.annotation_runs / lineage["run_id"]
    items = {r.item_id: r for r in read_records(run_dir / "items.csv", AnnotationItemRecord)}
    boxes: dict[str, list[BoxOutcome]] = {}
    for p in read_records(run_dir / "predictions.csv", PredictionRecord):
        xyxy = (p.x1, p.y1, p.x2, p.y2) if p.x1 is not None else None
        det = RawDetection(
            p.backend_label,
            p.backend_label_id,
            p.confidence if p.confidence is not None else float("nan"),
            xyxy if xyxy else (float("nan"),) * 4,  # type: ignore[arg-type]
        )
        boxes.setdefault(p.item_id, []).append(
            BoxOutcome(det, p.class_id, p.decision, p.review_flag, xyxy, p.clipped)  # type: ignore[arg-type]
        )
    return items, boxes


def _blank_record(resolver: LabelResolver, iid: str, reason: str) -> AnnotationItemRecord:
    item = resolver.state.items[iid]
    return AnnotationItemRecord(
        item_id=iid,
        source_id=item.source_id,
        source_type=item.source_type,
        source_path=item.source_path,
        frame_index=item.frame_index,
        split_group=resolver.state.dedup[iid].split_group,
        image_path=item.path,
        file_sha256=item.sha256,
        input_pixel_sha256="",
        width=item.width,
        height=item.height,
        run_id=MANUAL_RUN,
        backend="",
        backend_version="",
        model_id="",
        model_sha256="",
        inference_status=NOT_ANNOTATED,
        error="",
        n_raw=0,
        n_candidates=0,
        n_discarded=0,
        n_review_flagged=0,
        review_status=ReviewStatus.NEEDS_REVIEW,
        review_reasons=reason,
        provenance=Provenance.MACHINE,
        label_path="",
    )
