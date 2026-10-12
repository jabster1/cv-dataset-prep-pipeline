"""Review queue export: ``review/`` in a layout any YOLO-capable labeling tool can open.

```
review/
    images/<item_id>.<ext>      copies of queued images
    labels/<item_id>.txt        machine labels as a starting point (YOLO format)
    previews/<item_id>.jpg      predictions drawn on the image
    classes.txt                 project classes, one per line, in class-id order
    review_manifest.csv         what is queued and why
    .export_state.csv           sha256 of each label file as last written by the pipeline
```

Human edits are never overwritten: a label file is (re)written only if it does not exist
or still has exactly the content the pipeline last wrote. Otherwise it is kept and marked
``label_state=edited``. Nothing under ``review/`` is ever deleted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from cv_pipeline.annotation.policy import BoxOutcome
from cv_pipeline.annotation.previews import render_preview
from cv_pipeline.annotation.schema import (
    AnnotationItemRecord,
    BoxDecision,
    InferenceStatus,
    ReviewStatus,
)
from cv_pipeline.config import PipelineConfig
from cv_pipeline.manifest import read_records, write_records
from cv_pipeline.utils.hashing import sha256_bytes, sha256_file
from cv_pipeline.utils.io import (
    atomic_copy,
    atomic_write_bytes,
    atomic_write_text,
    encode_image,
    read_image,
)
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("annotation.review")


class LabelState:
    MACHINE = "machine"  # file holds exactly what the pipeline last wrote
    EDITED = "edited"  # changed by someone else since; preserved
    NONE = "none"  # no label written (unreadable image)


@dataclass
class ReviewManifestRecord:
    filename: str
    item_id: str
    based_on_run_id: str
    source_id: str
    source_path: str
    frame_index: int | None
    image_path: str
    label_path: str
    preview_path: str
    predicted_classes: str
    min_confidence: float | None
    max_confidence: float | None
    n_candidates: int
    inference_status: str
    review_status: str
    review_reasons: str
    label_state: str


@dataclass
class ExportStateRecord:
    item_id: str
    run_id: str
    label_sha256: str


@dataclass
class ReviewExportResult:
    queued: int = 0
    label_states: dict[str, int] = field(default_factory=dict)
    stale_labels: list[str] = field(default_factory=list)
    images_not_overwritten: list[str] = field(default_factory=list)


def label_text(boxes: list[BoxOutcome], width: int, height: int) -> str:
    lines = []
    for b in boxes:
        if b.decision != BoxDecision.CANDIDATE:
            continue
        cx, cy, w, h = b.yolo(width, height)  # type: ignore[misc]
        lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "".join(f"{line}\n" for line in lines)


def export_review_queue(
    cfg: PipelineConfig,
    ws: Workspace,
    records: list[AnnotationItemRecord],
    boxes: dict[str, list[BoxOutcome]],
) -> ReviewExportResult:
    review = ws.review
    img_dir, lbl_dir, prev_dir = review / "images", review / "labels", review / "previews"
    for d in (img_dir, lbl_dir, prev_dir):
        d.mkdir(parents=True, exist_ok=True)
    state_path = review / ".export_state.csv"
    state = (
        {r.item_id: r for r in read_records(state_path, ExportStateRecord)}
        if state_path.is_file()
        else {}
    )

    classes_txt = "".join(f"{c}\n" for c in cfg.classes)
    if (
        not (review / "classes.txt").is_file()
        or (review / "classes.txt").read_text(encoding="utf-8") != classes_txt
    ):
        atomic_write_text(review / "classes.txt", classes_txt)

    if cfg.review.export == "all":
        queue = list(records)
    else:
        queue = [r for r in records if r.review_status == ReviewStatus.NEEDS_REVIEW]

    result = ReviewExportResult(queued=len(queue))
    manifest: list[ReviewManifestRecord] = []
    for rec in queue:
        src = ws.root / rec.image_path
        readable = rec.inference_status != InferenceStatus.UNREADABLE_IMAGE
        image_rel = ""
        if readable and cfg.review.copy_images:
            dest = img_dir / f"{rec.item_id}{src.suffix.lower()}"
            if not dest.exists():
                atomic_copy(src, dest)
            elif sha256_file(dest) != rec.file_sha256:
                result.images_not_overwritten.append(rec.item_id)
            image_rel = dest.relative_to(ws.root).as_posix()
        elif readable:
            image_rel = rec.image_path

        label_rel, state_name = "", LabelState.NONE
        if readable:
            label_path = lbl_dir / f"{rec.item_id}.txt"
            text = label_text(boxes.get(rec.item_id, []), rec.width, rec.height)
            new_sha = sha256_bytes(text.encode("utf-8"))
            prev = state.get(rec.item_id)
            current: str | None = None
            if label_path.exists():
                current = sha256_file(label_path)
                pipeline_owned = prev is not None and prev.label_sha256 == current
            else:
                pipeline_owned = True
            if pipeline_owned:
                if current != new_sha:
                    atomic_write_text(label_path, text)
                state[rec.item_id] = ExportStateRecord(rec.item_id, rec.run_id, new_sha)
                state_name = LabelState.MACHINE
            else:
                state_name = LabelState.EDITED
                log_event(logger, "edited_label_preserved", logging.INFO, item_id=rec.item_id)
            label_rel = label_path.relative_to(ws.root).as_posix()

        preview_rel = ""
        if readable and cfg.review.previews:
            image = read_image(src)
            if image is not None:
                preview = render_preview(image, boxes.get(rec.item_id, []), tuple(cfg.classes))
                dest = prev_dir / f"{rec.item_id}.jpg"
                atomic_write_bytes(dest, encode_image(preview, ".jpg", 90))
                preview_rel = dest.relative_to(ws.root).as_posix()

        cands = [b for b in boxes.get(rec.item_id, []) if b.decision == BoxDecision.CANDIDATE]
        confs = [b.detection.confidence for b in cands]
        manifest.append(
            ReviewManifestRecord(
                filename=Path(image_rel).name if image_rel else "",
                item_id=rec.item_id,
                based_on_run_id=rec.run_id,
                source_id=rec.source_id,
                source_path=rec.source_path,
                frame_index=rec.frame_index,
                image_path=image_rel,
                label_path=label_rel,
                preview_path=preview_rel,
                predicted_classes=";".join(sorted({cfg.classes[b.class_id] for b in cands})),  # type: ignore[index]
                min_confidence=min(confs) if confs else None,
                max_confidence=max(confs) if confs else None,
                n_candidates=len(cands),
                inference_status=rec.inference_status,
                review_status=rec.review_status,
                review_reasons=rec.review_reasons,
                label_state=state_name,
            )
        )
        result.label_states[state_name] = result.label_states.get(state_name, 0) + 1

    write_records(review / "review_manifest.csv", manifest, ReviewManifestRecord)
    write_records(state_path, sorted(state.values(), key=lambda r: r.item_id), ExportStateRecord)

    queued_ids = {r.item_id for r in queue}
    result.stale_labels = sorted(p.stem for p in lbl_dir.glob("*.txt") if p.stem not in queued_ids)
    if result.stale_labels:
        log_event(
            logger,
            "review_labels_not_in_queue",
            logging.WARNING,
            count=len(result.stale_labels),
            hint="kept on disk; not part of the current review queue",
        )
    if result.images_not_overwritten:
        log_event(
            logger,
            "review_images_differ",
            logging.WARNING,
            count=len(result.images_not_overwritten),
            hint="existing review/images files differ from the workspace image; left untouched",
        )
    return result
