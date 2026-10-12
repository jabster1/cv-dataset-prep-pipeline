"""Confidence / review policy — pure functions, no I/O, no model.

Per box (first matching rule wins):
  1. non-finite or zero-area after clipping to the image -> discarded_invalid_box
  2. label maps to no project class                      -> discarded_unmapped_class
  3. confidence < confidence_threshold                   -> discarded_below_min_confidence
  4. otherwise candidate; review_flag = confidence < review_threshold

Per item, any of these puts it in the review queue (``needs_review``):
  inference_failed / unreadable_image, low_confidence (any flagged candidate),
  no_detections (if review_empty), unmapped_class (if review_unmapped_classes),
  audit_sample (deterministic sample of otherwise-unflagged items).
Everything else is ``unreviewed`` — machine-generated and not verified by anyone.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from cv_pipeline.annotation.schema import (
    COORD_DECIMALS,
    BoxDecision,
    InferenceStatus,
    RawDetection,
    ReviewReason,
    ReviewStatus,
)
from cv_pipeline.config import PipelineConfig


@dataclass(frozen=True)
class PolicySettings:
    classes: tuple[str, ...]
    class_map: tuple[tuple[str, str], ...]
    confidence_threshold: float
    review_threshold: float
    review_empty: bool
    review_unmapped_classes: bool
    review_sample_rate: float
    seed: int

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> PolicySettings:
        a = cfg.annotation
        return cls(
            classes=tuple(cfg.classes),
            class_map=tuple(sorted(a.class_map.items())),
            confidence_threshold=a.confidence_threshold,
            review_threshold=a.review_threshold,
            review_empty=a.review_empty,
            review_unmapped_classes=a.review_unmapped_classes,
            review_sample_rate=a.review_sample_rate,
            seed=cfg.project.seed,
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "class_map": dict(self.class_map),
            "confidence_threshold": self.confidence_threshold,
            "review_threshold": self.review_threshold,
            "review_empty": self.review_empty,
            "review_unmapped_classes": self.review_unmapped_classes,
            "review_sample_rate": self.review_sample_rate,
            "seed": self.seed,
        }

    def class_id(self, label: str) -> int | None:
        target = dict(self.class_map).get(label, label)
        try:
            return self.classes.index(target)
        except ValueError:
            return None


@dataclass(frozen=True)
class BoxOutcome:
    detection: RawDetection
    class_id: int | None
    decision: str
    review_flag: bool
    xyxy: tuple[float, float, float, float] | None  # clipped, pixels
    clipped: bool

    def yolo(self, width: int, height: int) -> tuple[float, float, float, float] | None:
        if self.xyxy is None:
            return None
        x1, y1, x2, y2 = self.xyxy
        return (
            round((x1 + x2) / 2 / width, 6),
            round((y1 + y2) / 2 / height, 6),
            round((x2 - x1) / width, 6),
            round((y2 - y1) / height, 6),
        )


def _clip(
    xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[tuple[float, float, float, float] | None, bool]:
    if not all(math.isfinite(v) for v in xyxy):
        return None, False
    x1, y1, x2, y2 = xyxy
    cx1, cy1 = min(max(x1, 0.0), width), min(max(y1, 0.0), height)
    cx2, cy2 = min(max(x2, 0.0), width), min(max(y2, 0.0), height)
    clipped = (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2)
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        return None, clipped
    box = tuple(round(v, COORD_DECIMALS) for v in (cx1, cy1, cx2, cy2))
    return box, clipped  # type: ignore[return-value]


def evaluate_boxes(
    detections: list[RawDetection], width: int, height: int, settings: PolicySettings
) -> list[BoxOutcome]:
    """Apply the per-box policy. ``detections`` must already be canonical (sorted)."""
    out = []
    for det in detections:
        box, clipped = _clip(det.xyxy, width, height)
        class_id = settings.class_id(det.label)
        review_flag = False
        if box is None or not math.isfinite(det.confidence):
            decision = BoxDecision.DISCARDED_INVALID_BOX
        elif class_id is None:
            decision = BoxDecision.DISCARDED_UNMAPPED_CLASS
        elif det.confidence < settings.confidence_threshold:
            decision = BoxDecision.DISCARDED_BELOW_MIN_CONFIDENCE
        else:
            decision = BoxDecision.CANDIDATE
            review_flag = det.confidence < settings.review_threshold
        out.append(BoxOutcome(det, class_id, decision, review_flag, box, clipped))
    return out


def audit_sampled(item_id: str, seed: int, rate: float) -> bool:
    """Deterministic, uniform-ish per-item draw in [0, 1) compared against ``rate``."""
    if rate <= 0:
        return False
    h = hashlib.sha256(f"audit:{seed}:{item_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < rate


def item_review(
    item_id: str, status: str, boxes: list[BoxOutcome], settings: PolicySettings
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if status == InferenceStatus.FAILED:
        reasons.append(ReviewReason.INFERENCE_FAILED)
    elif status == InferenceStatus.UNREADABLE_IMAGE:
        reasons.append(ReviewReason.UNREADABLE_IMAGE)
    else:
        candidates = [b for b in boxes if b.decision == BoxDecision.CANDIDATE]
        if any(b.review_flag for b in candidates):
            reasons.append(ReviewReason.LOW_CONFIDENCE)
        if settings.review_empty and not candidates:
            reasons.append(ReviewReason.NO_DETECTIONS)
        if settings.review_unmapped_classes and any(
            b.decision == BoxDecision.DISCARDED_UNMAPPED_CLASS for b in boxes
        ):
            reasons.append(ReviewReason.UNMAPPED_CLASS)
        if not reasons and audit_sampled(item_id, settings.seed, settings.review_sample_rate):
            reasons.append(ReviewReason.AUDIT_SAMPLE)
    status_out = ReviewStatus.NEEDS_REVIEW if reasons else ReviewStatus.UNREVIEWED
    return status_out, reasons
