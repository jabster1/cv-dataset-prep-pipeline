"""Annotation data model: normalized backend output, run manifests, status vocabularies.

Everything a backend produces is converted to :class:`RawDetection` before the pipeline
sees it, so the rest of the annotation stage never depends on a specific detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "cvp-annotation-v1"

CONFIDENCE_DECIMALS = 6
COORD_DECIMALS = 3


class InferenceStatus:
    OK = "ok"
    FAILED = "failed"
    UNREADABLE_IMAGE = "unreadable_image"


class ReviewStatus:
    """Machine annotations are either flagged for review or simply not yet reviewed.

    Nothing produced by the annotate stage is ever "verified". Human review (a later
    phase) adds ``reviewed_accepted`` / ``reviewed_corrected`` / ``reviewed_rejected``.
    """

    NEEDS_REVIEW = "needs_review"
    UNREVIEWED = "unreviewed"


class ReviewReason:
    INFERENCE_FAILED = "inference_failed"
    UNREADABLE_IMAGE = "unreadable_image"
    LOW_CONFIDENCE = "low_confidence"
    NO_DETECTIONS = "no_detections"
    UNMAPPED_CLASS = "unmapped_class"
    AUDIT_SAMPLE = "audit_sample"


class Provenance:
    MACHINE = "machine"
    HUMAN = "human"


class BoxDecision:
    CANDIDATE = "candidate"
    DISCARDED_BELOW_MIN_CONFIDENCE = "discarded_below_min_confidence"
    DISCARDED_UNMAPPED_CLASS = "discarded_unmapped_class"
    DISCARDED_INVALID_BOX = "discarded_invalid_box"


def _round(value: float, digits: int) -> float:
    return round(float(value), digits) if math.isfinite(value) else float("nan")


@dataclass(frozen=True)
class RawDetection:
    """One detection as reported by a backend, before any policy is applied.

    ``xyxy`` is in absolute pixels of the image passed to the backend. ``label`` is the
    backend's class name (for open-vocabulary detectors, the matched prompt phrase).
    """

    label: str
    label_id: int | None
    confidence: float
    xyxy: tuple[float, float, float, float]

    def normalized(self) -> RawDetection:
        return RawDetection(
            label=str(self.label),
            label_id=None if self.label_id is None else int(self.label_id),
            confidence=_round(self.confidence, CONFIDENCE_DECIMALS),
            xyxy=tuple(_round(v, COORD_DECIMALS) for v in self.xyxy),  # type: ignore[arg-type]
        )

    def sort_key(self) -> tuple[Any, ...]:
        finite = [v if math.isfinite(v) else math.inf for v in self.xyxy]
        return (self.label, -self.confidence if math.isfinite(self.confidence) else 0.0, *finite)

    def to_json(self) -> dict[str, Any]:
        def num(v: float) -> float | None:
            return v if math.isfinite(v) else None

        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": num(self.confidence),
            "xyxy": [num(v) for v in self.xyxy],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RawDetection:
        def num(v: float | None) -> float:
            return float("nan") if v is None else float(v)

        x1, y1, x2, y2 = (num(v) for v in data["xyxy"])
        return cls(data["label"], data["label_id"], num(data["confidence"]), (x1, y1, x2, y2))


def canonical_detections(dets: list[RawDetection]) -> list[RawDetection]:
    """Round and sort so results are independent of backend output order."""
    return sorted((d.normalized() for d in dets), key=RawDetection.sort_key)


@dataclass(frozen=True)
class BackendInfo:
    """Everything about a backend that can change its outputs."""

    name: str
    version: str
    model_id: str
    model_sha256: str | None
    class_names: tuple[str, ...] | None  # None => open-vocabulary
    params: dict[str, Any] = field(default_factory=dict)
    frameworks: dict[str, str] = field(default_factory=dict)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "backend": self.name,
            "backend_version": self.version,
            "frameworks": dict(sorted(self.frameworks.items())),
            "model_sha256": self.model_sha256,
            "params": self.params,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "class_names": list(self.class_names) if self.class_names is not None else None,
            "params": self.params,
            "frameworks": dict(sorted(self.frameworks.items())),
        }


# --------------------------------------------------------------------------------------
# Run manifests (annotations/runs/<run_id>/items.csv and predictions.csv)
# --------------------------------------------------------------------------------------


@dataclass
class AnnotationItemRecord:
    item_id: str
    source_id: str
    source_type: str
    source_path: str
    frame_index: int | None
    image_path: str
    file_sha256: str
    input_pixel_sha256: str
    width: int
    height: int
    run_id: str
    backend: str
    backend_version: str
    model_id: str
    model_sha256: str
    inference_status: str
    error: str
    n_raw: int
    n_candidates: int
    n_discarded: int
    n_review_flagged: int
    review_status: str
    review_reasons: str  # ';'-separated ReviewReason values
    provenance: str
    label_path: str


@dataclass
class PredictionRecord:
    item_id: str
    run_id: str
    pred_index: int
    class_id: int | None
    class_name: str
    backend_label: str
    backend_label_id: int | None
    confidence: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    cx: float | None
    cy: float | None
    w: float | None
    h: float | None
    clipped: bool
    decision: str
    review_flag: bool
    provenance: str
