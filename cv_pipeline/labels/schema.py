"""Label provenance vocabulary and the source-independent label representation."""

from __future__ import annotations

from dataclasses import dataclass, field

from cv_pipeline.utils.fingerprint import digest


class LabelProvenance:
    HUMAN_REVIEWED_CORRECTED = "human_reviewed_corrected"
    HUMAN_REVIEWED_ACCEPTED = "human_reviewed_accepted"
    HUMAN_IMPORTED = "human_imported"
    IMPORTED_UNVERIFIED = "imported_unverified"
    MACHINE_UNREVIEWED = "machine_unreviewed"


# Higher wins when several usable labels exist for one item.
PRIORITY = {
    LabelProvenance.HUMAN_REVIEWED_CORRECTED: 5,
    LabelProvenance.HUMAN_REVIEWED_ACCEPTED: 4,
    LabelProvenance.HUMAN_IMPORTED: 3,
    LabelProvenance.IMPORTED_UNVERIFIED: 2,
    LabelProvenance.MACHINE_UNREVIEWED: 1,
}
TRUSTED = frozenset(
    {
        LabelProvenance.HUMAN_REVIEWED_CORRECTED,
        LabelProvenance.HUMAN_REVIEWED_ACCEPTED,
        LabelProvenance.HUMAN_IMPORTED,
    }
)


class ResolutionStatus:
    TRUSTED = "trusted"  # human provenance
    OPT_IN = "opt_in"  # usable only because the policy explicitly allowed it
    UNAVAILABLE = "unavailable"
    CONFLICT = "conflict"


IGNORED_ONLY_NOTE = "ignored_only_source_annotation"


class UnavailableReason:
    REJECTED = "rejected_by_reviewer"
    NOT_REVIEWED = "machine_label_not_reviewed"
    MACHINE_EMPTY = "machine_zero_detections_not_a_negative"
    EMPTY_NOT_CONFIRMED = "empty_label_not_confirmed_negative"
    IGNORED_ONLY = "ignored_only_unconfirmed_negative"
    IMPORTED_UNVERIFIED = "imported_label_not_trusted"
    INVALID_LABEL = "invalid_imported_label"
    MISSING_LABEL = "imported_label_file_missing"
    NO_LABEL = "no_label"


@dataclass(frozen=True, order=True)
class LabelBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def make(cls, class_id: int, cx: float, cy: float, w: float, h: float) -> LabelBox:
        return cls(int(class_id), round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


@dataclass(frozen=True)
class LabelCandidate:
    """A label for one image from one source, in normalised YOLO coordinates.

    ``negative`` marks an explicit, confirmed "no target objects" label. An empty
    ``boxes`` tuple without ``negative`` is NOT a usable negative.
    """

    item_id: str
    provenance: str
    boxes: tuple[LabelBox, ...]
    negative: bool
    source_ref: str
    file_sha256: str
    width: int
    height: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label_sha256(self) -> str:
        return digest(
            {
                "boxes": [[b.class_id, b.cx, b.cy, b.w, b.h] for b in sorted(self.boxes)],
                "negative": self.negative,
            }
        )


@dataclass(frozen=True)
class Resolution:
    item_id: str
    status: str
    label: LabelCandidate | None
    reason: str = ""
    candidates: tuple[str, ...] = ()  # source_refs considered, for audit

    @property
    def usable(self) -> bool:
        return self.status in (ResolutionStatus.TRUSTED, ResolutionStatus.OPT_IN)
