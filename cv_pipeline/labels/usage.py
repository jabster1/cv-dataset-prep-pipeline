"""Shared rules for *using* resolved labels (augmentation, export).

One definition of: label policy from config, skip categories for items without a usable
label, and the resolution fingerprint that ties derived outputs to a label state.
"""

from __future__ import annotations

from collections.abc import Iterable

from cv_pipeline.annotation.schema import AnnotationItemRecord, InferenceStatus
from cv_pipeline.labels.resolver import LabelPolicy, LabelResolver
from cv_pipeline.labels.schema import Resolution, ResolutionStatus, UnavailableReason
from cv_pipeline.manifest import read_records
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.workspace import Workspace


class SkipCategory:
    LABEL_UNAVAILABLE = "label_unavailable"
    LABEL_UNTRUSTED = "label_untrusted"
    LABEL_CONFLICT = "label_conflict"
    ANNOTATION_FAILED = "annotation_failed"


UNTRUSTED_REASONS = frozenset(
    {
        UnavailableReason.NOT_REVIEWED,
        UnavailableReason.IMPORTED_UNVERIFIED,
        UnavailableReason.MACHINE_EMPTY,
        UnavailableReason.EMPTY_NOT_CONFIRMED,
        UnavailableReason.IGNORED_ONLY,
        UnavailableReason.REJECTED,
    }
)


def policy_from_settings(label_policy: str, allow_machine_empty: bool) -> LabelPolicy:
    machine = label_policy == "allow_machine_unreviewed"
    return LabelPolicy(
        allow_machine_unreviewed=machine, allow_machine_empty=machine and allow_machine_empty
    )


def skip_category(res: Resolution, annotation_failed: bool) -> str:
    if res.status == ResolutionStatus.CONFLICT:
        return SkipCategory.LABEL_CONFLICT
    if res.reason in UNTRUSTED_REASONS:
        return SkipCategory.LABEL_UNTRUSTED
    if annotation_failed:
        return SkipCategory.ANNOTATION_FAILED
    return SkipCategory.LABEL_UNAVAILABLE


def failed_inference_items(ws: Workspace, resolver: LabelResolver) -> set[str]:
    """Items whose inference failed in the active annotation run (if it is current)."""
    lineage = resolver.annotation
    if lineage.get("status") != "current":
        return set()
    path = ws.annotation_runs / lineage["run_id"] / "items.csv"
    return {
        r.item_id
        for r in read_records(path, AnnotationItemRecord)
        if r.inference_status != InferenceStatus.OK
    }


def resolution_digest(resolutions: Iterable[Resolution]) -> str:
    """Fingerprint of resolved labels: per item status, provenance, label sha and reason."""
    return digest(
        sorted(
            [
                r.item_id,
                r.status,
                r.label.provenance if r.label else "",
                r.label.label_sha256 if r.label else "",
                r.reason,
            ]
            for r in resolutions
        )
    )
