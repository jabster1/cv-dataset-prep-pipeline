"""The single label-resolution interface used by later stages (augmentation, export).

    resolver = LabelResolver.load(cfg, LabelPolicy())       # trusted-only by default
    res = resolver.resolve(item_id)  -> Resolution(status, label, reason)

Callers never need to know whether a label came from a Roboflow/YOLO import, a human
review decision, or (with explicit opt-in) a machine annotation run.

Candidate sources for an eligible item — the item itself plus every item dedup excluded
as an exact (byte/pixel-identical) duplicate of it, since identical pixels share
coordinates:

* imports           human_imported (labels_trusted) or imported_unverified
* review decisions  human_reviewed_accepted / human_reviewed_corrected; ``rejected``
                    suppresses that run's machine label under every policy
* active annotation run, only if its snapshot matches the current dataset state:
                    machine_unreviewed

Selection: the highest-priority *usable* candidate allowed by the policy wins; several
different labels at that priority are a ``conflict`` (no label). An empty label is usable
only as an explicit negative: a trusted import with ``empty_labels_are_negatives``, or a
human decision. A machine run with zero detections is never a negative unless the policy
sets ``allow_machine_empty``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cv_pipeline.annotation.review_queue import ExportStateRecord
from cv_pipeline.annotation.schema import (
    AnnotationItemRecord,
    BoxDecision,
    InferenceStatus,
    PredictionRecord,
)
from cv_pipeline.config import PipelineConfig
from cv_pipeline.dataset_state import (
    DatasetState,
    load_dataset_state,
    load_split_input,
    verify_workspace_files,
)
from cv_pipeline.imports.importer import ImportedDataset, LabelStatus, load_imports
from cv_pipeline.imports.yolo import parse_label
from cv_pipeline.labels.decisions import InvalidDecision, ReviewDecision, read_decisions
from cv_pipeline.labels.schema import (
    IGNORED_ONLY_NOTE,
    PRIORITY,
    TRUSTED,
    LabelBox,
    LabelCandidate,
    LabelProvenance,
    Resolution,
    ResolutionStatus,
    UnavailableReason,
)
from cv_pipeline.manifest import ExcludeReason, read_records
from cv_pipeline.splitting.split import annotation_lineage
from cv_pipeline.utils.hashing import sha256_bytes
from cv_pipeline.workspace import Workspace

# ``based_on_run_id`` for a human label made without any machine pre-annotation.
MANUAL_RUN = "manual"

# Reason precedence when no usable label exists (most informative first).
_REASON_ORDER = [
    UnavailableReason.REJECTED,
    UnavailableReason.NOT_REVIEWED,
    UnavailableReason.MACHINE_EMPTY,
    UnavailableReason.EMPTY_NOT_CONFIRMED,
    UnavailableReason.IGNORED_ONLY,
    UnavailableReason.IMPORTED_UNVERIFIED,
    UnavailableReason.INVALID_LABEL,
    UnavailableReason.MISSING_LABEL,
]


@dataclass(frozen=True)
class LabelPolicy:
    """Default: trusted (human) labels only. Every relaxation is an explicit opt-in."""

    allow_machine_unreviewed: bool = False
    allow_imported_unverified: bool = False
    allow_machine_empty: bool = False  # machine zero detections as negatives

    def allows(self, provenance: str) -> bool:
        if provenance in TRUSTED:
            return True
        if provenance == LabelProvenance.MACHINE_UNREVIEWED:
            return self.allow_machine_unreviewed
        if provenance == LabelProvenance.IMPORTED_UNVERIFIED:
            return self.allow_imported_unverified
        return False

    def as_dict(self) -> dict[str, bool]:
        return {
            "allow_machine_unreviewed": self.allow_machine_unreviewed,
            "allow_imported_unverified": self.allow_imported_unverified,
            "allow_machine_empty": self.allow_machine_empty,
        }


@dataclass
class _Evidence:
    candidates: list[LabelCandidate] = field(default_factory=list)
    reasons: set[str] = field(default_factory=set)


@dataclass
class LabelResolver:
    cfg: PipelineConfig
    policy: LabelPolicy
    state: DatasetState
    snapshot_fp: str
    annotation: dict[str, Any]
    evidence: dict[str, _Evidence]
    members: dict[str, list[str]]  # eligible item -> itself + exact duplicates
    invalid_decisions: list[InvalidDecision]
    applied_decisions: dict[str, ReviewDecision]

    @classmethod
    def load(cls, cfg: PipelineConfig, policy: LabelPolicy | None = None) -> LabelResolver:
        policy = policy or LabelPolicy()
        state = load_dataset_state(cfg)
        verify_workspace_files(cfg, state, state.items)  # integrity before deriving labels
        split_input = load_split_input(cfg)
        ws = Workspace(cfg.workspace_dir)

        evidence: dict[str, _Evidence] = defaultdict(_Evidence)
        _from_imports(load_imports(cfg), state, evidence)
        lineage = annotation_lineage(ws, split_input)
        rejected_runs: dict[str, set[str]] = defaultdict(set)  # item -> rejected run ids
        decisions, invalid = read_decisions(ws.review / "decisions.csv")
        applied = _from_decisions(ws, cfg, state, decisions, evidence, rejected_runs, invalid)
        if lineage.get("status") == "current":
            _from_machine_run(ws, lineage["run_id"], state, evidence, rejected_runs)

        members: dict[str, list[str]] = {iid: [iid] for iid in state.eligible_ids}
        for iid, d in sorted(state.dedup.items()):
            if (
                d.excluded
                and d.exclude_reason == ExcludeReason.EXACT_DUPLICATE
                and d.duplicate_of in members
            ):
                members[d.duplicate_of].append(iid)
        return cls(
            cfg=cfg,
            policy=policy,
            state=state,
            snapshot_fp=split_input.snapshot_fp,
            annotation=lineage,
            evidence=dict(evidence),
            members=members,
            invalid_decisions=invalid,
            applied_decisions=applied,
        )

    @property
    def eligible_ids(self) -> tuple[str, ...]:
        return self.state.eligible_ids

    def resolve(self, item_id: str) -> Resolution:
        if item_id not in self.members:
            return Resolution(item_id, ResolutionStatus.UNAVAILABLE, None, "not_an_eligible_item")
        candidates: list[LabelCandidate] = []
        reasons: set[str] = set()
        for member in self.members[item_id]:
            ev = self.evidence.get(member)
            if ev:
                candidates.extend(ev.candidates)
                reasons |= ev.reasons
        considered = tuple(sorted(c.source_ref for c in candidates))

        usable = []
        for c in candidates:
            if not self.policy.allows(c.provenance):
                reasons.add(
                    UnavailableReason.NOT_REVIEWED
                    if c.provenance == LabelProvenance.MACHINE_UNREVIEWED
                    else UnavailableReason.IMPORTED_UNVERIFIED
                )
                continue
            if c.boxes or c.negative:
                usable.append(c)
            elif c.provenance == LabelProvenance.MACHINE_UNREVIEWED:
                if self.policy.allow_machine_empty:
                    usable.append(
                        LabelCandidate(
                            **{
                                **c.__dict__,
                                "negative": True,
                                "notes": (*c.notes, "machine_empty_allowed_by_policy"),
                            }
                        )
                    )
                else:
                    reasons.add(UnavailableReason.MACHINE_EMPTY)
            elif IGNORED_ONLY_NOTE in c.notes:
                reasons.add(UnavailableReason.IGNORED_ONLY)
            else:
                reasons.add(UnavailableReason.EMPTY_NOT_CONFIRMED)

        if not usable:
            reason = next((r for r in _REASON_ORDER if r in reasons), UnavailableReason.NO_LABEL)
            return Resolution(item_id, ResolutionStatus.UNAVAILABLE, None, reason, considered)

        best = max(PRIORITY[c.provenance] for c in usable)
        top = [c for c in usable if PRIORITY[c.provenance] == best]
        if len({c.label_sha256 for c in top}) > 1:
            return Resolution(
                item_id,
                ResolutionStatus.CONFLICT,
                None,
                f"{len(top)} different {top[0].provenance} labels",
                considered,
            )
        chosen = sorted(top, key=lambda c: c.source_ref)[0]
        status = (
            ResolutionStatus.TRUSTED if chosen.provenance in TRUSTED else ResolutionStatus.OPT_IN
        )
        return Resolution(item_id, status, chosen, "", considered)

    def resolve_all(self) -> list[Resolution]:
        return [self.resolve(i) for i in self.eligible_ids]


# --------------------------------------------------------------------------------------
# Candidate sources
# --------------------------------------------------------------------------------------


def _from_imports(
    imports: list[ImportedDataset], state: DatasetState, evidence: dict[str, _Evidence]
) -> None:
    for ds in imports:
        prov = (
            LabelProvenance.HUMAN_IMPORTED
            if ds.labels_trusted
            else LabelProvenance.IMPORTED_UNVERIFIED
        )
        boxes_by_rel: dict[str, list[LabelBox]] = defaultdict(list)
        for b in ds.boxes:
            if not b.ignored and b.class_id is not None:
                boxes_by_rel[b.rel_path].append(LabelBox.make(b.class_id, b.cx, b.cy, b.w, b.h))
        for img in ds.images:
            if img.item_id not in state.items:
                continue  # e.g. rejected at ingest because the imported file changed
            ev = evidence[img.item_id]
            if img.label_status == LabelStatus.INVALID:
                ev.reasons.add(UnavailableReason.INVALID_LABEL)
                continue
            if img.label_status == LabelStatus.MISSING:
                ev.reasons.add(UnavailableReason.MISSING_LABEL)
                continue
            boxes = tuple(sorted(boxes_by_rel.get(img.rel_path, [])))
            notes: tuple[str, ...] = ("legacy_split=" + img.legacy_split,)
            if img.label_status == LabelStatus.EMPTY:
                # Genuinely empty label file: a negative only under the explicit policy.
                negative = ds.labels_trusted and ds.empty_labels_are_negatives
            elif not boxes:
                # Every box was in ignore_classes. That does not prove the project's targets
                # were looked for, so it is a negative only under its own explicit policy.
                negative = ds.labels_trusted and ds.ignored_only_labels_are_negatives
                notes = (*notes, IGNORED_ONLY_NOTE)
            else:
                negative = False
            ev.candidates.append(
                LabelCandidate(
                    item_id=img.item_id,
                    provenance=prov,
                    boxes=boxes,
                    negative=negative,
                    source_ref=f"import:{ds.name}/{img.rel_path}",
                    file_sha256=img.sha256,
                    width=img.width,
                    height=img.height,
                    notes=notes,
                )
            )


def _run_files(
    ws: Workspace, run_id: str
) -> tuple[dict[str, AnnotationItemRecord], dict[str, list[LabelBox]]] | None:
    run_dir = ws.annotation_runs / run_id
    try:
        identity = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["identity"]
        if identity.get("run_id") != run_id:
            return None
        items = {r.item_id: r for r in read_records(run_dir / "items.csv", AnnotationItemRecord)}
        preds = read_records(run_dir / "predictions.csv", PredictionRecord)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    boxes: dict[str, list[LabelBox]] = defaultdict(list)
    for p in preds:
        if p.decision == BoxDecision.CANDIDATE and p.class_id is not None:
            boxes[p.item_id].append(LabelBox.make(p.class_id, p.cx, p.cy, p.w, p.h))  # type: ignore[arg-type]
    return items, boxes


def _from_machine_run(
    ws: Workspace,
    run_id: str,
    state: DatasetState,
    evidence: dict[str, _Evidence],
    rejected_runs: dict[str, set[str]],
) -> None:
    loaded = _run_files(ws, run_id)
    if loaded is None:
        return
    items, boxes = loaded
    for iid, rec in items.items():
        if rec.inference_status != InferenceStatus.OK or iid not in state.items:
            continue
        if run_id in rejected_runs.get(iid, set()):
            evidence[iid].reasons.add(UnavailableReason.REJECTED)
            continue
        evidence[iid].candidates.append(
            LabelCandidate(
                item_id=iid,
                provenance=LabelProvenance.MACHINE_UNREVIEWED,
                boxes=tuple(sorted(boxes.get(iid, []))),
                negative=False,
                source_ref=f"run:{run_id}",
                file_sha256=rec.file_sha256,
                width=rec.width,
                height=rec.height,
            )
        )


def _from_decisions(
    ws: Workspace,
    cfg: PipelineConfig,
    state: DatasetState,
    decisions: dict[str, ReviewDecision],
    evidence: dict[str, _Evidence],
    rejected_runs: dict[str, set[str]],
    invalid: list[InvalidDecision],
) -> dict[str, ReviewDecision]:
    applied: dict[str, ReviewDecision] = {}
    runs: dict[str, Any] = {}
    export_state_path = ws.review / ".export_state.csv"
    exported = (
        {r.item_id: r.label_sha256 for r in read_records(export_state_path, ExportStateRecord)}
        if export_state_path.is_file()
        else {}
    )

    def bad(d: ReviewDecision, problem: str) -> None:
        invalid.append(InvalidDecision(d.row, d.item_id, problem))

    for iid, d in sorted(decisions.items()):
        item = state.items.get(iid)
        if item is None:
            bad(d, "unknown item_id (not in items.csv)")
            continue
        manual = d.based_on_run_id == MANUAL_RUN
        run_boxes: dict[str, list[LabelBox]] = {}
        if manual:
            # Labelled from scratch (no machine pre-annotation): only "corrected" is meaningful.
            if d.decision != "corrected":
                bad(d, f"based_on_run_id '{MANUAL_RUN}' is only valid with decision 'corrected'")
                continue
        else:
            if d.based_on_run_id not in runs:
                runs[d.based_on_run_id] = _run_files(ws, d.based_on_run_id)
            loaded = runs[d.based_on_run_id]
            if loaded is None:
                bad(d, f"annotation run {d.based_on_run_id} not found or not intact")
                continue
            run_items, run_boxes = loaded
            run_item = run_items.get(iid)
            if run_item is None:
                bad(d, f"item not in annotation run {d.based_on_run_id}")
                continue
            if run_item.file_sha256 != item.sha256:
                bad(d, "image content changed since the reviewed annotation run")
                continue
            if d.decision == "rejected":
                rejected_runs[iid].add(d.based_on_run_id)
                applied[iid] = d
                continue
            if run_item.inference_status != InferenceStatus.OK and d.decision == "accepted":
                bad(d, "cannot accept: inference failed for this item in that run")
                continue
        label_file = ws.review / "labels" / f"{iid}.txt"
        if d.decision == "accepted":
            edited = (
                label_file.is_file()
                and iid in exported
                and sha256_bytes(label_file.read_bytes()) != exported[iid]
            )
            if edited:
                bad(d, "accepted, but review/labels was edited — use 'corrected'")
                continue
            boxes = tuple(sorted(run_boxes.get(iid, [])))
            prov, ref = (
                LabelProvenance.HUMAN_REVIEWED_ACCEPTED,
                f"decision:accepted:{d.based_on_run_id}",
            )
        else:  # corrected
            if not label_file.is_file():
                bad(d, "corrected, but review/labels/<item_id>.txt does not exist")
                continue
            data = label_file.read_bytes()
            sha = sha256_bytes(data)
            if d.label_sha256 and d.label_sha256 != sha:
                bad(d, "corrected label file changed since the decision (label_sha256 mismatch)")
                continue
            parsed, err = parse_label(data.decode("utf-8", errors="replace"), len(cfg.classes))
            if err:
                bad(d, f"corrected label is invalid: {err}")
                continue
            boxes = tuple(
                sorted(LabelBox.make(b.source_class_id, b.cx, b.cy, b.w, b.h) for b in parsed)
            )
            prov, ref = LabelProvenance.HUMAN_REVIEWED_CORRECTED, f"decision:corrected:{sha[:12]}"
        evidence[iid].candidates.append(
            LabelCandidate(
                item_id=iid,
                provenance=prov,
                boxes=boxes,
                negative=not boxes,  # a human confirmed there is nothing to label
                source_ref=ref,
                file_sha256=item.sha256,
                width=item.width,
                height=item.height,
                notes=(f"decision_row={d.row}", f"reviewer={d.reviewer}")
                if d.reviewer
                else (f"decision_row={d.row}",),
            )
        )
        applied[iid] = d
    return applied


__all__ = ["LabelPolicy", "LabelResolver"]
