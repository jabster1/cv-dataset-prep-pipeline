"""``labels`` command: materialise label resolution for every eligible item, with audit.

Outputs (derived; rewritten each run, deterministic for the same inputs and policy):

    labels/resolved.csv   one row per eligible item: status, provenance, split, legacy splits
    labels/boxes.csv      boxes of every usable label
    reports/labels.{json,md}
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from cv_pipeline.config import PipelineConfig
from cv_pipeline.labels.resolver import LabelPolicy, LabelResolver
from cv_pipeline.labels.schema import Resolution
from cv_pipeline.manifest import (
    DuplicateKind,
    DuplicatePairRecord,
    ItemRecord,
    SourceRefRecord,
    read_records,
    write_records,
)
from cv_pipeline.splitting.history import SplitItemRecord, load_history
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("labels")


@dataclass
class ResolvedLabelRecord:
    item_id: str
    source_id: str
    split_group: str
    split: str  # current pipeline split, if the split matches this dataset state
    legacy_splits: str  # ';'-joined legacy splits of all references (imports)
    status: str
    provenance: str
    n_boxes: int
    negative: bool
    label_sha256: str
    source_ref: str
    reason: str


@dataclass
class ResolvedBoxRecord:
    item_id: str
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    provenance: str


@dataclass
class LabelsResult:
    records: list[ResolvedLabelRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def run_labels(cfg: PipelineConfig, policy: LabelPolicy | None = None) -> LabelsResult:
    policy = policy or LabelPolicy()
    ws = Workspace(cfg.workspace_dir).ensure()
    resolver = LabelResolver.load(cfg, policy)
    resolutions = resolver.resolve_all()

    split_of, split_note = _current_splits(ws, resolver.snapshot_fp)
    legacy = _legacy_splits(ws, resolver)
    records, boxes = [], []
    for res in resolutions:
        d = resolver.state.dedup[res.item_id]
        lab = res.label
        records.append(
            ResolvedLabelRecord(
                item_id=res.item_id,
                source_id=d.source_id,
                split_group=d.split_group,
                split=split_of.get(res.item_id, ""),
                legacy_splits=";".join(sorted(legacy.get(res.item_id, set()))),
                status=res.status,
                provenance=lab.provenance if lab else "",
                n_boxes=len(lab.boxes) if lab else 0,
                negative=bool(lab and lab.negative),
                label_sha256=lab.label_sha256 if lab else "",
                source_ref=lab.source_ref if lab else "",
                reason=res.reason,
            )
        )
        if lab:
            boxes.extend(
                ResolvedBoxRecord(
                    res.item_id,
                    b.class_id,
                    cfg.classes[b.class_id],
                    b.cx,
                    b.cy,
                    b.w,
                    b.h,
                    lab.provenance,
                )
                for b in lab.boxes
            )

    write_records(ws.labels_dir / "resolved.csv", records, ResolvedLabelRecord)
    write_records(ws.labels_dir / "boxes.csv", boxes, ResolvedBoxRecord)
    report = _report(cfg, ws, resolver, policy, records, resolutions, split_note)
    atomic_write_json(ws.reports / "labels.json", report)
    atomic_write_text(ws.reports / "labels.md", _markdown(report))
    log_event(
        logger,
        "labels_resolved",
        items=len(records),
        **{f"status_{k}": v for k, v in sorted(Counter(r.status for r in records).items())},
    )
    return LabelsResult(records, report)


def _current_splits(ws: Workspace, snapshot_fp: str) -> tuple[dict[str, str], str]:
    if not ws.split_items_manifest.is_file():
        return {}, "no split yet (run `split`)"
    last = load_history(ws).last_revision
    if last is None or last.get("snapshot_fp") != snapshot_fp:
        return {}, "the split was made from a different dataset state (re-run `split`)"
    items = read_records(ws.split_items_manifest, SplitItemRecord)
    return {i.item_id: i.split for i in items}, f"split revision {last['revision']}"


def _legacy_splits(ws: Workspace, resolver: LabelResolver) -> dict[str, set[str]]:
    if not ws.source_refs_manifest.is_file():
        return {}
    by_source: dict[str, set[str]] = defaultdict(set)
    for r in read_records(ws.source_refs_manifest, SourceRefRecord):
        if r.legacy_split:
            by_source[r.source_id].add(r.legacy_split)
    out: dict[str, set[str]] = {}
    for iid, members in resolver.members.items():
        splits: set[str] = set()
        for m in members:
            splits |= by_source.get(resolver.state.items[m].source_id, set())
        if splits:
            out[iid] = splits
    return out


def _report(
    cfg: PipelineConfig,
    ws: Workspace,
    resolver: LabelResolver,
    policy: LabelPolicy,
    records: list[ResolvedLabelRecord],
    resolutions: list[Resolution],
    split_note: str,
) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    crosstab: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        by_split[r.split or "unassigned"][r.status] += 1
        for legacy in r.legacy_splits.split(";") if r.legacy_splits else ["(none)"]:
            crosstab[legacy][r.split or "unassigned"] += 1

    origins = {i.item_id: i.origin for i in read_records(ws.items_manifest, ItemRecord)}
    cross_origin_near = 0
    if ws.duplicate_pairs_manifest.is_file():
        for p in read_records(ws.duplicate_pairs_manifest, DuplicatePairRecord):
            if p.kind == DuplicateKind.NEAR_DUPLICATE and origins.get(p.item_a) != origins.get(
                p.item_b
            ):
                cross_origin_near += 1

    warnings = []
    if resolver.invalid_decisions:
        warnings.append(
            f"{len(resolver.invalid_decisions)} review decision(s) are invalid and were ignored."
        )
    conflicts = [r for r in resolutions if r.status == "conflict"]
    if conflicts:
        warnings.append(f"{len(conflicts)} item(s) have conflicting labels and get no label.")
    if cross_origin_near:
        warnings.append(
            f"{cross_origin_near} near-duplicate pair(s) link imported and other-origin images. "
            "They share a split group (no leakage) but both count as training examples unless "
            "dedup.near_duplicate_action is 'exclude'."
        )
    return {
        "policy": policy.as_dict(),
        "snapshot_fp": resolver.snapshot_fp,
        "annotation_lineage": resolver.annotation,
        "split": split_note,
        "items": len(records),
        "by_status": dict(sorted(Counter(r.status for r in records).items())),
        "by_provenance": dict(sorted(Counter(r.provenance or "-" for r in records).items())),
        "unavailable_reasons": dict(
            sorted(Counter(r.reason for r in records if r.status == "unavailable").items())
        ),
        "negatives": dict(sorted(Counter(r.provenance for r in records if r.negative).items())),
        "by_split": {k: dict(sorted(v.items())) for k, v in sorted(by_split.items())},
        "legacy_vs_current_split": {
            k: dict(sorted(v.items())) for k, v in sorted(crosstab.items())
        },
        "conflicts": [
            {"item_id": r.item_id, "reason": r.reason, "candidates": list(r.candidates)}
            for r in conflicts
        ],
        "invalid_decisions": [d.__dict__ for d in resolver.invalid_decisions],
        "applied_decisions": dict(
            sorted(Counter(d.decision for d in resolver.applied_decisions.values()).items())
        ),
        "cross_origin_near_duplicate_pairs": cross_origin_near,
        "warnings": warnings,
    }


def _markdown(rep: dict[str, Any]) -> str:
    lines = [
        "# Label resolution report",
        "",
        f"- Policy: {rep['policy']}",
        f"- Items: {rep['items']} (snapshot `{rep['snapshot_fp'][:16]}`)",
        f"- Split: {rep['split']}",
        f"- Annotation run: {rep['annotation_lineage'].get('status')}",
        "",
        "Trusted = human provenance (imported human labels, accepted or corrected reviews). "
        "`opt_in` labels are used only because the policy explicitly allowed them.",
        "",
        "| Status | Items |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rep["by_status"].items()]
    lines += ["", "| Provenance | Items |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rep["by_provenance"].items()]
    if rep["unavailable_reasons"]:
        lines += ["", "| Unavailable because | Items |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rep["unavailable_reasons"].items()]
    if rep["legacy_vs_current_split"]:
        cols = sorted({c for v in rep["legacy_vs_current_split"].values() for c in v})
        lines += [
            "",
            "## Legacy split vs current split (audit only)",
            "",
            "| Legacy \\ current | " + " | ".join(cols) + " |",
            "|---|" + "---|" * len(cols),
        ]
        for k, v in rep["legacy_vs_current_split"].items():
            lines.append(f"| {k} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |")
    if rep["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- ⚠️ {w}" for w in rep["warnings"]]
    if rep["invalid_decisions"]:
        lines += [
            "",
            "## Invalid review decisions",
            "",
            "| Row | Item | Problem |",
            "|---|---|---|",
        ]
        lines += [
            f"| {d['row']} | {d['item_id']} | {d['problem']} |" for d in rep["invalid_decisions"]
        ]
    return "\n".join(lines) + "\n"
