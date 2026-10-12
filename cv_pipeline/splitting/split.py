"""Split stage: validated dataset state + persistent history -> train/val/test assignment.

    current items.csv + dedup.csv
      -> validate consistency/freshness (cv_pipeline.dataset_state)
      -> split-input snapshot + fingerprint
      -> reconcile with persistent history (retain / inherit / conflict)
      -> allocate only new groups (deterministic best-fit)
      -> new revision (only if something changed) + materialised splits/items.csv + report

Priorities: prevent leakage > preserve historical assignments > match proportions.
Splitting never reads machine predictions; a current annotation run is recorded only as
optional lineage.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from cv_pipeline.annotation.schema import AnnotationItemRecord
from cv_pipeline.config import PipelineConfig
from cv_pipeline.dataset_state import SplitInput, SplitInputRow, load_split_input
from cv_pipeline.manifest import read_records, write_records
from cv_pipeline.splitting.history import (
    GroupHistoryRecord,
    History,
    SourceHistoryRecord,
    SplitItemRecord,
    archive_history,
    load_history,
    save_history,
    set_sha256,
    state_digest,
)
from cv_pipeline.splitting.reconcile import ReconcileResult, current_groups, reconcile
from cv_pipeline.splitting.report import build_report, render_markdown
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("split")

EXIT_SPLIT_CONFLICT = 4
TOKEN_LENGTH = 12


class SplitError(RuntimeError):
    """The split stage refused to run (e.g. an invalid resplit request)."""


@dataclass
class SplitResult:
    exit_code: int
    revision: int | None
    new_revision: bool
    written: bool
    items: list[SplitItemRecord] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    resplit_token: str | None = None


def run_split_stage(cfg: PipelineConfig, force: bool = False) -> SplitResult:
    """Stage entry point for ``prepare`` (incremental only; never resplits)."""
    return run_split(cfg)


def run_split(
    cfg: PipelineConfig,
    dry_run: bool = False,
    resplit: bool = False,
    reason: str | None = None,
    confirm: str | None = None,
) -> SplitResult:
    ws = Workspace(cfg.workspace_dir).ensure()
    inp = load_split_input(cfg)
    history = load_history(ws)
    lineage = annotation_lineage(ws, inp)
    seed = cfg.split.seed if cfg.split.seed is not None else cfg.project.seed
    fractions = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    revision = history.next_revision

    token = None
    if resplit:
        if history.empty:
            raise SplitError("There is no split history to resplit; run `split` normally.")
        if not reason or not reason.strip():
            raise SplitError("--resplit requires a non-empty --reason.")
        token = history.state_digest()[:TOKEN_LENGTH]

    rec = reconcile(
        inp.rows,
        history,
        fractions,
        seed,
        cfg.split.require_nonempty,
        dict(cfg.split.conflict_resolutions),
        revision,
        resplit=resplit,
    )
    if rec.allocation is not None:
        for d in rec.allocation.decisions:
            log_event(logger, "allocation_decision", logging.DEBUG, **d)

    if rec.conflicts:
        report = build_report(cfg, inp, history, rec, lineage, None, seed, None, None)
        if not dry_run:
            _write_report(ws, report)
        for c in rec.conflicts:
            log_event(logger, "split_conflict", logging.ERROR, group=c.group_id, detail=c.message)
        return SplitResult(EXIT_SPLIT_CONFLICT, None, False, False, report=report)

    groups, sources, items = _build_state(rec, inp, history, revision, resplit)
    new_digest = state_digest(groups, sources)
    last = history.last_revision
    changed = (
        resplit
        or last is None
        or new_digest != last["state_digest"]
        or inp.snapshot_fp != last.get("snapshot_fp")
    ) and bool(inp.rows or not history.empty)

    movement = _movement(history, inp.rows, rec) if resplit else None
    revision_record = None
    if changed:
        mode = "resplit" if resplit else ("initial" if history.empty else "incremental")
        revision_record = _revision_record(
            revision, mode, inp, lineage, cfg, seed, rec, items, new_digest, reason, movement
        )

    if resplit and confirm != token:
        report = build_report(
            cfg, inp, history, rec, lineage, revision_record, seed, items, movement
        )
        report["resplit_preview"] = {"token": token, "confirmed": False}
        if confirm is not None:
            raise SplitError(
                "Confirmation token does not match the current split history; re-run "
                "`split --resplit --reason ...` without --confirm to see the current token."
            )
        log_event(
            logger,
            "resplit_preview",
            logging.WARNING,
            token=token,
            moved=movement,
            hint="nothing written; re-run with --confirm TOKEN to apply",
        )
        return SplitResult(0, None, False, False, items, report, token)

    report = build_report(cfg, inp, history, rec, lineage, revision_record, seed, items, movement)
    if dry_run:
        log_event(logger, "split_dry_run", revision=revision if changed else None)
        return SplitResult(0, revision if changed else None, changed, False, items, report)

    if changed:
        assert revision_record is not None
        if resplit:
            revision_record["resplit"]["archive"] = archive_history(ws, history.next_revision - 1)
            report["revision"]["record"] = revision_record
        save_history(ws, groups, sources, items, [*history.revisions, revision_record])
    else:
        write_records(ws.split_items_manifest, items, SplitItemRecord)
    _write_report(ws, report)

    log_event(
        logger,
        "split_finished",
        revision=revision if changed else (last or {}).get("revision"),
        new_revision=changed,
        items=len(items),
        **{f"items_{k}": v for k, v in Counter(i.split for i in items).items()},
    )
    for w in report["warnings"]:
        log_event(logger, "split_warning", logging.WARNING, detail=w)
    final_revision = revision if changed else (last or {}).get("revision")
    return SplitResult(0, final_revision, changed, True, items, report)


# --------------------------------------------------------------------------------------


def annotation_lineage(ws: Workspace, inp: SplitInput) -> dict[str, Any]:
    """Optional: is the active annotation run based on exactly this dataset state?"""
    if not ws.active_annotation_run.is_file():
        return {"status": "none"}
    try:
        run_id = json.loads(ws.active_annotation_run.read_text(encoding="utf-8"))["run_id"]
        run_dir = ws.annotation_runs / run_id
        identity = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["identity"]
        if identity.get("run_id") != run_id:
            return {"status": "unavailable", "run_id": run_id}
        recs = read_records(run_dir / "items.csv", AnnotationItemRecord)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return {"status": "unavailable"}
    projected = tuple(
        SplitInputRow(
            r.item_id, r.source_id, r.source_type, r.frame_index, r.split_group, r.file_sha256
        )
        for r in sorted(recs, key=lambda r: r.item_id)
    )
    return {"status": "current" if projected == inp.rows else "stale", "run_id": run_id}


def _build_state(
    rec: ReconcileResult,
    inp: SplitInput,
    history: History,
    revision: int,
    resplit: bool,
) -> tuple[list[GroupHistoryRecord], list[SourceHistoryRecord], list[SplitItemRecord]]:
    old_groups = {} if resplit else history.groups
    old_sources = {} if resplit else history.sources

    def stamp(new: Any, old: Any) -> Any:
        """Keep old last_revision if nothing else changed; otherwise stamp this revision."""
        if old is not None:
            new.last_revision = old.last_revision
            if new == old:
                return new
        new.last_revision = revision
        return new

    groups: list[GroupHistoryRecord] = []
    sources: list[SourceHistoryRecord] = []
    for gid, g in rec.groups.items():
        d = rec.decisions[gid]
        groups.append(
            stamp(
                GroupHistoryRecord(
                    split_group=gid,
                    split=d.split,
                    status="active",
                    n_sources=len(g.sources),
                    source_set_sha256=set_sha256(g.sources),
                    n_items=g.n_items,
                    item_set_sha256=set_sha256(g.items),
                    reason=d.reason,
                    first_revision=d.first_revision,
                    last_revision=revision,
                ),
                old_groups.get(gid),
            )
        )
        for s in g.sources:
            old = old_sources.get(s)
            first = old.first_revision if old is not None and old.split == d.split else revision
            sources.append(
                stamp(SourceHistoryRecord(s, gid, d.split, "active", first, revision), old)
            )

    current_ids = set(rec.groups)
    for gid, old in sorted(old_groups.items()):
        if gid not in current_ids:
            row = GroupHistoryRecord(**{**old.__dict__, "status": "inactive"})
            groups.append(stamp(row, old))
    current_sources = {s for g in rec.groups.values() for s in g.sources}
    for sid, old in sorted(old_sources.items()):
        if sid not in current_sources:
            row = SourceHistoryRecord(**{**old.__dict__, "status": "inactive"})
            sources.append(stamp(row, old))

    items = [
        SplitItemRecord(
            item_id=r.item_id,
            source_id=r.source_id,
            source_type=r.source_type,
            split_group=r.split_group,
            split=rec.decisions[r.split_group].split,
            reason=rec.decisions[r.split_group].reason,
            assigned_in_revision=rec.decisions[r.split_group].first_revision,
        )
        for r in inp.rows
    ]
    return groups, sources, items


def _movement(
    history: History, rows: tuple[SplitInputRow, ...], rec: ReconcileResult
) -> dict[str, int]:
    moved: dict[str, int] = defaultdict(int)
    for r in rows:
        old = history.sources.get(r.source_id)
        new = rec.decisions[r.split_group].split
        key = f"{old.split}->{new}" if old else f"new->{new}"
        moved[key] += 1
    return dict(sorted(moved.items()))


def _revision_record(
    revision: int,
    mode: str,
    inp: SplitInput,
    lineage: dict[str, Any],
    cfg: PipelineConfig,
    seed: int,
    rec: ReconcileResult,
    items: list[SplitItemRecord],
    digest_: str,
    reason: str | None,
    movement: dict[str, int] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "revision": revision,
        "mode": mode,
        "snapshot_fp": inp.snapshot_fp,
        "dedup_settings_fp": inp.dedup_settings_fp,
        "n_items": inp.n_items,
        "n_groups": len(rec.groups),
        "annotation_lineage": lineage,
        "config": {
            "fractions": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
            "seed": seed,
            "require_nonempty": cfg.split.require_nonempty,
        },
        "changes": dict(sorted(Counter(d.category for d in rec.decisions.values()).items())),
        "conflict_resolutions_applied": dict(sorted(rec.applied_resolutions.items())),
        "items_by_split": dict(sorted(Counter(i.split for i in items).items())),
        "state_digest": digest_,
    }
    if mode == "resplit":
        record["resplit"] = {"reason": reason, "moved": movement, "archive": None}
    return record


def _write_report(ws: Workspace, report: dict[str, Any]) -> None:
    atomic_write_json(ws.reports / "split.json", report)
    atomic_write_text(ws.reports / "split.md", render_markdown(report))


__all__ = [
    "EXIT_SPLIT_CONFLICT",
    "SplitError",
    "SplitResult",
    "annotation_lineage",
    "current_groups",
    "run_split",
    "run_split_stage",
]
