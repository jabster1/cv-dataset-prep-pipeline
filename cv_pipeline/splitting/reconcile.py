"""Reconcile the current split groups with persistent history (pure; no I/O).

For every current group ``G`` the historical splits of its sources are collected
(``H = {history.sources[s].split for s in G}``, inactive sources included):

* ``len(H) == 0`` -> new group, allocated
* ``len(H) == 1`` -> keeps that split: retained / items_changed / reactivated when the
  group id (= source set) is known, otherwise inherited (regrouped or extended)
* ``len(H) >  1`` -> split-history conflict. Only an explicit
  ``split.conflict_resolutions`` entry naming the most protected historical split
  (test > val > train) resolves it; anything else stops the stage.

History is never moved to rebalance proportions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from cv_pipeline.config import ConfigError
from cv_pipeline.dataset_state import SplitInputRow
from cv_pipeline.splitting.allocate import AllocationResult, AllocGroup, allocate
from cv_pipeline.splitting.history import PROTECTION, History, Reason, set_sha256


class Category:
    RETAINED = "retained"
    ITEMS_CHANGED = "items_changed"  # same sources, different items (e.g. re-sampled video)
    REACTIVATED = "reactivated"  # group reappeared after being inactive
    REGROUPED = "regrouped"  # new group id, all sources known (e.g. a divided group)
    EXTENDED = "extended"  # known sources joined by new sources
    RESOLVED = "conflict_resolved"
    ALLOCATED = "allocated"
    RESERVED = "allocated_nonempty"
    RESPLIT = "resplit"


@dataclass(frozen=True)
class CurrentGroup:
    group_id: str
    sources: tuple[str, ...]
    items: tuple[str, ...]

    @property
    def n_items(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class GroupDecision:
    split: str
    reason: str
    first_revision: int
    category: str


@dataclass
class Conflict:
    group_id: str
    historical_splits: list[str]
    allowed_resolution: str
    requested_resolution: str | None
    status: str  # unresolved | rejected
    message: str
    sources: list[dict[str, Any]]
    items_by_historical_split: dict[str, int]


@dataclass
class ReconcileResult:
    groups: dict[str, CurrentGroup]
    decisions: dict[str, GroupDecision] = field(default_factory=dict)
    conflicts: list[Conflict] = field(default_factory=list)
    allocation: AllocationResult | None = None
    applied_resolutions: dict[str, str] = field(default_factory=dict)


def current_groups(
    rows: tuple[SplitInputRow, ...] | list[SplitInputRow],
) -> dict[str, CurrentGroup]:
    sources: dict[str, set[str]] = defaultdict(set)
    items: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        sources[r.split_group].add(r.source_id)
        items[r.split_group].append(r.item_id)
    return {
        gid: CurrentGroup(gid, tuple(sorted(sources[gid])), tuple(sorted(items[gid])))
        for gid in sorted(sources)
    }


def reconcile(
    rows: tuple[SplitInputRow, ...],
    history: History,
    fractions: dict[str, float],
    seed: int,
    require_nonempty: bool,
    conflict_resolutions: dict[str, str],
    revision: int,
    resplit: bool = False,
) -> ReconcileResult:
    groups = current_groups(rows)
    result = ReconcileResult(groups=groups)
    item_source = {r.item_id: r.source_id for r in rows}
    new: list[str] = []

    for gid, g in groups.items():
        if resplit:
            new.append(gid)
            continue
        known = [s for s in g.sources if s in history.sources]
        hist_splits = sorted({history.sources[s].split for s in known}, key=PROTECTION.get)
        if not hist_splits:
            new.append(gid)
        elif len(hist_splits) == 1:
            result.decisions[gid] = _single_history(g, hist_splits[0], history, revision, known)
        else:
            _conflict(g, hist_splits, history, item_source, conflict_resolutions, revision, result)

    if not resplit:
        _check_stale_resolutions(conflict_resolutions, result, history)
    if result.conflicts:
        return result

    fixed: dict[str, int] = defaultdict(int)
    for gid, d in result.decisions.items():
        fixed[d.split] += groups[gid].n_items
    alloc = allocate(
        [AllocGroup(gid, groups[gid].n_items) for gid in new],
        fractions,
        fixed,
        total_items=len(rows),
        seed=seed,
        require_nonempty=require_nonempty,
    )
    for gid, split in alloc.assignments.items():
        if resplit:
            reason, category = Reason.RESPLIT, Category.RESPLIT
        else:
            reason = Reason.ALLOCATED
            category = Category.RESERVED if gid in alloc.reserved else Category.ALLOCATED
        result.decisions[gid] = GroupDecision(split, reason, revision, category)
    result.allocation = alloc
    return result


def _single_history(
    g: CurrentGroup, split: str, history: History, revision: int, known: list[str]
) -> GroupDecision:
    old = history.groups.get(g.group_id)
    if old is not None and old.split == split:
        if old.status == "inactive":
            category = Category.REACTIVATED
        elif old.item_set_sha256 != set_sha256(g.items):
            category = Category.ITEMS_CHANGED
        else:
            category = Category.RETAINED
        return GroupDecision(split, old.reason, old.first_revision, category)
    category = Category.EXTENDED if len(known) < len(g.sources) else Category.REGROUPED
    return GroupDecision(split, Reason.INHERITED, revision, category)


def _conflict(
    g: CurrentGroup,
    hist_splits: list[str],
    history: History,
    item_source: dict[str, str],
    resolutions: dict[str, str],
    revision: int,
    result: ReconcileResult,
) -> None:
    allowed = hist_splits[-1]  # most protected historical split
    requested = resolutions.get(g.group_id)
    if requested == allowed:
        result.decisions[g.group_id] = GroupDecision(
            allowed, Reason.CONFLICT_RESOLUTION, revision, Category.RESOLVED
        )
        result.applied_resolutions[g.group_id] = allowed
        return

    by_split: dict[str, int] = defaultdict(int)
    for item in g.items:
        src = history.sources.get(item_source[item])
        by_split[src.split if src else "new"] += 1
    if requested is None:
        status = "unresolved"
        message = (
            f"Group {g.group_id} now joins sources previously assigned to "
            f"{' and '.join(hist_splits)}. Resolve with split.conflict_resolutions: "
            f"{{{g.group_id}: {allowed}}} or an explicit full resplit."
        )
    else:
        status = "rejected"
        inward = [s for s in hist_splits if PROTECTION[s] > PROTECTION[requested]]
        if inward:
            why = (
                f"it would move historical {'/'.join(inward)} material toward training; "
                "moving held-out data inward requires an explicit full resplit"
            )
        else:
            why = (
                f"targeted resolutions must keep the most protected historical split "
                f"({allowed!r}); other reassignments require an explicit full resplit"
            )
        message = (
            f"Resolution {requested!r} for {g.group_id} rejected: {why}. "
            f"The only accepted targeted resolution is {allowed!r}."
        )
    result.conflicts.append(
        Conflict(
            group_id=g.group_id,
            historical_splits=hist_splits,
            allowed_resolution=allowed,
            requested_resolution=requested,
            status=status,
            message=message,
            sources=[
                {
                    "source_id": s,
                    "historical_split": history.sources[s].split if s in history.sources else None,
                    "historical_group": (
                        history.sources[s].split_group if s in history.sources else None
                    ),
                }
                for s in g.sources
            ],
            items_by_historical_split=dict(sorted(by_split.items())),
        )
    )


def _check_stale_resolutions(
    resolutions: dict[str, str], result: ReconcileResult, history: History
) -> None:
    in_conflict = {c.group_id for c in result.conflicts}
    stale = []
    for gid, split in sorted(resolutions.items()):
        if gid in result.applied_resolutions or gid in in_conflict:
            continue
        old = history.groups.get(gid)
        already_applied = (
            old is not None and old.reason == Reason.CONFLICT_RESOLUTION and old.split == split
        )
        if not already_applied:
            stale.append(gid)
    if stale:
        raise ConfigError(
            f"split.conflict_resolutions has entries for groups that are not in conflict: "
            f"{stale}. Remove them (they would otherwise silently apply to future changes)."
        )
