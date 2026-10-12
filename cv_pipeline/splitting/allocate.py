"""Deterministic whole-group allocation (pure; no I/O).

Groups are indivisible. Groups already placed by history are passed in as fixed counts;
only new groups are allocated:

1. Every new group, largest first (seeded, stable tie-break), goes to the split with the
   smallest increase in target-relative squared error::

       delta_k = ((c_k + n - t_k)^2 - (c_k - t_k)^2) / t_k

   Ties prefer train, then val, then test. Zero-fraction splits never receive groups.
2. ``require_nonempty``: only if step 1 leaves a requested hold-out split empty, allocation
   is redone with a reservation first: every hold-out split that was empty before
   allocation (val first, then test) receives one of the smallest new groups, as long as a
   new group remains for an empty train split (smallest group -> smallest target); then
   step 1 places the rest.

The size of a group is a single number today (its item count). Class-aware stratification
can later generalise ``n`` and ``t_k`` to per-class vectors without changing the output
(group -> split).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

SPLIT_ORDER = ("train", "val", "test")
HOLDOUT_FILL_ORDER = ("val", "test")


@dataclass(frozen=True)
class AllocGroup:
    group_id: str
    n_items: int


@dataclass
class AllocationResult:
    assignments: dict[str, str] = field(default_factory=dict)
    reserved: dict[str, str] = field(default_factory=dict)  # group -> split (non-empty fill)
    unfilled: list[str] = field(default_factory=list)  # non-zero splits left empty
    decisions: list[dict[str, object]] = field(default_factory=list)  # for debug logging


def tie_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"split:{seed}:{group_id}".encode()).hexdigest()


def allocate(
    new_groups: list[AllocGroup],
    fractions: dict[str, float],
    fixed_counts: dict[str, int],
    total_items: int,
    seed: int,
    require_nonempty: bool,
) -> AllocationResult:
    active = [s for s in SPLIT_ORDER if fractions.get(s, 0.0) > 0]
    targets = {s: fractions[s] * total_items for s in active}
    base = {s: fixed_counts.get(s, 0) for s in SPLIT_ORDER}

    plain = _allocate(new_groups, active, targets, base, seed, reserve=[])
    if not require_nonempty or not new_groups:
        return plain
    if not any(s in plain.unfilled for s in HOLDOUT_FILL_ORDER):
        return plain
    # Best-fit starved a requested hold-out: reserve one group for EVERY hold-out that was
    # empty before allocation (reserving only the starved one can starve another).
    empty_before = [s for s in HOLDOUT_FILL_ORDER if s in active and base[s] == 0]
    train_empty = "train" in active and base["train"] == 0
    spare = len(new_groups) - (1 if train_empty else 0)
    to_fill = empty_before[: max(0, min(len(empty_before), spare))]
    if not to_fill:
        return plain
    return _allocate(new_groups, active, targets, base, seed, reserve=to_fill)


def _allocate(
    new_groups: list[AllocGroup],
    active: list[str],
    targets: dict[str, float],
    base: dict[str, int],
    seed: int,
    reserve: list[str],
) -> AllocationResult:
    result = AllocationResult()
    counts = dict(base)
    remaining = list(new_groups)
    if reserve:
        smallest = sorted(remaining, key=lambda g: (g.n_items, tie_key(seed, g.group_id)))
        fill_order = sorted(reserve, key=lambda s: (targets[s], SPLIT_ORDER.index(s)))
        for group, split in zip(smallest, fill_order, strict=False):
            result.assignments[group.group_id] = split
            result.reserved[group.group_id] = split
            counts[split] += group.n_items
            result.decisions.append(
                {
                    "group": group.group_id,
                    "n_items": group.n_items,
                    "split": split,
                    "rule": "require_nonempty",
                }
            )
        remaining = [g for g in remaining if g.group_id not in result.reserved]

    for group in sorted(remaining, key=lambda g: (-g.n_items, tie_key(seed, g.group_id))):
        n = group.n_items
        deltas = {s: (2 * n * (counts[s] - targets[s]) + n * n) / targets[s] for s in active}
        split = min(active, key=lambda s: (deltas[s], SPLIT_ORDER.index(s)))
        result.assignments[group.group_id] = split
        counts[split] += n
        result.decisions.append(
            {
                "group": group.group_id,
                "n_items": n,
                "split": split,
                "rule": "best_fit",
                "deltas": {s: round(d, 6) for s, d in deltas.items()},
            }
        )

    result.unfilled = [s for s in active if counts[s] == 0]
    return result
