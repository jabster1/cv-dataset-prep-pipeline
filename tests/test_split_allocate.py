"""Pure tests for whole-group allocation."""

from __future__ import annotations

import random

from cv_pipeline.splitting.allocate import AllocGroup, allocate

F = {"train": 0.8, "val": 0.1, "test": 0.1}


def run(groups, fractions=F, fixed=None, seed=0, nonempty=True):
    total = sum(n for _, n in groups) + sum((fixed or {}).values())
    return allocate(
        [AllocGroup(g, n) for g, n in groups], fractions, fixed or {}, total, seed, nonempty
    )


def counts(result, groups, fixed=None):
    out = {"train": 0, "val": 0, "test": 0, **(fixed or {})}
    sizes = dict(groups)
    for g, s in result.assignments.items():
        out[s] += sizes[g]
    return out


def test_many_small_groups_hit_targets() -> None:
    groups = [(f"g{i:02d}", 1) for i in range(30)]
    r = run(groups)
    assert counts(r, groups) == {"train": 24, "val": 3, "test": 3}
    assert r.reserved == {} and r.unfilled == []


def test_large_group_goes_to_train() -> None:
    groups = [("video", 20)] + [(f"g{i}", 1) for i in range(80)]
    r = run(groups)
    assert r.assignments["video"] == "train"
    c = counts(r, groups)
    assert c["val"] == 10 and c["test"] == 10


def test_five_similar_groups_keep_holdouts_nonempty() -> None:
    groups = [(f"g{i}", 10) for i in range(5)]
    plain = run(groups, nonempty=False)
    assert plain.unfilled == ["test"]  # best-fit alone leaves test empty
    r = run(groups)
    c = counts(r, groups)
    assert c == {"train": 30, "val": 10, "test": 10}
    assert sorted(r.reserved.values()) == ["test", "val"]
    assert r.unfilled == []


def test_reservation_uses_smallest_new_groups() -> None:
    groups = [("big1", 30), ("big2", 30), ("big3", 30), ("small1", 2), ("small2", 3)]
    r = run(groups, fractions={"train": 0.9, "val": 0.05, "test": 0.05})
    if r.reserved:
        assert set(r.reserved) <= {"small1", "small2"}
    c = counts(r, groups)
    assert c["val"] > 0 and c["test"] > 0


def test_fewer_groups_than_nonzero_splits() -> None:
    two = [("a", 5), ("b", 5)]
    r = run(two)
    assert sorted(r.assignments.values()) == ["train", "val"]  # train kept, val before test
    assert r.unfilled == ["test"]
    one = run([("a", 5)])
    assert one.assignments == {"a": "train"} and one.unfilled == ["val", "test"]


def test_incremental_empty_split_without_new_groups_is_left_empty() -> None:
    r = run([], fixed={"train": 40, "val": 5})
    assert r.assignments == {} and r.unfilled == ["test"]


def test_incremental_new_group_fills_empty_split() -> None:
    r = run([("new", 3)], fixed={"train": 40, "val": 5})
    assert r.assignments == {"new": "test"}


def test_zero_fraction_split_never_receives_groups() -> None:
    groups = [(f"g{i}", 1) for i in range(10)]
    r = run(groups, fractions={"train": 0.8, "val": 0.2, "test": 0.0})
    assert "test" not in r.assignments.values()
    assert "test" not in r.unfilled


def test_order_independent_and_seeded() -> None:
    groups = [(f"g{i:02d}", 1 + i % 3) for i in range(25)]
    shuffled = groups[:]
    random.Random(1).shuffle(shuffled)
    assert run(groups).assignments == run(shuffled).assignments
    seeds = {tuple(sorted(run(groups, seed=s).assignments.items())) for s in range(5)}
    assert len(seeds) > 1  # seed changes tie-breaks among equal-size groups
