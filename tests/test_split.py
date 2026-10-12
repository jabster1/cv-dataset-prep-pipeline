"""Split stage tests (no annotation required)."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from cv_pipeline.annotation import run_annotate
from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError, load_config
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.dedup import run_dedup
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import DedupRecord, ItemRecord, read_records, write_records
from cv_pipeline.splitting import EXIT_SPLIT_CONFLICT, SplitError, SplitHistoryError, run_split
from cv_pipeline.splitting.history import (
    SplitItemRecord,
    load_history,
    save_history,
)
from cv_pipeline.synthetic import render_scene, write_synthetic_video
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes import det

F811 = {"split": {"train": 0.8, "val": 0.1, "test": 0.1}}


def scene(seed: int) -> np.ndarray:
    return render_scene(np.random.default_rng(seed))[0]


def save(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(buf.tobytes())
    return path


class Project:
    def __init__(self, tmp_path: Path, write_config) -> None:
        self.tmp, self._write = tmp_path, write_config
        self.raw = tmp_path / "raw" / "images"
        self.raw.mkdir(parents=True)

    def config(self, overrides: dict[str, Any] | None = None, name: str = "config.yaml"):
        return load_config(self._write(deep_update(F811, overrides or {}), name=name))

    def prepare(self, overrides: dict[str, Any] | None = None):
        cfg = self.config(overrides)
        run_ingest(cfg)
        run_dedup(cfg)
        return cfg

    def images(self, seeds) -> None:
        for s in seeds:
            save(self.raw / f"s{s:03d}.png", scene(s))

    def video(self, name: str, frames: int, seed: int = 0) -> None:
        write_synthetic_video(
            self.tmp / "raw" / "videos" / name, frames, 10.0, np.random.default_rng(seed)
        )


@pytest.fixture
def proj(tmp_path, write_config) -> Project:
    return Project(tmp_path, write_config)


def ws_of(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def splits_by_item(cfg) -> dict[str, SplitItemRecord]:
    return {r.item_id: r for r in read_records(ws_of(cfg).split_items_manifest, SplitItemRecord)}


def source_paths(cfg) -> dict[str, str]:
    return {
        i.item_id: i.source_path.removeprefix("images/")
        for i in read_records(ws_of(cfg).items_manifest, ItemRecord)
    }


def assert_groups_intact(items) -> None:
    by_group: dict[str, set[str]] = defaultdict(set)
    by_source: dict[str, set[str]] = defaultdict(set)
    for r in items:
        by_group[r.split_group].add(r.split)
        by_source[r.source_id].add(r.split)
    assert all(len(v) == 1 for v in by_group.values())
    assert all(len(v) == 1 for v in by_source.values())


def split_files(cfg) -> dict[str, bytes]:
    root = ws_of(cfg).splits
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def force_history_split(cfg, source_ids: set[str], split: str) -> None:
    """Test helper: simulate a historical assignment by rewriting split history."""
    ws = ws_of(cfg)
    h = load_history(ws)
    groups_hit = set()
    for s in source_ids:
        h.sources[s].split = split
        groups_hit.add(h.sources[s].split_group)
    for g in groups_hit:
        h.groups[g].split = split
    items = [
        SplitItemRecord(**{**r.__dict__, "split": split}) if r.source_id in source_ids else r
        for r in read_records(ws.split_items_manifest, SplitItemRecord)
    ]
    revisions = list(h.revisions)
    revisions[-1] = {**revisions[-1], "state_digest": h.state_digest()}
    save_history(ws, list(h.groups.values()), list(h.sources.values()), items, revisions)


# --------------------------------------------------------------------------------------
# Group integrity and proportions
# --------------------------------------------------------------------------------------


def test_video_frames_and_near_duplicates_stay_together(proj) -> None:
    proj.images(range(10))
    base = scene(99)
    save(proj.raw / "near_a.png", base)
    save(proj.raw / "near_b.png", cv2.convertScaleAbs(base, beta=12))
    proj.video("clip.mp4", 20)
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 2}}})

    result = run_split(cfg)
    assert result.exit_code == 0 and result.revision == 1
    items = result.items
    assert_groups_intact(items)
    frames = [r for r in items if r.source_type == "video_frame"]
    assert len(frames) == 10 and len({r.split for r in frames}) == 1
    paths = source_paths(cfg)
    near = [r for r in items if paths[r.item_id] in {"near_a.png", "near_b.png"}]
    assert len(near) == 2 and near[0].split_group == near[1].split_group
    assert near[0].split == near[1].split


def test_normal_80_10_10(proj) -> None:
    proj.images(range(30))
    cfg = proj.prepare()
    report = run_split(cfg).report
    assert {s: report["splits"][s]["items"] for s in ("train", "val", "test")} == {
        "train": 24,
        "val": 3,
        "test": 3,
    }


def test_group_larger_than_test_fraction_is_not_split(proj) -> None:
    proj.images(range(4))  # 14 items: targets 11.2 / 1.4 / 1.4 cannot be met exactly
    proj.video("big.mp4", 10)
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1}}})
    result = run_split(cfg)
    frames = [r for r in result.items if r.source_type == "video_frame"]
    assert {r.split for r in frames} == {"train"}
    report = result.report
    big = [g for g in report["largest_groups"] if g["exceeds_smallest_holdout_target"]]
    assert big and big[0]["items"] == 10
    assert any("cannot be split" in e for e in report["explanation"])
    assert report["splits"]["val"]["items"] > 0 and report["splits"]["test"]["items"] > 0
    assert any(report["splits"][s]["deviation_items"] != 0 for s in report["splits"])


def test_zero_fraction_split(proj) -> None:
    proj.images(range(10))
    cfg = proj.prepare({"split": {"train": 0.8, "val": 0.2, "test": 0.0}})
    result = run_split(cfg)
    assert {r.split for r in result.items} == {"train", "val"}
    assert not any("'test'" in w for w in result.report["warnings"])


# --------------------------------------------------------------------------------------
# Small datasets and require_nonempty
# --------------------------------------------------------------------------------------


def test_empty_dataset(proj) -> None:
    cfg = proj.prepare()
    result = run_split(cfg)
    assert result.exit_code == 0 and result.items == [] and result.revision is None
    assert "No eligible items" in " ".join(result.report["warnings"])
    assert not ws_of(cfg).split_revisions.exists()


def test_one_group_only(proj) -> None:
    proj.images([1])
    cfg = proj.prepare()
    result = run_split(cfg)
    assert [r.split for r in result.items] == ["train"]
    warnings = " ".join(result.report["warnings"])
    assert "'val'" in warnings and "'test'" in warnings and "too few groups" in warnings


def test_three_groups_fill_all_splits(proj) -> None:
    proj.images([1, 2, 3])
    cfg = proj.prepare()
    result = run_split(cfg)
    assert sorted(r.split for r in result.items) == ["test", "train", "val"]


def test_five_similar_groups_give_nonempty_holdouts(proj) -> None:
    for i in range(5):
        proj.video(f"v{i}.mp4", 10, seed=i)
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1}}})
    report = run_split(cfg).report
    assert report["splits"]["val"]["items"] == 10 and report["splits"]["test"]["items"] == 10
    assert len(report["reserved_for_nonempty"]) == 2
    off = run_split(
        proj.prepare(
            {
                "split": {"require_nonempty": False},
                "project": {"workspace": "work_off"},
                "ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1}},
            }
        )
    )
    assert off.report["splits"]["test"]["items"] == 0
    assert any("require_nonempty is false" in w for w in off.report["warnings"])


def test_incremental_empty_split_without_safe_new_group(proj) -> None:
    proj.images([1, 2])
    cfg = proj.prepare({"split": {"require_nonempty": False}})
    first = run_split(cfg)
    assert {r.split for r in first.items} == {"train"}

    cfg = proj.config()  # require_nonempty back on, no new data
    second = run_split(cfg)
    assert {r.split for r in second.items} == {"train"}  # history never moved
    assert not second.new_revision
    warnings = " ".join(second.report["warnings"])
    assert "historical assignments are never moved" in warnings

    proj.images([3])
    cfg = proj.prepare()
    third = run_split(cfg)
    new_item = next(r for r in third.items if r.assigned_in_revision == 2)
    assert new_item.split == "val"  # first empty requested hold-out gets the new group
    assert {r.item_id: r.split for r in first.items}.items() <= {
        r.item_id: r.split for r in third.items
    }.items()


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_deterministic_fresh_runs_and_idempotent_rerun(proj) -> None:
    proj.images(range(12))
    proj.video("v.mp4", 8)
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 2}}})
    run_split(cfg)
    files = split_files(cfg)
    report = (ws_of(cfg).reports / "split.json").read_bytes()

    again = run_split(cfg)
    assert not again.new_revision and split_files(cfg) == files

    cfg2 = proj.prepare(
        {
            "project": {"workspace": "work2"},
            "ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 2}},
        }
    )
    run_split(cfg2)
    assert split_files(cfg2) == files
    assert (ws_of(cfg2).reports / "split.json").read_bytes() == report


def test_manifest_row_order_does_not_matter(proj) -> None:
    proj.images(range(15))
    cfg = proj.prepare()
    ws = ws_of(cfg)
    items_bytes = ws.items_manifest.read_bytes()
    dedup_bytes = ws.dedup_manifest.read_bytes()
    run_split(cfg)
    files = split_files(cfg)

    shutil.rmtree(ws.splits)
    for path in (ws.items_manifest, ws.dedup_manifest):
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        random.Random(3).shuffle(body)
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerows([header, *body])
    assert ws.items_manifest.read_bytes() != items_bytes
    assert ws.dedup_manifest.read_bytes() != dedup_bytes
    run_split(cfg)
    assert split_files(cfg) == files


# --------------------------------------------------------------------------------------
# Incremental history
# --------------------------------------------------------------------------------------


def test_new_data_keeps_existing_assignments(proj) -> None:
    proj.images(range(20))
    cfg = proj.prepare()
    first = {r.item_id: r for r in run_split(cfg).items}

    proj.images(range(20, 30))
    cfg = proj.prepare()
    second = run_split(cfg)
    assert second.revision == 2 and second.new_revision
    now = {r.item_id: r for r in second.items}
    for iid, r in first.items():
        assert now[iid].split == r.split and now[iid].assigned_in_revision == 1
    new = [r for iid, r in now.items() if iid not in first]
    assert len(new) == 10 and {r.assigned_in_revision for r in new} == {2}
    changes = second.report["changes"]
    assert changes["retained"]["count"] == 20
    assert sum(changes[k]["count"] for k in changes if k.startswith("allocated")) == 10


def test_group_growth_inherits_split(proj) -> None:
    base = scene(50)
    save(proj.raw / "a.png", base)
    proj.images(range(10))
    cfg = proj.prepare()
    first = run_split(cfg)
    a_item = next(i for i, p in source_paths(cfg).items() if p == "a.png")
    a_split = next(r.split for r in first.items if r.item_id == a_item)

    save(proj.raw / "b.png", cv2.convertScaleAbs(base, beta=12))  # near-duplicate of a
    cfg = proj.prepare()
    second = run_split(cfg)
    paths = source_paths(cfg)
    ab = [r for r in second.items if paths[r.item_id] in {"a.png", "b.png"}]
    assert {r.split for r in ab} == {a_split}
    assert ab[0].split_group.startswith("sg_")
    assert second.report["changes"]["extended"]["count"] == 1
    assert {r.reason for r in ab} == {"inherited"}


def test_divided_group_keeps_split(proj) -> None:
    base = scene(50)
    save(proj.raw / "a.png", base)
    save(proj.raw / "b.png", cv2.convertScaleAbs(base, beta=12))
    proj.images(range(10))
    cfg = proj.prepare()
    first = run_split(cfg)
    paths = source_paths(cfg)
    ab_split = {r.split for r in first.items if paths[r.item_id] in {"a.png", "b.png"}}
    assert len(ab_split) == 1

    cfg = proj.prepare({"dedup": {"group_near_duplicates": False}})
    second = run_split(cfg)
    assert second.exit_code == 0
    ab = [r for r in second.items if paths[r.item_id] in {"a.png", "b.png"}]
    assert len({r.split_group for r in ab}) == 2
    assert {r.split for r in ab} == ab_split
    assert second.report["changes"]["regrouped"]["count"] == 2


def test_vanished_group_is_kept_and_restored(proj) -> None:
    proj.images(range(10))
    cfg = proj.prepare()
    first = {r.item_id: r.split for r in run_split(cfg).items}
    gone = proj.raw / "s003.png"
    keep = gone.read_bytes()
    gone_item = next(i for i, p in source_paths(cfg).items() if p == "s003.png")
    gone.unlink()

    cfg = proj.prepare()
    second = run_split(cfg)
    assert gone_item not in {r.item_id for r in second.items}
    hist = load_history(ws_of(cfg))
    assert hist.groups[gone_item].status == "inactive"
    assert hist.groups[gone_item].split == first[gone_item]

    gone.write_bytes(keep)
    cfg = proj.prepare()
    third = run_split(cfg)
    restored = next(r for r in third.items if r.item_id == gone_item)
    assert restored.split == first[gone_item]
    assert third.report["changes"]["reactivated"]["count"] == 1


def test_resampled_video_keeps_split(proj) -> None:
    proj.images(range(10))
    proj.video("v.mp4", 20)
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 5}}})
    first = run_split(cfg)
    vsplit = {r.split for r in first.items if r.source_type == "video_frame"}

    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 2}}})
    second = run_split(cfg)
    frames = [r for r in second.items if r.source_type == "video_frame"]
    assert len(frames) == 10 and {r.split for r in frames} == vsplit
    assert second.report["changes"]["items_changed"]["count"] == 1


# --------------------------------------------------------------------------------------
# Conflicts
# --------------------------------------------------------------------------------------


def _conflict_setup(proj, split_a: str, split_b: str):
    base = scene(50)
    save(proj.raw / "a.png", base)
    save(proj.raw / "b.png", cv2.convertScaleAbs(base, beta=12))
    proj.images(range(10))
    cfg = proj.prepare({"dedup": {"group_near_duplicates": False}})
    run_split(cfg)
    paths = source_paths(cfg)
    items = {r.item_id: r for r in read_records(ws_of(cfg).items_manifest, ItemRecord)}
    a = next(items[i].source_id for i, p in paths.items() if p == "a.png")
    b = next(items[i].source_id for i, p in paths.items() if p == "b.png")
    force_history_split(cfg, {a}, split_a)
    force_history_split(cfg, {b}, split_b)
    cfg = proj.prepare()  # grouping on: a and b merge
    merged = {
        d.split_group
        for d in read_records(ws_of(cfg).dedup_manifest, DedupRecord)
        if d.source_id in {a, b}
    }
    (gid,) = merged
    return cfg, gid, a, b


def test_train_test_merge_conflict_stops_without_mutation(proj) -> None:
    cfg, gid, a, b = _conflict_setup(proj, "train", "test")
    before = split_files(cfg)
    result = run_split(cfg)
    assert result.exit_code == EXIT_SPLIT_CONFLICT
    assert split_files(cfg) == before
    (conflict,) = result.report["conflicts"]
    assert conflict["group_id"] == gid
    assert conflict["historical_splits"] == ["train", "test"]
    assert conflict["allowed_resolution"] == "test" and conflict["status"] == "unresolved"
    md = (ws_of(cfg).reports / "split.md").read_text()
    assert "Split-history conflicts" in md and gid in md

    cfg_path = cfg.root / "config.yaml"
    assert main(["split", "--config", str(cfg_path), "--log-level", "CRITICAL"]) == 4


@pytest.mark.parametrize(
    "hist, allowed",
    [(("train", "test"), "test"), (("train", "val"), "val"), (("val", "test"), "test")],
)
def test_targeted_resolution_only_toward_protection(proj, hist, allowed) -> None:
    cfg, gid, a, b = _conflict_setup(proj, *hist)
    for bad in {"train", "val", "test"} - {allowed}:
        rejected = run_split(proj.config({"split": {"conflict_resolutions": {gid: bad}}}))
        assert rejected.exit_code == EXIT_SPLIT_CONFLICT
        assert rejected.report["conflicts"][0]["status"] == "rejected"
        assert "rejected" in rejected.report["conflicts"][0]["message"]

    cfg = proj.config({"split": {"conflict_resolutions": {gid: allowed}}})
    resolved = run_split(cfg)
    assert resolved.exit_code == 0
    group_items = [r for r in resolved.items if r.split_group == gid]
    assert {r.split for r in group_items} == {allowed}
    assert {r.reason for r in group_items} == {"conflict_resolution"}
    rev = load_history(ws_of(cfg)).last_revision
    assert rev["conflict_resolutions_applied"] == {gid: allowed}

    again = run_split(cfg)  # entry stays valid once applied
    assert again.exit_code == 0 and not again.new_revision


def test_unsafe_resolution_message(proj) -> None:
    cfg, gid, a, b = _conflict_setup(proj, "train", "test")
    result = run_split(proj.config({"split": {"conflict_resolutions": {gid: "train"}}}))
    assert (
        "move historical test material toward training" in result.report["conflicts"][0]["message"]
    )


def test_stale_conflict_resolution_is_config_error(proj) -> None:
    proj.images(range(5))
    cfg = proj.prepare()
    run_split(cfg)
    stale = proj.config({"split": {"conflict_resolutions": {"sg_doesnotexist": "test"}}})
    with pytest.raises(ConfigError, match="not in conflict"):
        run_split(stale)
    path = stale.root / "config.yaml"
    assert main(["split", "--config", str(path), "--log-level", "CRITICAL"]) == 2


# --------------------------------------------------------------------------------------
# Upstream state
# --------------------------------------------------------------------------------------


def test_stale_dedup_state_refuses(proj) -> None:
    proj.images(range(5))
    cfg = proj.prepare()
    changed_cfg = proj.config({"dedup": {"near_duplicate_threshold": 3}})
    with pytest.raises(DatasetStateError, match="different dedup settings"):
        run_split(changed_cfg)

    proj.images([7])
    run_ingest(cfg)  # no dedup afterwards
    with pytest.raises(DatasetStateError, match="re-run the dedup stage"):
        run_split(cfg)
    run_dedup(cfg)

    ws = ws_of(cfg)
    items = read_records(ws.items_manifest, ItemRecord)
    items[0].sha256 = "f" * 64  # e.g. a frame re-encoded by a different OpenCV
    write_records(ws.items_manifest, items, ItemRecord)
    with pytest.raises(DatasetStateError, match="different file content"):
        run_split(cfg)
    assert (
        main(["split", "--config", str(cfg.root / "config.yaml"), "--log-level", "CRITICAL"]) == 1
    )


def test_split_without_annotation_via_prepare(proj) -> None:
    proj.images(range(6))
    path = proj._write(deep_update(F811, {"annotation": {"enabled": False}}))
    assert main(["prepare", "--config", str(path), "--log-level", "CRITICAL"]) == 0
    cfg = load_config(path)
    ws = ws_of(cfg)
    assert ws.split_items_manifest.is_file()
    assert not ws.active_annotation_run.exists()
    rev = load_history(ws).last_revision
    assert rev["annotation_lineage"] == {"status": "none"}


def test_annotation_lineage_current_then_stale(proj, scripted) -> None:
    base = scene(50)
    save(proj.raw / "a.png", base)
    save(proj.raw / "b.png", cv2.convertScaleAbs(base, beta=12))
    proj.images(range(6))
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    ann = {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}}
    cfg = proj.prepare(ann)
    run_id = run_annotate(cfg).run_id
    current = run_split(cfg)
    assert current.report["annotation_lineage"] == {"status": "current", "run_id": run_id}
    assert load_history(ws_of(cfg)).last_revision["annotation_lineage"]["run_id"] == run_id

    cfg = proj.prepare(deep_update(ann, {"dedup": {"group_near_duplicates": False}}))
    stale = run_split(cfg)  # annotation not refreshed: split still proceeds
    assert stale.exit_code == 0
    assert stale.report["annotation_lineage"] == {"status": "stale", "run_id": run_id}
    assert any("re-run annotate" in n for n in stale.report["notes"])


def test_tampered_history_is_detected(proj) -> None:
    proj.images(range(5))
    cfg = proj.prepare()
    run_split(cfg)
    path = ws_of(cfg).split_groups_manifest
    path.write_text(path.read_text().replace(",train,", ",test,", 1))
    with pytest.raises(SplitHistoryError, match="does not match its last revision"):
        run_split(cfg)


# --------------------------------------------------------------------------------------
# Resplit and dry run
# --------------------------------------------------------------------------------------


def test_resplit_requires_reason_and_token(proj, capsys) -> None:
    proj.images(range(20))
    cfg = proj.prepare()
    run_split(cfg)
    force_history_split(
        cfg, {r.source_id for r in read_records(ws_of(cfg).items_manifest, ItemRecord)}, "test"
    )
    before = split_files(cfg)

    with pytest.raises(SplitError, match="--reason"):
        run_split(cfg, resplit=True)
    preview = run_split(cfg, resplit=True, reason="rebalance")
    assert preview.resplit_token and not preview.written
    assert split_files(cfg) == before
    assert any("TOWARD TRAINING" in w for w in preview.report["warnings"])
    with pytest.raises(SplitError, match="does not match"):
        run_split(cfg, resplit=True, reason="rebalance", confirm="0" * 12)
    assert split_files(cfg) == before

    applied = run_split(cfg, resplit=True, reason="rebalance", confirm=preview.resplit_token)
    assert applied.written and applied.new_revision
    rev = load_history(ws_of(cfg)).last_revision
    assert rev["mode"] == "resplit" and rev["resplit"]["reason"] == "rebalance"
    assert rev["resplit"]["moved"]["test->train"] > 0
    archive = ws_of(cfg).root / rev["resplit"]["archive"]
    assert (archive / "groups.csv").read_bytes() == before["groups.csv"]
    assert {r.reason for r in applied.items} == {"resplit"}
    assert {r.split for r in applied.items} == {"train", "val", "test"}

    path = cfg.root / "config.yaml"
    code = main(
        ["split", "--config", str(path), "--resplit", "--reason", "x", "--log-level", "CRITICAL"]
    )
    assert code == 0 and "--confirm" in capsys.readouterr().out


def test_resplit_without_history(proj) -> None:
    proj.images(range(3))
    cfg = proj.prepare()
    with pytest.raises(SplitError, match="no split history"):
        run_split(cfg, resplit=True, reason="x")


def test_dry_run_writes_nothing(proj) -> None:
    proj.images(range(10))
    cfg = proj.prepare()
    result = run_split(cfg, dry_run=True)
    assert result.items and not result.written
    assert not ws_of(cfg).splits.exists()
    assert not (ws_of(cfg).reports / "split.json").exists()


# --------------------------------------------------------------------------------------
# Config and raw data
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override, message",
    [
        ({"split": {"conflict_resolutions": {"sg_x": "holdout"}}}, "must be one of"),
        ({"split": {"train": 0.0, "val": 0.5, "test": 0.5}}, "split.train must be > 0"),
        ({"split": {"seed": "abc"}}, "expected an integer"),
    ],
)
def test_split_config_validation(proj, override, message) -> None:
    with pytest.raises(ConfigError, match=message):
        proj.config(override)


def test_raw_data_untouched(proj) -> None:
    proj.images(range(8))
    proj.video("v.mp4", 6)
    cfg = proj.prepare()
    before = snapshot_tree(proj.tmp / "raw")
    run_split(cfg)
    run_split(cfg, dry_run=True)
    assert snapshot_tree(proj.tmp / "raw") == before
    assert json.loads((ws_of(cfg).reports / "split.json").read_text())["snapshot"]["items"] > 0
