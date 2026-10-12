"""Annotate and split share one dataset-state validation and must agree on staleness."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cv_pipeline.annotation import run_annotate
from cv_pipeline.cli import main
from cv_pipeline.dataset_state import DatasetStateError, load_dataset_state
from cv_pipeline.dedup import run_dedup
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import DedupRecord, ItemRecord, read_records, write_records
from cv_pipeline.splitting import run_split
from cv_pipeline.workspace import Workspace

from .fakes import det
from .test_annotate import Project, save, scene


@pytest.fixture
def proj(tmp_path, write_config, scripted) -> Project:
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    p = Project(tmp_path, write_config)
    for s in range(4):
        save(p.raw / f"s{s}.png", scene(s))
    return p


def _ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def _new_item_without_dedup(proj: Project):
    cfg = proj.prepare()
    save(proj.raw / "late.png", scene(9))
    run_ingest(cfg)
    return cfg


def _dedup_settings_changed(proj: Project):
    proj.prepare()
    return proj.config({"dedup": {"near_duplicate_threshold": 3}})


def _file_content_changed(proj: Project):
    cfg = proj.prepare()
    items = read_records(_ws(cfg).items_manifest, ItemRecord)
    items[0].sha256 = "f" * 64  # e.g. a frame re-encoded with the same item_id
    write_records(_ws(cfg).items_manifest, items, ItemRecord)
    return cfg


def _source_id_mismatch(proj: Project):
    cfg = proj.prepare()
    dedup = read_records(_ws(cfg).dedup_manifest, DedupRecord)
    dedup[0].source_id = "img_0000000000000000"
    write_records(_ws(cfg).dedup_manifest, dedup, DedupRecord)
    return cfg


def _dedup_missing(proj: Project):
    cfg = proj.config()
    run_ingest(cfg)
    return cfg


def _hash_cache_missing(proj: Project):
    cfg = proj.prepare()
    _ws(cfg).image_hash_cache.unlink()
    return cfg


STALE_STATES: dict[str, tuple[Callable[[Project], object], str]] = {
    "items_vs_dedup": (_new_item_without_dedup, "dedup.csv does not match items.csv"),
    "dedup_settings": (_dedup_settings_changed, "different dedup settings"),
    "file_content": (_file_content_changed, "different file content"),
    "source_ids": (_source_id_mismatch, "source_id differs"),
    "dedup_missing": (_dedup_missing, "run the dedup stage first"),
    "hash_cache_missing": (_hash_cache_missing, "image_hashes.csv not found"),
}


@pytest.mark.parametrize("state", sorted(STALE_STATES))
def test_annotate_and_split_reject_the_same_stale_states(proj, scripted, state) -> None:
    make, message = STALE_STATES[state]
    cfg = make(proj)
    errors = []
    for stage in (run_annotate, run_split):
        with pytest.raises(DatasetStateError) as exc:
            stage(cfg)
        errors.append(str(exc.value))
    assert errors[0] == errors[1]  # one shared definition, one message
    assert message in errors[0]
    assert scripted.calls == []  # annotate refused before any inference


def test_cli_exit_code_for_stale_state_is_1_for_both(proj) -> None:
    cfg = _dedup_settings_changed(proj)
    path = cfg.root / "config.yaml"
    for command in ("annotate", "split"):
        assert main([command, "--config", str(path), "--log-level", "CRITICAL"]) == 1


def test_fresh_state_is_accepted_by_both(proj) -> None:
    cfg = proj.prepare()
    state = load_dataset_state(cfg)
    assert len(state.eligible_ids) == 4 and state.excluded == 0
    assert run_annotate(cfg).exit_code == 0
    assert run_split(cfg).exit_code == 0


def test_rerunning_dedup_makes_state_valid_again(proj) -> None:
    cfg = _dedup_settings_changed(proj)
    run_dedup(cfg)
    assert run_annotate(cfg).exit_code == 0
    assert run_split(cfg).exit_code == 0
