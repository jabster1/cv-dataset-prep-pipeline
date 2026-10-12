from __future__ import annotations

from pathlib import Path

import pytest

from cv_pipeline.config import ConfigError, load_config

from .conftest import REPO_ROOT


@pytest.mark.parametrize("name", ["example.yaml", "loto.example.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(REPO_ROOT / "configs" / name)
    assert cfg.classes
    assert cfg.root == REPO_ROOT
    assert cfg.workspace_dir.is_relative_to(REPO_ROOT / "workspaces")
    assert cfg.training.runs_dir.is_relative_to(cfg.workspace_dir)


def test_defaults_and_normalisation(write_config) -> None:
    cfg = load_config(
        write_config({"ingest": {"image_extensions": ["JPG", ".Png"], "video_extensions": ["mp4"]}})
    )
    assert cfg.ingest.image_extensions == [".jpg", ".png"]
    assert cfg.ingest.video_extensions == [".mp4"]
    assert (cfg.split.train, cfg.split.val, cfg.split.test) == (0.7, 0.2, 0.1)
    assert cfg.project.seed == 7


def test_paths_resolve_relative_to_config_file_not_cwd(write_config, tmp_path, monkeypatch) -> None:
    path = write_config()
    monkeypatch.chdir(Path("/"))
    cfg = load_config(path)
    assert cfg.workspace_dir == (tmp_path / "work").resolve()
    assert cfg.raw_dir == (tmp_path / "raw").resolve()


def test_unknown_key_is_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config({"annotation": {"confidance_threshold": 0.3}}))


def test_wrong_type_is_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(write_config({"project": {"seed": "abc"}}))


def test_bool_is_not_accepted_as_number(write_config) -> None:
    with pytest.raises(ConfigError):
        load_config(write_config({"split": {"train": True}}))


def test_literal_choices_enforced(write_config) -> None:
    with pytest.raises(ConfigError, match="must be one of"):
        load_config(write_config({"ingest": {"video": {"mode": "keyframes"}}}))


def test_split_must_sum_to_one(write_config) -> None:
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(write_config({"split": {"train": 0.8, "val": 0.2, "test": 0.1}}))


def test_custom_split_fractions(write_config) -> None:
    cfg = load_config(write_config({"split": {"train": 0.8, "val": 0.1, "test": 0.1}}))
    assert cfg.split.train == pytest.approx(0.8)


def test_duplicate_and_empty_classes_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="duplicate class"):
        load_config(write_config({"classes": ["a", "b", "a"]}))
    with pytest.raises(ConfigError, match="at least one class"):
        load_config(write_config({"classes": []}))


def test_missing_required_section(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("classes: [a]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="project"):
        load_config(p)


def test_workspace_inside_raw_dir_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="immutable"):
        load_config(write_config({"project": {"workspace": "raw/out"}}))


def test_class_map_targets_must_exist(write_config) -> None:
    with pytest.raises(ConfigError, match="class_map"):
        load_config(write_config({"annotation": {"class_map": {"lock": "padlock"}}}))


def test_review_threshold_not_below_confidence(write_config) -> None:
    with pytest.raises(ConfigError, match="review_threshold"):
        load_config(
            write_config({"annotation": {"confidence_threshold": 0.5, "review_threshold": 0.3}})
        )


def test_missing_file_and_bad_yaml(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)


def test_to_dict_roundtrip_is_plain(write_config) -> None:
    import yaml

    cfg = load_config(write_config())
    dumped = yaml.safe_dump(cfg.to_dict())
    assert "work" in dumped
