from __future__ import annotations

import json

import yaml

from cv_pipeline.cli import main
from cv_pipeline.workspace import Workspace


def test_prepare_runs_and_records_run_metadata(synthetic_project) -> None:
    cfg = synthetic_project()
    config_path = cfg.root / "config.yaml"
    assert main(["prepare", "--config", str(config_path), "--log-level", "WARNING"]) == 0

    ws = Workspace(cfg.workspace_dir)
    assert ws.items_manifest.is_file()
    resolved = yaml.safe_load(ws.resolved_config.read_text(encoding="utf-8"))
    assert resolved["project"]["name"] == "test"
    log_lines = (ws.logs / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in log_lines]
    assert "ingest_finished" in events
    assert "prepare_finished" in events


def test_single_stage_command(synthetic_project) -> None:
    cfg = synthetic_project()
    assert main(["ingest", "--config", str(cfg.root / "config.yaml"), "--log-level", "ERROR"]) == 0
    assert Workspace(cfg.workspace_dir).items_manifest.is_file()


def test_show_config(write_config, capsys) -> None:
    assert main(["show-config", "--config", str(write_config())]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["classes"] == ["square", "circle", "triangle"]


def test_invalid_config_exit_code(write_config, capsys) -> None:
    path = write_config({"split": {"train": 0.9}})
    assert main(["ingest", "--config", str(path)]) == 2
    assert "sum to 1.0" in capsys.readouterr().err


def test_missing_raw_dir_exit_code(write_config) -> None:
    assert main(["ingest", "--config", str(write_config()), "--log-level", "CRITICAL"]) == 1
