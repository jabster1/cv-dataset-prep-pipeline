"""Optional REAL Ultralytics training smoke test — skipped by default.

    CV_PIPELINE_TRAIN_SMOKE_MODEL=~/.cache/cv-pipeline-smoke/yolo11n.pt \
        pytest tests/test_training_smoke.py

Trains 1 epoch on CPU on a tiny synthetic export. It only proves the integration invokes
Ultralytics correctly (managed output dir, artifacts, metrics); it says nothing about
model quality. Weights are never downloaded by this test.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml

from cv_pipeline.training import run_train
from cv_pipeline.workspace import Workspace

from .conftest import deep_update
from .test_export import NO_AUG, TRUSTED_IMPORT, Env, balanced, labelled, setup

MODEL = Path(os.environ.get("CV_PIPELINE_TRAIN_SMOKE_MODEL", "")).expanduser()

pytestmark = pytest.mark.skipif(
    not os.environ.get("CV_PIPELINE_TRAIN_SMOKE_MODEL") or not MODEL.is_file(),
    reason="set CV_PIPELINE_TRAIN_SMOKE_MODEL to a local stock Ultralytics .pt to run",
)


def test_real_one_epoch_cpu_training(tmp_path, write_config, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # any stray Ultralytics 'runs/' would appear here
    env = Env(tmp_path, write_config)
    cfg = setup(env, n=14)
    balanced(cfg)
    cfg = labelled(env, cfg)
    from cv_pipeline.export import run_export

    exp = run_export(cfg).run_id
    (tmp_path / "models").mkdir()
    shutil.copyfile(MODEL, tmp_path / "models" / "base.pt")
    training = {
        "initialization": {"model": "models/base.pt", "mode": "pretrained"},
        "epochs": 1,
        "imgsz": 160,
        "batch": 4,
        "device": "cpu",
        "workers": 0,
        "trainer_args": {},  # pipeline_only + offline defaults decide augmentation/amp/plots
    }
    cfg = env.config(deep_update(NO_AUG, {"training": training}), imports=[TRUSTED_IMPORT])
    export_dir = Workspace(cfg.workspace_dir).root / "export" / "runs" / exp
    export_files = sorted(p.relative_to(export_dir) for p in export_dir.rglob("*"))

    result = run_train(cfg, exp)
    run = json.loads((result.run_dir / "run.json").read_text())
    assert result.status == "completed", run.get("error")
    assert (result.run_dir / "weights" / "best.pt").stat().st_size > 0
    assert (result.run_dir / "weights" / "last.pt").stat().st_size > 0
    assert "metrics/mAP50(B)" in run["metrics"]["validation"]
    assert run["metrics"]["test"] is not None
    assert run["device"]["resolved"] == "cpu"
    # Ultralytics output stayed inside the managed run; export untouched (caches in staging)
    assert not (tmp_path / "runs").exists()
    assert sorted(p.relative_to(export_dir) for p in export_dir.rglob("*")) == export_files
    assert list((result.run_dir / "dataset" / "labels").glob("*.cache"))
    # Ultralytics actually received the pipeline_only augmentation and offline settings
    used = yaml.safe_load((result.run_dir / "ultralytics" / "train" / "args.yaml").read_text())
    for key in (
        "mosaic",
        "fliplr",
        "flipud",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
    ):
        assert used[key] == 0, key
    assert used.get("augmentations") in ([], None)
    assert used["amp"] is False and used["plots"] is False
    assert run["training_augmentation"]["policy"] == "pipeline_only"
