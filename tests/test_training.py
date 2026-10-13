"""Phase 7: training from an explicit, re-verified export run (fake trainer backend)."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError
from cv_pipeline.export import run_export
from cv_pipeline.training import (
    ClassMismatchError,
    ExportIntegrityError,
    TrainingSetupError,
    UltralyticsTrainer,
    check_class_compatibility,
    resolve_device,
    run_train,
)
from cv_pipeline.training.coco import COCO80
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes_training import FakeTrainer
from .test_export import NO_AUG, TRUSTED_IMPORT, Env, balanced, labelled, place, raw_ids, setup

CLASSES = ["square", "circle", "triangle"]


@pytest.fixture
def env(tmp_path, write_config) -> Env:
    return Env(tmp_path, write_config)


def ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def model_file(env: Env, name: str = "base.pt", content: bytes = b"weights-v1") -> str:
    path = env.tmp / "models" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return f"models/{name}"


def exported(env: Env, export_over: dict[str, Any] | None = None, heldout_gap: bool = False):
    cfg = setup(env, raw=1 if heldout_gap else 0)
    balanced(cfg)
    if heldout_gap:
        place(cfg, {raw_ids(cfg)[0]: "val"})
    cfg = labelled(env, cfg, export_over)
    return cfg, run_export(cfg).run_id


def train_cfg(env: Env, training: dict[str, Any] | None = None, export_over=None):
    base = {
        "training": {
            "initialization": {"model": model_file(env), "mode": "pretrained"},
            "device": "cpu",
            "epochs": 3,
        }
    }
    merged = deep_update(base, {"training": training or {}})
    return env.config(
        deep_update(deep_update(NO_AUG, export_over or {}), merged), imports=[TRUSTED_IMPORT]
    )


@pytest.fixture
def ready(env):
    cfg, exp = exported(env)
    return train_cfg(env), exp


# --------------------------------------------------------------------------------------
# Explicit export + re-verification
# --------------------------------------------------------------------------------------


def test_explicit_export_run_required(ready, capsys) -> None:
    cfg, _ = ready
    with pytest.raises(ExportIntegrityError, match="explicit export run id"):
        run_train(cfg, "", backend=FakeTrainer())
    with pytest.raises(ExportIntegrityError, match="not found"):
        run_train(cfg, "exp_doesnotexist", backend=FakeTrainer())
    with pytest.raises(SystemExit):
        main(["train", "--config", str(cfg.root / "config.yaml")])


def test_valid_export_accepted_dry_run_starts_nothing(ready) -> None:
    cfg, exp = ready
    fake = FakeTrainer()
    result = run_train(cfg, exp, dry_run=True, backend=fake)
    assert result.status == "planned" and result.run_dir is None
    assert result.plan["export"]["run_id"] == exp
    assert result.plan["export"]["classes"] == CLASSES
    assert not fake.called("train") and not fake.called("evaluate")
    assert not (ws(cfg).root / "training").exists()


@pytest.mark.parametrize("tamper", ["image", "label", "extra_file", "dataset_yaml", "run_json"])
def test_changed_export_refused(ready, tamper) -> None:
    cfg, exp = ready
    run_dir = ws(cfg).root / "export" / "runs" / exp
    if tamper == "image":
        img = next((run_dir / "images" / "train").iterdir())
        img.write_bytes(img.read_bytes() + b"x")
        match = "changed images/train"
    elif tamper == "label":
        lbl = next(p for p in (run_dir / "labels" / "val").iterdir() if p.read_text())
        lbl.write_text("0 0.5 0.5 0.1 0.1\n")
        match = "changed labels/val"
    elif tamper == "extra_file":
        (run_dir / "labels" / "train.cache").write_bytes(b"x")
        match = "unexpected files"
    elif tamper == "dataset_yaml":
        y = run_dir / "dataset.yaml"
        y.write_text(y.read_text().replace("0: square", "0: lock"))
        match = "dataset.yaml differs"
    else:
        rj = run_dir / "run.json"
        data = json.loads(rj.read_text())
        data["identity"]["config"]["classes"] = ["lock", "loto_tag", "x"]
        rj.write_text(json.dumps(data))
        match = "does not hash to the run id"
    fake = FakeTrainer()
    with pytest.raises(ExportIntegrityError, match=match):
        run_train(cfg, exp, backend=fake)
    assert not fake.called("train")


def test_manifest_leakage_rechecked(ready) -> None:
    cfg, exp = ready
    manifest = ws(cfg).root / "export" / "runs" / exp / "manifest.csv"
    lines = manifest.read_text().splitlines()
    val_row = next(i for i, line in enumerate(lines) if line.startswith("val,"))
    train_group = next(line.split(",")[8] for line in lines[1:] if line.startswith("train,"))
    parts = lines[val_row].split(",")
    parts[8] = train_group  # a val image now shares a split group with train
    lines[val_row] = ",".join(parts)
    manifest.write_text("\n".join(lines) + "\n")
    with pytest.raises(ExportIntegrityError, match="appears in several splits"):
        run_train(cfg, exp, backend=FakeTrainer())


# --------------------------------------------------------------------------------------
# Incomplete evaluation sets
# --------------------------------------------------------------------------------------


def test_incomplete_evaluation_refused_by_default(env) -> None:
    _, exp = exported(env, {"export": {"evaluation_policy": "non_strict"}}, heldout_gap=True)
    cfg = train_cfg(env, export_over={"export": {"evaluation_policy": "non_strict"}})
    fake = FakeTrainer()
    with pytest.raises(TrainingSetupError, match="incomplete_evaluation_set: true"):
        run_train(cfg, exp, backend=fake)
    assert not fake.called("train")


def test_incomplete_evaluation_override_recorded(env) -> None:
    over = {"export": {"evaluation_policy": "non_strict"}}
    _, exp = exported(env, over, heldout_gap=True)
    cfg = train_cfg(env, {"allow_incomplete_evaluation": True}, export_over=over)
    result = run_train(cfg, exp, backend=FakeTrainer())
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["overrides"]["allow_incomplete_evaluation"] is True
    assert any("INCOMPLETE" in w for w in run["warnings"])
    assert run["metrics"]["holdout_complete"] is False
    assert "NOT a complete holdout" in (result.run_dir / "report.md").read_text()


# --------------------------------------------------------------------------------------
# Initialization and classes
# --------------------------------------------------------------------------------------


def test_class_compatibility_rules() -> None:
    assert "identical" in check_class_compatibility("finetune", CLASSES, CLASSES)
    with pytest.raises(ClassMismatchError, match="DIFFERENT order"):
        check_class_compatibility("finetune", ["circle", "square", "triangle"], CLASSES)
    with pytest.raises(ClassMismatchError) as exc:
        check_class_compatibility(
            "finetune", ["isolation_point", "lock", "loto_tag"], ["lock", "loto_tag"]
        )
    msg = str(exc.value)
    assert "only in checkpoint: ['isolation_point']" in msg and "never renamed" in msg
    with pytest.raises(ClassMismatchError, match="never renamed or remapped"):
        check_class_compatibility("finetune", ["padlock", "lockout_tag"], ["lock", "loto_tag"])
    assert "COCO-80" in check_class_compatibility("pretrained", list(COCO80), CLASSES)
    with pytest.raises(ClassMismatchError, match="custom classes"):
        check_class_compatibility("pretrained", ["isolation_point", "lock", "loto_tag"], CLASSES)


def test_matching_custom_checkpoint_finetunes(env) -> None:
    _, exp = exported(env)
    cfg = train_cfg(
        env, {"initialization": {"model": model_file(env, "best_custom.pt"), "mode": "finetune"}}
    )
    result = run_train(cfg, exp, backend=FakeTrainer(names=list(CLASSES)))
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["initialization"]["mode"] == "finetune"
    assert run["initialization"]["checkpoint_classes"] == CLASSES
    assert run["initialization"]["resume"] is False


def test_mismatched_custom_checkpoint_refused(env) -> None:
    _, exp = exported(env)
    cfg = train_cfg(
        env, {"initialization": {"model": model_file(env, "best_loto.pt"), "mode": "finetune"}}
    )
    fake = FakeTrainer(names=["isolation_point", "square", "circle"])
    with pytest.raises(ClassMismatchError):
        run_train(cfg, exp, backend=fake)
    assert not fake.called("train") and not (ws(cfg).root / "training").exists()


def test_standard_pretrained_supported_and_finetune_is_not_resume(ready) -> None:
    cfg, exp = ready
    fake = FakeTrainer()
    result = run_train(cfg, exp, backend=fake)
    (train,) = fake.called("train")
    assert "resume" not in train["args"] and train["args"]["pretrained"] is True
    run = json.loads((result.run_dir / "run.json").read_text())
    assert "COCO-80" in run["initialization"]["class_check"]
    assert run["initialization"]["resume"] is False


def test_ultralytics_backend_never_resumes(monkeypatch, tmp_path) -> None:
    captured: dict[str, Any] = {}

    class FakeYOLO:
        def __init__(self, path):
            captured["init"] = path

        def train(self, **kwargs):
            captured.update(kwargs)
            self.trainer = types.SimpleNamespace(
                save_dir=tmp_path, best=tmp_path / "b", last=tmp_path / "l"
            )

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    UltralyticsTrainer().train(
        tmp_path / "best_loto.pt", tmp_path / "d.yaml", tmp_path / "proj", "train", {"epochs": 1}
    )
    assert captured["resume"] is False and captured["exist_ok"] is False
    assert captured["project"] == str(tmp_path / "proj") and captured["name"] == "train"


def test_missing_model_and_download_policy(env) -> None:
    _, exp = exported(env)
    cfg = train_cfg(env, {"initialization": {"model": "yolo11n.pt", "mode": "pretrained"}})
    with pytest.raises(TrainingSetupError, match="never downloaded implicitly"):
        run_train(cfg, exp, dry_run=True, backend=FakeTrainer())
    stock = env.tmp / "cache" / "yolo11n.pt"
    stock.parent.mkdir()
    stock.write_bytes(b"stock")
    cfg = train_cfg(env, {"initialization": {"model": "yolo11n.pt", "allow_download": True}})
    fake = FakeTrainer(downloads={"yolo11n.pt": stock})
    plan = run_train(cfg, exp, dry_run=True, backend=fake).plan
    assert fake.called("download") and plan["initialization"]["sha256"] == sha256_file(stock)


def test_non_detection_checkpoint_refused(ready) -> None:
    cfg, exp = ready
    with pytest.raises(TrainingSetupError, match="expected 'detect'"):
        run_train(cfg, exp, dry_run=True, backend=FakeTrainer(task="segment"))


# --------------------------------------------------------------------------------------
# Devices
# --------------------------------------------------------------------------------------


def test_device_resolution() -> None:
    none = {"cuda_available": False, "mps_available": False}
    mac = {"cuda_available": False, "mps_available": True}
    gpu = {"cuda_available": True, "cuda_device_count": 2, "mps_available": False}
    assert resolve_device("cpu", gpu) == "cpu"
    assert resolve_device(None, none) == "cpu"
    assert resolve_device("auto", mac) == "mps"
    assert resolve_device(None, gpu) == "0"
    assert resolve_device("mps", mac) == "mps"
    with pytest.raises(TrainingSetupError, match="MPS is not available"):
        resolve_device("mps", none)
    assert resolve_device("0,1", gpu) == "0,1" and resolve_device("cuda:1", gpu) == "1"
    with pytest.raises(TrainingSetupError, match="not present"):
        resolve_device("2", gpu)
    with pytest.raises(TrainingSetupError, match="CUDA is not available"):
        resolve_device("0", mac)
    with pytest.raises(TrainingSetupError, match="Unsupported device"):
        resolve_device("tpu", gpu)


def test_device_recorded_and_cli_override(ready) -> None:
    cfg, exp = ready
    gpu = FakeTrainer(
        devices={
            "torch": "2.14.1",
            "cuda_available": True,
            "cuda_version": "12.4",
            "cuda_device_count": 1,
            "gpu_names": ["NVIDIA A100"],
            "mps_available": False,
            "platform": "linux",
            "machine": "x86_64",
        }
    )
    plan = run_train(cfg, exp, dry_run=True, device="0", backend=gpu).plan
    assert plan["device"]["requested"] == "0" and plan["device"]["resolved"] == "0"
    assert (
        plan["device"]["gpu_names"] == ["NVIDIA A100"] and plan["device"]["cuda_version"] == "12.4"
    )
    assert plan["trainer_args"]["device"] == "0"
    with pytest.raises(TrainingSetupError, match="MPS"):
        run_train(cfg, exp, dry_run=True, device="mps", backend=FakeTrainer())


# --------------------------------------------------------------------------------------
# Successful runs, artifacts, metrics
# --------------------------------------------------------------------------------------


def test_successful_run_records_everything(ready) -> None:
    cfg, exp = ready
    fake = FakeTrainer()
    result = run_train(cfg, exp, backend=fake)
    assert result.status == "completed" and result.exit_code == 0
    run = json.loads((result.run_dir / "run.json").read_text())
    model = cfg.root / "models" / "base.pt"
    assert run["initialization"]["sha256"] == sha256_file(model)
    assert run["trainer_args"]["epochs"] == 3 and run["trainer_args"]["seed"] == cfg.project.seed
    assert run["versions"]["ultralytics"] == "8.4.175" and "cv_pipeline" in run["versions"]
    assert run["device"]["resolved"] == "cpu"
    assert run["export"]["run_id"] == exp and run["export"]["labels_fp"]
    assert run["export"]["dataset_snapshot_fp"] and run["export"]["split_revision"] == 1
    assert run["export"]["class_mapping"] == {c: i for i, c in enumerate(CLASSES)}
    assert run["started_at"] and run["ended_at"]
    for w in ("best.pt", "last.pt"):
        assert (result.run_dir / "weights" / w).is_file()
    paths = {a["path"] for a in run["artifacts"]}
    assert {"weights/best.pt", "weights/last.pt", "ultralytics/train/results.csv"} <= paths
    assert (result.run_dir / "config.yaml").is_file()


def test_ultralytics_output_redirected_into_run(ready) -> None:
    cfg, exp = ready
    fake = FakeTrainer()
    export_dir = ws(cfg).root / "export" / "runs" / exp
    before = snapshot_tree(export_dir)
    result = run_train(cfg, exp, backend=fake)
    (train,) = fake.called("train")
    assert train["project"] == result.run_dir / "ultralytics" and train["name"] == "train"
    assert train["data"] == result.run_dir / "dataset" / "dataset.yaml"
    assert (result.run_dir / "dataset" / "dataset.yaml").read_bytes() == (
        export_dir / "dataset.yaml"
    ).read_bytes()
    assert {c["name"] for c in fake.called("evaluate")} == {"val", "test"}
    assert snapshot_tree(export_dir) == before  # export untouched


def test_validation_and_test_metrics_kept_separate(ready) -> None:
    cfg, exp = ready
    result = run_train(cfg, exp, backend=FakeTrainer())
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["validation"]["metrics/mAP50(B)"] == 0.79
    assert metrics["test"]["metrics/mAP50(B)"] == 0.71
    assert metrics["holdout_complete"] is True
    assert "selected by Ultralytics on val" in metrics["checkpoint"]


def test_evaluate_test_can_be_disabled(env) -> None:
    _, exp = exported(env)
    cfg = train_cfg(env, {"evaluate_test": False})
    fake = FakeTrainer()
    result = run_train(cfg, exp, backend=fake)
    assert json.loads((result.run_dir / "metrics.json").read_text())["test"] is None
    assert [c["split"] for c in fake.called("evaluate")] == ["val"]


# --------------------------------------------------------------------------------------
# Failure, immutability, identity
# --------------------------------------------------------------------------------------


def test_failed_training_not_marked_complete_and_kept(ready) -> None:
    cfg, exp = ready
    result = run_train(cfg, exp, backend=FakeTrainer(fail=RuntimeError("CUDA out of memory")))
    assert result.status == "failed" and result.exit_code == 1
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["status"] == "failed" and "CUDA out of memory" in run["error"]
    assert "Traceback" in (result.run_dir / "error.txt").read_text()
    assert not (result.run_dir / "weights").exists()
    with pytest.raises(TrainingSetupError, match="kept for\\s+diagnosis|never overwritten"):
        run_train(cfg, exp, backend=FakeTrainer())
    retry = run_train(cfg, exp, attempt=2, backend=FakeTrainer())
    assert retry.status == "completed" and retry.run_id != result.run_id
    assert json.loads((result.run_dir / "run.json").read_text())["status"] == "failed"


def test_interrupted_training(ready) -> None:
    cfg, exp = ready
    result = run_train(cfg, exp, backend=FakeTrainer(fail=KeyboardInterrupt()))
    assert result.status == "interrupted" and result.exit_code == 130
    assert json.loads((result.run_dir / "run.json").read_text())["status"] == "interrupted"


def test_export_changed_during_training_fails_run(ready) -> None:
    cfg, exp = ready
    export_dir = ws(cfg).root / "export" / "runs" / exp

    def tamper(project):
        (export_dir / "labels" / "train.cache").write_bytes(b"oops")

    result = run_train(cfg, exp, backend=FakeTrainer(on_train=tamper))
    assert result.status == "failed" and result.exit_code == 1
    run = json.loads((result.run_dir / "run.json").read_text())
    assert run["status"] == "failed"
    assert "re-verification" in run["error"] and "unexpected files" in run["error"]


def test_completed_run_not_overwritten(ready) -> None:
    cfg, exp = ready
    first = run_train(cfg, exp, backend=FakeTrainer())
    content = snapshot_tree(first.run_dir)
    fake = FakeTrainer()
    again = run_train(cfg, exp, backend=fake)
    assert again.status == "existing" and again.run_id == first.run_id
    assert not fake.called("train") and snapshot_tree(first.run_dir) == content


def test_changed_inputs_create_distinct_runs(env) -> None:
    _, exp = exported(env)
    base = run_train(train_cfg(env), exp, dry_run=True, backend=FakeTrainer()).run_id
    epochs = run_train(
        train_cfg(env, {"epochs": 5}), exp, dry_run=True, backend=FakeTrainer()
    ).run_id
    seed = run_train(train_cfg(env, {"seed": 99}), exp, dry_run=True, backend=FakeTrainer()).run_id
    args = run_train(
        train_cfg(env, {"trainer_args": {"fliplr": 0.0}}), exp, dry_run=True, backend=FakeTrainer()
    ).run_id
    v2 = model_file(env, "base_v2.pt", b"weights-v2")  # different weights, same mode
    model = run_train(
        train_cfg(env, {"initialization": {"model": v2}}), exp, dry_run=True, backend=FakeTrainer()
    ).run_id
    assert len({base, epochs, seed, args, model}) == 5
    cfg2 = labelled(env, None, {"export": {"classes": ["circle", "square", "triangle"]}})
    exp2 = run_export(cfg2).run_id
    other_export = run_train(
        train_cfg(env, export_over={"export": {"classes": ["circle", "square", "triangle"]}}),
        exp2,
        dry_run=True,
        backend=FakeTrainer(),
    ).run_id
    assert exp2 != exp and other_export not in {base, epochs, seed, args, model}


def test_no_production_model_replaced(env) -> None:
    _, exp = exported(env)
    rel = model_file(env, "best_loto.pt", b"production-weights")
    cfg = train_cfg(env, {"initialization": {"model": rel, "mode": "finetune"}})
    models_before = snapshot_tree(env.tmp / "models")
    result = run_train(cfg, exp, backend=FakeTrainer(names=list(CLASSES)))
    assert result.status == "completed"
    assert snapshot_tree(env.tmp / "models") == models_before
    assert (result.run_dir / "weights" / "best.pt").read_bytes() != b"production-weights"


def test_trainer_args_validation(env) -> None:
    with pytest.raises(ConfigError, match="unsupported key"):
        env.config({"training": {"trainer_args": {"resume": True}}})
    with pytest.raises(ConfigError, match="unsupported key"):
        env.config({"training": {"trainer_args": {"project": "/tmp/x"}}})
    with pytest.raises(ConfigError, match="must be in"):
        env.config({"training": {"trainer_args": {"fliplr": 2.0}}})
    with pytest.raises(ConfigError, match="must be one of"):
        env.config({"training": {"trainer_args": {"optimizer": "Lion"}}})
    with pytest.raises(ConfigError, match="must be one of"):
        env.config({"training": {"initialization": {"mode": "resume"}}})


def test_cli_dry_run(ready, capsys) -> None:
    cfg, exp = ready
    import cv_pipeline.cli as cli

    original = cli.run_train
    cli.run_train = lambda *a, **k: original(*a, backend=FakeTrainer(), **k)
    try:
        code = main(
            [
                "train",
                "--config",
                str(cfg.root / "config.yaml"),
                "--export-run",
                exp,
                "--dry-run",
                "--log-level",
                "CRITICAL",
            ]
        )
    finally:
        cli.run_train = original
    assert code == 0 and "DRY RUN" in capsys.readouterr().out
    assert not (ws(cfg).root / "training").exists()
