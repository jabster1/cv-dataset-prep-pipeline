"""Train/fine-tune YOLO from an EXPLICIT, re-verified Phase 6 export run.

    verify export (identity, dataset.yaml, hashes, leakage)  ─┐
    incomplete-evaluation policy                              │ preflight (also --dry-run)
    resolve + hash initialization model, inspect its classes  │
    class compatibility (mode-specific, never remapped)       │
    device resolution, effective trainer args, run identity  ─┘
              │
    training/runs/<run_id>/   status: running → completed | failed | interrupted
      dataset/ (staged view: Ultralytics caches land here, never in the export)
      ultralytics/{train,val,test}/  weights/best.pt, last.pt  metrics.json  run.json

Run identity is for lineage of configuration and inputs; training itself (especially on
GPUs) is not bit-for-bit reproducible, and nothing here claims it is. New weights stay in
their run — nothing is promoted or copied over an existing model.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cv_pipeline import __version__
from cv_pipeline.config import PipelineConfig
from cv_pipeline.training.args import validate_trainer_args
from cv_pipeline.training.backend import (
    TrainerBackend,
    TrainingSetupError,
    UltralyticsTrainer,
    resolve_device,
)
from cv_pipeline.training.coco import COCO80
from cv_pipeline.training.verify_export import VerifiedExport, verify_export_run
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("training")

TRAIN_SCHEMA = "cvp-training-v1"


class ClassMismatchError(TrainingSetupError):
    pass


@dataclass
class TrainResult:
    run_id: str
    run_dir: Path | None
    status: str  # planned | completed | failed | interrupted | existing
    created: bool
    plan: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# Preflight helpers
# --------------------------------------------------------------------------------------


def resolve_init_model(cfg: PipelineConfig, backend: TrainerBackend) -> Path:
    init = cfg.training.initialization
    path = Path(init.model).expanduser()
    if not path.is_absolute():
        path = cfg.root / path
    if path.is_file():
        return path.resolve()
    if init.allow_download and "/" not in init.model and "\\" not in init.model:
        return backend.download(init.model)
    raise TrainingSetupError(
        f"Initialization model not found: {path} (training.initialization.model="
        f"{init.model!r}). Weights are never downloaded implicitly; place the file there or "
        "set allow_download: true for a stock Ultralytics model name."
    )


def check_class_compatibility(mode: str, checkpoint: list[str], export: list[str]) -> str:
    """Return a short description of the outcome, or raise ClassMismatchError."""
    if mode == "finetune":
        if checkpoint == export:
            return "identical classes (names and order)"
        if sorted(checkpoint) == sorted(export):
            raise ClassMismatchError(
                "finetune: checkpoint and export have the same class names in a DIFFERENT "
                f"order — class ids would change meaning.\n  checkpoint: {checkpoint}\n  "
                f"export:     {export}\nFix the order in export.classes (or the upstream "
                "mapping); nothing is reordered automatically."
            )
        only_ckpt = [c for c in checkpoint if c not in export]
        only_export = [c for c in export if c not in checkpoint]
        raise ClassMismatchError(
            "finetune: checkpoint classes differ from the export classes.\n"
            f"  checkpoint: {checkpoint}\n  export:     {export}\n"
            f"  only in checkpoint: {only_ckpt}\n  only in export: {only_export}\n"
            "Classes are never renamed or remapped silently. Align the taxonomy upstream "
            "(import class_map / export.classes), or train from a standard base model with "
            "mode: pretrained. Deliberate head replacement is not supported."
        )
    # pretrained
    if checkpoint == export:
        return "identical classes"
    if list(checkpoint) == list(COCO80):
        return "standard COCO-80 base weights; a new detection head is built for the export classes"
    raise ClassMismatchError(
        f"pretrained: the checkpoint has custom classes {checkpoint}, not the stock COCO-80 "
        "base classes. Using it as a 'pretrained base' would silently replace its detection "
        "head. Use mode: finetune (requires identical classes) or a standard base model."
    )


def effective_args(cfg: PipelineConfig, device: str, seed: int) -> dict[str, Any]:
    t = cfg.training
    args: dict[str, Any] = {
        "epochs": t.epochs,
        "imgsz": t.imgsz,
        "batch": t.batch,
        "device": device,
        "workers": t.workers,
        "patience": t.patience,
        "seed": seed,
        "deterministic": t.deterministic,
        "cache": False if t.cache == "none" else t.cache,
        "pretrained": True,
        "val": True,
    }
    args.update(validate_trainer_args(t.trainer_args))
    return args


def _stage_dataset(export: VerifiedExport, dest: Path) -> str:
    """Read-only view of the export for Ultralytics (its .cache files land here)."""
    method = "hardlink"
    dest.mkdir(parents=True)
    shutil.copyfile(export.run_dir / "dataset.yaml", dest / "dataset.yaml")
    for r in export.records:
        for rel in (r.export_image, r.export_label):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(export.run_dir / rel, target)
            except OSError:
                shutil.copyfile(export.run_dir / rel, target)
                method = "copy"
    for s in ("train", "val", "test"):
        (dest / "images" / s).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / s).mkdir(parents=True, exist_ok=True)
    return method


def _artifacts(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for base in ("weights", "ultralytics"):
        root = run_dir / base
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    out.append(
                        {
                            "path": p.relative_to(run_dir).as_posix(),
                            "sha256": sha256_file(p),
                            "bytes": p.stat().st_size,
                        }
                    )
    return out


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_train(
    cfg: PipelineConfig,
    export_run: str,
    dry_run: bool = False,
    attempt: int = 1,
    device: str | None = None,
    backend: TrainerBackend | None = None,
) -> TrainResult:
    ws = Workspace(cfg.workspace_dir).ensure()
    t = cfg.training
    backend = backend or UltralyticsTrainer()
    if attempt < 1:
        raise TrainingSetupError("--attempt must be >= 1")

    # 1-2. explicit export, re-verified ---------------------------------------------------
    export = verify_export_run(ws, export_run)
    warnings: list[str] = []
    if export.incomplete_evaluation_set:
        if not t.allow_incomplete_evaluation:
            raise TrainingSetupError(
                f"Export {export.run_id} is marked incomplete_evaluation_set: true. Refusing to "
                "train/evaluate on an incomplete holdout. Complete the val/test labels and "
                "re-export, or set training.allow_incomplete_evaluation: true (development "
                "only; metrics will be flagged as NOT from a complete holdout)."
            )
        warnings.append(
            "OVERRIDE: training on an export with an INCOMPLETE evaluation set "
            "(training.allow_incomplete_evaluation: true). Validation/test metrics do not "
            "come from a complete holdout."
        )
        log_event(logger, "incomplete_evaluation_override", logging.WARNING, export=export.run_id)

    # 3. initialization model + class compatibility --------------------------------------
    init = t.initialization
    model_path = resolve_init_model(cfg, backend)
    model_sha = sha256_file(model_path)
    ckpt = backend.inspect_checkpoint(model_path)
    if ckpt.task != "detect":
        raise TrainingSetupError(f"Initialization model task is {ckpt.task!r}; expected 'detect'.")
    class_check = check_class_compatibility(init.mode, list(ckpt.names), export.classes)

    # 4. device + args -------------------------------------------------------------------
    requested = device if device is not None else t.device
    dev_info = backend.device_info()
    resolved = resolve_device(requested, dev_info)
    seed = t.seed if t.seed is not None else cfg.project.seed
    args = effective_args(cfg, resolved, seed)
    versions = backend.versions()

    # 5. identity --------------------------------------------------------------------------
    identity = {
        "export_run_id": export.run_id,
        "export_fingerprint": export.fingerprint,
        "initialization": {"mode": init.mode, "model_sha256": model_sha},
        "trainer_args": {**args, "device": requested or "auto"},
        "evaluate_test": t.evaluate_test,
        "allow_incomplete_evaluation": t.allow_incomplete_evaluation,
        "versions": {"ultralytics": versions.get("ultralytics"), "torch": versions.get("torch")},
        "backend": backend.name,
        "attempt": attempt,
    }
    run_id = "trn_" + digest({"schema": TRAIN_SCHEMA, **identity})[:16]
    run_dir = ws.root / "training" / "runs" / run_id
    ex_id = export.identity
    plan = {
        "schema": TRAIN_SCHEMA,
        "run_id": run_id,
        "identity": identity,
        "export": {
            "run_id": export.run_id,
            "fingerprint": export.fingerprint,
            "dataset_snapshot_fp": ex_id.get("snapshot_fp"),
            "split_revision": ex_id.get("split_revision"),
            "labels_fp": ex_id.get("labels_fp"),
            "augmentation_run_id": (ex_id.get("augmentation") or {}).get("run_id"),
            "classes": export.classes,
            "class_mapping": (ex_id.get("config") or {}).get("class_mapping"),
            "counts": export.counts,
            "incomplete_evaluation_set": export.incomplete_evaluation_set,
        },
        "initialization": {
            "mode": init.mode,
            "reference": init.model,
            "resolved_path": str(model_path),
            "sha256": model_sha,
            "checkpoint_classes": list(ckpt.names),
            "checkpoint_task": ckpt.task,
            "checkpoint_metadata": ckpt.metadata,
            "class_check": class_check,
            "resume": False,
        },
        "trainer_args": args,
        "seed": seed,
        "device": {"requested": requested or "auto", "resolved": resolved, **dev_info},
        "versions": {**versions, "cv_pipeline": __version__},
        "overrides": {
            "allow_incomplete_evaluation": t.allow_incomplete_evaluation
            and export.incomplete_evaluation_set
        },
        "warnings": warnings,
        "reproducibility_note": "Run identity records configuration and inputs. Training "
        "results are not guaranteed bit-for-bit identical, especially across GPUs.",
    }
    if run_dir.exists():
        existing = _read_json(run_dir / "run.json")
        status = existing.get("status", "unknown")
        if status == "completed":
            log_event(logger, "training_run_exists", run_id=run_id)
            return TrainResult(run_id, run_dir, "existing", False, existing)
        raise TrainingSetupError(
            f"Training run {run_id} already exists with status {status!r}; it is kept for "
            "diagnosis and never overwritten. Use --attempt N to start a new run."
        )
    if dry_run:
        log_event(logger, "training_dry_run", run_id=run_id, device=resolved)
        return TrainResult(run_id, None, "planned", False, plan)

    # 6. run -------------------------------------------------------------------------------
    run_dir.mkdir(parents=True)
    atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    record: dict[str, Any] = {**plan, "status": "running", "started_at": _now(), "ended_at": None}
    record["dataset_staging"] = _stage_dataset(export, run_dir / "dataset")
    atomic_write_json(run_dir / "run.json", record)
    data_yaml = run_dir / "dataset" / "dataset.yaml"
    project = run_dir / "ultralytics"
    t0 = time.perf_counter()
    log_event(logger, "training_started", run_id=run_id, export=export.run_id, device=resolved)
    try:
        outcome = backend.train(model_path, data_yaml, project, "train", args)
        weights = run_dir / "weights"
        weights.mkdir()
        shutil.copyfile(outcome.best, weights / "best.pt")
        shutil.copyfile(outcome.last, weights / "last.pt")
        eval_args = {
            "imgsz": t.imgsz,
            "batch": t.batch if t.batch > 0 else 16,
            "device": resolved,
            "workers": t.workers,
        }
        metrics: dict[str, Any] = {
            "checkpoint": "weights/best.pt (selected by Ultralytics on val fitness)",
            "holdout_complete": not export.incomplete_evaluation_set,
            "validation": backend.evaluate(
                weights / "best.pt", data_yaml, "val", project, "val", eval_args
            ),
            "test": None,
        }
        if t.evaluate_test and export.has_test:
            metrics["test"] = backend.evaluate(
                weights / "best.pt", data_yaml, "test", project, "test", eval_args
            )
        verify_export_run(ws, export.run_id)  # the export must be unchanged after training
        if sha256_file(model_path) != model_sha:
            raise TrainingSetupError("initialization model changed during training")
        atomic_write_json(run_dir / "metrics.json", metrics)
        record.update(status="completed", metrics=metrics, artifacts=_artifacts(run_dir))
        status, code = "completed", 0
    except KeyboardInterrupt:
        record.update(status="interrupted", error="KeyboardInterrupt")
        status, code = "interrupted", 130
    except Exception as exc:  # noqa: BLE001 - recorded, never marked complete
        atomic_write_text(run_dir / "error.txt", traceback.format_exc())
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        status, code = "failed", 1
    record.update(ended_at=_now(), duration_s=round(time.perf_counter() - t0, 1))
    record.setdefault("artifacts", _artifacts(run_dir))
    atomic_write_json(run_dir / "run.json", record)
    atomic_write_text(run_dir / "report.md", render_report(record))
    log_event(logger, "training_finished", run_id=run_id, status=status)
    return TrainResult(run_id, run_dir, status, True, record, code)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt_metrics(m: dict[str, Any] | None) -> list[str]:
    if not m:
        return ["(none)"]
    keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    rows = [f"| {k.split('/')[1]} | {m[k]:.4f} |" for k in keys if k in m]
    return ["| Metric | Value |", "|---|---|", *rows] if rows else ["(no standard metrics)"]


def render_report(r: dict[str, Any]) -> str:
    m = r.get("metrics") or {}
    init = r["initialization"]
    lines = [
        f"# Training run {r['run_id']}",
        "",
        f"- Status: **{r['status']}** ({r.get('started_at')} → {r.get('ended_at')})",
        f"- Export: `{r['export']['run_id']}` (classes {r['export']['classes']}, counts "
        f"{r['export']['counts']})",
        f"- Initialization: {init['mode']} from `{init['reference']}` "
        f"(sha256 `{init['sha256'][:12]}`) — {init['class_check']}",
        f"- Device: {r['device']['requested']} → {r['device']['resolved']}",
        f"- Seed: {r['seed']}; versions: {r['versions']}",
        "- Weights stay in this run; no model is promoted automatically.",
        "",
    ]
    if r.get("warnings"):
        lines += ["## Warnings", ""] + [f"- ⚠️ {w}" for w in r["warnings"]] + [""]
    if r["status"] == "completed":
        holdout = "" if m.get("holdout_complete") else " — ⚠️ NOT a complete holdout"
        lines += [f"## Validation (best.pt){holdout}", "", *_fmt_metrics(m.get("validation")), ""]
        lines += [f"## Test (best.pt){holdout}", "", *_fmt_metrics(m.get("test")), ""]
    elif r.get("error"):
        lines += ["## Error", "", f"`{r['error']}` (see error.txt)", ""]
    return "\n".join(lines)
