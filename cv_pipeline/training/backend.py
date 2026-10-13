"""Trainer backend abstraction (Ultralytics today; fakes in tests).

The pipeline owns validation, identity, lineage and run management; the backend only
inspects checkpoints, reports devices/versions and runs train/val. Ultralytics and torch
are imported lazily.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol


class TrainingSetupError(RuntimeError):
    """Training cannot start (model, classes, device, export, existing run...)."""


@dataclass(frozen=True)
class CheckpointInfo:
    names: list[str]
    task: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainOutcome:
    save_dir: Path
    best: Path
    last: Path


class TrainerBackend(Protocol):
    name: str

    def versions(self) -> dict[str, str]: ...

    def trainer_defaults(self) -> dict[str, Any]: ...

    def device_info(self) -> dict[str, Any]: ...

    def download(self, model_name: str) -> Path: ...

    def inspect_checkpoint(self, path: Path) -> CheckpointInfo: ...

    def train(
        self, model: Path, data_yaml: Path, project: Path, name: str, args: dict[str, Any]
    ) -> TrainOutcome: ...

    def evaluate(
        self,
        weights: Path,
        data_yaml: Path,
        split: str,
        project: Path,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------------------
# Device resolution (pure; the backend only reports availability)
# --------------------------------------------------------------------------------------

_CUDA = re.compile(r"^(?:cuda:)?(\d+(?:,\d+)*)$")


def resolve_device(requested: str | None, info: dict[str, Any]) -> str:
    """Return the Ultralytics device string, or raise if the request cannot be met."""
    req = (requested or "auto").strip().lower()
    cuda_n = int(info.get("cuda_device_count", 0)) if info.get("cuda_available") else 0
    if req == "auto":
        if cuda_n:
            return "0"
        return "mps" if info.get("mps_available") else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "mps":
        if not info.get("mps_available"):
            raise TrainingSetupError("device 'mps' requested but Apple MPS is not available.")
        return "mps"
    if req == "cuda":
        req = "0"
    m = _CUDA.match(req)
    if not m:
        raise TrainingSetupError(
            f"Unsupported device {requested!r}; use auto, cpu, mps, 0, 0,1 or cuda:0."
        )
    ids = [int(x) for x in m.group(1).split(",")]
    if not cuda_n:
        raise TrainingSetupError(f"CUDA device {requested!r} requested but CUDA is not available.")
    missing = [i for i in ids if i >= cuda_n]
    if missing:
        raise TrainingSetupError(f"CUDA device(s) {missing} not present ({cuda_n} visible).")
    return ",".join(str(i) for i in ids)


# --------------------------------------------------------------------------------------
# Ultralytics
# --------------------------------------------------------------------------------------


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


class UltralyticsTrainer:
    name = "ultralytics"

    def versions(self) -> dict[str, str]:
        return {
            "ultralytics": _version("ultralytics"),
            "torch": _version("torch"),
            "python": platform.python_version(),
        }

    def trainer_defaults(self) -> dict[str, Any]:
        from ultralytics.cfg import get_cfg

        return dict(vars(get_cfg()))

    def device_info(self) -> dict[str, Any]:
        import torch

        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        return {
            "torch": torch.__version__,
            "cuda_available": cuda,
            "cuda_version": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count() if cuda else 0,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda
            else [],
            "mps_available": mps,
            "platform": platform.platform(),
            "machine": platform.machine(),
        }

    def download(self, model_name: str) -> Path:
        from ultralytics import YOLO

        model = YOLO(model_name)  # fetches stock weights into Ultralytics' weights dir
        return Path(model.ckpt_path).resolve()

    def inspect_checkpoint(self, path: Path) -> CheckpointInfo:
        from ultralytics import YOLO

        model = YOLO(str(path))
        raw = model.names
        names = [raw[i] for i in sorted(raw)] if isinstance(raw, dict) else list(raw)
        ckpt = model.ckpt if isinstance(model.ckpt, dict) else {}
        train_args = ckpt.get("train_args") or {}
        return CheckpointInfo(
            names=names,
            task=str(model.task),
            metadata={
                "ultralytics_version": ckpt.get("version"),
                "date": ckpt.get("date"),
                "train_data": train_args.get("data") if isinstance(train_args, dict) else None,
                "epoch": ckpt.get("epoch"),
            },
        )

    def train(
        self, model: Path, data_yaml: Path, project: Path, name: str, args: dict[str, Any]
    ) -> TrainOutcome:
        from ultralytics import YOLO

        yolo = YOLO(str(model))
        yolo.train(
            data=str(data_yaml),
            project=str(project),
            name=name,
            exist_ok=False,
            resume=False,
            **args,
        )
        trainer = yolo.trainer
        return TrainOutcome(Path(trainer.save_dir), Path(trainer.best), Path(trainer.last))

    def evaluate(
        self,
        weights: Path,
        data_yaml: Path,
        split: str,
        project: Path,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        from ultralytics import YOLO

        yolo = YOLO(str(weights))
        result = yolo.val(
            data=str(data_yaml),
            split=split,
            project=str(project),
            name=name,
            exist_ok=False,
            **args,
        )
        metrics = {k: float(v) for k, v in result.results_dict.items()}
        names = yolo.names
        maps = getattr(getattr(result, "box", None), "maps", None)
        if maps is not None:
            metrics["per_class_mAP50-95"] = {
                str(names[i]): float(v) for i, v in enumerate(list(maps)) if i in names
            }
        return metrics
