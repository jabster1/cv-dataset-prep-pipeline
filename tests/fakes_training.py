"""Deterministic fake trainer backend (no torch, no Ultralytics, no GPU)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cv_pipeline.training.args import KNOWN_TRAINER_KEYS
from cv_pipeline.training.backend import CheckpointInfo, TrainOutcome
from cv_pipeline.training.coco import COCO80

# Real Ultralytics 8.4.175 defaults for the settings the pipeline cares about.
ULTRALYTICS_DEFAULTS: dict[str, Any] = {
    **{k: None for k in KNOWN_TRAINER_KEYS},
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "close_mosaic": 10,
    "amp": True,
    "plots": True,
}

METRICS = {
    "metrics/precision(B)": 0.81,
    "metrics/recall(B)": 0.74,
    "metrics/mAP50(B)": 0.79,
    "metrics/mAP50-95(B)": 0.52,
    "fitness": 0.55,
}


@dataclass
class FakeTrainer:
    names: list[str] = field(default_factory=lambda: list(COCO80))
    task: str = "detect"
    devices: dict[str, Any] = field(
        default_factory=lambda: {
            "torch": "2.14.1",
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "gpu_names": [],
            "mps_available": False,
            "platform": "test",
            "machine": "test",
        }
    )
    fail: BaseException | None = None
    on_train: Any = None  # optional callable(project) run during training
    metrics: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "val": dict(METRICS),
            "test": {**METRICS, "metrics/mAP50(B)": 0.71},
        }
    )
    downloads: dict[str, Path] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=lambda: dict(ULTRALYTICS_DEFAULTS))
    version: str = "8.4.175"
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    name: str = "fake"

    def versions(self) -> dict[str, str]:
        return {"ultralytics": self.version, "torch": "2.14.1", "python": "3.11"}

    def trainer_defaults(self) -> dict[str, Any]:
        return dict(self.defaults)

    def device_info(self) -> dict[str, Any]:
        return dict(self.devices)

    def download(self, model_name: str) -> Path:
        self.calls.append(("download", {"name": model_name}))
        return self.downloads[model_name]

    def inspect_checkpoint(self, path: Path) -> CheckpointInfo:
        self.calls.append(("inspect", {"path": path}))
        return CheckpointInfo(list(self.names), self.task, {"ultralytics_version": "8.4.175"})

    def train(
        self, model: Path, data_yaml: Path, project: Path, name: str, args: dict[str, Any]
    ) -> TrainOutcome:
        self.calls.append(
            (
                "train",
                {
                    "model": model,
                    "data": data_yaml,
                    "project": project,
                    "name": name,
                    "args": dict(args),
                },
            )
        )
        if self.on_train:
            self.on_train(project)
        if self.fail is not None:
            raise self.fail
        save = project / name
        (save / "weights").mkdir(parents=True)
        (save / "weights" / "best.pt").write_bytes(b"best:" + model.read_bytes()[:16])
        (save / "weights" / "last.pt").write_bytes(b"last:" + model.read_bytes()[:16])
        (save / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.79\n")
        return TrainOutcome(save, save / "weights" / "best.pt", save / "weights" / "last.pt")

    def evaluate(
        self,
        weights: Path,
        data_yaml: Path,
        split: str,
        project: Path,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            ("evaluate", {"weights": weights, "split": split, "name": name, "args": dict(args)})
        )
        (project / name).mkdir(parents=True)
        return dict(self.metrics[split])

    def called(self, kind: str) -> list[dict[str, Any]]:
        return [c for k, c in self.calls if k == kind]
