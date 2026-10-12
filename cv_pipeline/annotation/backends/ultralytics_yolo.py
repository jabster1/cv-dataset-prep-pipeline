"""Ultralytics YOLO detection backend (``annotation.backend: yolo``).

* Only an existing local ``.pt`` file is accepted — nothing is ever downloaded.
* ``ultralytics``/``torch`` are imported on the first ``predict`` call, not at import or
  ``info()`` time. Versions come from package metadata, and model class names are cached
  per model SHA-256, so fully cached runs never load the model.
* Additional ``annotation.options`` are forwarded to ``YOLO.predict`` (e.g.
  ``agnostic_nms``, ``half``) and are part of the inference fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from cv_pipeline.annotation.backend import (
    AnnotationSetupError,
    BackendContext,
    model_reference,
    package_version,
    register_backend,
    resolve_model_path,
)
from cv_pipeline.annotation.schema import BackendInfo, RawDetection
from cv_pipeline.config import AnnotationConfig
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.utils.io import atomic_write_json
from cv_pipeline.utils.logging import get_logger, log_event

logger = get_logger("annotation.yolo")

SUPPORTED_SUFFIXES = (".pt",)  # ONNX / TensorRT can be added as further backends


class UltralyticsYoloBackend:
    name = "ultralytics_yolo"

    def __init__(self, cfg: AnnotationConfig, ctx: BackendContext) -> None:
        self._cfg = cfg
        self._model_path = resolve_model_path(cfg.model, ctx.project_root)
        if self._model_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise AnnotationSetupError(
                f"Unsupported model format {self._model_path.suffix!r} for the yolo backend; "
                f"supported: {list(SUPPORTED_SUFFIXES)}"
            )
        self._model_sha = sha256_file(self._model_path)
        self._model_ref = model_reference(str(cfg.model), ctx.project_root)
        self._meta_path = ctx.cache_dir / f"{self._model_sha}.json"
        self._model: Any = None
        self._names: tuple[str, ...] | None = None

    # ------------------------------------------------------------------------------
    def info(self) -> BackendInfo:
        cfg = self._cfg
        return BackendInfo(
            name=self.name,
            version=package_version("ultralytics"),
            model_id=self._model_ref,
            model_sha256=self._model_sha,
            class_names=self._class_names(),
            params={
                "imgsz": cfg.imgsz,
                "iou": cfg.iou_threshold,
                "conf": cfg.inference_confidence_floor,
                "max_det": cfg.max_detections,
                "batch": cfg.batch_size,
                "device": cfg.device,
                "options": cfg.options,
            },
            frameworks={"torch": package_version("torch")},
        )

    def predict(self, images: Sequence[np.ndarray]) -> list[list[RawDetection]]:
        model = self._load()
        cfg = self._cfg
        results = model.predict(
            source=list(images),  # numpy arrays are treated as BGR by Ultralytics
            conf=cfg.inference_confidence_floor,
            iou=cfg.iou_threshold,
            imgsz=cfg.imgsz,
            max_det=cfg.max_detections,
            device=cfg.device,
            batch=len(images),
            verbose=False,
            **cfg.options,
        )
        names = self._class_names() or ()
        out: list[list[RawDetection]] = []
        for res in results:
            boxes = res.boxes
            dets: list[RawDetection] = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy().astype(float)
                conf = boxes.conf.cpu().numpy().astype(float)
                cls = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
                    label = names[k] if 0 <= k < len(names) else str(k)
                    dets.append(RawDetection(label, int(k), float(c), (x1, y1, x2, y2)))
            out.append(dets)
        return out

    def close(self) -> None:
        self._model = None

    # ------------------------------------------------------------------------------
    def _class_names(self) -> tuple[str, ...]:
        if self._names is None and self._meta_path.is_file():
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if data.get("model_sha256") == self._model_sha:
                self._names = tuple(data["class_names"])
        if self._names is None:
            self._load()
        assert self._names is not None
        return self._names

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise AnnotationSetupError(
                "The yolo backend requires the 'ultralytics' package."
            ) from exc
        log_event(logger, "model_loading", model=str(self._model_path))
        try:
            model = YOLO(str(self._model_path))
        except Exception as exc:
            raise AnnotationSetupError(
                f"Could not load YOLO model {self._model_path}: {exc}"
            ) from exc
        if getattr(model, "task", "detect") != "detect":
            raise AnnotationSetupError(
                f"Model task is {model.task!r}; the yolo backend supports detection models only."
            )
        raw_names = model.names
        names = (
            tuple(raw_names[i] for i in sorted(raw_names))
            if isinstance(raw_names, dict)
            else tuple(raw_names)
        )
        self._names = names
        atomic_write_json(
            self._meta_path, {"model_sha256": self._model_sha, "class_names": list(names)}
        )
        self._model = model
        return model


@register_backend("yolo")
def create_yolo_backend(cfg: AnnotationConfig, ctx: BackendContext) -> UltralyticsYoloBackend:
    return UltralyticsYoloBackend(cfg, ctx)
