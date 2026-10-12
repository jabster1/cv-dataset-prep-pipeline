"""Deterministic scripted annotation backend for tests (no model, no torch)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from cv_pipeline.annotation.backend import BackendContext, model_reference
from cv_pipeline.annotation.identity import input_pixel_sha256
from cv_pipeline.annotation.schema import BackendInfo, RawDetection
from cv_pipeline.config import AnnotationConfig
from cv_pipeline.synthetic import SYNTHETIC_CLASSES

Response = list[RawDetection] | Exception | Callable[[np.ndarray], list[RawDetection]]


def det(label: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> RawDetection:
    return RawDetection(label, None, conf, (x1, y1, x2, y2))


@dataclass
class Script:
    """Responses keyed by the input pixel hash; ``default`` for everything else."""

    responses: dict[str, Response] = field(default_factory=dict)
    default: Response = field(default_factory=list)
    class_names: tuple[str, ...] | None = SYNTHETIC_CLASSES
    version: str = "1.0"
    model_sha256: str = "0" * 64
    fail_multi_image_batches: bool = False
    calls: list[list[str]] = field(default_factory=list)  # pixel hashes per predict()
    created: int = 0

    @property
    def inferred(self) -> list[str]:
        return [k for call in self.calls for k in call]


class ScriptedBackend:
    def __init__(self, script: Script, cfg: AnnotationConfig, ctx: BackendContext) -> None:
        self.script, self.cfg = script, cfg
        self.model_ref = model_reference(cfg.model, ctx.project_root) if cfg.model else "scripted"
        script.created += 1

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="scripted",
            version=self.script.version,
            model_id=self.model_ref,
            model_sha256=self.script.model_sha256,
            class_names=self.script.class_names,
            params={
                "imgsz": self.cfg.imgsz,
                "conf": self.cfg.inference_confidence_floor,
                "batch": self.cfg.batch_size,
                "options": self.cfg.options,
            },
        )

    def predict(self, images: Sequence[np.ndarray]) -> list[list[RawDetection]]:
        keys = [input_pixel_sha256(img) for img in images]
        self.script.calls.append(keys)
        if self.script.fail_multi_image_batches and len(images) > 1:
            raise RuntimeError("batch failure")
        out = []
        for key, img in zip(keys, images, strict=True):
            response = self.script.responses.get(key, self.script.default)
            if callable(response) and not isinstance(response, Exception):
                response = response(img)
            if isinstance(response, Exception):
                raise response
            out.append(list(response))
        return out

    def close(self) -> None:
        pass
