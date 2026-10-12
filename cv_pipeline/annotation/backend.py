"""Annotation backend abstraction and registry.

A backend turns decoded images into :class:`RawDetection` lists and describes itself via
:class:`BackendInfo`. Adding a new detector (e.g. an open-vocabulary model such as
Grounding DINO) means writing one module that calls :func:`register_backend`; the
annotate stage, policy, manifests and review queue stay unchanged.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from cv_pipeline.annotation.schema import BackendInfo, RawDetection
from cv_pipeline.config import AnnotationConfig


class AnnotationSetupError(RuntimeError):
    """The annotation stage cannot start (missing model, unknown backend, bad mapping)."""


@dataclass(frozen=True)
class BackendContext:
    project_root: Path  # base for relative model paths
    cache_dir: Path  # backend-owned cache space (e.g. model metadata)


@runtime_checkable
class AnnotationBackend(Protocol):
    def info(self) -> BackendInfo:
        """Describe the backend without loading heavy frameworks where possible."""
        ...

    def predict(self, images: Sequence[np.ndarray]) -> list[list[RawDetection]]:
        """One detection list per input. Inputs are uint8 BGR (H, W, 3), EXIF-applied."""
        ...

    def close(self) -> None: ...


BackendFactory = Callable[[AnnotationConfig, BackendContext], AnnotationBackend]

_REGISTRY: dict[str, BackendFactory] = {}
_BUILTIN_MODULES = ("cv_pipeline.annotation.backends.ultralytics_yolo",)


def register_backend(name: str) -> Callable[[BackendFactory], BackendFactory]:
    def decorator(factory: BackendFactory) -> BackendFactory:
        _REGISTRY[name] = factory
        return factory

    return decorator


def unregister_backend(name: str) -> None:
    _REGISTRY.pop(name, None)


def available_backends() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def create_backend(cfg: AnnotationConfig, ctx: BackendContext) -> AnnotationBackend:
    _load_builtins()
    factory = _REGISTRY.get(cfg.backend)
    if factory is None:
        raise AnnotationSetupError(
            f"Unknown annotation backend {cfg.backend!r}; available: {sorted(_REGISTRY)}"
        )
    return factory(cfg, ctx)


def _load_builtins() -> None:
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)  # light: heavy frameworks are imported lazily


def resolve_model_path(model: str | None, project_root: Path) -> Path:
    """Resolve and validate ``annotation.model``. Never downloads anything."""
    if not model:
        raise AnnotationSetupError(
            "annotation.model is required to run annotation (path to a weights file)."
        )
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        raise AnnotationSetupError(
            f"Annotation model not found: {path} (from annotation.model={model!r}). "
            "Models are never downloaded automatically; place the weights file there or "
            "fix the path."
        )
    return path.resolve()


def model_reference(model: str, project_root: Path) -> str:
    """Machine-independent reference to a model file, for run metadata and identity.

    * inside the project root -> root-relative POSIX path, e.g. ``models/loto.pt``
      (the same whether the config wrote it relative or absolute)
    * outside the project root -> ``external:<file name>``, e.g. ``external:yolo11n.pt``

    Absolute prefixes (home directories, checkout locations) never leak into run
    identity. What the model *is* is identified separately by ``model_sha256``.
    """
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return f"external:{resolved.name}"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise AnnotationSetupError(
            f"Package {name!r} is required by this annotation backend but is not installed."
        ) from exc
