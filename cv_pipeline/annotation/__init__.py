"""Automatic annotation: pluggable backends, confidence/review policy, immutable run
snapshots and a review queue. Machine annotations are never treated as verified."""

from cv_pipeline.annotation.annotate import AnnotationResult, run_annotate, run_annotate_stage
from cv_pipeline.annotation.backend import (
    AnnotationBackend,
    AnnotationSetupError,
    BackendContext,
    register_backend,
)
from cv_pipeline.annotation.schema import BackendInfo, RawDetection

__all__ = [
    "AnnotationBackend",
    "AnnotationResult",
    "AnnotationSetupError",
    "BackendContext",
    "BackendInfo",
    "RawDetection",
    "register_backend",
    "run_annotate",
    "run_annotate_stage",
]
