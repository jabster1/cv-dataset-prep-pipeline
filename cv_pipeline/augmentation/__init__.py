"""Train-only, label-aware, deterministic augmentation."""

from cv_pipeline.augmentation.augment import (
    AugmentationError,
    AugmentedItemRecord,
    AugmentResult,
    RejectedVariantRecord,
    SkippedItemRecord,
    run_augment,
    run_augment_stage,
)
from cv_pipeline.labels.usage import SkipCategory

__all__ = [
    "AugmentResult",
    "AugmentationError",
    "AugmentedItemRecord",
    "RejectedVariantRecord",
    "SkipCategory",
    "SkippedItemRecord",
    "run_augment",
    "run_augment_stage",
]
