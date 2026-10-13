"""Train-only, label-aware, deterministic augmentation."""

from cv_pipeline.augmentation.augment import (
    AugmentationError,
    AugmentedItemRecord,
    AugmentResult,
    RejectedVariantRecord,
    SkipCategory,
    SkippedItemRecord,
    run_augment,
    run_augment_stage,
)

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
