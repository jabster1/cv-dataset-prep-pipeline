"""Source-aware, history-preserving train/val/test splitting."""

from cv_pipeline.splitting.history import SplitHistoryError
from cv_pipeline.splitting.split import (
    EXIT_SPLIT_CONFLICT,
    SplitError,
    SplitResult,
    run_split,
    run_split_stage,
)

__all__ = [
    "EXIT_SPLIT_CONFLICT",
    "SplitError",
    "SplitHistoryError",
    "SplitResult",
    "run_split",
    "run_split_stage",
]
