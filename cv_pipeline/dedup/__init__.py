"""Duplicate detection: exact (byte/pixel) and perceptual near-duplicates, split grouping."""

from cv_pipeline.dedup.dedup import DedupResult, run_dedup

__all__ = ["DedupResult", "run_dedup"]
