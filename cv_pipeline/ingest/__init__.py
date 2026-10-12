"""Ingestion: discover raw images/videos, copy images, sample video frames, record lineage."""

from cv_pipeline.ingest.ingest import IngestResult, run_ingest

__all__ = ["IngestResult", "run_ingest"]
