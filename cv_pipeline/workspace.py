"""Workspace layout: every generated artifact for a project lives under one directory.

The raw input directory is never written to; everything here is derived and can be
regenerated from raw data + config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Workspace:
    root: Path

    # Stage outputs -----------------------------------------------------------------
    @property
    def interim_images(self) -> Path:
        return self.root / "interim" / "images"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def annotations(self) -> Path:
        return self.root / "annotations"

    @property
    def annotation_runs(self) -> Path:
        return self.annotations / "runs"

    @property
    def inference_cache(self) -> Path:
        return self.annotations / "cache" / "inference"

    @property
    def model_cache(self) -> Path:
        return self.annotations / "cache" / "models"

    @property
    def active_annotation_run(self) -> Path:
        return self.annotations / "active_run.json"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # Manifests ---------------------------------------------------------------------
    @property
    def items_manifest(self) -> Path:
        return self.manifests / "items.csv"

    @property
    def rejected_manifest(self) -> Path:
        return self.manifests / "rejected.csv"

    @property
    def dedup_manifest(self) -> Path:
        return self.manifests / "dedup.csv"

    @property
    def duplicate_pairs_manifest(self) -> Path:
        return self.manifests / "duplicate_pairs.csv"

    @property
    def image_hash_cache(self) -> Path:
        return self.manifests / "image_hashes.csv"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def split_groups_manifest(self) -> Path:
        return self.splits / "groups.csv"

    @property
    def split_sources_manifest(self) -> Path:
        return self.splits / "group_sources.csv"

    @property
    def split_revisions(self) -> Path:
        return self.splits / "revisions.jsonl"

    @property
    def split_items_manifest(self) -> Path:
        return self.splits / "items.csv"

    @property
    def imports(self) -> Path:
        return self.root / "imports"

    @property
    def source_refs_manifest(self) -> Path:
        return self.manifests / "source_refs.csv"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def resolved_config(self) -> Path:
        return self.root / "config.resolved.yaml"

    def ensure(self) -> Workspace:
        for d in (self.interim_images, self.manifests, self.reports, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self
