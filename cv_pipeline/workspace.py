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
    def labels(self) -> Path:
        return self.root / "labels"

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
    def resolved_config(self) -> Path:
        return self.root / "config.resolved.yaml"

    def ensure(self) -> Workspace:
        for d in (self.interim_images, self.manifests, self.reports, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        return self
