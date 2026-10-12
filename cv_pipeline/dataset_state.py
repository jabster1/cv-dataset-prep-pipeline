"""Validated view of the current dataset state (ingest + dedup), shared by later stages.

Splitting (and later export) depend only on *which items exist, how they are grouped, and
which file content they refer to* — never on machine predictions. This module checks that
``items.csv`` and ``dedup.csv`` agree with each other, with the current dedup config and
with the files actually on disk, and builds a deterministic snapshot fingerprint of the
result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cv_pipeline.annotation.identity import digest
from cv_pipeline.config import PipelineConfig
from cv_pipeline.dedup.report import dedup_settings
from cv_pipeline.manifest import DedupRecord, ImageHashRecord, ItemRecord, read_records
from cv_pipeline.workspace import Workspace

SPLIT_INPUT_SCHEMA = "cvp-split-input-v1"


class DatasetStateError(RuntimeError):
    """Upstream manifests are missing, inconsistent, or stale relative to config/files."""


@dataclass(frozen=True)
class SplitInputRow:
    """The fields of an eligible item that splitting depends on."""

    item_id: str
    source_id: str
    source_type: str
    frame_index: int | None
    split_group: str
    file_sha256: str


@dataclass(frozen=True)
class SplitInput:
    rows: tuple[SplitInputRow, ...]  # eligible items, sorted by item_id
    snapshot_fp: str
    dedup_settings_fp: str
    excluded: int

    @property
    def n_items(self) -> int:
        return len(self.rows)


def snapshot_fingerprint(rows: list[SplitInputRow] | tuple[SplitInputRow, ...]) -> str:
    return digest({"schema": SPLIT_INPUT_SCHEMA, "items": [asdict(r) for r in rows]})


def load_split_input(cfg: PipelineConfig) -> SplitInput:
    ws = Workspace(cfg.workspace_dir)
    for path, stage in ((ws.items_manifest, "ingest"), (ws.dedup_manifest, "dedup")):
        if not path.is_file():
            raise DatasetStateError(f"{path} not found — run the {stage} stage first.")
    items = {r.item_id: r for r in read_records(ws.items_manifest, ItemRecord)}
    dedup = {d.item_id: d for d in read_records(ws.dedup_manifest, DedupRecord)}

    if set(dedup) != set(items):
        added = len(set(items) - set(dedup))
        removed = len(set(dedup) - set(items))
        raise DatasetStateError(
            f"dedup.csv does not match items.csv ({added} items not deduplicated, {removed} "
            "no longer ingested) — re-run the dedup stage."
        )
    mismatched = sorted(i for i in items if items[i].source_id != dedup[i].source_id)
    if mismatched:
        raise DatasetStateError(
            f"dedup.csv source_id differs from items.csv for {len(mismatched)} items "
            f"(e.g. {mismatched[0]}) — re-run the dedup stage."
        )

    settings = dedup_settings(cfg.dedup)
    _check_dedup_settings(ws, settings)
    _check_dedup_content(ws, cfg, items, dedup)

    rows = tuple(
        SplitInputRow(
            item_id=i.item_id,
            source_id=i.source_id,
            source_type=i.source_type,
            frame_index=i.frame_index,
            split_group=dedup[i.item_id].split_group,
            file_sha256=i.sha256,
        )
        for i in sorted(items.values(), key=lambda r: r.item_id)
        if not dedup[i.item_id].excluded
    )
    return SplitInput(
        rows=rows,
        snapshot_fp=snapshot_fingerprint(rows),
        dedup_settings_fp=digest({"schema": SPLIT_INPUT_SCHEMA, "dedup": settings}),
        excluded=len(items) - len(rows),
    )


def _check_dedup_settings(ws: Workspace, expected: dict[str, Any]) -> None:
    report = ws.reports / "duplicates.json"
    try:
        recorded = json.loads(report.read_text(encoding="utf-8"))["settings"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise DatasetStateError(
            f"Cannot read dedup settings from {report} ({exc}) — re-run the dedup stage."
        ) from exc
    if recorded != expected:
        changed = sorted(k for k in expected if recorded.get(k) != expected[k])
        raise DatasetStateError(
            f"dedup.csv was produced with different dedup settings ({', '.join(changed)}) "
            "than the current config — re-run the dedup stage."
        )


def _check_dedup_content(
    ws: Workspace,
    cfg: PipelineConfig,
    items: dict[str, ItemRecord],
    dedup: dict[str, DedupRecord],
) -> None:
    """dedup.csv must describe the files currently referenced by items.csv."""
    if not cfg.dedup.enabled:
        regrouped = sorted(i for i, d in dedup.items() if d.split_group != d.source_id)
        if regrouped:
            raise DatasetStateError(
                "dedup is disabled but dedup.csv contains merged split groups — re-run dedup."
            )
        return
    cache_path: Path = ws.image_hash_cache
    if not cache_path.is_file():
        raise DatasetStateError(f"{cache_path} not found — re-run the dedup stage.")
    cache = {
        r.sha256: r
        for r in read_records(cache_path, ImageHashRecord)
        if r.hash_size == cfg.dedup.hash_size
    }
    stale = sorted(
        i
        for i, item in items.items()
        if item.sha256 not in cache or cache[item.sha256].pixel_sha256 != dedup[i].pixel_sha256
    )
    if stale:
        raise DatasetStateError(
            f"dedup.csv was computed from different file content for {len(stale)} items "
            f"(e.g. {stale[0]}; workspace files changed since dedup) — re-run the dedup stage."
        )
