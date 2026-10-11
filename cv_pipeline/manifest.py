"""Typed CSV manifests.

Manifests are the contract between stages: each stage reads the previous stage's
manifest and writes its own. CSV keeps them diffable and spreadsheet-friendly.
``None`` is stored as an empty cell.
"""

from __future__ import annotations

import csv
import types
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

from cv_pipeline.utils.io import atomic_open

T = TypeVar("T")


class SourceType:
    IMAGE = "image"
    VIDEO_FRAME = "video_frame"
    AUGMENTED = "augmented"


@dataclass
class ItemRecord:
    """One image known to the pipeline, with full lineage back to raw data.

    ``source_id`` is the leakage-prevention key: every item derived from the same raw
    file (all frames of a video, all augmentations of an image) shares it.
    """

    item_id: str
    path: str  # relative to the workspace root, POSIX separators
    source_id: str
    source_type: str  # SourceType
    source_path: str  # relative to ingest.raw_dir, POSIX separators
    source_sha256: str
    sha256: str
    width: int
    height: int
    frame_index: int | None = None
    timestamp_s: float | None = None
    parent_item_id: str | None = None


@dataclass
class RejectedRecord:
    """A raw file (or derived item) that was not carried forward, and why."""

    source_path: str
    reason: str  # RejectReason
    detail: str = ""
    duplicate_of: str | None = None


class RejectReason:
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNREADABLE_IMAGE = "unreadable_image"
    UNREADABLE_VIDEO = "unreadable_video"
    NO_FRAMES_EXTRACTED = "no_frames_extracted"
    EXACT_DUPLICATE = "exact_duplicate"


def write_records(path: Path, records: Iterable[T], cls: type[T]) -> int:
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    count = 0
    with atomic_open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for rec in records:
            writer.writerow({n: _to_cell(getattr(rec, n)) for n in names})
            count += 1
    return count


def read_records(path: Path, cls: type[T]) -> list[T]:
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]  # type: ignore[arg-type]
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {n for n in names if _is_required(cls, n)} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: manifest is missing columns {sorted(missing)}")
        return [
            cls(**{n: _from_cell(hints[n], row[n]) for n in names if n in row})  # type: ignore[call-arg]
            for row in reader
        ]


def _is_required(cls: type, name: str) -> bool:
    f = next(f for f in fields(cls) if f.name == name)
    return f.default is MISSING and f.default_factory is MISSING


def _to_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(round(value, 6))
    return value


def _from_cell(tp: Any, raw: str | None) -> Any:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        if raw is None or raw == "":
            return None
        (inner,) = [a for a in get_args(tp) if a is not type(None)]
        return _from_cell(inner, raw)
    if raw is None:
        raw = ""
    if tp is int:
        return int(raw)
    if tp is float:
        return float(raw)
    if tp is bool:
        return raw.strip().lower() in {"1", "true", "yes"}
    return raw
