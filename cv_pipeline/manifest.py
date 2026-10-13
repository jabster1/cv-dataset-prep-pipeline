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
    origin: str = "raw"  # "raw" or "import:<name>"; source_path is relative to that root


@dataclass
class RejectedRecord:
    """A raw file (or derived item) that was not carried forward, and why."""

    source_path: str
    reason: str  # RejectReason
    detail: str = ""
    duplicate_of: str | None = None
    origin: str = "raw"


class RejectReason:
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNREADABLE_IMAGE = "unreadable_image"
    UNREADABLE_VIDEO = "unreadable_video"
    NO_FRAMES_EXTRACTED = "no_frames_extracted"
    EXACT_DUPLICATE = "exact_duplicate"
    IMPORT_FILE_MODIFIED = "import_file_modified"


RAW_ORIGIN = "raw"


def import_origin(name: str) -> str:
    return f"import:{name}"


@dataclass
class SourceRefRecord:
    """Every file that maps to a source, from any origin. Byte-identical files share a
    ``source_id``; exactly one reference per source is ``canonical`` (the one in items.csv),
    the rest are ``duplicate``. Provenance from all of them is preserved here."""

    source_id: str
    origin: str
    source_path: str
    source_sha256: str
    role: str  # canonical | duplicate
    legacy_split: str = ""  # train | val | test for imported datasets
    family_key: str = ""  # must-stay-together lineage family (see cv_pipeline.lineage)


@dataclass
class ImageHashRecord:
    """Cache of per-file hashes, keyed by (sha256, hash_size)."""

    sha256: str
    hash_size: int
    phash: str
    dhash: str
    pixel_sha256: str


class DuplicateKind:
    BYTE_IDENTICAL = "byte_identical"
    PIXEL_IDENTICAL = "pixel_identical"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass
class DuplicatePairRecord:
    """A linked pair of items. Same-source near-duplicates (e.g. adjacent video frames)
    are deliberately not listed — they are summarised per source in the report."""

    item_a: str
    item_b: str
    kind: str  # DuplicateKind
    phash_distance: int
    dhash_distance: int
    same_source: bool


class ExcludeReason:
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass
class DedupRecord:
    """Dedup outcome for one item. ``excluded`` items stay on disk and in ``items.csv``;
    later stages skip them. ``split_group`` is the unit the splitter assigns to a split."""

    item_id: str
    source_id: str
    split_group: str
    phash: str
    dhash: str
    pixel_sha256: str
    dup_group: str | None = None
    excluded: bool = False
    exclude_reason: str | None = None  # ExcludeReason
    duplicate_of: str | None = None


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
