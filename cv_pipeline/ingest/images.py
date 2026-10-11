"""Still-image ingestion: verify the file decodes, then copy it byte-for-byte."""

from __future__ import annotations

from pathlib import Path

from cv_pipeline.manifest import ItemRecord, SourceType
from cv_pipeline.utils.io import atomic_copy, read_image, rel_posix


class ImageReadError(RuntimeError):
    pass


def image_item_id(sha256: str) -> str:
    return f"img_{sha256[:16]}"


def ingest_image(
    src: Path,
    sha256: str,
    raw_dir: Path,
    workspace_root: Path,
    dest_dir: Path,
) -> ItemRecord:
    """Copy ``src`` into ``dest_dir`` under a content-addressed name.

    The copy is byte-identical (EXIF preserved). Width/height are measured after EXIF
    orientation is applied, matching what OpenCV/Ultralytics will see downstream.
    """
    image = read_image(src)
    if image is None:
        raise ImageReadError("file could not be decoded as an image")
    height, width = image.shape[:2]

    item_id = image_item_id(sha256)
    dest = dest_dir / f"{item_id}{src.suffix.lower()}"
    atomic_copy(src, dest)

    return ItemRecord(
        item_id=item_id,
        path=rel_posix(dest, workspace_root),
        source_id=item_id,
        source_type=SourceType.IMAGE,
        source_path=rel_posix(src, raw_dir),
        source_sha256=sha256,
        sha256=sha256,
        width=int(width),
        height=int(height),
    )
