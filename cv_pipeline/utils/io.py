"""Filesystem helpers. All writes go through atomic temp-file + rename."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import cv2
import numpy as np


@contextmanager
def atomic_open(path: Path, mode: str = "w", **kwargs: Any) -> Iterator[IO[Any]]:
    """Open a temp file next to ``path`` and move it into place only on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, **kwargs) as fh:
            yield fh
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    with atomic_open(path, "wb") as fh:
        fh.write(data)


def atomic_write_text(path: Path, text: str) -> None:
    with atomic_open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy file contents (not permissions/metadata) from ``src`` to ``dst`` atomically.

    ``src`` is only ever opened for reading.
    """
    with src.open("rb") as fin, atomic_open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def rel_posix(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` as a forward-slash string (stable across OSes)."""
    return path.resolve().relative_to(base.resolve()).as_posix()


def is_relative_to(path: Path, other: Path) -> bool:
    return path.resolve().is_relative_to(other.resolve())


def read_image(path: Path) -> np.ndarray | None:
    """Decode an image as BGR (EXIF orientation applied). Returns ``None`` if undecodable.

    Uses ``imdecode`` on raw bytes so non-ASCII paths work on every platform.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def encode_image(image: np.ndarray, ext: str, jpeg_quality: int = 95) -> bytes:
    ext = ext if ext.startswith(".") else f".{ext}"
    params: list[int] = []
    if ext.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"OpenCV could not encode image as {ext}")
    return buf.tobytes()
