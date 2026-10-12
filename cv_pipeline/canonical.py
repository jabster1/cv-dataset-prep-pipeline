"""Deterministic choice of the canonical copy among identical files.

Used for byte-identical raw files (ingest) and byte/pixel-identical items (dedup). Since
the candidates are identical, the choice only affects which path lineage points to — so
the rule is deliberately small and predictable. Lower sort key wins:

1. paths that do not look like a copy: no directory or file-stem word equal to
   duplicate(s) / dup(s) / copy / copies / backup, and no ``name (1).jpg`` suffix
2. (dedup only) video frames before still images — a still that is pixel-identical to an
   extracted frame was exported from it, and keeping the frame keeps the sequence whole
3. fewer directory levels
4. normalised path (Unicode NFC, case-folded)
5. the raw path itself, then (dedup only) frame index and item id — makes the order total
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

COPY_MARKERS = frozenset({"duplicate", "duplicates", "dup", "dups", "copy", "copies", "backup"})
_WORD = re.compile(r"[a-z]+")
_NUMBERED_COPY = re.compile(r"\(\d+\)$")  # "photo (1).jpg" — browser/OS download copies


def looks_like_copy(rel_path: str) -> bool:
    """True if any directory or the file stem marks the file as a copy."""
    p = PurePosixPath(rel_path)
    components = [*p.parent.parts, p.stem]
    for comp in components:
        lowered = comp.casefold()
        if COPY_MARKERS.intersection(_WORD.findall(lowered)):
            return True
    return bool(_NUMBERED_COPY.search(p.stem.strip()))


def normalize_path(rel_path: str) -> str:
    return unicodedata.normalize("NFC", rel_path).casefold()


def canonical_path_key(rel_path: str) -> tuple[bool, int, str, str]:
    """Sort key for a raw-relative POSIX path; the minimum is the canonical copy."""
    depth = len(PurePosixPath(rel_path).parts) - 1
    return (looks_like_copy(rel_path), depth, normalize_path(rel_path), rel_path)
