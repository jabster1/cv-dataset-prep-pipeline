"""Canonical JSON digests shared by every stage that fingerprints its inputs or outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, compact separators, ASCII only, NaN/Infinity rejected."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()
