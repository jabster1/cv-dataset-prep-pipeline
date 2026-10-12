"""Deterministic identities for annotation inputs, settings, results and runs.

All fingerprints are SHA-256 over canonical JSON (sorted keys, compact separators,
ASCII). Floats are rounded before they reach this module.

Identity scheme
---------------
* ``inference_fp``  — backend + framework versions + model weights + inference params
* ``policy_fp``     — classes, class_map, thresholds, review settings, seed
* ``spec_fp``       — sha256(inference_fp, policy_fp)
* ``snapshot_fp``   — ordered eligible items with lineage and input identities
* ``result_digest`` — ordered per-item inference outcomes (status, error, raw detections)
* ``run_id``        — "ann_" + sha256(spec_fp, snapshot_fp, result_digest)[:16]

A run directory's contents are a pure function of these inputs, so a given ``run_id``
always denotes the same bytes and completed runs never need to be rewritten.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from cv_pipeline.annotation.schema import SCHEMA_VERSION

PIXEL_HASH_VERSION = "cvp-input-pixels-v1"


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def input_pixel_sha256(image: np.ndarray) -> str:
    """Identity of the exact array handed to ``AnnotationBackend.predict``.

    The array is the ``uint8`` BGR ``(H, W, 3)`` result of ``cv2.imdecode(...,
    IMREAD_COLOR)``. It is hashed as::

        sha256(b"cvp-input-pixels-v1\\n"
               + f"{H} {W} 3 uint8 BGR HWC C\\n".encode("ascii")
               + C-contiguous bytes: rows top->bottom, pixels left->right, B,G,R per pixel)
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 (H, W, 3) BGR image, got {image.dtype} {image.shape}")
    h, w = image.shape[:2]
    hasher = hashlib.sha256()
    hasher.update(f"{PIXEL_HASH_VERSION}\n".encode("ascii"))
    hasher.update(f"{h} {w} 3 uint8 BGR HWC C\n".encode("ascii"))
    hasher.update(np.ascontiguousarray(image).tobytes())
    return hasher.hexdigest()


def combine(*parts: str) -> str:
    return digest({"schema": SCHEMA_VERSION, "parts": list(parts)})


def make_run_id(spec_fp: str, snapshot_fp: str, result_digest: str) -> str:
    return "ann_" + combine(spec_fp, snapshot_fp, result_digest)[:16]
