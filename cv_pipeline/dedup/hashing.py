"""Perceptual and pixel hashing, plus fast pairwise Hamming-distance search.

Implemented with OpenCV + NumPy only (no ``imagehash`` dependency):

* **pHash** — grayscale, resize to ``4*hash_size`` square, 2-D DCT, keep the top-left
  ``hash_size x hash_size`` low-frequency block, threshold at its median (DC excluded).
  Robust to re-encoding, mild blur, brightness shifts and small resizes.
* **dHash** — grayscale, resize to ``(hash_size+1) x hash_size``, one bit per horizontal
  gradient sign. Captures structure that pHash can miss; requiring *both* to match cuts
  false positives.
* **pixel hash** — SHA-256 of the decoded pixels, so images that are pixel-identical but
  byte-different (re-saved PNG, stripped metadata) are recognised as exact duplicates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from cv_pipeline.utils.hashing import sha256_bytes

# popcount for every byte value
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ImageHashes:
    phash: str  # hex
    dhash: str  # hex
    pixel_sha256: str


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8).ravel()).tobytes().hex()


def phash(image: np.ndarray, hash_size: int = 8) -> str:
    side = hash_size * 4
    small = cv2.resize(_gray(image), (side, side), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))[:hash_size, :hash_size]
    flat = dct.ravel()
    median = np.median(flat[1:])  # exclude DC (overall brightness)
    return _bits_to_hex(flat > median)


def dhash(image: np.ndarray, hash_size: int = 8) -> str:
    small = cv2.resize(_gray(image), (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    small = small.astype(np.int16)
    return _bits_to_hex(small[:, 1:] > small[:, :-1])


def pixel_sha256(image: np.ndarray) -> str:
    header = f"{image.shape}|{image.dtype}|".encode()
    return sha256_bytes(header + np.ascontiguousarray(image).tobytes())


def compute_hashes(image: np.ndarray, hash_size: int = 8) -> ImageHashes:
    return ImageHashes(phash(image, hash_size), dhash(image, hash_size), pixel_sha256(image))


def hamming(a_hex: str, b_hex: str) -> int:
    a = np.frombuffer(bytes.fromhex(a_hex), dtype=np.uint8)
    b = np.frombuffer(bytes.fromhex(b_hex), dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("hashes have different lengths (different hash_size?)")
    return int(_POPCOUNT[a ^ b].sum())


def hex_matrix(hashes: list[str]) -> np.ndarray:
    """Stack hex hashes into an ``(n, n_bytes)`` uint8 matrix."""
    if not hashes:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hashes])


def near_pairs(
    phashes: np.ndarray,
    dhashes: np.ndarray,
    phash_threshold: int,
    dhash_threshold: int | None,
    groups: np.ndarray | None = None,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(i, j, phash_dist, dhash_dist)`` for ``i < j`` within both thresholds.

    If ``groups`` is given, pairs whose group labels are equal are skipped (used to
    ignore same-source pairs without materialising them).

    Vectorised one row at a time: O(n^2) comparisons but each is a few bytes, which is
    fine for tens of thousands of images on a laptop.
    """
    n = phashes.shape[0]
    for i in range(n - 1):
        dp = _POPCOUNT[phashes[i + 1 :] ^ phashes[i]].sum(axis=1, dtype=np.int32)
        dd = _POPCOUNT[dhashes[i + 1 :] ^ dhashes[i]].sum(axis=1, dtype=np.int32)
        mask = dp <= phash_threshold
        if dhash_threshold is not None:
            mask &= dd <= dhash_threshold
        if groups is not None:
            mask &= groups[i + 1 :] != groups[i]
        for off in np.flatnonzero(mask):
            yield i, i + 1 + int(off), int(dp[off]), int(dd[off])
