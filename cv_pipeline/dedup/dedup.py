"""Dedup stage: ``items.csv`` -> ``dedup.csv`` + ``duplicate_pairs.csv`` + duplicates report.

Policy
------
* **Exact duplicates** (byte- or pixel-identical, any source) carry no extra information.
  ``exact_duplicate_action: exclude`` (default) keeps one canonical copy (see
  :func:`_priority`) and marks the rest excluded with ``duplicate_of``.
* **Near-duplicates across different sources** (two photos of the same scene, a photo and
  a video frame) are a leakage risk. They are always recorded and, with
  ``group_near_duplicates``, forced into the same ``split_group``. They are only excluded
  if ``near_duplicate_action: exclude``.
* **Near-duplicates within one source** (adjacent video frames) are never linked or
  excluded: the source already keeps them in one split, and a sequence of similar frames
  is often exactly the useful data. They are summarised per source in the report.

Nothing is deleted. Excluded items remain on disk and in ``items.csv``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from cv_pipeline.canonical import canonical_path_key
from cv_pipeline.config import DedupConfig, PipelineConfig
from cv_pipeline.dedup.hashing import ImageHashes, compute_hashes, hamming, hex_matrix, near_pairs
from cv_pipeline.dedup.report import build_report, render_markdown
from cv_pipeline.manifest import (
    DedupRecord,
    DuplicateKind,
    DuplicatePairRecord,
    ExcludeReason,
    ImageHashRecord,
    ItemRecord,
    RejectedRecord,
    RejectReason,
    SourceType,
    read_records,
    write_records,
)
from cv_pipeline.utils.hashing import sha256_bytes
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text, read_image
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("dedup")

_TYPE_RANK = {SourceType.VIDEO_FRAME: 0, SourceType.IMAGE: 1, SourceType.AUGMENTED: 2}


@dataclass
class DedupResult:
    records: list[DedupRecord]
    pairs: list[DuplicatePairRecord]
    report: dict[str, object]


def run_dedup(cfg: PipelineConfig, force: bool = False) -> DedupResult:
    ws = Workspace(cfg.workspace_dir).ensure()
    if not ws.items_manifest.is_file():
        raise FileNotFoundError(f"{ws.items_manifest} not found — run the ingest stage first.")
    t0 = time.perf_counter()
    dcfg = cfg.dedup

    items = sorted(read_records(ws.items_manifest, ItemRecord), key=lambda r: r.item_id)
    raw_duplicates = _raw_duplicates(ws)
    log_event(logger, "dedup_started", items=len(items), enabled=dcfg.enabled)

    if dcfg.enabled:
        hashes = _hash_items(items, ws, dcfg.hash_size, force)
        pairs, intra_counts = _find_pairs(items, hashes, dcfg)
    else:
        hashes, pairs, intra_counts = {}, [], {}

    by_id = {r.item_id: r for r in items}
    exclusions = _assign_exclusions(items, pairs, dcfg)
    dup_groups = _dup_groups(pairs)
    split_groups = _split_groups(items, pairs, by_id, dcfg)

    records = []
    for item in items:
        h = hashes.get(item.item_id)
        excl = exclusions.get(item.item_id)
        records.append(
            DedupRecord(
                item_id=item.item_id,
                source_id=item.source_id,
                split_group=split_groups[item.source_id],
                phash=h.phash if h else "",
                dhash=h.dhash if h else "",
                pixel_sha256=h.pixel_sha256 if h else "",
                dup_group=dup_groups.get(item.item_id),
                excluded=excl is not None,
                exclude_reason=excl[0] if excl else None,
                duplicate_of=excl[1] if excl else None,
            )
        )

    intra = _intra_source_summary(items, hashes, dcfg) if dcfg.enabled else []
    report = build_report(cfg, items, records, pairs, raw_duplicates, intra, intra_counts)

    write_records(ws.dedup_manifest, records, DedupRecord)
    write_records(ws.duplicate_pairs_manifest, pairs, DuplicatePairRecord)
    atomic_write_json(ws.reports / "duplicates.json", report)
    atomic_write_text(ws.reports / "duplicates.md", render_markdown(report))

    for warning in report["warnings"]:  # type: ignore[union-attr]
        log_event(logger, "dedup_warning", logging.WARNING, detail=warning)
    log_event(
        logger,
        "dedup_finished",
        seconds=round(time.perf_counter() - t0, 2),
        pairs=len(pairs),
        excluded=sum(r.excluded for r in records),
        dup_groups=len(set(dup_groups.values())),
        split_groups=len(set(split_groups.values())),
    )
    return DedupResult(records, pairs, report)


# --------------------------------------------------------------------------------------
# Hashing (with cache)
# --------------------------------------------------------------------------------------


def _hash_items(
    items: list[ItemRecord], ws: Workspace, hash_size: int, force: bool
) -> dict[str, ImageHashes]:
    cache: dict[tuple[str, int], ImageHashRecord] = {}
    if not force and ws.image_hash_cache.is_file():
        try:
            for rec in read_records(ws.image_hash_cache, ImageHashRecord):
                cache[(rec.sha256, rec.hash_size)] = rec
        except (ValueError, KeyError) as exc:
            log_event(logger, "hash_cache_unreadable", logging.WARNING, error=str(exc))

    out: dict[str, ImageHashes] = {}
    used: dict[str, ImageHashRecord] = {}
    computed = 0
    for item in items:
        cached = cache.get((item.sha256, hash_size)) or used.get(item.sha256)
        if cached is None:
            path = ws.root / item.path
            image = read_image(path)
            if image is None:
                raise FileNotFoundError(
                    f"Cannot read {path} for item {item.item_id}; re-run ingest (with --force)."
                )
            h = compute_hashes(image, hash_size)
            cached = ImageHashRecord(item.sha256, hash_size, h.phash, h.dhash, h.pixel_sha256)
            computed += 1
        used[item.sha256] = cached
        out[item.item_id] = ImageHashes(cached.phash, cached.dhash, cached.pixel_sha256)

    write_records(ws.image_hash_cache, sorted(used.values(), key=lambda r: r.sha256), ImageHashRecord)
    log_event(logger, "hashes_ready", computed=computed, cached=len(items) - computed)
    return out


# --------------------------------------------------------------------------------------
# Pair finding
# --------------------------------------------------------------------------------------


def _find_pairs(
    items: list[ItemRecord], hashes: dict[str, ImageHashes], dcfg: DedupConfig
) -> tuple[list[DuplicatePairRecord], dict[str, int]]:
    """Return linked pairs and, per source, the number of same-source near pairs skipped."""
    pairs: dict[tuple[str, str], DuplicatePairRecord] = {}

    def add(a: ItemRecord, b: ItemRecord, kind: str) -> None:
        a, b = (a, b) if a.item_id < b.item_id else (b, a)
        key = (a.item_id, b.item_id)
        if key in pairs:
            return
        ha, hb = hashes[a.item_id], hashes[b.item_id]
        pairs[key] = DuplicatePairRecord(
            item_a=a.item_id,
            item_b=b.item_id,
            kind=kind,
            phash_distance=hamming(ha.phash, hb.phash),
            dhash_distance=hamming(ha.dhash, hb.dhash),
            same_source=a.source_id == b.source_id,
        )

    # Exact: byte-identical first, then pixel-identical (different bytes, same pixels).
    for group in _group_by(items, lambda r: r.sha256):
        for a, b in combinations(group, 2):
            add(a, b, DuplicateKind.BYTE_IDENTICAL)
    for group in _group_by(items, lambda r: hashes[r.item_id].pixel_sha256):
        for a, b in combinations(group, 2):
            add(a, b, DuplicateKind.PIXEL_IDENTICAL)

    # Near: vectorised search, cross-source only. Same-source pairs are counted, not linked.
    sources = np.array([r.source_id for r in items])
    ph = hex_matrix([hashes[r.item_id].phash for r in items])
    dh = hex_matrix([hashes[r.item_id].dhash for r in items])
    thresholds = (dcfg.near_duplicate_threshold, dcfg.dhash_threshold)
    for i, j, _, _ in near_pairs(ph, dh, *thresholds, groups=sources):
        add(items[i], items[j], DuplicateKind.NEAR_DUPLICATE)

    position = {r.item_id: i for i, r in enumerate(items)}
    intra_counts: dict[str, int] = {}
    for src_items in _group_by(items, lambda r: r.source_id):
        idx = np.array([position[r.item_id] for r in src_items])
        n = sum(1 for _ in near_pairs(ph[idx], dh[idx], *thresholds))
        if n:
            intra_counts[src_items[0].source_id] = n

    return sorted(pairs.values(), key=lambda p: (p.item_a, p.item_b)), intra_counts


def _group_by(
    items: list[ItemRecord], key: Callable[[ItemRecord], str]
) -> Iterable[list[ItemRecord]]:
    """Groups of 2+ items sharing ``key``, in sorted key order."""
    groups: dict[str, list[ItemRecord]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return (g for _, g in sorted(groups.items()) if len(g) > 1)


# --------------------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------------------


def _priority(item: ItemRecord) -> tuple[object, ...]:
    """Which copy to keep; lower wins. See :mod:`cv_pipeline.canonical` for the rule."""
    copy_like, depth, normalized, path = canonical_path_key(item.source_path)
    return (
        copy_like,
        _TYPE_RANK.get(item.source_type, 9),
        depth,
        normalized,
        path,
        item.frame_index if item.frame_index is not None else -1,
        item.item_id,
    )


def _assign_exclusions(
    items: list[ItemRecord], pairs: list[DuplicatePairRecord], dcfg: DedupConfig
) -> dict[str, tuple[str, str]]:
    """Greedy, order-deterministic: an item is excluded if it duplicates an already-kept item."""
    exact_adj: dict[str, set[str]] = defaultdict(set)
    near_adj: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        if p.kind == DuplicateKind.NEAR_DUPLICATE:
            near_adj[p.item_a].add(p.item_b)
            near_adj[p.item_b].add(p.item_a)
        else:
            exact_adj[p.item_a].add(p.item_b)
            exact_adj[p.item_b].add(p.item_a)

    order = sorted(items, key=_priority)
    rank = {r.item_id: i for i, r in enumerate(order)}
    kept: set[str] = set()
    excluded: dict[str, tuple[str, str]] = {}
    for item in order:
        iid = item.item_id
        if dcfg.exact_duplicate_action == "exclude":
            hits = sorted(exact_adj[iid] & kept, key=rank.__getitem__)
            if hits:
                excluded[iid] = (ExcludeReason.EXACT_DUPLICATE, hits[0])
                continue
        if dcfg.near_duplicate_action == "exclude":
            hits = sorted(near_adj[iid] & kept, key=rank.__getitem__)
            if hits:
                excluded[iid] = (ExcludeReason.NEAR_DUPLICATE, hits[0])
                continue
        kept.add(iid)
    return excluded


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root: lexicographically smallest.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo

    def components(self) -> dict[str, list[str]]:
        comps: dict[str, list[str]] = defaultdict(list)
        for x in self.parent:
            comps[self.find(x)].append(x)
        return {root: sorted(m) for root, m in comps.items()}


def _group_id(prefix: str, members: list[str]) -> str:
    return f"{prefix}_{sha256_bytes(','.join(sorted(members)).encode())[:12]}"


def _dup_groups(pairs: list[DuplicatePairRecord]) -> dict[str, str]:
    """item_id -> duplicate-group id, for items linked by any recorded pair."""
    uf = _UnionFind()
    for p in pairs:
        uf.union(p.item_a, p.item_b)
    out: dict[str, str] = {}
    for members in uf.components().values():
        gid = _group_id("dg", members)
        for m in members:
            out[m] = gid
    return out


def _split_groups(
    items: list[ItemRecord],
    pairs: list[DuplicatePairRecord],
    by_id: dict[str, ItemRecord],
    dcfg: DedupConfig,
) -> dict[str, str]:
    """source_id -> split-group id. Sources joined by exact pairs (always) and cross-source
    near-duplicates (when ``group_near_duplicates``) share a group."""
    uf = _UnionFind()
    for item in items:
        uf.find(item.source_id)
    for p in pairs:
        if p.kind == DuplicateKind.NEAR_DUPLICATE and not dcfg.group_near_duplicates:
            continue
        uf.union(by_id[p.item_a].source_id, by_id[p.item_b].source_id)
    out: dict[str, str] = {}
    for members in uf.components().values():
        gid = members[0] if len(members) == 1 else _group_id("sg", members)
        for m in members:
            out[m] = gid
    return out


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------


def _raw_duplicates(ws: Workspace) -> list[RejectedRecord]:
    if not ws.rejected_manifest.is_file():
        return []
    return [
        r
        for r in read_records(ws.rejected_manifest, RejectedRecord)
        if r.reason == RejectReason.EXACT_DUPLICATE
    ]


def _intra_source_summary(
    items: list[ItemRecord], hashes: dict[str, ImageHashes], dcfg: DedupConfig
) -> list[dict[str, object]]:
    """Redundancy of consecutive frames within each video — informational only."""
    by_source: dict[str, list[ItemRecord]] = defaultdict(list)
    for item in items:
        if item.source_type == SourceType.VIDEO_FRAME:
            by_source[item.source_id].append(item)

    rows: list[dict[str, object]] = []
    for source_id, frames in sorted(by_source.items()):
        frames.sort(key=lambda r: r.frame_index or 0)
        dists = []
        similar = 0
        for a, b in zip(frames, frames[1:], strict=False):
            dp = hamming(hashes[a.item_id].phash, hashes[b.item_id].phash)
            dd = hamming(hashes[a.item_id].dhash, hashes[b.item_id].dhash)
            dists.append(dp)
            if dp <= dcfg.near_duplicate_threshold and (
                dcfg.dhash_threshold is None or dd <= dcfg.dhash_threshold
            ):
                similar += 1
        n_consecutive = max(len(frames) - 1, 0)
        rows.append(
            {
                "source_id": source_id,
                "source_path": frames[0].source_path,
                "frames": len(frames),
                "similar_consecutive_pairs": similar,
                "consecutive_pairs": n_consecutive,
                "similar_ratio": round(similar / n_consecutive, 3) if n_consecutive else 0.0,
                "min_phash_distance": min(dists) if dists else None,
                "median_phash_distance": float(np.median(dists)) if dists else None,
            }
        )
    return rows
