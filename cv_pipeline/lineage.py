"""Must-stay-together lineage relationships between sources (leakage prevention only).

Today the only relationship is a Roboflow derivative family inside one imported dataset
(see ``imports.yolo.family_key``). It is recorded per file reference in
``manifests/source_refs.csv`` and dedup unions every family into one split group.

A lineage link never makes items duplicates, never excludes anything, and never deletes
anything; it only forces related sources into the same split.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from cv_pipeline.manifest import SourceRefRecord, read_records
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.workspace import Workspace

LINEAGE_SCHEMA = "cvp-lineage-v1"


def family_links(ws: Workspace, present_sources: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """family key -> sorted source ids, for families linking 2+ present sources."""
    present = set(present_sources)
    members: dict[str, set[str]] = defaultdict(set)
    if ws.source_refs_manifest.is_file():
        for r in read_records(ws.source_refs_manifest, SourceRefRecord):
            if r.family_key and r.source_id in present:
                members[r.family_key].add(r.source_id)
    return {k: tuple(sorted(v)) for k, v in sorted(members.items()) if len(v) > 1}


def lineage_fp(links: dict[str, tuple[str, ...]]) -> str:
    return digest({"schema": LINEAGE_SCHEMA, "families": {k: list(v) for k, v in links.items()}})


def lineage_summary(links: dict[str, tuple[str, ...]]) -> dict[str, object]:
    return {
        "families": len(links),
        "sources": sum(len(v) for v in links.values()),
        "lineage_fp": lineage_fp(links),
    }


def components(sources: Iterable[str], links: dict[str, tuple[str, ...]]) -> dict[str, str]:
    """source -> smallest source id of its lineage component (deterministic)."""
    parent = {s: s for s in sources}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in links.values():
        for other in members[1:]:
            a, b = find(members[0]), find(other)
            if a != b:
                lo, hi = sorted((a, b))
                parent[hi] = lo
    return {s: find(s) for s in parent}
