"""Duplicates report: one deterministic JSON document + a Markdown rendering of it."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from cv_pipeline.config import PipelineConfig
from cv_pipeline.manifest import (
    DedupRecord,
    DuplicateKind,
    DuplicatePairRecord,
    ItemRecord,
    RejectedRecord,
)

REDUNDANT_VIDEO_RATIO = 0.5
_MD_MAX_GROUPS = 200


def build_report(
    cfg: PipelineConfig,
    items: list[ItemRecord],
    records: list[DedupRecord],
    pairs: list[DuplicatePairRecord],
    raw_duplicates: list[RejectedRecord],
    intra: list[dict[str, Any]],
    intra_counts: dict[str, int],
) -> dict[str, Any]:
    d = cfg.dedup
    by_id = {i.item_id: i for i in items}
    rec_by_id = {r.item_id: r for r in records}

    kinds = Counter(p.kind for p in pairs)
    cross_source_near = sum(1 for p in pairs if p.kind == DuplicateKind.NEAR_DUPLICATE)
    excluded = [r for r in records if r.excluded]

    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.dup_group:
            groups[r.dup_group].append(r.item_id)
    group_kinds: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        group_kinds[rec_by_id[p.item_a].dup_group or ""].add(p.kind)

    dup_groups = []
    for gid, members in sorted(groups.items()):
        dup_groups.append(
            {
                "group": gid,
                "kinds": sorted(group_kinds[gid]),
                "members": [
                    {
                        "item_id": m,
                        "source_path": by_id[m].source_path,
                        "source_type": by_id[m].source_type,
                        "frame_index": by_id[m].frame_index,
                        "excluded": rec_by_id[m].excluded,
                        "exclude_reason": rec_by_id[m].exclude_reason,
                        "duplicate_of": rec_by_id[m].duplicate_of,
                    }
                    for m in sorted(members)
                ],
            }
        )

    split_sizes = Counter(r.split_group for r in records)
    sources_per_group: dict[str, set[str]] = defaultdict(set)
    for r in records:
        sources_per_group[r.split_group].add(r.source_id)
    largest_id, largest_n = (
        min(split_sizes.items(), key=lambda kv: (-kv[1], kv[0])) if split_sizes else ("", 0)
    )
    largest_fraction = round(largest_n / len(records), 4) if records else 0.0

    warnings: list[str] = []
    notes: list[str] = []
    holdout = [f for f in (cfg.split.val, cfg.split.test) if f > 0]
    if holdout and records and largest_fraction > min(holdout):
        warnings.append(
            f"Split group {largest_id} holds {largest_fraction:.0%} of items, more than the "
            f"smallest hold-out split ({min(holdout):.0%}); split fractions will deviate. "
            "Add more independent sources or lower video sampling for that source."
        )
    if cross_source_near and not d.group_near_duplicates and d.near_duplicate_action == "keep":
        warnings.append(
            f"{cross_source_near} cross-source near-duplicate pairs are kept and NOT grouped "
            "(dedup.group_near_duplicates: false); they may land in different splits and leak."
        )
    for row in intra:
        if row["frames"] >= 3 and row["similar_ratio"] >= REDUNDANT_VIDEO_RATIO:
            notes.append(
                f"{row['source_path']}: {row['similar_consecutive_pairs']}/"
                f"{row['consecutive_pairs']} consecutive frames are near-identical. Frames are "
                "kept; consider lowering ingest.video.fps if this is more data than needed."
            )

    return {
        "settings": {
            "enabled": d.enabled,
            "hash_size": d.hash_size,
            "phash_threshold": d.near_duplicate_threshold,
            "dhash_threshold": d.dhash_threshold,
            "exact_duplicate_action": d.exact_duplicate_action,
            "near_duplicate_action": d.near_duplicate_action,
            "group_near_duplicates": d.group_near_duplicates,
        },
        "items_analysed": len(records),
        "items_kept": len(records) - len(excluded),
        "raw_exact_duplicates": [
            {"source_path": r.source_path, "duplicate_of": r.duplicate_of} for r in raw_duplicates
        ],
        "pairs": {
            DuplicateKind.BYTE_IDENTICAL: kinds.get(DuplicateKind.BYTE_IDENTICAL, 0),
            DuplicateKind.PIXEL_IDENTICAL: kinds.get(DuplicateKind.PIXEL_IDENTICAL, 0),
            "near_duplicate_cross_source": cross_source_near,
            "near_duplicate_same_source_not_linked": sum(intra_counts.values()),
        },
        "excluded": {
            "total": len(excluded),
            "by_reason": dict(sorted(Counter(r.exclude_reason for r in excluded).items())),
        },
        "duplicate_groups": dup_groups,
        "video_redundancy": intra,
        "split_groups": {
            "count": len(split_sizes),
            "multi_source": sum(1 for s in sources_per_group.values() if len(s) > 1),
            "largest": {"split_group": largest_id, "items": largest_n, "fraction": largest_fraction},
        },
        "warnings": warnings,
        "notes": notes,
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def render_markdown(report: dict[str, Any]) -> str:
    s = report["settings"]
    p = report["pairs"]
    lines = [
        "# Duplicates report",
        "",
        "Nothing has been deleted. Excluded items remain on disk and in `items.csv`; "
        "later stages skip them. See `manifests/dedup.csv` and `manifests/duplicate_pairs.csv`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items analysed | {report['items_analysed']} |",
        f"| Items kept | {report['items_kept']} |",
        f"| Items excluded | {report['excluded']['total']} |",
        f"| Raw files skipped at ingest as byte-identical | {len(report['raw_exact_duplicates'])} |",
        f"| Byte-identical pairs | {p[DuplicateKind.BYTE_IDENTICAL]} |",
        f"| Pixel-identical pairs | {p[DuplicateKind.PIXEL_IDENTICAL]} |",
        f"| Near-duplicate pairs across sources | {p['near_duplicate_cross_source']} |",
        f"| Near-duplicate pairs within a source (not linked) | "
        f"{p['near_duplicate_same_source_not_linked']} |",
        f"| Duplicate groups | {len(report['duplicate_groups'])} |",
        f"| Split groups | {report['split_groups']['count']} "
        f"({report['split_groups']['multi_source']} span multiple sources) |",
        "",
        f"Settings: hash_size={s['hash_size']}, pHash ≤ {s['phash_threshold']}, "
        f"dHash ≤ {s['dhash_threshold']}, exact → {s['exact_duplicate_action']}, "
        f"near → {s['near_duplicate_action']}, group near-duplicates: "
        f"{s['group_near_duplicates']}.",
        "",
    ]
    if not s["enabled"]:
        lines += ["> Dedup is **disabled**; only source-based split groups were assigned.", ""]

    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- ⚠️ {w}" for w in report["warnings"]] + [""]
    if report["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in report["notes"]] + [""]

    if report["raw_exact_duplicates"]:
        lines += [
            "## Byte-identical raw files (skipped at ingest)",
            "",
            "| Raw file | Duplicate of item |",
            "|---|---|",
        ]
        lines += [
            f"| {_cell(r['source_path'])} | {_cell(r['duplicate_of'])} |"
            for r in report["raw_exact_duplicates"]
        ]
        lines.append("")

    groups = report["duplicate_groups"]
    if groups:
        lines += ["## Duplicate groups", ""]
        for g in groups[:_MD_MAX_GROUPS]:
            lines += [
                f"### {g['group']} ({', '.join(g['kinds'])})",
                "",
                "| Item | Source | Frame | Status |",
                "|---|---|---|---|",
            ]
            for m in g["members"]:
                status = (
                    f"excluded ({m['exclude_reason']}, duplicate of {m['duplicate_of']})"
                    if m["excluded"]
                    else "kept"
                )
                lines.append(
                    f"| {m['item_id']} | {_cell(m['source_path'])} | {_cell(m['frame_index'])} "
                    f"| {status} |"
                )
            lines.append("")
        if len(groups) > _MD_MAX_GROUPS:
            lines += [f"_{len(groups) - _MD_MAX_GROUPS} more groups in duplicates.json._", ""]

    if report["video_redundancy"]:
        lines += [
            "## Video frame redundancy (informational — frames are never excluded for this)",
            "",
            "| Video | Frames | Near-identical consecutive pairs | Median pHash distance |",
            "|---|---|---|---|",
        ]
        for r in report["video_redundancy"]:
            lines.append(
                f"| {_cell(r['source_path'])} | {r['frames']} | "
                f"{r['similar_consecutive_pairs']}/{r['consecutive_pairs']} | "
                f"{_cell(r['median_phash_distance'])} |"
            )
        lines.append("")
    return "\n".join(lines)
