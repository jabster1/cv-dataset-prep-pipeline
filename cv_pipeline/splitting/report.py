"""Split report (``reports/split.{json,md}``): requested vs achieved, why they differ,
what changed relative to history, warnings and conflicts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from cv_pipeline.config import PipelineConfig
from cv_pipeline.splitting.allocate import SPLIT_ORDER
from cv_pipeline.splitting.history import PROTECTION

if TYPE_CHECKING:
    from cv_pipeline.dataset_state import SplitInput
    from cv_pipeline.splitting.history import History, SplitItemRecord
    from cv_pipeline.splitting.reconcile import ReconcileResult


def build_report(
    cfg: PipelineConfig,
    inp: SplitInput,
    history: History,
    rec: ReconcileResult,
    lineage: dict[str, Any],
    revision_record: dict[str, Any] | None,
    seed: int,
    items: list[SplitItemRecord] | None,
    movement: dict[str, int] | None,
) -> dict[str, Any]:
    fractions = {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test}
    n = inp.n_items
    last = history.last_revision
    report: dict[str, Any] = {
        "snapshot": {
            "snapshot_fp": inp.snapshot_fp,
            "dedup_settings_fp": inp.dedup_settings_fp,
            "items": n,
            "groups": len(rec.groups),
            "excluded_by_dedup": inp.excluded,
        },
        "annotation_lineage": lineage,
        "revision": {
            "number": revision_record["revision"]
            if revision_record
            else (last or {}).get("revision"),
            "new": revision_record is not None,
            "record": revision_record,
        },
        "config": {
            "fractions": fractions,
            "seed": seed,
            "require_nonempty": cfg.split.require_nonempty,
            "conflict_resolutions": dict(sorted(cfg.split.conflict_resolutions.items())),
        },
        "conflicts": [asdict(c) for c in rec.conflicts],
        "warnings": [],
        "notes": [],
    }
    if lineage.get("status") == "stale":
        report["notes"].append(
            f"Annotation run {lineage['run_id']} was made from a different dataset state; its "
            "labels must be refreshed (re-run annotate) before being combined with these splits."
        )
    if rec.conflicts:
        report["warnings"].append(
            f"{len(rec.conflicts)} split-history conflict(s); nothing in splits/ was changed."
        )
        return report

    assert items is not None
    by_group = Counter(i.split_group for i in items)
    split_of = {i.split_group: i.split for i in items}
    targets = {s: fractions[s] * n for s in SPLIT_ORDER}
    stats: dict[str, Any] = {}
    for s in SPLIT_ORDER:
        members = [i for i in items if i.split == s]
        stats[s] = {
            "requested_fraction": fractions[s],
            "target_items": round(targets[s], 2),
            "items": len(members),
            "fraction": round(len(members) / n, 4) if n else 0.0,
            "deviation_items": round(len(members) - targets[s], 2),
            "groups": len({i.split_group for i in members}),
            "sources": len({i.source_id for i in members}),
            "by_source_type": dict(sorted(Counter(i.source_type for i in members).items())),
        }
    report["splits"] = stats

    changes: dict[str, list[str]] = defaultdict(list)
    for gid, d in sorted(rec.decisions.items()):
        changes[d.category].append(gid)
    report["changes"] = {k: {"count": len(v), "groups": v} for k, v in sorted(changes.items())}
    reserved = rec.allocation.reserved if rec.allocation else {}
    report["reserved_for_nonempty"] = dict(sorted(reserved.items()))
    if movement is not None:
        report["resplit_movement"] = movement

    holdout_targets = [targets[s] for s in ("val", "test") if fractions[s] > 0]
    smallest_holdout = min(holdout_targets) if holdout_targets else None
    largest = sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    report["largest_groups"] = [
        {
            "split_group": gid,
            "items": cnt,
            "fraction": round(cnt / n, 4),
            "split": split_of[gid],
            "exceeds_smallest_holdout_target": smallest_holdout is not None
            and cnt > smallest_holdout,
        }
        for gid, cnt in largest
    ]

    explanation = []
    fixed_items = sum(
        rec.groups[g].n_items
        for g, d in rec.decisions.items()
        if d.category not in ("allocated", "allocated_nonempty", "resplit")
    )
    if fixed_items:
        explanation.append(
            f"{fixed_items} of {n} items are in groups whose split is fixed by history; only "
            f"{n - fixed_items} items in new groups were allocated."
        )
    if n:
        explanation.append(
            f"Groups are indivisible: {len(by_group)} groups, largest {largest[0][1]} items "
            f"({largest[0][1] / n:.1%})."
        )
    big = [g for g in report["largest_groups"] if g["exceeds_smallest_holdout_target"]]
    for g in big:
        explanation.append(
            f"Group {g['split_group']} ({g['items']} items, {g['fraction']:.1%}) is larger than "
            f"the smallest hold-out target ({smallest_holdout:.1f} items) and cannot be split."
        )
    if reserved:
        explanation.append(
            f"{len(reserved)} small group(s) were placed to keep requested splits non-empty."
        )
    report["explanation"] = explanation

    warnings = report["warnings"]
    if n == 0:
        warnings.append("No eligible items: nothing to split.")
    unfilled = rec.allocation.unfilled if rec.allocation else []
    new_groups = sum(
        1 for d in rec.decisions.values() if d.category in ("allocated", "allocated_nonempty")
    )
    for s in unfilled:
        if n == 0:
            continue
        if not cfg.split.require_nonempty:
            why = "split.require_nonempty is false"
        elif new_groups == 0 and not history.empty:
            why = "no new groups were available and historical assignments are never moved"
        else:
            why = "there are too few groups to give every requested split one"
        warnings.append(f"Split '{s}' has fraction {fractions[s]} but no items: {why}.")
    for s in SPLIT_ORDER:
        if fractions[s] == 0 and stats[s]["items"]:
            warnings.append(
                f"Split '{s}' has fraction 0 but keeps {stats[s]['items']} historically assigned "
                "items (history is never moved)."
            )
    if movement:
        inward = {
            k: v
            for k, v in movement.items()
            if not k.startswith("new->")
            and PROTECTION[k.split("->")[1]] < PROTECTION[k.split("->")[0]]
        }
        if inward:
            warnings.append(
                "RESPLIT MOVES HELD-OUT DATA TOWARD TRAINING: "
                + ", ".join(f"{k}: {v} items" for k, v in inward.items())
                + ". Evaluation results from earlier revisions are no longer comparable."
            )
    if last is not None and (revision_record is None or revision_record["mode"] != "resplit"):
        old_cfg = last.get("config", {})
        new_cfg = {
            "fractions": fractions,
            "seed": seed,
            "require_nonempty": cfg.split.require_nonempty,
        }
        changed = sorted(k for k in new_cfg if old_cfg.get(k) != new_cfg[k])
        if changed:
            report["notes"].append(
                f"Split settings changed since revision {last['revision']} ({', '.join(changed)}); "
                "only new groups use the new settings. Use --resplit to rebalance deliberately."
            )
    return report


def _fmt_pct(x: float) -> str:
    return f"{x:.1%}"


def render_markdown(report: dict[str, Any]) -> str:
    snap, rev, lin = report["snapshot"], report["revision"], report["annotation_lineage"]
    lines = [
        "# Split report",
        "",
        f"- Items: {snap['items']} in {snap['groups']} split groups "
        f"({snap['excluded_by_dedup']} excluded by dedup)",
        f"- Snapshot: `{snap['snapshot_fp'][:16]}`",
        f"- Revision: {rev['number']} ({'new' if rev['new'] else 'unchanged'})",
        f"- Annotation lineage: {lin.get('status')}"
        + (f" (`{lin['run_id']}`)" if lin.get("run_id") else ""),
        "",
    ]
    if report["conflicts"]:
        lines += ["## Split-history conflicts", "", "Nothing in `splits/` was changed.", ""]
        for c in report["conflicts"]:
            lines += [
                f"### {c['group_id']} ({c['status']})",
                "",
                c["message"],
                "",
                "| Source | Historical split | Historical group |",
                "|---|---|---|",
            ]
            lines += [
                f"| {s['source_id']} | {s['historical_split'] or 'new'} | "
                f"{s['historical_group'] or ''} |"
                for s in c["sources"]
            ]
            items = ", ".join(f"{k}: {v}" for k, v in c["items_by_historical_split"].items())
            lines += ["", f"Items by historical split: {items}", ""]
    if "splits" in report:
        lines += [
            "## Requested vs achieved",
            "",
            "| Split | Requested | Target items | Items | Achieved | Deviation | Groups "
            "| Sources |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s, st in report["splits"].items():
            lines.append(
                f"| {s} | {_fmt_pct(st['requested_fraction'])} | {st['target_items']} | "
                f"{st['items']} | {_fmt_pct(st['fraction'])} | {st['deviation_items']:+} | "
                f"{st['groups']} | {st['sources']} |"
            )
        lines.append("")
        if report["explanation"]:
            lines += ["### Why proportions differ", ""]
            lines += [f"- {e}" for e in report["explanation"]] + [""]
        if report["changes"]:
            lines += ["## Changes", "", "| Category | Groups |", "|---|---|"]
            lines += [f"| {k} | {v['count']} |" for k, v in report["changes"].items()]
            lines.append("")
        if report.get("resplit_movement"):
            lines += ["## Resplit movement (items)", "", "| From -> to | Items |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in report["resplit_movement"].items()]
            lines.append("")
    if report["warnings"]:
        lines += ["## Warnings", ""] + [f"- ⚠️ {w}" for w in report["warnings"]] + [""]
    if report["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in report["notes"]] + [""]
    return "\n".join(lines)
