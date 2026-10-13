"""Augmentation report (``reports/augmentation.{json,md}``)."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from cv_pipeline.labels.schema import LabelProvenance


def build_report(
    ctx: dict[str, Any],
    items: list[Any],
    rejected: list[Any],
    dry_run: bool,
    created: bool,
) -> dict[str, Any]:
    cfg = ctx["cfg"]
    parents = ctx["parents"]
    skipped = ctx["skipped"]
    split_counts = ctx["split_counts"]
    variants = cfg.augmentation.variants_per_item

    parent_split = Counter(p.split.split for p in parents)
    generated_split = Counter(i.split for i in items)
    assertion = {
        "augmented_parents_by_split": {s: parent_split.get(s, 0) for s in ("train", "val", "test")},
        "generated_items_by_split": {
            s: generated_split.get(s, 0) for s in ("train", "val", "test")
        },
    }
    assertion["only_train"] = all(assertion[k][s] == 0 for k in assertion for s in ("val", "test"))

    provenance = Counter(p.resolution.label.provenance for p in parents)
    transforms_applied: Counter[str] = Counter()
    for i in items:
        for step in json.loads(i.recipe):
            transforms_applied[step["transform"]] += 1

    by_origin = Counter("raw" if p.item.origin == "raw" else "imported" for p in parents)
    family_parents = sum(1 for p in parents if p.family_key)
    warnings = []
    if not parents and split_counts.get("train", 0):
        warnings.append(
            "No train item has a usable label under the current policy; nothing to augment. "
            "Import trusted labels or record review decisions (review/decisions.csv)."
        )
    if provenance.get(LabelProvenance.MACHINE_UNREVIEWED):
        warnings.append(
            f"{provenance[LabelProvenance.MACHINE_UNREVIEWED]} train item(s) use UNREVIEWED "
            "machine labels (explicit opt-in). Not suitable for a production dataset."
        )
    if rejected:
        warnings.append(f"{len(rejected)} variant(s) were rejected after all retry attempts.")

    return {
        "run_id": ctx["run_id"],
        "new_run": created,
        "dry_run": dry_run,
        "inputs": {**ctx["identity"], "annotation_lineage": ctx["lineage"]},
        "label_policy": ctx["config"]["label_policy"],
        "splits": {s: split_counts.get(s, 0) for s in ("train", "val", "test")},
        "split_assertion": assertion,
        "train_items": {
            "total": split_counts.get("train", 0),
            "eligible": len(parents),
            "skipped": len(skipped),
            "skipped_by_category": dict(sorted(Counter(s.category for s in skipped).items())),
            "skipped_by_reason": dict(sorted(Counter(s.reason for s in skipped).items())),
        },
        "parents": {
            "by_provenance": dict(sorted(provenance.items())),
            "trusted_negatives": sum(
                1
                for p in parents
                if p.resolution.label.negative and p.resolution.status == "trusted"
            ),
            "machine_unreviewed_used": provenance.get(LabelProvenance.MACHINE_UNREVIEWED, 0),
            "by_origin": dict(sorted(by_origin.items())),
            "roboflow_family_members": family_parents,
        },
        "variants": {
            "per_item": variants,
            "requested": len(parents) * variants,
            "generated": len(items),
            "rejected": len(rejected),
            "rejected_reasons": dict(sorted(Counter(r.reason for r in rejected).items())),
            "generated_after_retry": sum(1 for i in items if i.attempt > 0),
        },
        "boxes": {
            "before": sum(i.boxes_before for i in items),
            "after": sum(i.boxes_after for i in items),
            "box_free_outputs_from_trusted_negatives": sum(
                1 for i in items if i.parent_negative and i.boxes_after == 0
            ),
        },
        "transforms_enabled": [t["name"] for t in ctx["config"]["transforms"]],
        "transforms_applied": dict(sorted(transforms_applied.items())),
        "training_items_by_source": {
            "raw_ingestion": by_origin.get("raw", 0),
            "historical_import": by_origin.get("imported", 0),
            "of_which_roboflow_family_members": family_parents,
            "generated_augmentation": len(items),
        },
        "warnings": warnings,
    }


def render_markdown(r: dict[str, Any]) -> str:
    t, p, v, a = r["train_items"], r["parents"], r["variants"], r["split_assertion"]
    state = "dry run" if r["dry_run"] else "new" if r["new_run"] else "existing, reused"
    snap = r["inputs"]["snapshot_fp"][:16]
    lines = [
        "# Augmentation report",
        "",
        f"- Run: `{r['run_id']}` ({state})",
        f"- Split revision: {r['inputs']['split_revision']} (snapshot `{snap}`)",
        f"- Label policy: {r['label_policy']}",
        f"- Splits: {r['splits']}",
        f"- **Train-only check: {'PASSED' if a['only_train'] else 'FAILED'}** — augmented parents "
        f"by split {a['augmented_parents_by_split']}",
        "",
        "Train items without a usable label are skipped, never treated as background.",
        "",
        "## Train items",
        "",
        "| | Items |",
        "|---|---|",
        f"| Train items | {t['total']} |",
        f"| Eligible (usable label) | {t['eligible']} |",
        f"| Skipped | {t['skipped']} |",
    ]
    lines += [f"| Skipped: {k} | {n} |" for k, n in t["skipped_by_category"].items()]
    lines += [
        f"| Trusted negatives augmented | {p['trusted_negatives']} |",
        f"| Machine-unreviewed labels used (opt-in) | {p['machine_unreviewed_used']} |",
        "",
        "## Label provenance of augmented parents",
        "",
        "| Provenance | Items |",
        "|---|---|",
    ]
    lines += [f"| {k} | {n} |" for k, n in p["by_provenance"].items()]
    lines += [
        "",
        "## Variants",
        "",
        f"Requested {v['requested']}, generated {v['generated']}, rejected {v['rejected']} "
        f"({v['generated_after_retry']} generated after a deterministic retry).",
        "",
    ]
    if v["rejected_reasons"]:
        lines += ["| Rejection reason | Variants |", "|---|---|"]
        lines += [f"| {k} | {n} |" for k, n in v["rejected_reasons"].items()]
        lines.append("")
    lines += ["## Transforms applied", "", "| Transform | Times |", "|---|---|"]
    lines += [f"| {k} | {n} |" for k, n in r["transforms_applied"].items()]
    s = r["training_items_by_source"]
    lines += [
        "",
        "## Training items by source",
        "",
        "| Source | Items |",
        "|---|---|",
        f"| Raw ingestion | {s['raw_ingestion']} |",
        f"| Historical import | {s['historical_import']} |",
        "| … of which Roboflow derivative-family members | "
        f"{s['of_which_roboflow_family_members']} |",
        f"| Generated by this augmentation run | {s['generated_augmentation']} |",
    ]
    if r["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- ⚠️ {w}" for w in r["warnings"]]
    return "\n".join(lines) + "\n"
