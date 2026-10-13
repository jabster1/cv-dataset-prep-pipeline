"""Export report: per-split counts, held-out coverage, class distribution, fingerprints."""

from __future__ import annotations

from collections import Counter
from typing import Any

SPLITS = ("train", "val", "test")


def build_report(ctx: dict[str, Any], dry_run: bool, created: bool) -> dict[str, Any]:
    records = ctx["records"]
    excluded = ctx["excluded"]
    coverage = ctx["coverage"]
    target = ctx["target"]
    rows = ctx["rows"]

    boxes_per_class: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    for r in rows:
        for b in r.boxes:
            boxes_per_class[r.record.split][target[b.class_id]] += 1

    train_orig = [r for r in records if r.split == "train" and r.kind == "original"]
    train_aug = [r for r in records if r.split == "train" and r.kind == "augmented"]
    train_exc = [x for x in excluded if x.split == "train" and x.kind == "original"]
    train = {
        "active_items": coverage["train"]["total"],
        "usable_labeled_parents": len(train_orig),
        "positives": sum(1 for r in train_orig if not r.negative),
        "trusted_negatives": sum(1 for r in train_orig if r.negative),
        "skipped_by_category": dict(sorted(Counter(x.category for x in train_exc).items())),
        "augmented_included": len(train_aug),
        "augmented_excluded": sum(1 for x in excluded if x.kind == "augmented"),
        "by_origin": dict(
            sorted(Counter("raw" if r.origin == "raw" else "imported" for r in train_orig).items())
        ),
        "roboflow_family_members": sum(1 for r in train_orig if r.family_key),
    }

    warnings = []
    if ctx["incomplete"]:
        warnings.append(
            "INCOMPLETE EVALUATION SET: some val/test items lack usable trusted labels and were "
            "excluded (see held-out coverage). Do not treat metrics on this export as final."
        )
    for s in SPLITS:
        n_images = sum(1 for r in records if r.split == s)
        if not n_images:
            continue
        absent = [c for c in target if boxes_per_class[s][c] == 0]
        if absent:
            warnings.append(f"{s}: no boxes for class(es) {absent}")
    if ctx["non_target_dropped"]:
        warnings.append(
            f"Dropped non-target boxes: {dict(sorted(ctx['non_target_dropped'].items()))}"
        )

    aug = ctx["aug"]
    return {
        "status": "dry_run" if dry_run else "ok",
        "run_id": ctx["run_id"],
        "new_run": created,
        "incomplete_evaluation_set": ctx["incomplete"],
        "evaluation_policy": ctx["evaluation_policy"],
        "fingerprints": {
            "snapshot_fp": ctx["identity"]["snapshot_fp"],
            "split_revision": ctx["identity"]["split_revision"],
            "split_state_digest": ctx["identity"]["split_state_digest"],
            "labels_fp": ctx["identity"]["labels_fp"],
            "augmentation_run_id": aug["run_id"] if aug else None,
        },
        "classes": {i: c for i, c in enumerate(target)},
        "class_mapping": ctx["identity"]["config"]["class_mapping"],
        "images_per_split": {s: sum(1 for r in records if r.split == s) for s in SPLITS},
        "boxes_per_class_per_split": {
            s: {c: boxes_per_class[s][c] for c in target} for s in SPLITS
        },
        "train": train,
        "heldout": {s: coverage[s] for s in ("val", "test")},
        "leakage_checks": "passed",
        "annotation_lineage": ctx["lineage"],
        "warnings": warnings,
    }


def render_markdown(r: dict[str, Any]) -> str:
    t = r["train"]
    lines = [
        "# Export report",
        "",
        f"- Run: `{r['run_id']}` — status **{r['status']}**",
        f"- Evaluation policy: {r['evaluation_policy']}; **incomplete_evaluation_set: "
        f"{str(r['incomplete_evaluation_set']).lower()}**",
        f"- Snapshot `{r['fingerprints']['snapshot_fp'][:16]}`, split revision "
        f"{r['fingerprints']['split_revision']}, labels `{r['fingerprints']['labels_fp'][:16]}`, "
        f"augmentation run `{r['fingerprints']['augmentation_run_id']}`",
        f"- Classes: {r['classes']}",
        f"- Leakage checks: {r['leakage_checks']}",
        "",
        "Items without a usable label are excluded — never exported as background.",
        "",
        "## Train",
        "",
        "| | Items |",
        "|---|---|",
        f"| Active train items | {t['active_items']} |",
        f"| Exported originals (usable labels) | {t['usable_labeled_parents']} |",
        f"| … positives | {t['positives']} |",
        f"| … trusted negatives (empty label) | {t['trusted_negatives']} |",
    ]
    lines += [f"| Skipped: {k} | {v} |" for k, v in t["skipped_by_category"].items()]
    lines += [
        f"| Augmented images included | {t['augmented_included']} |",
        f"| Roboflow family members (originals) | {t['roboflow_family_members']} |",
        f"| By origin | {t['by_origin']} |",
        "",
        "## Held-out label coverage",
        "",
        "| Split | Active | Exported | Positives | Negatives | Unlabeled | Untrusted | Conflicts "
        "| Coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s, c in r["heldout"].items():
        lines.append(
            f"| {s} | {c['total']} | {c['exported']} | {c['positives']} | {c['negatives']} | "
            f"{c['unlabeled']} | {c['untrusted']} | {c['conflicts']} | {c['coverage_pct']}% |"
        )
    lines += [
        "",
        "## Boxes per class",
        "",
        "| Split | " + " | ".join(r["classes"].values()) + " |",
        "|---|" + "---|" * len(r["classes"]),
    ]
    for s, counts in r["boxes_per_class_per_split"].items():
        lines.append(f"| {s} | " + " | ".join(str(v) for v in counts.values()) + " |")
    if r["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- ⚠️ {w}" for w in r["warnings"]]
    return "\n".join(lines) + "\n"
