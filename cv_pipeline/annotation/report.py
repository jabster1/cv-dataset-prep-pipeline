"""Per-invocation annotation report (``reports/annotation.{json,md}``).

Unlike the run directory, this report describes one invocation (cache hits, whether the
run was new, forced-inference differences), so it is not part of run identity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from cv_pipeline.annotation.schema import BackendInfo, BoxDecision, InferenceStatus
from cv_pipeline.config import PipelineConfig

if TYPE_CHECKING:
    from cv_pipeline.annotation.annotate import AnnotationResult

_MD_MAX_ROWS = 100


def build_report(
    cfg: PipelineConfig,
    result: AnnotationResult,
    info: BackendInfo,
    identity: dict[str, Any],
    previous_run_id: str | None,
    force: bool,
    limit: int | None,
    n_eligible: int,
) -> dict[str, Any]:
    items, preds = result.items, result.predictions
    per_class: dict[str, list[float]] = defaultdict(list)
    for p in preds:
        if p.decision == BoxDecision.CANDIDATE and p.confidence is not None:
            per_class[p.class_name].append(p.confidence)
    reasons = Counter(r for i in items for r in i.review_reasons.split(";") if r)

    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "new_run": result.created,
        "previous_active_run_id": previous_run_id,
        "identity": identity,
        "invocation": {
            "force": force,
            "limit": limit,
            "cache_hits": result.cache_hits,
            "inferred": result.inferred,
            "forced_changes": result.forced_changes,
            "forced_reproduced_exactly": force and not result.forced_changes,
        },
        "backend": info.to_json(),
        "decoder": {"opencv": cv2.__version__, "numpy": np.__version__},
        "settings": {
            "confidence_threshold": cfg.annotation.confidence_threshold,
            "review_threshold": cfg.annotation.review_threshold,
            "inference_confidence_floor": cfg.annotation.inference_confidence_floor,
            "review_empty": cfg.annotation.review_empty,
            "review_unmapped_classes": cfg.annotation.review_unmapped_classes,
            "review_sample_rate": cfg.annotation.review_sample_rate,
            "class_map": cfg.annotation.class_map,
        },
        "items": {
            "eligible": n_eligible,
            "skipped_excluded_by_dedup": result.skipped_excluded,
            "annotated": len(items),
            "inference_status": dict(sorted(Counter(i.inference_status for i in items).items())),
            "with_no_candidates": sum(
                1 for i in items if i.inference_status == InferenceStatus.OK and i.n_candidates == 0
            ),
        },
        "boxes": {
            "by_decision": dict(sorted(Counter(p.decision for p in preds).items())),
            "candidates_by_class": {
                c: {
                    "count": len(v),
                    "min_confidence": min(v),
                    "median_confidence": round(median(v), 6),
                }
                for c, v in sorted(per_class.items())
            },
        },
        "review": {
            "queued": result.review.queued,
            "by_reason": dict(sorted(reasons.items())),
            "label_states": dict(sorted(result.review.label_states.items())),
            "labels_not_in_queue": len(result.review.stale_labels),
        },
        "failures": [
            {
                "item_id": i.item_id,
                "source_path": i.source_path,
                "status": i.inference_status,
                "error": i.error,
            }
            for i in result.failures
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    b, it, rv, inv = report["backend"], report["items"], report["review"], report["invocation"]
    lines = [
        "# Annotation report",
        "",
        "All boxes are **machine-generated** (provenance `machine`). Nothing here has been "
        "verified by a person; `unreviewed` means not yet reviewed, not correct.",
        "",
        f"- Run: `{report['run_id']}` ({'new' if report['new_run'] else 'existing, reused'})",
        f"- Active run before this invocation: `{report['previous_active_run_id']}`",
        f"- Backend: {b['name']} {b['version']}, model `{b['model_id']}` "
        f"(sha256 `{(b['model_sha256'] or '')[:12]}`)",
        f"- Cache hits: {inv['cache_hits']}, inferred: {inv['inferred']}",
    ]
    if inv["force"]:
        lines.append(
            "- Forced re-inference reproduced cached results exactly."
            if inv["forced_reproduced_exactly"]
            else f"- Forced re-inference changed results for {len(inv['forced_changes'])} items."
        )
    lines += [
        "",
        "## Items",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Eligible | {it['eligible']} |",
        f"| Skipped (excluded by dedup) | {it['skipped_excluded_by_dedup']} |",
        f"| Annotated | {it['annotated']} |",
    ]
    lines += [f"| Status `{k}` | {v} |" for k, v in it["inference_status"].items()]
    lines += [
        f"| OK with no candidate boxes | {it['with_no_candidates']} |",
        "",
        "## Boxes",
        "",
        "| Decision | Count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in report["boxes"]["by_decision"].items()]
    if report["boxes"]["candidates_by_class"]:
        lines += ["", "| Class | Candidates | Min conf | Median conf |", "|---|---|---|---|"]
        lines += [
            f"| {c} | {v['count']} | {v['min_confidence']:.3f} | {v['median_confidence']:.3f} |"
            for c, v in report["boxes"]["candidates_by_class"].items()
        ]
    lines += ["", "## Review queue", "", f"Queued: {rv['queued']}", ""]
    if rv["by_reason"]:
        lines += ["| Reason | Items |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rv["by_reason"].items()]
        lines.append("")
    if rv["label_states"].get("edited"):
        lines += [f"Human-edited labels preserved: {rv['label_states']['edited']}", ""]
    if report["failures"]:
        lines += ["## Failures", "", "| Item | Source | Status | Error |", "|---|---|---|---|"]
        for f in report["failures"][:_MD_MAX_ROWS]:
            err = f["error"].replace("|", "\\|")
            lines.append(f"| {f['item_id']} | {f['source_path']} | {f['status']} | {err} |")
        lines.append("")
    return "\n".join(lines)
