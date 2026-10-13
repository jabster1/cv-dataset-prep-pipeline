"""Re-verify a Phase 6 export run before (and after) training.

Uses only the export run directory, so it works wherever the export is copied (e.g. a GPU
machine). Nothing from Phase 6's earlier validation is trusted blindly:

* run.json identity recomputes to the run id
* dataset.yaml equals the regenerated text byte-for-byte (class mapping, paths, flags)
* every manifest image/label exists and matches its recorded sha256
* every label parses, uses valid class ids, matches the box count / negative flag
* images/ and labels/ contain exactly the manifest files (nothing added)
* per-split counts match run.json
* manifest-level leakage invariants still hold (one split per item/group/source/family;
  augmentations only in train with exported train parents)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cv_pipeline.export.export import (
    EXPORT_SCHEMA,
    SPLITS,
    ExportRecord,
    dataset_yaml,
    manifest_leakage_problems,
)
from cv_pipeline.imports.yolo import parse_label
from cv_pipeline.manifest import read_records
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.workspace import Workspace

_TOP_LEVEL_FILES = {
    "dataset.yaml",
    "manifest.csv",
    "excluded.csv",
    "run.json",
    "report.json",
    "report.md",
}


class ExportIntegrityError(RuntimeError):
    """The selected export run is missing, incomplete or has changed since export."""


@dataclass(frozen=True)
class VerifiedExport:
    run_id: str
    run_dir: Path
    identity: dict[str, Any]
    classes: list[str]
    incomplete_evaluation_set: bool
    records: list[ExportRecord]
    counts: dict[str, int]

    @property
    def has_test(self) -> bool:
        return self.counts.get("test", 0) > 0

    @property
    def fingerprint(self) -> str:
        return digest({"schema": EXPORT_SCHEMA, **self.identity})


def verify_export_run(ws: Workspace, run_id: str) -> VerifiedExport:
    if not run_id:
        raise ExportIntegrityError("An explicit export run id is required (--export-run exp_...).")
    run_dir = ws.root / "export" / "runs" / run_id
    if not run_dir.is_dir():
        raise ExportIntegrityError(f"Export run {run_id} not found in {run_dir.parent}.")
    problems: list[str] = []

    try:
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportIntegrityError(
            f"Export run {run_id} has no readable run.json (incomplete?)."
        ) from exc
    identity = run.get("identity") or {}
    if run.get("schema") != EXPORT_SCHEMA or run.get("run_id") != run_id:
        problems.append("run.json schema/run_id mismatch")
    if "exp_" + digest({"schema": EXPORT_SCHEMA, **identity})[:16] != run_id:
        problems.append("run.json identity does not hash to the run id (edited?)")

    classes = list((identity.get("config") or {}).get("classes") or [])
    incomplete = bool(run.get("incomplete_evaluation_set"))
    records = (
        read_records(run_dir / "manifest.csv", ExportRecord)
        if (run_dir / "manifest.csv").is_file()
        else []
    )
    if not records:
        problems.append("manifest.csv missing or empty")
    counts = {s: sum(1 for r in records if r.split == s) for s in SPLITS}
    if counts != {s: int((run.get("counts") or {}).get(s, -1)) for s in SPLITS}:
        problems.append(f"per-split counts {counts} differ from run.json {run.get('counts')}")

    expected_yaml = dataset_yaml(classes, counts["test"] > 0, incomplete, run_id)
    yaml_path = run_dir / "dataset.yaml"
    if not yaml_path.is_file() or yaml_path.read_text(encoding="utf-8") != expected_yaml:
        problems.append("dataset.yaml differs from the export identity (classes/paths/flags)")

    expected_files = set()
    for r in records:
        for rel, sha in ((r.export_image, r.image_sha256), (r.export_label, r.label_sha256)):
            expected_files.add(rel)
            path = run_dir / rel
            if not path.is_file():
                problems.append(f"missing {rel}")
            elif sha256_file(path) != sha:
                problems.append(f"changed {rel}")
        label = run_dir / r.export_label
        if label.is_file():
            boxes, err = parse_label(label.read_text(encoding="utf-8"), max(len(classes), 1))
            if err:
                problems.append(f"invalid label {r.export_label}: {err}")
            elif len(boxes) != r.n_boxes or (not boxes) != r.negative:
                problems.append(f"label {r.export_label} disagrees with the manifest")

    actual_files = {
        p.relative_to(run_dir).as_posix()
        for sub in ("images", "labels")
        for p in (run_dir / sub).rglob("*")
        if p.is_file()
    }
    extra = sorted(actual_files - expected_files)
    if extra:
        problems.append(f"unexpected files in the export (e.g. {extra[0]}; {len(extra)} total)")
    stray_top = sorted(
        p.name for p in run_dir.iterdir() if p.is_file() and p.name not in _TOP_LEVEL_FILES
    )
    if stray_top:
        problems.append(f"unexpected top-level files {stray_top}")
    for s in ("train", "val"):
        if not (run_dir / "images" / s).is_dir() or not (run_dir / "labels" / s).is_dir():
            problems.append(f"missing images/{s} or labels/{s}")

    problems += manifest_leakage_problems(records)
    if problems:
        raise ExportIntegrityError(
            f"Export run {run_id} failed re-verification — refusing to train:\n  "
            + "\n  ".join(problems[:20])
            + ("\n  ..." if len(problems) > 20 else "")
        )
    return VerifiedExport(run_id, run_dir, identity, classes, incomplete, records, counts)
