"""Explicit import of an existing labeled dataset into the workspace.

``import-dataset --name N`` validates the export (read-only), then writes an immutable
snapshot to ``<workspace>/imports/<N>/``:

    files/        byte copies of images, label files and data.yaml (verified by sha256)
    import.json   settings used, class names/mapping, counts, content fingerprint
    images.csv    one row per image (item_id, size, legacy split, label status, ...)
    boxes.csv     one row per validated label line (mapped to project classes)
    rejected.csv  images that could not be imported, orphan label files

Ingest then treats each configured import as an additional root (origin
``import:<N>``), so imported images get normal content-addressed item ids and flow through
dedup, annotation and splitting with everything else.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cv_pipeline.config import ImportConfig, PipelineConfig
from cv_pipeline.imports.exif import exif_orientation
from cv_pipeline.imports.yolo import (
    ExportFormatError,
    discover,
    family_key,
    parse_label,
    roboflow_key,
)
from cv_pipeline.ingest.images import image_item_id
from cv_pipeline.manifest import read_records, write_records
from cv_pipeline.utils.fingerprint import digest
from cv_pipeline.utils.hashing import sha256_bytes, sha256_file
from cv_pipeline.utils.io import atomic_write_json, atomic_write_text, decode_image
from cv_pipeline.utils.logging import get_logger, log_event
from cv_pipeline.workspace import Workspace

logger = get_logger("imports")

IMPORT_SCHEMA = "cvp-import-v1"


class ImportError_(RuntimeError):  # noqa: N801 - avoid shadowing builtin ImportError
    """The dataset cannot be imported (format, taxonomy, conflicting re-import)."""


class LabelStatus:
    LABELED = "labeled"
    EMPTY = "empty"
    MISSING = "missing"
    INVALID = "invalid"


@dataclass
class ImportImageRecord:
    rel_path: str
    item_id: str
    sha256: str
    width: int
    height: int
    legacy_split: str
    label_path: str
    label_status: str
    label_sha256: str
    n_boxes: int
    n_ignored: int
    label_error: str
    roboflow_key: str
    family_key: str = ""  # leakage family (see imports.yolo.family_key); "" if none
    exif_orientation: int = 0  # 0 = no EXIF orientation tag, else 1..8


@dataclass
class ImportBoxRecord:
    rel_path: str
    item_id: str
    line: int
    source_class_id: int
    source_class_name: str
    class_id: int | None
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    ignored: bool


@dataclass
class ImportRejectRecord:
    rel_path: str
    reason: str
    detail: str


@dataclass
class ImportResult:
    name: str
    created: bool
    dry_run: bool
    record: dict[str, Any]
    images: list[ImportImageRecord] = field(default_factory=list)
    boxes: list[ImportBoxRecord] = field(default_factory=list)
    rejected: list[ImportRejectRecord] = field(default_factory=list)


def import_settings(imp: ImportConfig, classes: list[str]) -> dict[str, Any]:
    """Settings that determine how labels are interpreted; frozen into import.json."""
    return {
        "format": imp.format,
        "labels_trusted": imp.labels_trusted,
        "empty_labels_are_negatives": imp.empty_labels_are_negatives,
        "ignored_only_labels_are_negatives": imp.ignored_only_labels_are_negatives,
        "class_map": dict(sorted(imp.class_map.items())),
        "ignore_classes": sorted(imp.ignore_classes),
        "project_classes": list(classes),
    }


_SETTING_DEFAULTS = {"ignored_only_labels_are_negatives": False}


def get_import_config(cfg: PipelineConfig, name: str) -> ImportConfig:
    for imp in cfg.imports:
        if imp.name == name:
            return imp
    raise ImportError_(
        f"No import named {name!r} in the config; configured: {[i.name for i in cfg.imports]}"
    )


def map_taxonomy(
    names: tuple[str, ...], imp: ImportConfig, classes: list[str]
) -> dict[int, int | None]:
    """Source class id -> project class id (None = deliberately ignored)."""
    unknown_keys = sorted(set(imp.class_map) - set(names))
    if unknown_keys:
        raise ImportError_(f"class_map has source classes not in data.yaml: {unknown_keys}")
    unknown_ignored = sorted(set(imp.ignore_classes) - set(names))
    if unknown_ignored:
        raise ImportError_(f"ignore_classes not in data.yaml: {unknown_ignored}")
    mapping: dict[int, int | None] = {}
    conflicts = []
    for i, name in enumerate(names):
        if name in imp.ignore_classes:
            mapping[i] = None
        elif name in imp.class_map:
            mapping[i] = classes.index(imp.class_map[name])
        elif name in classes:
            mapping[i] = classes.index(name)
        else:
            conflicts.append(name)
    if conflicts:
        raise ImportError_(
            f"Imported taxonomy conflicts with project classes {classes}: source classes "
            f"{conflicts} are not project classes. Map them with class_map or drop them "
            "explicitly with ignore_classes."
        )
    targets = [t for t in mapping.values() if t is not None]
    merged = sorted({classes[t] for t in targets if targets.count(t) > 1})
    if merged:
        raise ImportError_(f"Several source classes map to the same project class {merged}")
    return mapping


def run_import(cfg: PipelineConfig, name: str, dry_run: bool = False) -> ImportResult:
    imp = get_import_config(cfg, name)
    ws = Workspace(cfg.workspace_dir).ensure()
    root = imp.path
    try:
        layout = discover(root, cfg.ingest.image_extensions)
    except ExportFormatError as exc:
        raise ImportError_(f"{name}: {exc}") from exc
    mapping = map_taxonomy(layout.names, imp, cfg.classes)
    log_event(logger, "import_started", name=name, path=str(root), images=len(layout.images))

    images: list[ImportImageRecord] = []
    boxes: list[ImportBoxRecord] = []
    rejected: list[ImportRejectRecord] = [
        ImportRejectRecord(p, "orphan_label", "label file without a matching image")
        for p in layout.orphan_labels
    ]
    copied: list[str] = [layout.data_yaml]
    for img in layout.images:
        data = (root / img.rel_path).read_bytes()
        decoded = decode_image(data)
        if decoded is None:
            rejected.append(ImportRejectRecord(img.rel_path, "unreadable_image", "cannot decode"))
            continue
        sha = sha256_bytes(data)
        item_id = image_item_id(sha)
        h, w = decoded.shape[:2]
        label_file = root / img.label_rel_path
        status, label_sha, error, n_ignored = LabelStatus.MISSING, "", "", 0
        parsed = []
        if label_file.is_file():
            text_bytes = label_file.read_bytes()
            label_sha = sha256_bytes(text_bytes)
            copied.append(img.label_rel_path)
            try:
                text = text_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text, error = "", "label file is not UTF-8"
            if not error:
                parsed, err = parse_label(text, len(layout.names))
                error = err or ""
            if error:
                status, parsed = LabelStatus.INVALID, []
            else:
                status = LabelStatus.LABELED if parsed else LabelStatus.EMPTY
        for b in parsed:
            target = mapping[b.source_class_id]
            n_ignored += target is None
            boxes.append(
                ImportBoxRecord(
                    rel_path=img.rel_path,
                    item_id=item_id,
                    line=b.line,
                    source_class_id=b.source_class_id,
                    source_class_name=layout.names[b.source_class_id],
                    class_id=target,
                    class_name=cfg.classes[target] if target is not None else "",
                    cx=b.cx,
                    cy=b.cy,
                    w=b.w,
                    h=b.h,
                    ignored=target is None,
                )
            )
        images.append(
            ImportImageRecord(
                rel_path=img.rel_path,
                item_id=item_id,
                sha256=sha,
                width=int(w),
                height=int(h),
                legacy_split=img.legacy_split,
                label_path=img.label_rel_path if label_file.is_file() else "",
                label_status=status,
                label_sha256=label_sha,
                n_boxes=len(parsed),
                n_ignored=n_ignored,
                label_error=error,
                roboflow_key=roboflow_key(img.rel_path),
                family_key=family_key(name, img.rel_path),
                exif_orientation=exif_orientation(data),
            )
        )
        copied.append(img.rel_path)

    content = sorted((p, sha256_file(root / p)) for p in copied)
    settings = import_settings(imp, cfg.classes)
    record = {
        "schema": IMPORT_SCHEMA,
        "name": name,
        "settings": settings,
        "source_class_names": list(layout.names),
        "class_mapping": {
            layout.names[i]: (cfg.classes[t] if t is not None else None)
            for i, t in sorted(mapping.items())
        },
        "content_fp": digest(content),
        "import_fp": digest({"content": content, "settings": settings}),
        "counts": _counts(images, rejected),
    }
    result = ImportResult(name, False, dry_run, record, images, boxes, rejected)

    dest = ws.root / "imports" / name
    if dest.exists():
        existing = _read_json(dest / "import.json")
        if existing.get("import_fp") == record["import_fp"]:
            log_event(logger, "import_unchanged", name=name)
            return result
        raise ImportError_(
            f"Import {name!r} already exists with different content or settings. Imports are "
            "immutable; give the new version a new name (e.g. a version suffix)."
        )
    if dry_run:
        return result

    _write_snapshot(ws, dest, root, record, content, images, boxes, rejected)
    _write_report(ws, record, images, rejected)
    result.created = True
    for img in images:
        if img.exif_orientation not in (0, 1):
            log_event(
                logger,
                "exif_orientation",
                logging.WARNING,
                name=name,
                file=img.rel_path,
                orientation=img.exif_orientation,
            )
    log_event(logger, "import_finished", name=name, **record["counts"])
    return result


def family_stats(images: list[ImportImageRecord] | tuple[ImportImageRecord, ...]) -> dict[str, int]:
    keys = [i.family_key for i in images if i.family_key]
    sizes: dict[str, int] = {}
    for k in keys:
        sizes[k] = sizes.get(k, 0) + 1
    multi = {k: n for k, n in sizes.items() if n > 1}
    return {"lineage_families": len(multi), "lineage_family_images": sum(multi.values())}


def _counts(images: list[ImportImageRecord], rejected: list[ImportRejectRecord]) -> dict[str, Any]:
    def tally(values: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in values:
            out[v] = out.get(v, 0) + 1
        return dict(sorted(out.items()))

    return {
        "images": len(images),
        "boxes": sum(i.n_boxes for i in images),
        "ignored_boxes": sum(i.n_ignored for i in images),
        "ignored_only_labels": sum(
            1 for i in images if i.label_status == LabelStatus.LABELED and i.n_boxes == i.n_ignored
        ),
        "label_status": tally([i.label_status for i in images]),
        "legacy_split": tally([i.legacy_split for i in images]),
        "rejected": tally([r.reason for r in rejected]),
        **family_stats(images),
        "exif_rotated_images": sum(1 for i in images if i.exif_orientation not in (0, 1)),
    }


def _write_snapshot(
    ws: Workspace,
    dest: Path,
    root: Path,
    record: dict[str, Any],
    content: list[tuple[str, str]],
    images: list[ImportImageRecord],
    boxes: list[ImportBoxRecord],
    rejected: list[ImportRejectRecord],
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{dest.name}-", dir=dest.parent))
    try:
        for rel, sha in content:
            target = tmp / "files" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / rel, target)
            if sha256_file(target) != sha:
                raise ImportError_(
                    f"Copy of {rel} does not match its source (changed during import?)"
                )
        write_records(tmp / "images.csv", images, ImportImageRecord)
        write_records(tmp / "boxes.csv", boxes, ImportBoxRecord)
        write_records(tmp / "rejected.csv", rejected, ImportRejectRecord)
        atomic_write_json(tmp / "import.json", record)
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _write_report(
    ws: Workspace,
    record: dict[str, Any],
    images: list[ImportImageRecord],
    rejected: list[ImportRejectRecord],
) -> None:
    name, c = record["name"], record["counts"]
    s = record["settings"]
    lines = [
        f"# Import report: {name}",
        "",
        f"- Labels trusted (human-made): {s['labels_trusted']}  -> provenance "
        f"`{'human_imported' if s['labels_trusted'] else 'imported_unverified'}`",
        f"- Empty label files count as negatives: {s['empty_labels_are_negatives']}",
        f"- Images: {c['images']}, boxes: {c['boxes']} ({c['ignored_boxes']} in ignored classes)",
        f"- Label status: {c['label_status']}",
        f"- Legacy split (metadata only; never used as the pipeline split): {c['legacy_split']}",
        f"- Rejected: {c['rejected']}",
        "",
        "## Class mapping",
        "",
        "| Source class | Project class |",
        "|---|---|",
    ]
    lines += [
        f"| {k} | {v if v is not None else '(ignored)'} |"
        for k, v in record["class_mapping"].items()
    ]
    if c["lineage_families"]:
        lines += [
            "",
            "## Roboflow derivative families (kept together)",
            "",
            f"{c['lineage_family_images']} images belong to {c['lineage_families']} families "
            "of files renamed by Roboflow from one original (`<stem>_<ext>.rf.<hash>`), e.g. "
            "augmented copies made at export time. Every family is forced into one split "
            "group, so its members can never be split across train/val/test. Nothing is "
            "excluded or deleted because of this.",
            "",
            "Recommended: export future Roboflow versions **without augmentation** and let the "
            "pipeline own train-only augmentation.",
        ]
    rotated = [i for i in images if i.exif_orientation not in (0, 1)]
    if rotated:
        lines += [
            "",
            "## ⚠️ EXIF orientation",
            "",
            f"{len(rotated)} image(s) carry an EXIF orientation tag other than 1. They are "
            "decoded rotated (as OpenCV/Ultralytics do); check that their labels were made on "
            "the rotated view. Images were not modified.",
            "",
            "| File | Orientation |",
            "|---|---|",
        ]
        lines += [f"| {i.rel_path} | {i.exif_orientation} |" for i in rotated[:200]]
    invalid = [i for i in images if i.label_status == "invalid"]
    if invalid or rejected:
        lines += ["", "## Problems", "", "| File | Problem |", "|---|---|"]
        lines += [f"| {i.rel_path} | invalid label: {i.label_error} |" for i in invalid[:200]]
        lines += [f"| {r.rel_path} | {r.reason}: {r.detail} |" for r in rejected[:200]]
    atomic_write_json(ws.reports / f"import_{name}.json", record)
    atomic_write_text(ws.reports / f"import_{name}.md", "\n".join(lines) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------------------
# Registry used by ingest and label resolution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportedDataset:
    name: str
    dir: Path
    record: dict[str, Any]
    images: tuple[ImportImageRecord, ...]
    boxes: tuple[ImportBoxRecord, ...]

    @property
    def files_root(self) -> Path:
        return self.dir / "files"

    @property
    def labels_trusted(self) -> bool:
        return bool(self.record["settings"]["labels_trusted"])

    @property
    def empty_labels_are_negatives(self) -> bool:
        return bool(self.record["settings"]["empty_labels_are_negatives"])

    @property
    def ignored_only_labels_are_negatives(self) -> bool:
        return bool(self.record["settings"]["ignored_only_labels_are_negatives"])


def load_imports(cfg: PipelineConfig) -> list[ImportedDataset]:
    """Configured imports that have been imported, sorted by name.

    Raises if an import's frozen settings differ from the current config. Imports present
    in the workspace but no longer configured are ignored (with a warning).
    """
    ws = Workspace(cfg.workspace_dir)
    base = ws.root / "imports"
    configured = {i.name: i for i in cfg.imports}
    out = []
    for name in sorted(configured):
        d = base / name
        if not (d / "import.json").is_file():
            log_event(
                logger,
                "import_not_yet_imported",
                name=name,
                hint=f"run: import-dataset --name {name}",
            )
            continue
        record = _read_json(d / "import.json")
        expected = import_settings(configured[name], cfg.classes)
        # Settings added after an import was made take their (conservative) default.
        recorded = {**_SETTING_DEFAULTS, **record.get("settings", {})}
        if recorded != expected:
            raise ImportError_(
                f"Settings for import {name!r} changed since it was imported (labels_trusted, "
                "empty/ignored-only negatives, class_map, ignore_classes or project classes). "
                "Imports are immutable: re-import under a new name."
            )
        record = {**record, "settings": recorded}
        images = []
        for img in read_records(d / "images.csv", ImportImageRecord):
            derived = family_key(name, img.rel_path)  # pure function of name + path
            if img.family_key and img.family_key != derived:
                raise ImportError_(f"Import {name!r}: images.csv family_key was modified")
            img.family_key = derived
            images.append(img)
        out.append(
            ImportedDataset(
                name=name,
                dir=d,
                record=record,
                images=tuple(images),
                boxes=tuple(read_records(d / "boxes.csv", ImportBoxRecord)),
            )
        )
    if base.is_dir():
        stray = sorted(
            p.name
            for p in base.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in configured
        )
        if stray:
            log_event(logger, "imports_not_configured_ignored", names=stray)
    return out
