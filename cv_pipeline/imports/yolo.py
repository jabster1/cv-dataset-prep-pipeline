"""Read-only discovery and validation of YOLO detection exports (incl. Roboflow).

Supported layouts (exactly one per export):

    Roboflow style                 Ultralytics style
    data.yaml                      data.yaml
    train/images  train/labels     images/train  labels/train
    valid/images  valid/labels     images/val    labels/val
    test/images   test/labels      images/test   labels/test

``valid`` and ``val`` both mean the legacy validation split. Labels are matched to images
by relative path with the extension replaced by ``.txt``. The ``train:``/``val:`` paths
inside data.yaml are not used (Roboflow exports often contain stale relative paths); only
``names`` (and ``nc`` if present) are read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

LEGACY_SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
COORD_TOLERANCE = 1e-6
# Roboflow renames exported files to ``<original stem>_<original ext>.rf.<32 hex>.<ext>``.
# Deliberately strict: anything else is not treated as a Roboflow family member.
_ROBOFLOW_STEM = re.compile(
    r"^(?P<orig>[^/]*[^/_])_(?:jpe?g|png|webp|bmp|tiff?)\.rf\.[0-9a-fA-F]{32}$", re.IGNORECASE
)


class ExportFormatError(ValueError):
    """The export cannot be imported as a YOLO detection dataset."""


@dataclass(frozen=True)
class ExportImage:
    rel_path: str  # POSIX, relative to the export root
    legacy_split: str  # train | val | test
    label_rel_path: str  # where the label file is expected (may not exist)


@dataclass(frozen=True)
class ExportLayout:
    data_yaml: str
    names: tuple[str, ...]
    images: tuple[ExportImage, ...]
    orphan_labels: tuple[str, ...]  # label files with no matching image


@dataclass(frozen=True)
class ParsedBox:
    line: int
    source_class_id: int
    cx: float
    cy: float
    w: float
    h: float


def roboflow_key(rel_path: str) -> str:
    """Original file stem for Roboflow-renamed files (``x_jpg.rf.<32 hex>.jpg`` -> ``x``);
    empty when the name does not match the Roboflow pattern exactly."""
    stem = Path(rel_path).stem
    m = _ROBOFLOW_STEM.match(stem)
    return m.group("orig") if m else ""


def family_key(import_name: str, rel_path: str) -> str:
    """Leakage-prevention family of an imported file: derivatives of one original image
    within ONE imported dataset. Namespaced by dataset so unrelated imports never collide.
    Empty when the relationship cannot be established confidently."""
    key = roboflow_key(rel_path)
    return f"import:{import_name}:roboflow:{key}" if key else ""


def parse_names(data: object) -> tuple[str, ...]:
    if not isinstance(data, dict) or "names" not in data:
        raise ExportFormatError("data.yaml has no 'names' entry")
    raw = data["names"]
    if isinstance(raw, list):
        names = [str(n) for n in raw]
    elif isinstance(raw, dict):
        try:
            ids = sorted(int(k) for k in raw)
        except (TypeError, ValueError) as exc:
            raise ExportFormatError("data.yaml 'names' keys must be integers") from exc
        if ids != list(range(len(ids))):
            raise ExportFormatError(f"data.yaml 'names' ids must be 0..{len(ids) - 1}, got {ids}")
        names = [str(raw[k] if k in raw else raw[str(k)]) for k in ids]
    else:
        raise ExportFormatError("data.yaml 'names' must be a list or an id -> name mapping")
    if not names:
        raise ExportFormatError("data.yaml 'names' is empty")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ExportFormatError(f"data.yaml has duplicate class names {dupes}")
    if "nc" in data and int(data["nc"]) != len(names):  # type: ignore[call-overload]
        raise ExportFormatError(f"data.yaml nc={data['nc']} but {len(names)} names")
    return tuple(names)


def discover(root: Path, image_extensions: list[str]) -> ExportLayout:
    if not root.is_dir():
        raise ExportFormatError(f"Export directory not found: {root}")
    yamls = [p for p in (root / "data.yaml", root / "data.yml") if p.is_file()]
    if not yamls:
        raise ExportFormatError(f"No data.yaml found in {root}")
    try:
        names = parse_names(yaml.safe_load(yamls[0].read_text(encoding="utf-8")))
    except yaml.YAMLError as exc:
        raise ExportFormatError(f"Invalid data.yaml: {exc}") from exc

    pairs: list[tuple[str, Path, Path]] = []  # (legacy split, images dir, labels dir)
    for d in sorted(LEGACY_SPLIT_NAMES):
        if (root / d / "images").is_dir():
            pairs.append((LEGACY_SPLIT_NAMES[d], root / d / "images", root / d / "labels"))
        if (root / "images" / d).is_dir():
            pairs.append((LEGACY_SPLIT_NAMES[d], root / "images" / d, root / "labels" / d))
    if not pairs:
        raise ExportFormatError(
            f"No YOLO split directories in {root} (expected e.g. train/images or images/train)"
        )
    seen_splits = [s for s, _, _ in pairs]
    clash = sorted({s for s in seen_splits if seen_splits.count(s) > 1})
    if clash:
        raise ExportFormatError(
            f"Ambiguous export layout: split(s) {clash} found more than once "
            "(e.g. both 'valid' and 'val', or both layouts)"
        )

    exts = {e.lower() for e in image_extensions}
    images: list[ExportImage] = []
    expected_labels: set[str] = set()
    all_labels: set[str] = set()
    for split, img_dir, lbl_dir in pairs:
        for p in sorted(img_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
                rel_in_split = p.relative_to(img_dir).with_suffix(".txt")
                label = (lbl_dir / rel_in_split).relative_to(root).as_posix()
                images.append(ExportImage(p.relative_to(root).as_posix(), split, label))
                expected_labels.add(label)
        if lbl_dir.is_dir():
            all_labels.update(
                p.relative_to(root).as_posix() for p in lbl_dir.rglob("*.txt") if p.is_file()
            )
    images.sort(key=lambda i: i.rel_path)
    return ExportLayout(
        data_yaml=yamls[0].relative_to(root).as_posix(),
        names=names,
        images=tuple(images),
        orphan_labels=tuple(sorted(all_labels - expected_labels)),
    )


def parse_label(text: str, n_classes: int) -> tuple[list[ParsedBox], str | None]:
    """Parse a YOLO detection label file. Returns (boxes, error); any bad line -> error."""
    boxes: list[ParsedBox] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return [], f"line {i}: expected 5 values (class cx cy w h), got {len(parts)}"
        try:
            cls = int(parts[0])
        except ValueError:
            return [], f"line {i}: class id {parts[0]!r} is not an integer"
        if not 0 <= cls < n_classes:
            return [], f"line {i}: class id {cls} outside 0..{n_classes - 1}"
        try:
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            return [], f"line {i}: coordinates are not numbers"
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            return [], f"line {i}: non-finite coordinates"
        t = COORD_TOLERANCE
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            return [], f"line {i}: centre ({cx}, {cy}) outside [0, 1]"
        if not (0 < w <= 1 + t and 0 < h <= 1 + t):
            return [], f"line {i}: width/height ({w}, {h}) must be in (0, 1]"
        if cx - w / 2 < -t or cx + w / 2 > 1 + t or cy - h / 2 < -t or cy + h / 2 > 1 + t:
            return [], f"line {i}: box extends outside the image"
        boxes.append(ParsedBox(i, cls, cx, cy, w, h))
    return boxes, None
