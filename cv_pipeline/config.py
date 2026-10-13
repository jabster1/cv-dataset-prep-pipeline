"""YAML configuration: typed schema, strict loading, semantic validation, path resolution.

Design notes
------------
* Unknown keys are an error — a typo like ``confidance_threshold`` must not silently fall
  back to a default.
* Relative paths are resolved against ``root``, which itself is resolved relative to the
  config file's directory (default ``.``). This makes a config behave the same no matter
  which directory the CLI is launched from.
* The schema covers every stage up front so config files stay stable as stages land.
"""

from __future__ import annotations

import math
import re
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised for malformed or semantically invalid configuration."""


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    name: str
    workspace: Path = Path("workspace")
    seed: int = 42


@dataclass
class VideoSamplingConfig:
    mode: Literal["fps", "every_n_frames"] = "fps"
    fps: float = 1.0
    every_n_frames: int = 30
    max_frames: int | None = None
    max_frames_strategy: Literal["uniform", "first"] = "uniform"
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95


@dataclass
class IngestConfig:
    raw_dir: Path = Path("raw")
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    video_extensions: list[str] = field(default_factory=lambda: [".mp4", ".mov"])
    video: VideoSamplingConfig = field(default_factory=VideoSamplingConfig)


@dataclass
class DedupConfig:
    enabled: bool = True
    hash_size: int = 8
    # A pair is a near-duplicate when pHash distance <= near_duplicate_threshold AND
    # dHash distance <= dhash_threshold (null = pHash only). Distances are in bits out of
    # hash_size**2.
    near_duplicate_threshold: int = 6
    dhash_threshold: int | None = 10
    # Actions only mark items in manifests/dedup.csv; files are never deleted.
    exact_duplicate_action: Literal["exclude", "keep"] = "exclude"
    near_duplicate_action: Literal["keep", "exclude"] = "keep"
    group_near_duplicates: bool = True


@dataclass
class AnnotationConfig:
    # ``prepare`` runs the annotate stage only when enabled; ``annotate`` always runs.
    enabled: bool = False
    backend: str = "yolo"
    # Weights file, resolved against ``root``. Validated only when annotation runs.
    # Models are never downloaded implicitly.
    model: str | None = None
    # Boxes below this are discarded from candidates (still recorded for audit).
    confidence_threshold: float = 0.25
    # Candidates below this put their item in the review queue.
    review_threshold: float = 0.5
    # Lowest confidence requested from the backend; must be <= confidence_threshold.
    inference_confidence_floor: float = 0.05
    iou_threshold: float = 0.7
    imgsz: int = 640
    batch_size: int = 1
    max_detections: int = 300
    device: str | None = None
    # Backend label -> project class. Labels equal to a project class map implicitly.
    class_map: dict[str, str] = field(default_factory=dict)
    review_empty: bool = True
    review_unmapped_classes: bool = False
    # Deterministic audit sample of otherwise-unflagged items sent to review.
    review_sample_rate: float = 0.05
    # Backend-specific settings (e.g. prompts for an open-vocabulary detector).
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportConfig:
    """An existing labeled dataset to import explicitly (``import-dataset --name``)."""

    name: str
    path: Path
    format: Literal["yolo"] = "yolo"
    # True: labels were made by people -> provenance human_imported (trusted).
    # False: provenance imported_unverified (needs explicit opt-in to be used).
    labels_trusted: bool = True
    # Empty label files count as human-confirmed negatives (only if labels_trusted).
    empty_labels_are_negatives: bool = False
    # Label files whose boxes are ALL in ignore_classes count as negatives for the project
    # classes (only if labels_trusted). Off by default: dropping other classes does not
    # prove the project's targets were looked for and found absent.
    ignored_only_labels_are_negatives: bool = False
    class_map: dict[str, str] = field(default_factory=dict)  # source name -> project class
    ignore_classes: list[str] = field(default_factory=list)  # source classes to drop


@dataclass
class ReviewConfig:
    export: Literal["flagged", "all"] = "flagged"
    copy_images: bool = True
    previews: bool = True


SplitName = Literal["train", "val", "test"]


@dataclass
class SplitConfig:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    # Tie-break seed for allocating NEW groups; null -> project.seed.
    seed: int | None = None
    # Give every split with fraction > 0 at least one new group when feasible.
    require_nonempty: bool = True
    # Explicit resolutions for split-history conflicts: {split_group: split}. Only the
    # most protected historical split (test > val > train) is accepted.
    conflict_resolutions: dict[str, SplitName] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Train-only augmentation (see cv_pipeline/augmentation)."""

    enabled: bool = False  # prepare runs augment only when true; `augment` always runs
    # trusted_only: human_imported / human_reviewed_accepted / human_reviewed_corrected.
    # allow_machine_unreviewed: development/POC opt-in for unreviewed machine labels.
    label_policy: Literal["trusted_only", "allow_machine_unreviewed"] = "trusted_only"
    allow_machine_empty: bool = False  # machine zero detections as negatives (opt-in only)
    variants_per_item: int = 2
    seed: int | None = None  # null -> project.seed
    min_visibility: float = 0.5  # boxes less visible than this after a transform are lost
    min_box_area: float = 16.0  # pixels^2; smaller boxes are lost
    allow_partial_box_loss: bool = False  # losing ALL boxes of a positive is never allowed
    max_attempts: int = 5  # deterministic retries per variant before it is rejected
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95
    # Overrides of the conservative defaults in cv_pipeline/augmentation/transforms.py
    transforms: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExportConfig:
    """Validated YOLO dataset export (cv_pipeline/export)."""

    enabled: bool = False  # prepare runs export only when true; `export` always runs
    label_policy: Literal["trusted_only", "allow_machine_unreviewed"] = "trusted_only"
    allow_machine_empty: bool = False  # only with allow_machine_unreviewed
    # strict: refuse if any active val/test item lacks a usable trusted label.
    # non_strict: export usable held-out items only, marked incomplete_evaluation_set.
    evaluation_policy: Literal["strict", "non_strict"] = "strict"
    augmentation_run: str = "latest"  # latest | none | <augmentation run id>
    # Training taxonomy: ordered subset of project classes (null = all project classes).
    # Export class ids are indices into this list.
    classes: list[str] | None = None
    # Boxes of project classes outside `classes`: error (refuse) | drop_boxes.
    non_target_classes: Literal["error", "drop_boxes"] = "error"


@dataclass
class TrainingConfig:
    model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str | None = None
    patience: int = 50
    runs_dir: Path = Path("runs")
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    project: ProjectConfig
    classes: list[str]
    root: Path = Path(".")
    log_level: str = "INFO"
    ingest: IngestConfig = field(default_factory=IngestConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    imports: list[ImportConfig] = field(default_factory=list)
    split: SplitConfig = field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def workspace_dir(self) -> Path:
        return self.project.workspace

    @property
    def raw_dir(self) -> Path:
        return self.ingest.raw_dir

    def to_dict(self) -> dict[str, Any]:
        """Plain-YAML-serialisable representation (paths as strings)."""
        return _to_plain(asdict(self))


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    return config_from_dict(data or {}, base_dir=path.resolve().parent)


def config_from_dict(data: dict[str, Any], base_dir: Path) -> PipelineConfig:
    """Build, resolve and validate a config. ``base_dir`` anchors a relative ``root``."""
    cfg = _build(PipelineConfig, data, "config")
    _resolve_paths(cfg, base_dir)
    _normalize(cfg)
    validate_config(cfg)
    return cfg


def _resolve_paths(cfg: PipelineConfig, base_dir: Path) -> None:
    root = cfg.root if cfg.root.is_absolute() else base_dir / cfg.root
    cfg.root = root.resolve()

    def under_root(p: Path) -> Path:
        return (p if p.is_absolute() else cfg.root / p).resolve()

    cfg.project.workspace = under_root(cfg.project.workspace)
    cfg.ingest.raw_dir = under_root(cfg.ingest.raw_dir)
    for imp in cfg.imports:
        imp.path = under_root(imp.path)
    if not cfg.training.runs_dir.is_absolute():
        cfg.training.runs_dir = (cfg.project.workspace / cfg.training.runs_dir).resolve()


def _normalize(cfg: PipelineConfig) -> None:
    def norm_ext(ext: str) -> str:
        ext = ext.strip().lower()
        return ext if ext.startswith(".") else f".{ext}"

    cfg.ingest.image_extensions = sorted({norm_ext(e) for e in cfg.ingest.image_extensions})
    cfg.ingest.video_extensions = sorted({norm_ext(e) for e in cfg.ingest.video_extensions})
    cfg.classes = [c.strip() for c in cfg.classes]
    cfg.log_level = cfg.log_level.upper()


def validate_config(cfg: PipelineConfig) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    # Classes
    check(len(cfg.classes) > 0, "classes: at least one class name is required")
    check(all(cfg.classes), "classes: class names must be non-empty")
    dupes = sorted({c for c in cfg.classes if cfg.classes.count(c) > 1})
    check(not dupes, f"classes: duplicate class names {dupes}")
    check(cfg.project.name.strip() != "", "project.name must be non-empty")

    # Paths: raw data must stay untouched, so the workspace may not overlap it.
    ws, raw = cfg.workspace_dir, cfg.raw_dir
    check(
        not (ws.is_relative_to(raw) or raw.is_relative_to(ws)),
        f"project.workspace ({ws}) and ingest.raw_dir ({raw}) must not contain each other; "
        "raw data is treated as immutable",
    )

    # Ingest
    overlap = set(cfg.ingest.image_extensions) & set(cfg.ingest.video_extensions)
    check(not overlap, f"ingest: extensions listed as both image and video: {sorted(overlap)}")
    v = cfg.ingest.video
    check(v.fps > 0 and math.isfinite(v.fps), "ingest.video.fps must be > 0")
    check(v.every_n_frames >= 1, "ingest.video.every_n_frames must be >= 1")
    check(v.max_frames is None or v.max_frames >= 1, "ingest.video.max_frames must be >= 1 or null")
    check(1 <= v.jpeg_quality <= 100, "ingest.video.jpeg_quality must be in [1, 100]")

    # Dedup
    d = cfg.dedup
    bits = d.hash_size**2
    check(4 <= d.hash_size <= 32, "dedup.hash_size must be in [4, 32]")
    check(
        0 <= d.near_duplicate_threshold < bits,
        f"dedup.near_duplicate_threshold must be in [0, {bits}) for hash_size {d.hash_size}",
    )
    check(
        d.dhash_threshold is None or 0 <= d.dhash_threshold < bits,
        f"dedup.dhash_threshold must be null or in [0, {bits}) for hash_size {d.hash_size}",
    )

    # Annotation
    a = cfg.annotation
    for name in ("inference_confidence_floor", "confidence_threshold", "review_threshold"):
        check(0.0 <= getattr(a, name) <= 1.0, f"annotation.{name} must be in [0, 1]")
    check(
        a.inference_confidence_floor <= a.confidence_threshold <= a.review_threshold,
        "annotation thresholds must satisfy inference_confidence_floor <= "
        "confidence_threshold <= review_threshold",
    )
    check(0.0 < a.iou_threshold <= 1.0, "annotation.iou_threshold must be in (0, 1]")
    check(a.imgsz > 0, "annotation.imgsz must be > 0")
    check(a.batch_size >= 1, "annotation.batch_size must be >= 1")
    check(a.max_detections >= 1, "annotation.max_detections must be >= 1")
    check(0.0 <= a.review_sample_rate <= 1.0, "annotation.review_sample_rate must be in [0, 1]")
    check(a.backend.strip() != "", "annotation.backend must be non-empty")
    unknown_targets = sorted(set(a.class_map.values()) - set(cfg.classes))
    check(not unknown_targets, f"annotation.class_map maps to unknown classes {unknown_targets}")
    # Backend-specific requirements (``model`` for yolo, registered backend name) are
    # checked when annotation is invoked, so other stages run with an incomplete section.

    # Imports
    names = [i.name for i in cfg.imports]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    check(not dup_names, f"imports: duplicate import names {dup_names}")
    for imp in cfg.imports:
        where = f"imports[{imp.name}]"
        check(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", imp.name) is not None,
            f"{where}: name must match [a-z0-9][a-z0-9_-]*",
        )
        for other, label in ((ws, "project.workspace"), (raw, "ingest.raw_dir")):
            check(
                not (imp.path.is_relative_to(other) or other.is_relative_to(imp.path)),
                f"{where}: path must not overlap {label} ({other})",
            )
        bad_targets = sorted(set(imp.class_map.values()) - set(cfg.classes))
        check(not bad_targets, f"{where}: class_map maps to unknown classes {bad_targets}")
        targets = list(imp.class_map.values())
        merged = sorted({t for t in targets if targets.count(t) > 1})
        check(
            not merged,
            f"{where}: several source classes map to the same project class {merged}",
        )
        both = sorted(set(imp.class_map) & set(imp.ignore_classes))
        check(not both, f"{where}: classes both mapped and ignored: {both}")
        check(
            not (imp.empty_labels_are_negatives and not imp.labels_trusted),
            f"{where}: empty_labels_are_negatives requires labels_trusted",
        )
        check(
            not (imp.ignored_only_labels_are_negatives and not imp.labels_trusted),
            f"{where}: ignored_only_labels_are_negatives requires labels_trusted",
        )

    # Split
    s = cfg.split
    fractions = {"train": s.train, "val": s.val, "test": s.test}
    for name, frac in fractions.items():
        check(0.0 <= frac <= 1.0, f"split.{name} must be in [0, 1]")
    check(s.train > 0, "split.train must be > 0")
    check(
        math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-6),
        f"split fractions must sum to 1.0 (got {sum(fractions.values()):.6f})",
    )

    # Augmentation
    aug = cfg.augmentation
    check(aug.variants_per_item >= 1, "augmentation.variants_per_item must be >= 1")
    check(0.0 <= aug.min_visibility <= 1.0, "augmentation.min_visibility must be in [0, 1]")
    check(aug.min_box_area >= 0, "augmentation.min_box_area must be >= 0")
    check(aug.max_attempts >= 1, "augmentation.max_attempts must be >= 1")
    check(1 <= aug.jpeg_quality <= 100, "augmentation.jpeg_quality must be in [1, 100]")
    check(
        not (aug.allow_machine_empty and aug.label_policy != "allow_machine_unreviewed"),
        "augmentation.allow_machine_empty requires label_policy: allow_machine_unreviewed",
    )
    from cv_pipeline.augmentation.transforms import TransformConfigError, resolve_transforms

    try:
        resolve_transforms(aug.transforms)
    except TransformConfigError as exc:
        errors.append(str(exc))

    # Export
    e = cfg.export
    check(
        not (e.allow_machine_empty and e.label_policy != "allow_machine_unreviewed"),
        "export.allow_machine_empty requires label_policy: allow_machine_unreviewed",
    )
    if e.classes is not None:
        check(len(e.classes) > 0, "export.classes must not be empty (use null for all classes)")
        unknown = sorted(set(e.classes) - set(cfg.classes))
        check(not unknown, f"export.classes contains classes not in the project: {unknown}")
        dup = sorted({c for c in e.classes if e.classes.count(c) > 1})
        check(not dup, f"export.classes has duplicates: {dup}")
    check(e.augmentation_run.strip() != "", "export.augmentation_run must not be empty")

    # Training
    t = cfg.training
    check(t.epochs >= 1, "training.epochs must be >= 1")
    check(t.imgsz > 0, "training.imgsz must be > 0")
    check(t.batch == -1 or t.batch >= 1, "training.batch must be >= 1 (or -1 for auto)")
    check(t.patience >= 0, "training.patience must be >= 0")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------------------
# Strict dict -> dataclass builder
# --------------------------------------------------------------------------------------


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{where}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required fields
        raise ConfigError(f"{where}: {exc}") from exc


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)

    if tp is Any:
        return value
    if is_dataclass(tp):
        return _build(tp, {} if value is None else value, where)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{where}: value may not be null")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError("; ".join(errors))
    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{where}: must be one of {list(allowed)}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp)
        return [_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = get_args(tp)
        return {
            _coerce(key_tp, k, f"{where}.<key>"): _coerce(val_tp, v, f"{where}.{k}")
            for k, v in value.items()
        }
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return str(value)
    if tp is Path:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a path string, got {value!r}")
        return Path(value).expanduser()
    raise ConfigError(f"{where}: unsupported schema type {tp!r}")  # pragma: no cover


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
