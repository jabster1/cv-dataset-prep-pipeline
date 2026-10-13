"""Allowlisted Ultralytics trainer arguments, training augmentation policy, download policy.

Trainer arguments: only plain values (numbers, booleans, a few enumerated strings) for
well-known settings are accepted; anything else is a configuration error. Paths, resume,
data, devices and callbacks are owned by the pipeline.

Augmentation ownership (``training.augmentation_policy``):

* ``pipeline_only`` (default) — Phase 5 owns augmentation. Every Ultralytics stochastic
  train-time augmentation is explicitly disabled (not left to version defaults), including
  its built-in Albumentations step (blur/median-blur/to-gray/CLAHE, active whenever the
  ``albumentations`` package is installed), which is disabled with ``augmentations=[]``.
  Refuses to run if the installed Ultralytics has trainer settings unknown to this module.
* ``ultralytics`` — explicit opt-in to trainer-side augmentation: Ultralytics defaults plus
  any allowlisted ``trainer_args``.

Download policy: with ``initialization.allow_download: false`` the known optional download
paths are switched off — ``amp=False`` (the AMP check downloads a model on CUDA) and
``plots=False`` (plots may download fonts).
"""

from __future__ import annotations

from typing import Any


class TrainerArgsError(ValueError):
    pass


_FLOAT_0_1 = ("float", 0.0, 1.0)
ALLOWED: dict[str, tuple[Any, ...]] = {
    # optimisation
    "optimizer": ("choice", ("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp")),
    "lr0": ("float", 0.0, 1.0),
    "lrf": _FLOAT_0_1,
    "momentum": _FLOAT_0_1,
    "weight_decay": ("float", 0.0, 0.1),
    "warmup_epochs": ("float", 0.0, 100.0),
    "cos_lr": ("bool",),
    "close_mosaic": ("int", 0, 10_000),
    "freeze": ("int", 0, 1_000),
    "amp": ("bool",),
    "rect": ("bool",),
    "multi_scale": ("bool",),
    "dropout": _FLOAT_0_1,
    "plots": ("bool",),
    "save_period": ("int", -1, 10_000),
    # Ultralytics built-in train-time augmentation (only with augmentation_policy: ultralytics)
    "bgr": _FLOAT_0_1,
    "cutmix": _FLOAT_0_1,
    "fliplr": _FLOAT_0_1,
    "flipud": _FLOAT_0_1,
    "hsv_h": _FLOAT_0_1,
    "hsv_s": _FLOAT_0_1,
    "hsv_v": _FLOAT_0_1,
    "degrees": ("float", 0.0, 180.0),
    "translate": _FLOAT_0_1,
    "scale": ("float", 0.0, 1.0),
    "shear": ("float", 0.0, 90.0),
    "perspective": ("float", 0.0, 0.001),
    "mosaic": _FLOAT_0_1,
    "mixup": _FLOAT_0_1,
    "copy_paste": _FLOAT_0_1,
    "erasing": _FLOAT_0_1,
}


# Stochastic train-time augmentation settings and the value that disables each one.
AUGMENTATION_DISABLED: dict[str, Any] = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "bgr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "close_mosaic": 0,
    # Ultralytics' built-in Albumentations step: an empty list composes to a no-op.
    "augmentations": [],
}
BUILTIN_ALBUMENTATIONS_NOTE = (
    "Ultralytics default (Blur, MedianBlur, ToGray, CLAHE at p=0.01 when the albumentations "
    "package is installed)"
)


class AugmentationPolicyError(RuntimeError):
    """pipeline_only cannot be guaranteed with the installed trainer."""


def validate_trainer_args(
    args: dict[str, Any],
    augmentation_policy: str = "ultralytics",
    allow_download: bool = True,
) -> dict[str, Any]:
    """Validate trainer args, including conflicts with the augmentation/download policies."""
    out = _validate_values(args)
    if augmentation_policy == "pipeline_only":
        conflicts = sorted(
            k
            for k, v in out.items()
            if k in AUGMENTATION_DISABLED and v != AUGMENTATION_DISABLED[k]
        )
        if conflicts:
            raise TrainerArgsError(
                "training.trainer_args "
                + ", ".join(f"{k}={out[k]}" for k in conflicts)
                + " conflict with augmentation_policy: pipeline_only (Phase 5 owns "
                "augmentation; trainer-side augmentation is disabled). Remove them, or set "
                "augmentation_policy: ultralytics to opt in explicitly."
            )
    if not allow_download:
        for key in ("amp", "plots"):
            if out.get(key) is True:
                raise TrainerArgsError(
                    f"training.trainer_args.{key}: true can trigger Ultralytics downloads "
                    f"({'AMP check weights' if key == 'amp' else 'plot fonts'}) but "
                    "training.initialization.allow_download is false. Allow downloads or "
                    f"remove {key}."
                )
    return out


def resolve_training_args(
    trainer_defaults: dict[str, Any],
    trainer_version: str,
    augmentation_policy: str,
    allow_download: bool,
    trainer_args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """Return (args to pass, resolved augmentation record, download record, warnings)."""
    args = validate_trainer_args(trainer_args, augmentation_policy, allow_download)
    warnings: list[str] = []
    if augmentation_policy == "pipeline_only":
        unknown = sorted(set(trainer_defaults) - KNOWN_TRAINER_KEYS)
        missing = sorted(
            k for k in AUGMENTATION_DISABLED if k != "augmentations" and k not in trainer_defaults
        )
        if unknown or missing:
            raise AugmentationPolicyError(
                "augmentation_policy: pipeline_only cannot confirm that trainer-side "
                f"augmentation is disabled with Ultralytics {trainer_version}: "
                + (f"unknown trainer settings {unknown}; " if unknown else "")
                + (f"expected augmentation settings missing {missing}; " if missing else "")
                + f"this pipeline was validated against Ultralytics {VALIDATED_ULTRALYTICS_VERSION}"
                " (see constraints.txt). Review the new settings and update "
                "cv_pipeline/training/args.py before training."
            )
        if trainer_version != VALIDATED_ULTRALYTICS_VERSION:
            warnings.append(
                f"Ultralytics {trainer_version} differs from the validated "
                f"{VALIDATED_ULTRALYTICS_VERSION}; its settings match the known list, but "
                "verify that no new code-level augmentation was introduced."
            )
        args.update(AUGMENTATION_DISABLED)
    resolved = {
        k: args.get(k, trainer_defaults.get(k))
        for k in AUGMENTATION_DISABLED
        if k != "augmentations"
    }
    resolved["augmentations"] = args.get("augmentations", BUILTIN_ALBUMENTATIONS_NOTE)
    download: dict[str, Any] = {"allow_download": allow_download, "forced": {}}
    if not allow_download:
        for key, why in (
            ("amp", "AMP check downloads a model on CUDA"),
            ("plots", "plots may download font assets"),
        ):
            if args.get(key) is not False:
                args[key] = False
                download["forced"][key] = why
    download["amp"] = args.get("amp", trainer_defaults.get("amp"))
    download["plots"] = args.get("plots", trainer_defaults.get("plots"))
    download["not_guaranteed_offline"] = [
        "Ultralytics anonymous usage analytics follow Ultralytics' own `sync` setting "
        "(disable with `yolo settings sync=False`)",
        "other Ultralytics helper checks outside training (e.g. package update checks) are "
        "not controlled by the pipeline",
    ]
    return args, resolved, download, warnings


def _validate_values(args: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(args) - set(ALLOWED))
    if unknown:
        raise TrainerArgsError(
            f"training.trainer_args: unsupported key(s) {unknown}; allowed: {sorted(ALLOWED)}"
        )
    out: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        spec = ALLOWED[key]
        kind = spec[0]
        if kind == "bool":
            if not isinstance(value, bool):
                raise TrainerArgsError(f"training.trainer_args.{key} must be true/false")
        elif kind == "choice":
            if value not in spec[1]:
                raise TrainerArgsError(
                    f"training.trainer_args.{key} must be one of {list(spec[1])}"
                )
        else:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TrainerArgsError(f"training.trainer_args.{key} must be a number")
            if kind == "int" and not isinstance(value, int):
                raise TrainerArgsError(f"training.trainer_args.{key} must be an integer")
            lo, hi = spec[1], spec[2]
            if not lo <= value <= hi:
                raise TrainerArgsError(f"training.trainer_args.{key} must be in [{lo}, {hi}]")
        out[key] = value
    return out


# All Ultralytics trainer settings at the validated version. pipeline_only refuses to run
# if the installed Ultralytics has settings not listed here (one of them could be a new
# augmentation that this pipeline would otherwise silently leave enabled).
VALIDATED_ULTRALYTICS_VERSION = "8.4.175"
KNOWN_TRAINER_KEYS: frozenset[str] = frozenset(
    {
        "agnostic_nms",
        "amp",
        "angle",
        "augment",
        "auto_augment",
        "batch",
        "bgr",
        "box",
        "cache",
        "cfg",
        "channels_last",
        "classes",
        "close_mosaic",
        "cls",
        "cls_pw",
        "cls_remap",
        "compile",
        "conf",
        "copy_paste",
        "copy_paste_mode",
        "cos_lr",
        "cutmix",
        "data",
        "degrees",
        "deterministic",
        "device",
        "dfl",
        "dgrad",
        "dis",
        "distill_model",
        "dlam",
        "dlog",
        "dnn",
        "dropout",
        "dynamic",
        "embed",
        "epochs",
        "erasing",
        "exist_ok",
        "fliplr",
        "flipud",
        "format",
        "fraction",
        "freeze",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "imgsz",
        "iou",
        "kobj",
        "line_width",
        "lr0",
        "lrf",
        "mask_ratio",
        "max_det",
        "mixup",
        "mode",
        "model",
        "momentum",
        "mosaic",
        "multi_scale",
        "name",
        "nbs",
        "nms",
        "opset",
        "optimize",
        "optimizer",
        "overlap_mask",
        "patience",
        "perspective",
        "plots",
        "pose",
        "pretrained",
        "profile",
        "project",
        "quantize",
        "rect",
        "resume",
        "retina_masks",
        "rle",
        "save",
        "save_conf",
        "save_crop",
        "save_frames",
        "save_json",
        "save_period",
        "save_txt",
        "scale",
        "seed",
        "shear",
        "show",
        "show_boxes",
        "show_conf",
        "show_labels",
        "simplify",
        "single_cls",
        "source",
        "split",
        "stream_buffer",
        "task",
        "time",
        "tracker",
        "translate",
        "val",
        "verbose",
        "vid_stride",
        "visualize",
        "warmup_bias_lr",
        "warmup_epochs",
        "warmup_momentum",
        "weight_decay",
        "workers",
        "workspace",
    }
)
