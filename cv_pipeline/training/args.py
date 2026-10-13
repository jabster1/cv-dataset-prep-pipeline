"""Allowlisted extra Ultralytics trainer arguments.

Only plain values (numbers, booleans, a few enumerated strings) for well-known trainer
settings are accepted; anything else is a configuration error. This is deliberately not a
pass-through: arguments that control paths, resume, data, devices or callbacks are owned by
the pipeline.

Note: Ultralytics applies its OWN train-time augmentation by default (e.g. fliplr=0.5,
hsv_h=0.015, mosaic=1.0). Set these here when flips/hue/mosaic are semantically unsafe
for a project (the effective values are always recorded in the training run).
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
    # Ultralytics built-in train-time augmentation
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


def validate_trainer_args(args: dict[str, Any]) -> dict[str, Any]:
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
