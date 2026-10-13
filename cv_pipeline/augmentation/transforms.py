"""Registry of detection-safe transforms with conservative defaults.

This module does not import Albumentations (config validation must stay fast); the
pipeline is built lazily by :func:`build_pipeline`.

Every transform has a default enabled state, probability ``p`` and parameters. A config
entry under ``augmentation.transforms.<name>`` overrides only the keys it names. Transforms
are always applied in registry order (not config order) so results are deterministic.

Nothing here is project-specific. Transforms that can change what an object *means* —
flips (text, left/right, handle position), hue (colour-coded state), rotation — are
disabled by default; projects opt in deliberately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    enabled: bool
    p: float
    params: dict[str, Any]
    caution: str = ""


TRANSFORMS: dict[str, TransformSpec] = {
    "horizontal_flip": TransformSpec(
        False,
        0.5,
        {},
        "mirrors text, left/right placement and handle/lever positions",
    ),
    "brightness_contrast": TransformSpec(
        True, 0.5, {"brightness_limit": 0.15, "contrast_limit": 0.15}
    ),
    "gamma": TransformSpec(False, 0.3, {"gamma_limit": [85, 115]}),
    "hue_saturation_value": TransformSpec(
        True,
        0.3,
        {"hue_shift_limit": 0, "sat_shift_limit": 15, "val_shift_limit": 10},
        "hue shifts can change colour-coded state (e.g. red/green indicators)",
    ),
    "gaussian_blur": TransformSpec(True, 0.1, {"blur_limit": [3, 3]}),
    "gauss_noise": TransformSpec(True, 0.15, {"std_range": [0.01, 0.03]}),
    "image_compression": TransformSpec(True, 0.3, {"quality_range": [70, 95]}),
    "affine": TransformSpec(
        True, 0.5, {"scale": [0.9, 1.1], "translate_percent": [-0.05, 0.05], "rotate": [0, 0]}
    ),
    "rotate": TransformSpec(
        False,
        0.3,
        {"limit": 5},
        "axis-aligned boxes grow when rotated; orientation may carry meaning",
    ),
    "perspective": TransformSpec(False, 0.2, {"scale": [0.02, 0.05]}),
    "bbox_safe_crop": TransformSpec(False, 0.3, {"erosion_rate": 0.0}),
}


class TransformConfigError(ValueError):
    pass


def resolve_transforms(overrides: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge config overrides into the defaults. Returns the enabled transforms, in
    registry order, as ``{"name", "p", "params"}``. Raises on unknown names/keys."""
    unknown = sorted(set(overrides) - set(TRANSFORMS))
    if unknown:
        raise TransformConfigError(
            f"augmentation.transforms: unknown transform(s) {unknown}; known: {sorted(TRANSFORMS)}"
        )
    out = []
    for name, spec in TRANSFORMS.items():
        o = dict(overrides.get(name) or {})
        bad = sorted(set(o) - {"enabled", "p", *spec.params})
        if bad:
            raise TransformConfigError(
                f"augmentation.transforms.{name}: unknown key(s) {bad}; allowed: "
                f"{sorted({'enabled', 'p', *spec.params})}"
            )
        enabled = o.pop("enabled", spec.enabled)
        p = o.pop("p", spec.p)
        if not isinstance(enabled, bool):
            raise TransformConfigError(f"augmentation.transforms.{name}.enabled must be true/false")
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
            raise TransformConfigError(f"augmentation.transforms.{name}.p must be in [0, 1]")
        if enabled:
            out.append({"name": name, "p": float(p), "params": {**spec.params, **o}})
    return out


def _pair(v: Any) -> tuple[float, float]:
    if isinstance(v, list | tuple):
        return (float(v[0]), float(v[1]))
    return (-float(v), float(v))


def _build_one(A: Any, cv2: Any, t: dict[str, Any]) -> Any:  # noqa: N803 - albumentations alias
    n, p, k = t["name"], t["p"], t["params"]
    if n == "horizontal_flip":
        return A.HorizontalFlip(p=p)
    if n == "brightness_contrast":
        return A.RandomBrightnessContrast(
            brightness_limit=k["brightness_limit"], contrast_limit=k["contrast_limit"], p=p
        )
    if n == "gamma":
        return A.RandomGamma(gamma_limit=tuple(k["gamma_limit"]), p=p)
    if n == "hue_saturation_value":
        return A.HueSaturationValue(
            hue_shift_limit=k["hue_shift_limit"],
            sat_shift_limit=k["sat_shift_limit"],
            val_shift_limit=k["val_shift_limit"],
            p=p,
        )
    if n == "gaussian_blur":
        return A.GaussianBlur(blur_limit=tuple(int(x) for x in k["blur_limit"]), p=p)
    if n == "gauss_noise":
        return A.GaussNoise(std_range=tuple(k["std_range"]), p=p)
    if n == "image_compression":
        return A.ImageCompression(quality_range=tuple(int(x) for x in k["quality_range"]), p=p)
    if n == "affine":
        return A.Affine(
            scale=_pair(k["scale"]),
            translate_percent=_pair(k["translate_percent"]),
            rotate=_pair(k["rotate"]),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=p,
        )
    if n == "rotate":
        return A.Rotate(limit=_pair(k["limit"]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=p)
    if n == "perspective":
        return A.Perspective(scale=_pair(k["scale"]), border_mode=cv2.BORDER_CONSTANT, fill=0, p=p)
    if n == "bbox_safe_crop":
        return A.BBoxSafeRandomCrop(erosion_rate=float(k["erosion_rate"]), p=p)
    raise TransformConfigError(f"no builder for {n}")  # pragma: no cover


def build_pipeline(
    transforms: list[dict[str, Any]], min_visibility: float, min_box_area: float
) -> Any:
    """Albumentations Compose with YOLO-format boxes, clipping and visibility filtering."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no network version check
    import albumentations as A
    import cv2

    return A.Compose(
        [_build_one(A, cv2, t) for t in transforms],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=min_visibility,
            min_area=min_box_area,
            clip=True,
        ),
        seed=0,
        save_applied_params=True,
    )
