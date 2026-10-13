"""Synthetic YOLO / Roboflow exports for import tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from cv_pipeline.synthetic import SYNTHETIC_CLASSES, render_scene

ROBOFLOW_SPLITS = {"train": "train", "val": "valid", "test": "test"}


def export_image(seed: int) -> tuple[np.ndarray, list[str]]:
    img, boxes = render_scene(np.random.default_rng(1000 + seed))
    return img, [b.to_yolo_line() for b in boxes]


def make_export(
    root: Path,
    splits: dict[str, list[int]] | None = None,
    names: list[str] | tuple[str, ...] = SYNTHETIC_CLASSES,
    layout: str = "roboflow",
    empty: set[int] = frozenset(),  # type: ignore[assignment]
    missing: set[int] = frozenset(),  # type: ignore[assignment]
    label_override: dict[int, str] | None = None,
    names_as_dict: bool = False,
) -> dict[int, str]:
    """Write an export; returns seed -> image path relative to ``root``."""
    splits = splits or {"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]}
    label_override = label_override or {}
    root.mkdir(parents=True, exist_ok=True)
    names_value = {i: n for i, n in enumerate(names)} if names_as_dict else list(names)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names_value,
            }
        ),
        encoding="utf-8",
    )
    paths = {}
    for split, seeds in splits.items():
        d = ROBOFLOW_SPLITS[split] if layout == "roboflow" else split
        img_dir = root / d / "images" if layout == "roboflow" else root / "images" / d
        lbl_dir = root / d / "labels" if layout == "roboflow" else root / "labels" / d
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            img, lines = export_image(seed)
            name = f"img_{seed:03d}"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            assert ok
            (img_dir / f"{name}.jpg").write_bytes(buf.tobytes())
            paths[seed] = (img_dir / f"{name}.jpg").relative_to(root).as_posix()
            if seed in missing:
                continue
            text = "" if seed in empty else "\n".join(lines) + "\n"
            (lbl_dir / f"{name}.txt").write_text(label_override.get(seed, text), encoding="utf-8")
    return paths
