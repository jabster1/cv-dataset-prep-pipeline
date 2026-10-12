"""Tiny synthetic dataset generator for tests, demos and smoke-testing a config.

Scenes are noisy gradient backgrounds with simple coloured shapes. Ground-truth boxes are
known exactly, which later stages (annotation, augmentation, validation) use in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SYNTHETIC_CLASSES: tuple[str, ...] = ("square", "circle", "triangle")
_CLASS_COLORS = ((40, 40, 220), (40, 200, 40), (220, 80, 40))  # BGR
_IMAGE_EXTS = (".jpg", ".png", ".webp")


@dataclass(frozen=True)
class SyntheticBox:
    """A ground-truth box in YOLO format (normalised centre x/y, width, height)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def _background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Gradient in a random direction + soft blobs + sensor-like noise.

    Varying the low-frequency structure matters: perceptual hashes are dominated by it, so
    identical gradients would make unrelated synthetic scenes look like near-duplicates.
    """
    base = rng.integers(60, 200, size=3).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (np.cos(angle) * (xx / width - 0.5) + np.sin(angle) * (yy / height - 0.5)) * 80
    img = base[None, None, :] + ramp[:, :, None]
    for _ in range(int(rng.integers(2, 5))):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        sigma = rng.uniform(0.1, 0.3) * min(width, height)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        img += blob[:, :, None] * rng.uniform(-60, 60, size=3)[None, None, :]
    img = img + rng.normal(0, 6, size=(height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw(img: np.ndarray, class_id: int, x0: int, y0: int, size: int) -> None:
    color = _CLASS_COLORS[class_id % len(_CLASS_COLORS)]
    x1, y1 = x0 + size - 1, y0 + size - 1
    if class_id == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=-1)
    elif class_id == 1:
        r = size // 2
        cv2.circle(img, (x0 + r, y0 + r), r, color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        pts = np.array([[x0 + size // 2, y0], [x0, y1], [x1, y1]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color, lineType=cv2.LINE_AA)


def render_scene(
    rng: np.random.Generator,
    width: int = 320,
    height: int = 240,
    max_objects: int = 3,
    positions: list[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, list[SyntheticBox]]:
    """Render one scene. ``positions`` = explicit ``(class_id, x0, y0, size)`` tuples."""
    img = _background(rng, width, height)
    if positions is None:
        positions = []
        for _ in range(int(rng.integers(1, max_objects + 1))):
            size = int(rng.integers(min(width, height) // 8, min(width, height) // 3))
            x0 = int(rng.integers(0, width - size))
            y0 = int(rng.integers(0, height - size))
            positions.append((int(rng.integers(0, len(SYNTHETIC_CLASSES))), x0, y0, size))

    boxes: list[SyntheticBox] = []
    for class_id, x0, y0, size in positions:
        _draw(img, class_id, x0, y0, size)
        boxes.append(
            SyntheticBox(
                class_id=class_id,
                cx=(x0 + size / 2) / width,
                cy=(y0 + size / 2) / height,
                w=size / width,
                h=size / height,
            )
        )
    return img, boxes


def write_synthetic_raw(
    raw_dir: Path,
    n_images: int = 12,
    n_videos: int = 2,
    video_frames: int = 40,
    video_fps: float = 10.0,
    seed: int = 0,
    labels_dir: Path | None = None,
) -> dict[str, int]:
    """Write a small raw dataset: ``raw_dir/images/*`` (jpg/png/webp) and ``raw_dir/videos/*``
    (mp4/mov). If ``labels_dir`` is given, ground-truth YOLO labels for the still images are
    written there (never inside ``raw_dir``)."""
    rng = np.random.default_rng(seed)
    img_dir, vid_dir = raw_dir / "images", raw_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        img, boxes = render_scene(rng)
        ext = _IMAGE_EXTS[i % len(_IMAGE_EXTS)]
        name = f"scene_{i:03d}"
        ok, buf = cv2.imencode(ext, img)
        if not ok:  # pragma: no cover - depends on OpenCV build
            raise RuntimeError(f"OpenCV cannot encode {ext}")
        (img_dir / f"{name}{ext}").write_bytes(buf.tobytes())
        if labels_dir is not None:
            (labels_dir / f"{name}.txt").write_text(
                "\n".join(b.to_yolo_line() for b in boxes) + "\n", encoding="utf-8"
            )

    for v in range(n_videos):
        ext = ".mp4" if v % 2 == 0 else ".mov"
        write_synthetic_video(vid_dir / f"clip_{v:03d}{ext}", video_frames, video_fps, rng)

    return {"images": n_images, "videos": n_videos, "frames_per_video": video_frames}


def write_synthetic_video(
    path: Path,
    n_frames: int,
    fps: float,
    rng: np.random.Generator,
    width: int = 320,
    height: int = 240,
) -> None:
    """A square drifting across a static background."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on OpenCV build
        raise RuntimeError(f"OpenCV cannot write video {path}")
    bg = _background(rng, width, height)
    size = height // 4
    class_id = int(rng.integers(0, len(SYNTHETIC_CLASSES)))
    try:
        for f in range(n_frames):
            frame = bg.copy()
            x0 = int((width - size) * f / max(1, n_frames - 1))
            _draw(frame, class_id, x0, (height - size) // 2, size)
            writer.write(frame)
    finally:
        writer.release()
