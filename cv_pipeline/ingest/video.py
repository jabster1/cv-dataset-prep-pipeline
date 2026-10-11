"""Video probing and deterministic frame sampling.

Frame selection is a pure function of (frame_count, native_fps, sampling config) so it is
reproducible and unit-testable. Extraction then reads the video sequentially with
``grab()`` and only decodes selected frames, which is both accurate (no keyframe-seek
drift) and reasonably fast.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv_pipeline.config import VideoSamplingConfig
from cv_pipeline.utils.logging import get_logger, log_event

logger = get_logger("ingest.video")

FALLBACK_FPS = 30.0


class VideoReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    fps_is_estimated: bool = False

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("OpenCV could not open the video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    estimated = False
    if not math.isfinite(fps) or fps <= 0:
        log_event(
            logger,
            "video_fps_unknown",
            logging.WARNING,
            video=path.name,
            assumed_fps=FALLBACK_FPS,
        )
        fps, estimated = FALLBACK_FPS, True

    if frame_count <= 0:
        # Container metadata missing/unreliable: count frames by decoding headers only.
        frame_count = _count_frames(path)
    if frame_count <= 0:
        raise VideoReadError("video contains no readable frames")
    return VideoInfo(fps, frame_count, width, height, estimated)


def _count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n


def select_frame_indices(
    frame_count: int, native_fps: float, sampling: VideoSamplingConfig
) -> list[int]:
    """Indices of frames to extract, sorted ascending and unique."""
    if frame_count <= 0:
        return []

    if sampling.mode == "every_n_frames":
        indices = list(range(0, frame_count, sampling.every_n_frames))
    else:  # fps
        if sampling.fps >= native_fps:
            indices = list(range(frame_count))
        else:
            step = native_fps / sampling.fps
            n = math.ceil(frame_count / step)
            # +1e-9 guards against float error turning e.g. 2.9999999 into 2.
            indices = sorted({min(frame_count - 1, int(k * step + 1e-9)) for k in range(n)})

    max_frames = sampling.max_frames
    if max_frames is not None and len(indices) > max_frames:
        if sampling.max_frames_strategy == "first":
            indices = indices[:max_frames]
        else:  # uniform: spread the budget across the whole video
            picks = np.linspace(0, len(indices) - 1, num=max_frames)
            indices = sorted({indices[int(round(p))] for p in picks})
    return indices


def iter_frames(path: Path, indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR image)`` for each requested index, in order.

    Frames the decoder cannot deliver (truncated files, inaccurate frame counts) are
    logged as a warning rather than silently skipped.
    """
    if not indices:
        return
    wanted = set(indices)
    last = indices[-1]
    delivered = 0
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError("OpenCV could not open the video")
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    delivered += 1
                    yield idx, frame
                else:
                    log_event(logger, "frame_decode_failed", logging.WARNING, video=path.name, frame=idx)
            idx += 1
    finally:
        cap.release()

    if delivered < len(indices):
        log_event(
            logger,
            "video_frames_missing",
            logging.WARNING,
            video=path.name,
            requested=len(indices),
            delivered=delivered,
        )
