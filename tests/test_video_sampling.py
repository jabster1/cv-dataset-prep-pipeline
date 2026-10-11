from __future__ import annotations

import numpy as np
import pytest

from cv_pipeline.config import VideoSamplingConfig
from cv_pipeline.ingest.video import iter_frames, probe_video, select_frame_indices
from cv_pipeline.synthetic import write_synthetic_video


def test_every_n_frames() -> None:
    s = VideoSamplingConfig(mode="every_n_frames", every_n_frames=10)
    assert select_frame_indices(35, 30.0, s) == [0, 10, 20, 30]


def test_fps_mode_integer_ratio() -> None:
    s = VideoSamplingConfig(mode="fps", fps=2.0)
    # 30 fps -> one frame every 15; 3 seconds of video
    assert select_frame_indices(90, 30.0, s) == [0, 15, 30, 45, 60, 75]


def test_fps_mode_non_integer_ratio_stays_in_bounds() -> None:
    s = VideoSamplingConfig(mode="fps", fps=1.0)
    idx = select_frame_indices(300, 29.97, s)
    assert idx[0] == 0
    assert all(0 <= i < 300 for i in idx)
    assert len(idx) == 11  # ceil(300 / 29.97)
    assert idx == sorted(set(idx))


def test_fps_higher_than_native_takes_every_frame() -> None:
    s = VideoSamplingConfig(mode="fps", fps=60.0)
    assert select_frame_indices(12, 25.0, s) == list(range(12))


def test_max_frames_uniform_spreads_across_video() -> None:
    s = VideoSamplingConfig(mode="every_n_frames", every_n_frames=1, max_frames=5)
    idx = select_frame_indices(1000, 30.0, s)
    assert len(idx) == 5
    assert idx[0] == 0 and idx[-1] == 999


def test_max_frames_first() -> None:
    s = VideoSamplingConfig(mode="every_n_frames", every_n_frames=1, max_frames=5,
                            max_frames_strategy="first")
    assert select_frame_indices(1000, 30.0, s) == [0, 1, 2, 3, 4]


def test_selection_is_deterministic() -> None:
    s = VideoSamplingConfig(mode="fps", fps=3.3, max_frames=17)
    assert select_frame_indices(5000, 23.976, s) == select_frame_indices(5000, 23.976, s)


def test_empty_video() -> None:
    assert select_frame_indices(0, 30.0, VideoSamplingConfig()) == []


@pytest.mark.parametrize("ext", [".mp4", ".mov"])
def test_probe_and_iter_frames_real_file(tmp_path, ext) -> None:
    path = tmp_path / f"v{ext}"
    write_synthetic_video(path, n_frames=20, fps=10.0, rng=np.random.default_rng(0))
    info = probe_video(path)
    assert info.frame_count == 20
    assert info.fps == pytest.approx(10.0)
    frames = list(iter_frames(path, [0, 7, 19]))
    assert [i for i, _ in frames] == [0, 7, 19]
    assert frames[0][1].shape == (240, 320, 3)
    # the square moves, so sampled frames must differ
    assert not np.array_equal(frames[0][1], frames[2][1])
