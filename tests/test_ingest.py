from __future__ import annotations

import shutil
from collections import defaultdict

import pytest

from cv_pipeline.config import load_config
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import ItemRecord, RejectedRecord, RejectReason, SourceType, read_records
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.utils.io import read_image
from cv_pipeline.workspace import Workspace

from .conftest import snapshot_tree

N_IMAGES, N_VIDEOS, FRAMES, EVERY_N = 12, 2, 40, 5
FRAMES_PER_VIDEO = FRAMES // EVERY_N


def test_ingests_images_and_samples_videos(synthetic_project) -> None:
    cfg = synthetic_project()
    result = run_ingest(cfg)

    assert result.images_ingested == N_IMAGES
    assert result.videos_ingested == N_VIDEOS
    assert result.frames_extracted == N_VIDEOS * FRAMES_PER_VIDEO
    assert len(result.items) == N_IMAGES + N_VIDEOS * FRAMES_PER_VIDEO
    assert result.rejected == []

    ws = Workspace(cfg.workspace_dir)
    for rec in result.items:
        out = ws.root / rec.path
        assert out.is_file()
        assert sha256_file(out) == rec.sha256
        img = read_image(out)
        assert img is not None and img.shape[:2] == (rec.height, rec.width)


def test_image_copies_are_byte_identical_and_self_sourced(synthetic_project) -> None:
    cfg = synthetic_project()
    result = run_ingest(cfg)
    images = [r for r in result.items if r.source_type == SourceType.IMAGE]
    assert {r.source_path.rsplit(".", 1)[-1] for r in images} == {"jpg", "png", "webp"}
    for rec in images:
        raw = cfg.raw_dir / rec.source_path
        assert (cfg.workspace_dir / rec.path).read_bytes() == raw.read_bytes()
        assert rec.source_id == rec.item_id
        assert rec.sha256 == rec.source_sha256
        assert rec.frame_index is None and rec.parent_item_id is None


def test_video_frames_share_source_id_and_keep_lineage(synthetic_project) -> None:
    cfg = synthetic_project()
    result = run_ingest(cfg)
    by_source: dict[str, list[ItemRecord]] = defaultdict(list)
    for rec in result.items:
        if rec.source_type == SourceType.VIDEO_FRAME:
            by_source[rec.source_id].append(rec)

    assert len(by_source) == N_VIDEOS
    for source_id, frames in by_source.items():
        assert len(frames) == FRAMES_PER_VIDEO
        assert len({f.source_path for f in frames}) == 1
        assert len({f.source_sha256 for f in frames}) == 1
        assert sorted(f.frame_index for f in frames) == list(range(0, FRAMES, EVERY_N))
        for f in frames:
            assert f.item_id.startswith(source_id + "_f")
            assert f.timestamp_s == pytest.approx(f.frame_index / 10.0)
    assert {f[0].source_path for f in by_source.values()} == {
        "videos/clip_000.mp4",
        "videos/clip_001.mov",
    }


def test_raw_directory_is_never_modified(synthetic_project) -> None:
    cfg = synthetic_project()
    before = snapshot_tree(cfg.raw_dir)
    run_ingest(cfg)
    run_ingest(cfg, force=True)
    assert snapshot_tree(cfg.raw_dir) == before


def test_manifest_roundtrip(synthetic_project) -> None:
    cfg = synthetic_project()
    result = run_ingest(cfg)
    ws = Workspace(cfg.workspace_dir)
    assert read_records(ws.items_manifest, ItemRecord) == result.items
    assert read_records(ws.rejected_manifest, RejectedRecord) == []
    assert (ws.reports / "ingest_summary.json").is_file()


def test_ingest_is_deterministic(synthetic_project) -> None:
    cfg = synthetic_project()
    run_ingest(cfg)
    manifest = Workspace(cfg.workspace_dir).items_manifest
    first = manifest.read_bytes()
    run_ingest(cfg, force=True)
    assert manifest.read_bytes() == first

    # A fresh workspace from the same raw data yields the same manifest.
    cfg2 = synthetic_project({"project": {"workspace": "work2"}})
    run_ingest(cfg2)
    assert Workspace(cfg2.workspace_dir).items_manifest.read_bytes() == first


def test_rerun_reuses_existing_outputs(synthetic_project) -> None:
    cfg = synthetic_project()
    first = run_ingest(cfg)
    second = run_ingest(cfg)
    assert second.reused == len(first.items)
    assert second.items == first.items


def test_ids_survive_raw_file_rename(synthetic_project) -> None:
    cfg = synthetic_project()
    first = {r.item_id: r for r in run_ingest(cfg).items}
    (cfg.raw_dir / "images" / "scene_000.jpg").rename(cfg.raw_dir / "images" / "renamed.jpg")
    second = {r.item_id: r for r in run_ingest(cfg).items}
    assert first.keys() == second.keys()
    moved = [r for r in second.values() if r.source_path == "images/renamed.jpg"]
    assert len(moved) == 1


def test_problem_files_are_recorded_not_dropped(synthetic_project) -> None:
    cfg = synthetic_project()
    raw = cfg.raw_dir
    (raw / "images" / "corrupt.jpg").write_bytes(b"not really a jpeg")
    (raw / "images" / "notes.txt").write_text("hello", encoding="utf-8")
    (raw / "videos" / "broken.mp4").write_bytes(b"\x00" * 64)
    shutil.copyfile(raw / "images" / "scene_001.png", raw / "images" / "zz_copy.png")
    (raw / "images" / ".DS_Store").write_bytes(b"x")

    result = run_ingest(cfg)
    reasons = {r.source_path: r.reason for r in result.rejected}
    assert reasons == {
        "images/corrupt.jpg": RejectReason.UNREADABLE_IMAGE,
        "images/notes.txt": RejectReason.UNSUPPORTED_EXTENSION,
        "videos/broken.mp4": RejectReason.UNREADABLE_VIDEO,
        "images/zz_copy.png": RejectReason.EXACT_DUPLICATE,
    }
    dup = next(r for r in result.rejected if r.reason == RejectReason.EXACT_DUPLICATE)
    original = next(r for r in result.items if r.source_path == "images/scene_001.png")
    assert dup.duplicate_of == original.item_id
    assert result.hidden_files_ignored == 1
    # good files are still ingested
    assert result.images_ingested == N_IMAGES


def test_changing_sampling_changes_frames(synthetic_project) -> None:
    cfg = synthetic_project()
    run_ingest(cfg)
    cfg_fps = synthetic_project(
        {"ingest": {"video": {"mode": "fps", "fps": 2.0, "max_frames": 3}}}
    )
    result = run_ingest(cfg_fps)
    frames = [r for r in result.items if r.source_type == SourceType.VIDEO_FRAME]
    assert len(frames) == N_VIDEOS * 3
    # 2 fps of a 10 fps clip -> candidates 0,5,...,35; max_frames=3 (uniform) spans the clip
    assert sorted({f.frame_index for f in frames}) == [0, 20, 35]


def test_missing_raw_dir(write_config) -> None:
    cfg = load_config(write_config())
    with pytest.raises(FileNotFoundError, match="Raw data directory not found"):
        run_ingest(cfg)
