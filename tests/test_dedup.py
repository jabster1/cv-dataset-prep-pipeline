from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import cv_pipeline.dedup.dedup as dedup_mod
from cv_pipeline.config import PipelineConfig, load_config
from cv_pipeline.dedup import DedupResult, run_dedup
from cv_pipeline.dedup.hashing import compute_hashes, hamming
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import (
    DedupRecord,
    DuplicateKind,
    ExcludeReason,
    ItemRecord,
    RejectedRecord,
    RejectReason,
    read_records,
)
from cv_pipeline.synthetic import render_scene, write_synthetic_video
from cv_pipeline.workspace import Workspace

from .conftest import snapshot_tree

DEFAULT_PHASH_T, DEFAULT_DHASH_T = 6, 10


def scene(seed: int) -> np.ndarray:
    return render_scene(np.random.default_rng(seed))[0]


def save(path: Path, image: np.ndarray, params: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image, params or [])
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def brighter(img: np.ndarray, beta: int = 12) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def shifted(img: np.ndarray, dx: int) -> np.ndarray:
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def resized(img: np.ndarray, factor: float = 0.9) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)


@pytest.fixture
def project(tmp_path: Path, write_config: Callable[..., Path]):
    """Returns (raw_images_dir, run) where run(overrides, force) -> (DedupResult, cfg)."""
    raw_images = tmp_path / "raw" / "images"
    raw_images.mkdir(parents=True)

    def run(
        overrides: dict[str, Any] | None = None, force: bool = False
    ) -> tuple[DedupResult, PipelineConfig]:
        cfg = load_config(write_config(overrides))
        run_ingest(cfg)
        return run_dedup(cfg, force=force), cfg

    return raw_images, run


def by_source_path(result: DedupResult, cfg: PipelineConfig) -> dict[str, DedupRecord]:
    """Dedup records keyed by raw path relative to ``raw/images`` (videos keep full path)."""
    items = read_records(Workspace(cfg.workspace_dir).items_manifest, ItemRecord)
    path_of = {i.item_id: i.source_path.removeprefix("images/") for i in items}
    return {path_of[r.item_id]: r for r in result.records}


# --------------------------------------------------------------------------------------
# Hash primitives
# --------------------------------------------------------------------------------------


def test_hashes_are_stable_under_mild_changes_and_separate_different_scenes() -> None:
    base = scene(1)
    h0 = compute_hashes(base)
    jpeg = cv2.imdecode(cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 80])[1], 1)
    for variant in (base.copy(), brighter(base), resized(base), jpeg):
        h = compute_hashes(variant)
        assert hamming(h0.phash, h.phash) <= DEFAULT_PHASH_T
        assert hamming(h0.dhash, h.dhash) <= DEFAULT_DHASH_T

    for seed in range(2, 12):
        other = compute_hashes(scene(seed))
        assert hamming(h0.phash, other.phash) > DEFAULT_PHASH_T


def test_pixel_hash_ignores_encoding_differences() -> None:
    img = scene(4)
    a = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 0])[1].tobytes()
    b = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()
    assert a != b
    ha = compute_hashes(cv2.imdecode(np.frombuffer(a, np.uint8), 1))
    hb = compute_hashes(cv2.imdecode(np.frombuffer(b, np.uint8), 1))
    assert ha.pixel_sha256 == hb.pixel_sha256


def test_hash_size_changes_hash_length() -> None:
    assert len(compute_hashes(scene(1), hash_size=16).phash) == 16 * 16 // 4


# --------------------------------------------------------------------------------------
# Stage behaviour
# --------------------------------------------------------------------------------------


def test_byte_identical_and_renamed_files_are_reported(project) -> None:
    raw, run = project
    save(raw / "a.png", scene(1))
    save(raw / "b.png", scene(2))
    (raw / "nested").mkdir()
    shutil.copyfile(raw / "a.png", raw / "nested" / "renamed_copy.png")
    shutil.copyfile(raw / "b.png", raw / "z_same_name_diff_folder.png")

    result, cfg = run()
    raw_dups = {d["source_path"]: d["duplicate_of"] for d in result.report["raw_exact_duplicates"]}
    recs = by_source_path(result, cfg)
    assert raw_dups == {
        "images/nested/renamed_copy.png": recs["a.png"].item_id,
        "images/z_same_name_diff_folder.png": recs["b.png"].item_id,
    }
    # The copies were never ingested as separate items, so nothing is double-counted.
    assert len(result.records) == 2
    md = (Workspace(cfg.workspace_dir).reports / "duplicates.md").read_text(encoding="utf-8")
    assert "nested/renamed_copy.png" in md


def test_canonical_raw_copy_is_not_the_one_in_a_duplicates_folder(project) -> None:
    raw, run = project
    save(raw / "scene_001.png", scene(1))
    (raw / "duplicates").mkdir()
    # Both copies sort before "scene_001.png", which used to make one of them canonical.
    shutil.copyfile(raw / "scene_001.png", raw / "duplicates" / "renamed_copy_of_scene_001.png")
    shutil.copyfile(raw / "scene_001.png", raw / "scene_001 (1).png")

    result, cfg = run()
    recs = by_source_path(result, cfg)
    assert set(recs) == {"scene_001.png"}
    canonical_id = recs["scene_001.png"].item_id
    rejected = read_records(Workspace(cfg.workspace_dir).rejected_manifest, RejectedRecord)
    assert {r.source_path: (r.reason, r.duplicate_of, r.detail) for r in rejected} == {
        "images/duplicates/renamed_copy_of_scene_001.png": (
            RejectReason.EXACT_DUPLICATE, canonical_id, "byte-identical to images/scene_001.png"
        ),
        "images/scene_001 (1).png": (
            RejectReason.EXACT_DUPLICATE, canonical_id, "byte-identical to images/scene_001.png"
        ),
    }


def test_canonical_pixel_identical_item_prefers_non_copy_path(project) -> None:
    raw, run = project
    img = scene(5)
    save(raw / "backup" / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # sorts first
    save(raw / "z.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    result, cfg = run()
    recs = by_source_path(result, cfg)
    assert not recs["z.png"].excluded
    assert recs["backup/a.png"].excluded
    assert recs["backup/a.png"].duplicate_of == recs["z.png"].item_id


def test_canonical_pixel_identical_prefers_shallower_path(project) -> None:
    raw, run = project
    img = scene(5)
    save(raw / "site" / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(raw / "z.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    result, cfg = run()
    recs = by_source_path(result, cfg)
    assert recs["site/a.png"].duplicate_of == recs["z.png"].item_id


def test_video_frame_is_canonical_over_a_still_exported_from_it(project, tmp_path: Path) -> None:
    raw, run = project
    video = tmp_path / "raw" / "videos" / "clip.mp4"
    write_synthetic_video(video, n_frames=6, fps=10.0, rng=np.random.default_rng(2))
    cap = cv2.VideoCapture(str(video))
    for _ in range(4):
        ok, frame = cap.read()
    cap.release()
    assert ok
    # pixel-identical to frame 3, but different bytes from the pipeline's own PNG encoding
    save(raw / "a_exported_frame.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    result, cfg = run({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1,
                                            "image_format": "png"}}})
    recs = by_source_path(result, cfg)
    still = recs["a_exported_frame.png"]
    frames = {r.item_id: r for r in result.records if r.source_id.startswith("vid_")}
    assert len(frames) == 6 and not any(r.excluded for r in frames.values())
    assert still.excluded and still.exclude_reason == ExcludeReason.EXACT_DUPLICATE
    assert still.duplicate_of in frames and still.duplicate_of.endswith("_f000003")
    assert [p.kind for p in result.pairs] == [DuplicateKind.PIXEL_IDENTICAL]


def test_pixel_identical_files_excluded_by_default(project) -> None:
    raw, run = project
    img = scene(5)
    save(raw / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(raw / "b.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    save(raw / "c.png", scene(6))

    result, cfg = run()
    recs = by_source_path(result, cfg)
    assert [p.kind for p in result.pairs] == [DuplicateKind.PIXEL_IDENTICAL]
    assert not recs["a.png"].excluded
    assert recs["b.png"].excluded
    assert recs["b.png"].exclude_reason == ExcludeReason.EXACT_DUPLICATE
    assert recs["b.png"].duplicate_of == recs["a.png"].item_id
    assert recs["a.png"].dup_group == recs["b.png"].dup_group is not None
    assert not recs["c.png"].excluded and recs["c.png"].dup_group is None
    assert result.report["excluded"] == {"total": 1, "by_reason": {"exact_duplicate": 1}}


def test_exact_duplicates_can_be_kept_but_stay_in_one_split_group(project) -> None:
    raw, run = project
    img = scene(5)
    save(raw / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(raw / "b.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    result, cfg = run({"dedup": {"exact_duplicate_action": "keep"}})
    recs = by_source_path(result, cfg)
    assert not any(r.excluded for r in result.records)
    assert recs["a.png"].split_group == recs["b.png"].split_group


def test_visually_similar_images_are_grouped_not_excluded(project) -> None:
    raw, run = project
    base = scene(7)
    save(raw / "a_base.jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    save(raw / "b_brighter.jpg", brighter(base), [cv2.IMWRITE_JPEG_QUALITY, 85])
    save(raw / "c_resized.png", resized(base))
    save(raw / "d_unrelated.png", scene(8))

    result, cfg = run()
    recs = by_source_path(result, cfg)
    similar = ["a_base.jpg", "b_brighter.jpg", "c_resized.png"]

    assert {p.kind for p in result.pairs} == {DuplicateKind.NEAR_DUPLICATE}
    assert all(not p.same_source for p in result.pairs)
    assert len({recs[n].dup_group for n in similar}) == 1
    assert recs[similar[0]].dup_group is not None
    # same split group => cannot leak across train/val/test
    assert len({recs[n].split_group for n in similar}) == 1
    assert recs["d_unrelated.png"].split_group != recs["a_base.jpg"].split_group
    # default near_duplicate_action is keep: nothing excluded
    assert not any(r.excluded for r in result.records)


def test_near_duplicates_can_be_excluded_when_configured(project) -> None:
    raw, run = project
    base = scene(7)
    save(raw / "a_base.jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    save(raw / "b_brighter.jpg", brighter(base), [cv2.IMWRITE_JPEG_QUALITY, 85])

    result, cfg = run({"dedup": {"near_duplicate_action": "exclude"}})
    recs = by_source_path(result, cfg)
    assert not recs["a_base.jpg"].excluded
    assert recs["b_brighter.jpg"].excluded
    assert recs["b_brighter.jpg"].exclude_reason == ExcludeReason.NEAR_DUPLICATE
    assert recs["b_brighter.jpg"].duplicate_of == recs["a_base.jpg"].item_id
    # still on disk and in items.csv
    items = read_records(Workspace(cfg.workspace_dir).items_manifest, ItemRecord)
    assert len(items) == 2


def test_clearly_different_images_are_not_linked(project) -> None:
    raw, run = project
    for seed in range(20, 32):
        save(raw / f"s{seed}.png", scene(seed))
    result, _ = run()
    assert result.pairs == []
    assert all(r.dup_group is None and not r.excluded for r in result.records)
    assert all(r.split_group == r.source_id for r in result.records)


def test_thresholds_are_configurable(project) -> None:
    raw, run = project
    base = scene(9)
    variant = shifted(base, 4)  # slightly re-framed shot of the same scene
    dist = hamming(compute_hashes(base).phash, compute_hashes(variant).phash)
    assert 0 < dist <= DEFAULT_PHASH_T  # precondition for this test
    save(raw / "a.png", base)
    save(raw / "b.png", variant)

    linked, _ = run()
    assert [p.phash_distance for p in linked.pairs] == [dist]
    strict, _ = run({"dedup": {"near_duplicate_threshold": dist - 1}})
    assert strict.pairs == []
    phash_only, _ = run({"dedup": {"near_duplicate_threshold": dist, "dhash_threshold": None}})
    assert len(phash_only.pairs) == 1


def test_adjacent_video_frames_are_not_over_filtered(project, tmp_path: Path) -> None:
    raw, run = project
    write_synthetic_video(
        tmp_path / "raw" / "videos" / "slow.mp4", n_frames=30, fps=10.0,
        rng=np.random.default_rng(0),
    )
    save(raw / "unrelated.png", scene(3))

    result, cfg = run({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1}}})
    frames = [r for r in result.records if r.source_id.startswith("vid_")]
    assert len(frames) == 30
    assert not any(r.excluded for r in frames)
    assert all(r.dup_group is None for r in frames)
    assert not any(p.same_source for p in result.pairs if p.kind == DuplicateKind.NEAR_DUPLICATE)
    assert len({r.split_group for r in frames}) == 1

    (video,) = result.report["video_redundancy"]
    assert video["frames"] == 30
    assert video["similar_consecutive_pairs"] > 0  # they ARE similar — and still all kept
    assert result.report["pairs"]["near_duplicate_same_source_not_linked"] > 0


def test_still_image_of_a_video_frame_joins_the_video_split_group(project, tmp_path: Path) -> None:
    raw, run = project
    video = tmp_path / "raw" / "videos" / "clip.mp4"
    write_synthetic_video(video, n_frames=20, fps=10.0, rng=np.random.default_rng(1))
    cap = cv2.VideoCapture(str(video))
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    assert ok
    save(raw / "photo_of_same_scene.jpg", brighter(frame, 8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    save(raw / "unrelated.png", scene(3))

    result, cfg = run({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 1}}})
    recs = by_source_path(result, cfg)
    photo = recs["photo_of_same_scene.jpg"]
    frames = [r for r in result.records if r.source_id.startswith("vid_")]
    assert photo.dup_group is not None
    assert {r.split_group for r in frames} == {photo.split_group}
    assert recs["unrelated.png"].split_group != photo.split_group


def test_group_near_duplicates_can_be_disabled_with_warning(project) -> None:
    raw, run = project
    base = scene(7)
    save(raw / "a.jpg", base)
    save(raw / "b.jpg", brighter(base))
    result, cfg = run({"dedup": {"group_near_duplicates": False}})
    recs = by_source_path(result, cfg)
    assert recs["a.jpg"].dup_group == recs["b.jpg"].dup_group is not None
    assert recs["a.jpg"].split_group != recs["b.jpg"].split_group
    assert any("may land in different splits" in w for w in result.report["warnings"])


def test_disabled_dedup_still_assigns_source_split_groups(project) -> None:
    raw, run = project
    img = scene(5)
    save(raw / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(raw / "b.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    result, _ = run({"dedup": {"enabled": False}})
    assert result.pairs == []
    assert all(r.split_group == r.source_id and not r.excluded for r in result.records)


def test_deterministic_reruns(project, write_config) -> None:
    raw, run = project
    base = scene(7)
    save(raw / "a.jpg", base)
    save(raw / "b.jpg", brighter(base))
    img = scene(5)
    save(raw / "c.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(raw / "d.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    for seed in range(40, 45):
        save(raw / f"s{seed}.png", scene(seed))

    outputs = lambda ws: [  # noqa: E731
        ws.dedup_manifest,
        ws.duplicate_pairs_manifest,
        ws.reports / "duplicates.json",
        ws.reports / "duplicates.md",
    ]
    _, cfg = run()
    ws = Workspace(cfg.workspace_dir)
    first = [p.read_bytes() for p in outputs(ws)]

    run()  # cached hashes
    assert [p.read_bytes() for p in outputs(ws)] == first
    run(force=True)  # recomputed hashes
    assert [p.read_bytes() for p in outputs(ws)] == first

    _, cfg2 = run({"project": {"workspace": "work_fresh"}})
    assert [p.read_bytes() for p in outputs(Workspace(cfg2.workspace_dir))] == first


def test_hash_cache_is_reused(project, monkeypatch) -> None:
    raw, run = project
    for seed in range(3):
        save(raw / f"s{seed}.png", scene(seed))
    run()

    calls = []
    real = dedup_mod.compute_hashes
    monkeypatch.setattr(dedup_mod, "compute_hashes", lambda *a, **k: calls.append(1) or real(*a, **k))
    run()
    assert calls == []
    run({"dedup": {"hash_size": 16, "near_duplicate_threshold": 20, "dhash_threshold": 40}})
    assert len(calls) == 3  # different hash_size => cache miss


def test_dedup_never_modifies_raw_data(project, tmp_path: Path) -> None:
    raw, run = project
    save(raw / "a.png", scene(1))
    shutil.copyfile(raw / "a.png", raw / "copy.png")
    before = snapshot_tree(tmp_path / "raw")
    run({"dedup": {"near_duplicate_action": "exclude"}})
    assert snapshot_tree(tmp_path / "raw") == before


def test_dedup_requires_ingest(write_config) -> None:
    cfg = load_config(write_config())
    with pytest.raises(FileNotFoundError, match="run the ingest stage first"):
        run_dedup(cfg)
