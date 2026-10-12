"""Annotate stage tests using the deterministic scripted backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import cv_pipeline.annotation.annotate as annotate_mod
from cv_pipeline.annotation import AnnotationSetupError, run_annotate
from cv_pipeline.annotation.identity import input_pixel_sha256
from cv_pipeline.annotation.schema import (
    AnnotationItemRecord,
    BoxDecision,
    InferenceStatus,
    PredictionRecord,
    Provenance,
    ReviewReason,
    ReviewStatus,
)
from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError, load_config
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.dedup import run_dedup
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import ItemRecord, read_records
from cv_pipeline.synthetic import render_scene, write_synthetic_video
from cv_pipeline.utils.io import read_image
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes import det

ANN = {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}}


def scene(seed: int) -> np.ndarray:
    return render_scene(np.random.default_rng(seed))[0]


def save(path: Path, image: np.ndarray, params: list[int] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image, params or [])
    assert ok
    path.write_bytes(buf.tobytes())
    return path


class Project:
    def __init__(self, tmp_path: Path, write_config) -> None:
        self.tmp, self._write = tmp_path, write_config
        self.raw = tmp_path / "raw" / "images"
        self.raw.mkdir(parents=True)

    def config(self, overrides: dict[str, Any] | None = None):
        return load_config(self._write(deep_update(ANN, overrides or {})))

    def prepare(self, overrides: dict[str, Any] | None = None):
        cfg = self.config(overrides)
        run_ingest(cfg)
        run_dedup(cfg)
        return cfg

    def key(self, name: str) -> str:
        return input_pixel_sha256(read_image(self.raw / name))

    def item_id(self, cfg, name: str) -> str:
        items = read_records(Workspace(cfg.workspace_dir).items_manifest, ItemRecord)
        return next(i.item_id for i in items if i.source_path == f"images/{name}")


@pytest.fixture
def proj(tmp_path, write_config, scripted) -> Project:
    return Project(tmp_path, write_config)


def run_files(run_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(run_dir).as_posix(): p.read_bytes()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------------------
# Eligibility and lineage
# --------------------------------------------------------------------------------------


def test_excluded_duplicates_are_skipped_and_never_inferred(proj, scripted) -> None:
    img = scene(1)
    save(proj.raw / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(proj.raw / "b.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # pixel-identical -> excluded
    save(proj.raw / "c.png", scene(2))
    cfg = proj.prepare()

    result = run_annotate(cfg)
    ids = [r.item_id for r in result.items]
    assert ids == sorted([proj.item_id(cfg, "a.png"), proj.item_id(cfg, "c.png")])
    assert proj.item_id(cfg, "b.png") not in ids
    assert result.skipped_excluded == 1
    assert sorted(scripted.inferred) == sorted([proj.key("a.png"), proj.key("c.png")])


def test_eligible_items_annotated_with_lineage(proj, scripted, tmp_path) -> None:
    save(proj.raw / "a.png", scene(1))
    write_synthetic_video(tmp_path / "raw" / "videos" / "v.mp4", 10, 10.0, np.random.default_rng(0))
    scripted.default = [det("square", 0.9, 10, 10, 50, 50)]
    cfg = proj.prepare({"ingest": {"video": {"mode": "every_n_frames", "every_n_frames": 5}}})

    result = run_annotate(cfg)
    items = {
        i.item_id: i for i in read_records(Workspace(cfg.workspace_dir).items_manifest, ItemRecord)
    }
    assert {r.item_id for r in result.items} == set(items)
    for r in result.items:
        src = items[r.item_id]
        assert (r.source_id, r.source_type, r.source_path, r.frame_index, r.image_path) == (
            src.source_id,
            src.source_type,
            src.source_path,
            src.frame_index,
            src.path,
        )
        assert r.file_sha256 == src.sha256
        assert (r.width, r.height) == (src.width, src.height)
        assert r.provenance == Provenance.MACHINE
        assert r.inference_status == InferenceStatus.OK
    frames = [r for r in result.items if r.frame_index is not None]
    assert len(frames) == 2 and len({f.source_id for f in frames}) == 1


def test_manifests_on_disk_match_result(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [det("square", 0.9, 10, 10, 50, 50), det("circle", 0.3, 1, 1, 5, 5)]
    cfg = proj.prepare()
    result = run_annotate(cfg)
    assert read_records(result.run_dir / "items.csv", AnnotationItemRecord) == result.items
    assert read_records(result.run_dir / "predictions.csv", PredictionRecord) == result.predictions
    run_json = json.loads((result.run_dir / "run.json").read_text())
    assert run_json["identity"]["run_id"] == result.run_id
    assert run_json["backend"]["name"] == "scripted"
    active = json.loads(Workspace(cfg.workspace_dir).active_annotation_run.read_text())
    assert active["run_id"] == result.run_id


# --------------------------------------------------------------------------------------
# Detections and policy
# --------------------------------------------------------------------------------------


def test_multiple_detections_sorted_and_written_as_yolo(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))  # 320 x 240
    scripted.default = [
        det("triangle", 0.7, 0, 0, 32, 24),
        det("square", 0.6, 160, 120, 320, 240),
        det("square", 0.95, 32, 24, 96, 72),
    ]
    cfg = proj.prepare()
    result = run_annotate(cfg)
    preds = result.predictions
    assert [(p.class_name, p.confidence) for p in preds] == [
        ("square", 0.95),
        ("square", 0.6),
        ("triangle", 0.7),
    ]
    assert [p.pred_index for p in preds] == [0, 1, 2]
    (item,) = result.items
    assert item.n_candidates == 3 and item.review_status == ReviewStatus.UNREVIEWED
    label = (result.run_dir / item.label_path).read_text().splitlines()
    assert label == [
        "0 0.200000 0.200000 0.200000 0.200000",
        "0 0.750000 0.750000 0.500000 0.500000",
        "2 0.050000 0.050000 0.100000 0.100000",
    ]


def test_zero_detections(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    (item,) = run_annotate(cfg).items
    assert item.n_raw == 0 and item.n_candidates == 0
    assert item.review_status == ReviewStatus.NEEDS_REVIEW
    assert item.review_reasons == ReviewReason.NO_DETECTIONS
    assert (
        Workspace(cfg.workspace_dir).annotation_runs / item.run_id / item.label_path
    ).read_text() == ""

    cfg2 = proj.prepare({"annotation": {"review_empty": False}})
    (item2,) = run_annotate(cfg2).items
    assert item2.review_status == ReviewStatus.UNREVIEWED


def test_confidence_filtering_is_recorded_not_silent(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [
        det("square", 0.10, 0, 0, 10, 10),  # below min -> discarded
        det("circle", 0.30, 0, 0, 10, 10),  # review range
        det("triangle", 0.80, 0, 0, 10, 10),  # confident candidate
        det("zebra", 0.99, 0, 0, 10, 10),  # unmapped
    ]
    cfg = proj.prepare()
    result = run_annotate(cfg)
    by_label = {p.backend_label: p for p in result.predictions}
    assert by_label["square"].decision == BoxDecision.DISCARDED_BELOW_MIN_CONFIDENCE
    assert by_label["circle"].decision == BoxDecision.CANDIDATE and by_label["circle"].review_flag
    assert (
        by_label["triangle"].decision == BoxDecision.CANDIDATE
        and not by_label["triangle"].review_flag
    )
    assert by_label["zebra"].decision == BoxDecision.DISCARDED_UNMAPPED_CLASS
    (item,) = result.items
    assert (item.n_raw, item.n_candidates, item.n_discarded, item.n_review_flagged) == (4, 2, 2, 1)
    assert item.review_reasons == ReviewReason.LOW_CONFIDENCE
    assert result.report["boxes"]["by_decision"] == {
        "candidate": 2,
        "discarded_below_min_confidence": 1,
        "discarded_unmapped_class": 1,
    }
    label_lines = (result.run_dir / item.label_path).read_text().splitlines()
    assert len(label_lines) == 2


def test_high_confidence_is_unreviewed_never_verified_and_audit_sampled(proj, scripted) -> None:
    for s in range(12):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.default = [det("square", 0.99, 10, 10, 50, 50)]
    cfg = proj.prepare()
    result = run_annotate(cfg)
    assert {r.review_status for r in result.items} == {ReviewStatus.UNREVIEWED}
    assert {r.provenance for r in result.items} == {Provenance.MACHINE}
    assert {p.provenance for p in result.predictions} == {Provenance.MACHINE}

    all_audit = run_annotate(proj.prepare({"annotation": {"review_sample_rate": 1.0}}))
    assert {r.review_reasons for r in all_audit.items} == {ReviewReason.AUDIT_SAMPLE}
    assert all_audit.review.queued == 12


# --------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------


def test_backend_failure_is_isolated_reported_and_retried(proj, scripted) -> None:
    for s in range(4):
        save(proj.raw / f"s{s}.png", scene(s))
    bad = proj.key("s2.png")
    scripted.responses[bad] = RuntimeError("boom")
    scripted.default = [det("square", 0.9, 1, 1, 9, 9)]
    cfg = proj.prepare({"annotation": {"batch_size": 4}})

    result = run_annotate(cfg)
    assert result.exit_code == 3
    (failed,) = result.failures
    assert failed.item_id == proj.item_id(cfg, "s2.png")
    assert failed.inference_status == InferenceStatus.FAILED
    assert failed.error == "RuntimeError: boom"
    assert failed.review_reasons == ReviewReason.INFERENCE_FAILED
    assert failed.label_path == ""
    assert sum(r.inference_status == InferenceStatus.OK for r in result.items) == 3
    assert result.report["failures"][0]["error"] == "RuntimeError: boom"

    # failures are not cached: the next run retries only the failed image
    scripted.calls.clear()
    del scripted.responses[bad]
    retry = run_annotate(cfg)
    assert scripted.inferred == [bad]
    assert retry.exit_code == 0 and retry.run_id != result.run_id
    assert result.run_dir.is_dir()  # failed run snapshot preserved


def test_whole_batch_failure_falls_back_to_single_items(proj, scripted) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.fail_multi_image_batches = True
    cfg = proj.prepare({"annotation": {"batch_size": 3}})
    result = run_annotate(cfg)
    assert result.exit_code == 0
    assert [len(c) for c in scripted.calls] == [3, 1, 1, 1]


def test_cli_exit_code_3_on_item_failure(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = ValueError("cannot infer")
    cfg = proj.prepare()
    code = main(["annotate", "--config", str(cfg.root / "config.yaml"), "--log-level", "CRITICAL"])
    assert code == 3
    report = json.loads((Workspace(cfg.workspace_dir).reports / "annotation.json").read_text())
    assert report["failures"][0]["error"] == "ValueError: cannot infer"


def test_unreadable_inputs_are_reported(proj, scripted) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    cfg = proj.prepare()
    ws = Workspace(cfg.workspace_dir)
    items = {i.source_path: i for i in read_records(ws.items_manifest, ItemRecord)}
    (ws.root / items["images/s0.png"].path).write_bytes(b"garbage")
    (ws.root / items["images/s1.png"].path).unlink()

    result = run_annotate(cfg)
    by_id = {r.item_id: r for r in result.items}
    corrupted = by_id[items["images/s0.png"].item_id]
    missing = by_id[items["images/s1.png"].item_id]
    assert corrupted.inference_status == InferenceStatus.UNREADABLE_IMAGE
    assert "sha256 mismatch" in corrupted.error
    assert missing.inference_status == InferenceStatus.UNREADABLE_IMAGE
    assert "cannot read file" in missing.error
    assert by_id[items["images/s2.png"].item_id].inference_status == InferenceStatus.OK
    assert len(scripted.inferred) == 1
    assert result.exit_code == 3
    assert {r.review_reasons for r in (corrupted, missing)} == {ReviewReason.UNREADABLE_IMAGE}


# --------------------------------------------------------------------------------------
# Identity, determinism and immutability
# --------------------------------------------------------------------------------------


def test_deterministic_run_and_manifests(proj, scripted) -> None:
    for s in range(4):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.default = lambda img: [
        det("square", 0.9, 1, 1, 20, 20),
        det("circle", 0.3, 5, 5, 9, 9),
    ]
    cfg = proj.prepare()
    first = run_annotate(cfg)
    files = run_files(first.run_dir)

    again = run_annotate(cfg)
    assert again.run_id == first.run_id and not again.created
    assert run_files(again.run_dir) == files

    fresh = run_annotate(proj.prepare({"project": {"workspace": "work_fresh"}}))
    assert fresh.run_id == first.run_id
    assert run_files(fresh.run_dir) == files


def test_rerun_with_same_inputs_does_not_call_backend(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    run_annotate(cfg)
    scripted.calls.clear()
    result = run_annotate(cfg)
    assert scripted.calls == [] and result.cache_hits == 1 and not result.created


def test_new_items_create_new_run_and_leave_old_untouched(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    first = run_annotate(cfg)
    before = run_files(first.run_dir)

    save(proj.raw / "b.png", scene(2))
    cfg = proj.prepare()
    scripted.calls.clear()
    second = run_annotate(cfg)
    assert second.run_id != first.run_id and second.created
    assert scripted.inferred == [proj.key("b.png")]  # only the new image
    assert run_files(first.run_dir) == before
    active = json.loads(Workspace(cfg.workspace_dir).active_annotation_run.read_text())
    assert active["run_id"] == second.run_id and active["previous_run_id"] == first.run_id


def test_policy_change_new_run_without_inference(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [det("square", 0.4, 1, 1, 20, 20)]
    first = run_annotate(proj.prepare())
    scripted.calls.clear()
    second = run_annotate(
        proj.prepare({"annotation": {"confidence_threshold": 0.45, "review_threshold": 0.6}})
    )
    assert scripted.calls == []
    assert second.run_id != first.run_id
    assert first.predictions[0].decision == BoxDecision.CANDIDATE
    assert second.predictions[0].decision == BoxDecision.DISCARDED_BELOW_MIN_CONFIDENCE


def test_model_or_inference_change_reinfers(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    first = run_annotate(cfg)
    scripted.calls.clear()
    scripted.model_sha256 = "1" * 64
    second = run_annotate(cfg)
    assert len(scripted.inferred) == 1 and second.run_id != first.run_id
    scripted.calls.clear()
    third = run_annotate(proj.prepare({"annotation": {"imgsz": 1280}}))
    assert len(scripted.inferred) == 1 and third.run_id not in {first.run_id, second.run_id}


def test_force_reproducing_results_keeps_run(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    cfg = proj.prepare()
    first = run_annotate(cfg)
    scripted.calls.clear()
    forced = run_annotate(cfg, force=True)
    assert len(scripted.inferred) == 1
    assert forced.run_id == first.run_id and not forced.created
    assert forced.forced_changes == []
    assert forced.report["invocation"]["forced_reproduced_exactly"] is True


def test_force_with_changed_results_creates_new_run_and_preserves_old(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    cfg = proj.prepare()
    first = run_annotate(cfg)
    before = run_files(first.run_dir)

    scripted.default = [det("square", 0.8, 1, 1, 20, 20)]  # e.g. nondeterministic hardware
    forced = run_annotate(cfg, force=True)
    assert forced.run_id != first.run_id and forced.created
    assert forced.forced_changes == [proj.item_id(cfg, "a.png")]
    assert run_files(first.run_dir) == before
    # the cache now holds the new result, so a normal run reproduces the forced run
    assert run_annotate(cfg).run_id == forced.run_id


def test_existing_run_dir_is_never_rewritten(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    first = run_annotate(cfg)
    mtimes = {p: p.stat().st_mtime_ns for p in first.run_dir.rglob("*")}
    run_annotate(cfg)
    assert {p: p.stat().st_mtime_ns for p in first.run_dir.rglob("*")} == mtimes

    run_json = first.run_dir / "run.json"
    data = json.loads(run_json.read_text())
    data["identity"]["result_digest"] = "tampered"
    run_json.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_annotate(cfg)


def test_limit_changes_snapshot(proj, scripted) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    cfg = proj.prepare()
    limited = run_annotate(cfg, limit=2)
    full = run_annotate(cfg)
    assert len(limited.items) == 2 and len(full.items) == 3
    assert limited.run_id != full.run_id
    assert [r.item_id for r in limited.items] == [r.item_id for r in full.items][:2]


# --------------------------------------------------------------------------------------
# Cache keyed on decoded pixels
# --------------------------------------------------------------------------------------


def test_cache_is_keyed_by_input_pixels_not_file_bytes(proj, scripted, monkeypatch) -> None:
    img = scene(1)
    save(proj.raw / "a.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    save(proj.raw / "b.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cfg = proj.prepare({"dedup": {"exact_duplicate_action": "keep"}})
    result = run_annotate(cfg)
    assert len(result.items) == 2
    assert len(scripted.inferred) == 1  # identical pixels inferred once
    assert len({r.input_pixel_sha256 for r in result.items}) == 1
    assert len({r.file_sha256 for r in result.items}) == 2

    # Same file bytes, different decoded pixels (e.g. a decoder change) -> cache miss.
    real_decode = annotate_mod.decode_image

    def shifted_decode(data):
        out = real_decode(data)
        return None if out is None else cv2.add(out, 1)

    monkeypatch.setattr(annotate_mod, "decode_image", shifted_decode)
    scripted.calls.clear()
    changed = run_annotate(cfg)
    assert len(scripted.inferred) == 1
    assert changed.run_id != result.run_id


# --------------------------------------------------------------------------------------
# Setup validation
# --------------------------------------------------------------------------------------


def test_model_path_only_required_when_annotation_runs(tmp_path, write_config) -> None:
    raw = tmp_path / "raw" / "images"
    save(raw / "a.png", scene(1))
    path = write_config({"annotation": {"enabled": False, "backend": "yolo", "model": None}})
    assert main(["prepare", "--config", str(path), "--log-level", "CRITICAL"]) == 0

    cfg = load_config(path)
    with pytest.raises(AnnotationSetupError, match="annotation.model is required"):
        run_annotate(cfg)
    assert main(["annotate", "--config", str(path), "--log-level", "CRITICAL"]) == 1

    missing = write_config({"annotation": {"enabled": True, "backend": "yolo", "model": "nope.pt"}})
    with pytest.raises(AnnotationSetupError, match="never downloaded"):
        run_annotate(load_config(missing))
    assert main(["prepare", "--config", str(missing), "--log-level", "CRITICAL"]) == 1

    (tmp_path / "weights.onnx").write_bytes(b"x")
    onnx = write_config(
        {"annotation": {"enabled": True, "backend": "yolo", "model": "weights.onnx"}}
    )
    with pytest.raises(AnnotationSetupError, match="Unsupported model format"):
        run_annotate(load_config(onnx))


def test_unknown_backend(proj) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare({"annotation": {"backend": "grounding_dino"}})
    with pytest.raises(AnnotationSetupError, match="Unknown annotation backend"):
        run_annotate(cfg)


def test_class_mapping_must_cover_a_project_class(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.class_names = ("person", "car")
    with pytest.raises(AnnotationSetupError, match="class_map"):
        run_annotate(proj.prepare())
    scripted.default = [det("car", 0.9, 1, 1, 9, 9), det("person", 0.9, 1, 1, 9, 9)]
    result = run_annotate(proj.prepare({"annotation": {"class_map": {"car": "square"}}}))
    by_label = {p.backend_label: p for p in result.predictions}
    assert by_label["car"].class_name == "square"
    assert by_label["person"].decision == BoxDecision.DISCARDED_UNMAPPED_CLASS


def test_requires_current_dedup(proj) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.config()
    run_ingest(cfg)
    with pytest.raises(DatasetStateError, match="run the dedup stage first"):
        run_annotate(cfg)
    run_dedup(cfg)
    save(proj.raw / "b.png", scene(2))
    run_ingest(cfg)
    with pytest.raises(DatasetStateError, match="re-run the dedup stage"):
        run_annotate(cfg)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"inference_confidence_floor": 0.3}, "inference_confidence_floor <= confidence_threshold"),
        ({"confidence_threshold": 0.6}, "confidence_threshold <= review_threshold"),
        ({"batch_size": 0}, "batch_size"),
        ({"max_detections": 0}, "max_detections"),
        ({"review_sample_rate": 1.5}, "review_sample_rate"),
    ],
)
def test_annotation_config_validation(write_config, override, message) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write_config({"annotation": override}))


def test_review_config_validation(write_config) -> None:
    with pytest.raises(ConfigError, match="must be one of"):
        load_config(write_config({"review": {"export": "everything"}}))


def test_raw_data_untouched(proj, scripted, tmp_path) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.default = [det("square", 0.3, 1, 1, 9, 9)]
    cfg = proj.prepare({"review": {"export": "all"}})
    before = snapshot_tree(tmp_path / "raw")
    run_annotate(cfg)
    run_annotate(cfg, force=True)
    assert snapshot_tree(tmp_path / "raw") == before


# --------------------------------------------------------------------------------------
# Split-group snapshot
# --------------------------------------------------------------------------------------


def test_split_group_is_snapshotted_and_changes_create_new_run(proj, scripted) -> None:
    base = scene(7)
    save(proj.raw / "a_base.jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    save(proj.raw / "b_brighter.jpg", cv2.convertScaleAbs(base, beta=12))  # near-duplicate
    save(proj.raw / "c_other.png", scene(8))
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]

    grouped = proj.prepare({"dedup": {"group_near_duplicates": True}})
    first = run_annotate(grouped)
    a, b, c = (proj.item_id(grouped, n) for n in ("a_base.jpg", "b_brighter.jpg", "c_other.png"))
    groups_1 = {r.item_id: r.split_group for r in first.items}
    assert groups_1[a] == groups_1[b] != groups_1[c]
    assert groups_1[a].startswith("sg_")
    on_disk = read_records(first.run_dir / "items.csv", AnnotationItemRecord)
    assert {r.item_id: r.split_group for r in on_disk} == groups_1
    before = run_files(first.run_dir)

    # Re-running dedup with different grouping changes split_group for a and b only.
    scripted.calls.clear()
    ungrouped = proj.prepare({"dedup": {"group_near_duplicates": False}})
    second = run_annotate(ungrouped)
    groups_2 = {r.item_id: r.split_group for r in second.items}
    assert groups_2[a] != groups_2[b]
    assert groups_2[c] == groups_1[c]

    assert second.run_id != first.run_id and second.created
    assert scripted.calls == []  # same pixels + same model -> cached inference reused
    assert run_files(first.run_dir) == before  # historical grouping preserved
    assert {
        r.item_id: r.split_group
        for r in read_records(first.run_dir / "items.csv", AnnotationItemRecord)
    } == groups_1


# --------------------------------------------------------------------------------------
# Model reference portability
# --------------------------------------------------------------------------------------


def test_model_reference_is_machine_independent(tmp_path) -> None:
    from cv_pipeline.annotation.backend import model_reference

    root = tmp_path / "project"
    (root / "models").mkdir(parents=True)
    assert model_reference("models/m.pt", root) == "models/m.pt"
    assert model_reference(str(root / "models" / "m.pt"), root) == "models/m.pt"
    assert model_reference("models/../models/m.pt", root) == "models/m.pt"
    assert model_reference(str(tmp_path / "home1" / "m.pt"), root) == "external:m.pt"
    assert model_reference(str(tmp_path / "home2" / "m.pt"), root) == "external:m.pt"
    assert model_reference("~/anything/m.pt", root) == "external:m.pt"


def _portable_project(base: Path, model_abs: bool, external_model: Path | None = None):
    import yaml

    from .conftest import base_config_dict

    raw = base / "raw" / "images"
    save(raw / "a.png", scene(1))
    save(raw / "b.png", scene(2))
    if external_model is None:
        model = base / "models" / "m.pt"
        model_value = str(model) if model_abs else "models/m.pt"
    else:
        model = external_model
        model_value = str(model)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"weights")
    data = deep_update(base_config_dict(), deep_update(ANN, {"annotation": {"model": model_value}}))
    path = base / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(path)
    run_ingest(cfg)
    run_dedup(cfg)
    return cfg


def test_moving_the_project_does_not_change_run_identity(tmp_path, scripted) -> None:
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    here = run_annotate(_portable_project(tmp_path / "machine_a" / "repo", model_abs=True))
    there = run_annotate(
        _portable_project(tmp_path / "machine_b" / "elsewhere" / "repo", model_abs=True)
    )
    relative = run_annotate(_portable_project(tmp_path / "machine_c" / "repo", model_abs=False))
    assert here.run_id == there.run_id == relative.run_id
    assert {r.model_id for r in here.items} == {"models/m.pt"}
    assert run_files(here.run_dir) == run_files(there.run_dir) == run_files(relative.run_dir)


def test_external_model_location_does_not_change_run_identity(tmp_path, scripted) -> None:
    scripted.default = [det("square", 0.9, 1, 1, 20, 20)]
    one = run_annotate(
        _portable_project(
            tmp_path / "a" / "repo", True, tmp_path / "home_alice" / ".cache" / "m.pt"
        )
    )
    two = run_annotate(
        _portable_project(tmp_path / "b" / "repo", True, tmp_path / "home_bob" / "models" / "m.pt")
    )
    assert one.run_id == two.run_id
    assert {r.model_id for r in one.items} == {"external:m.pt"}
