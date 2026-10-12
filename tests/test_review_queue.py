from __future__ import annotations

import cv2
import pytest

from cv_pipeline.annotation import run_annotate
from cv_pipeline.annotation.review_queue import LabelState, ReviewManifestRecord
from cv_pipeline.manifest import ItemRecord, read_records
from cv_pipeline.utils.io import read_image
from cv_pipeline.workspace import Workspace

from .fakes import det
from .test_annotate import Project, save, scene


@pytest.fixture
def proj(tmp_path, write_config, scripted) -> Project:
    return Project(tmp_path, write_config)


def manifest(cfg) -> dict[str, ReviewManifestRecord]:
    rows = read_records(
        Workspace(cfg.workspace_dir).review / "review_manifest.csv", ReviewManifestRecord
    )
    return {r.item_id: r for r in rows}


def test_queue_contains_flagged_items_with_tool_friendly_layout(proj, scripted) -> None:
    save(proj.raw / "confident.png", scene(1))
    save(proj.raw / "unsure.png", scene(2))
    scripted.responses[proj.key("confident.png")] = [det("square", 0.95, 10, 10, 100, 100)]
    scripted.responses[proj.key("unsure.png")] = [
        det("circle", 0.3, 20, 20, 60, 60),
        det("square", 0.9, 100, 100, 200, 200),
    ]
    cfg = proj.prepare()
    result = run_annotate(cfg)
    ws = Workspace(cfg.workspace_dir)
    unsure = proj.item_id(cfg, "unsure.png")

    rows = manifest(cfg)
    assert list(rows) == [unsure]
    row = rows[unsure]
    assert row.based_on_run_id == result.run_id
    assert row.source_path == "images/unsure.png"
    assert row.predicted_classes == "circle;square"
    assert (row.min_confidence, row.max_confidence, row.n_candidates) == (0.3, 0.9, 2)
    assert row.review_reasons == "low_confidence"
    assert row.label_state == LabelState.MACHINE
    assert row.filename == f"{unsure}.png"

    assert (ws.root / row.image_path).read_bytes() == (proj.raw / "unsure.png").read_bytes()
    assert len((ws.root / row.label_path).read_text().splitlines()) == 2
    assert read_image(ws.root / row.preview_path) is not None
    assert (ws.review / "classes.txt").read_text() == "square\ncircle\ntriangle\n"


def test_export_all(proj, scripted) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.default = [det("square", 0.95, 10, 10, 100, 100)]
    cfg = proj.prepare({"review": {"export": "all", "copy_images": False, "previews": False}})
    run_annotate(cfg)
    rows = manifest(cfg)
    assert len(rows) == 3
    assert {r.review_status for r in rows.values()} == {"unreviewed"}
    ws = Workspace(cfg.workspace_dir)
    assert not any((ws.review / "images").iterdir())
    items = {i.item_id: i for i in read_records(ws.items_manifest, ItemRecord)}
    assert all(r.image_path == items[r.item_id].path for r in rows.values())


def test_human_edits_are_never_overwritten(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    save(proj.raw / "b.png", scene(2))
    scripted.default = [det("square", 0.3, 10, 10, 100, 100)]
    cfg = proj.prepare()
    run_annotate(cfg)
    ws = Workspace(cfg.workspace_dir)
    a, b = proj.item_id(cfg, "a.png"), proj.item_id(cfg, "b.png")
    edited = "1 0.500000 0.500000 0.200000 0.200000\n"
    (ws.review / "labels" / f"{a}.txt").write_text(edited)

    # Policy change -> new machine labels (box now has a different review flag but same
    # geometry); a model change produces different geometry.
    scripted.model_sha256 = "2" * 64
    scripted.default = [det("triangle", 0.35, 0, 0, 32, 24)]
    second = run_annotate(cfg)

    assert (ws.review / "labels" / f"{a}.txt").read_text() == edited
    assert (
        ws.review / "labels" / f"{b}.txt"
    ).read_text() == "2 0.050000 0.050000 0.100000 0.100000\n"
    rows = manifest(cfg)
    assert rows[a].label_state == LabelState.EDITED
    assert rows[b].label_state == LabelState.MACHINE
    assert rows[b].based_on_run_id == second.run_id
    assert second.review.label_states == {"edited": 1, "machine": 1}


def test_preexisting_label_files_are_treated_as_human(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    cfg = proj.prepare()
    ws = Workspace(cfg.workspace_dir)
    a = proj.item_id(cfg, "a.png")
    (ws.review / "labels").mkdir(parents=True)
    (ws.review / "labels" / f"{a}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    run_annotate(cfg)
    assert (ws.review / "labels" / f"{a}.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"
    assert manifest(cfg)[a].label_state == LabelState.EDITED


def test_items_leaving_the_queue_keep_their_files(proj, scripted) -> None:
    save(proj.raw / "a.png", scene(1))
    scripted.default = [det("square", 0.3, 10, 10, 100, 100)]
    cfg = proj.prepare()
    run_annotate(cfg)
    ws = Workspace(cfg.workspace_dir)
    a = proj.item_id(cfg, "a.png")
    label = ws.review / "labels" / f"{a}.txt"
    assert label.is_file()

    unflagged = run_annotate(proj.prepare({"annotation": {"review_threshold": 0.3}}))
    assert manifest(cfg) == {}
    assert label.is_file() and (ws.review / "images" / f"{a}.png").is_file()
    assert unflagged.review.stale_labels == [a]
    assert unflagged.report["review"]["labels_not_in_queue"] == 1


def test_failed_and_unreadable_items_in_queue(proj, scripted) -> None:
    save(proj.raw / "fail.png", scene(1))
    save(proj.raw / "gone.png", scene(2))
    scripted.responses[proj.key("fail.png")] = RuntimeError("boom")
    cfg = proj.prepare()
    ws = Workspace(cfg.workspace_dir)
    gone = proj.item_id(cfg, "gone.png")
    items = {i.item_id: i for i in read_records(ws.items_manifest, ItemRecord)}
    (ws.root / items[gone].path).unlink()

    run_annotate(cfg)
    rows = manifest(cfg)
    fail = rows[proj.item_id(cfg, "fail.png")]
    assert fail.inference_status == "failed" and fail.review_reasons == "inference_failed"
    assert (ws.root / fail.label_path).read_text() == ""  # blank starting point
    assert (ws.root / fail.image_path).is_file()
    assert rows[gone].label_state == LabelState.NONE
    assert rows[gone].image_path == "" and rows[gone].label_path == ""


def test_queue_export_is_deterministic(proj, scripted) -> None:
    for s in range(3):
        save(proj.raw / f"s{s}.png", scene(s))
    scripted.default = [det("circle", 0.3, 10, 10, 100, 100)]
    cfg = proj.prepare()
    run_annotate(cfg)
    review = Workspace(cfg.workspace_dir).review
    first = {p: p.read_bytes() for p in sorted(review.rglob("*")) if p.is_file()}
    run_annotate(cfg)
    assert {p: p.read_bytes() for p in sorted(review.rglob("*")) if p.is_file()} == first
    assert cv2.imread(str(next((review / "previews").iterdir()))) is not None
