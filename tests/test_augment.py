"""Phase 5: train-only, label-aware, deterministic augmentation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import pytest

from cv_pipeline.augmentation import (
    AugmentationError,
    AugmentedItemRecord,
    SkipCategory,
    run_augment,
)
from cv_pipeline.augmentation.transforms import TRANSFORMS
from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.dedup import run_dedup
from cv_pipeline.imports.yolo import parse_label
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import ItemRecord, SourceRefRecord, read_records
from cv_pipeline.splitting import run_split
from cv_pipeline.splitting.history import SplitItemRecord
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes import det
from .test_import_lineage import add_family
from .test_imports import IMPORT, Env
from .yolo_export import export_image, make_export

AUG = {"augmentation": {"variants_per_item": 2}}
ALL_TRAIN = {"split": {"train": 1.0, "val": 0.0, "test": 0.0}}


@pytest.fixture
def env(tmp_path, write_config) -> Env:
    return Env(tmp_path, write_config)


def ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def prepared(env: Env, overrides: dict[str, Any] | None = None, imports=None, annotate=False):
    cfg = env.build(deep_update(AUG, overrides or {}), imports=imports, annotate=annotate)
    assert run_split(cfg).exit_code == 0
    return cfg


def split_of(cfg) -> dict[str, str]:
    return {r.item_id: r.split for r in read_records(ws(cfg).split_items_manifest, SplitItemRecord)}


def run_files(run_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(run_dir).as_posix(): p.read_bytes()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


def decide(cfg, rows: list[dict[str, str]]) -> None:
    path = ws(cfg).review / "decisions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "based_on_run_id", "decision"])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------------------
# Train only
# --------------------------------------------------------------------------------------


def test_only_train_items_are_augmented(env) -> None:
    make_export(env.export, splits={"train": list(range(20)), "val": [20, 21], "test": [22, 23]})
    cfg = prepared(env)
    splits = split_of(cfg)
    result = run_augment(cfg)
    assert result.created and result.items
    assert {splits[i.parent_item_id] for i in result.items} == {"train"}
    assert {i.split for i in result.items} == {"train"}
    held_out = {i for i, s in splits.items() if s != "train"}
    assert held_out  # val/test exist and all have trusted labels...
    assert not held_out & {i.parent_item_id for i in result.items}  # ...but are never used
    assert not held_out & {s.item_id for s in result.skipped}  # not even considered
    a = result.report["split_assertion"]
    assert a["only_train"] and a["augmented_parents_by_split"]["val"] == 0
    assert a["augmented_parents_by_split"]["test"] == 0
    train = sum(1 for s in splits.values() if s == "train")
    assert result.report["train_items"]["total"] == train


def test_augmented_items_get_no_split_identity(env) -> None:
    make_export(env.export, splits={"train": list(range(8))})
    cfg = prepared(env, ALL_TRAIN)
    before = {p: p.read_bytes() for p in ws(cfg).splits.rglob("*") if p.is_file()}
    items_before = ws(cfg).items_manifest.read_bytes()
    result = run_augment(cfg)
    aug_ids = {i.aug_item_id for i in result.items}
    assert {p: p.read_bytes() for p in ws(cfg).splits.rglob("*") if p.is_file()} == before
    assert ws(cfg).items_manifest.read_bytes() == items_before
    assert not aug_ids & set(split_of(cfg))


# --------------------------------------------------------------------------------------
# Label trust
# --------------------------------------------------------------------------------------


def test_trusted_imported_labels_are_used(env) -> None:
    make_export(env.export, splits={"train": list(range(6))})
    cfg = prepared(env, ALL_TRAIN)
    result = run_augment(cfg)
    assert {i.label_provenance for i in result.items} == {"human_imported"}
    assert result.report["parents"]["by_provenance"] == {"human_imported": 6}
    assert len(result.items) == 12


def _raw_project(env, scripted, responses, policy=None, n=4):
    for s in range(n):
        cv2.imwrite(str(env.raw / f"field_{s}.png"), export_image(300 + s)[0])
    scripted.default = responses
    over = deep_update(
        ALL_TRAIN,
        {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}},
    )
    if policy:
        over = deep_update(over, {"augmentation": policy})
    cfg = prepared(env, over, imports=[], annotate=True)
    run_id = json.loads(ws(cfg).active_annotation_run.read_text())["run_id"]
    ids = sorted(i.item_id for i in read_records(ws(cfg).items_manifest, ItemRecord))
    return cfg, ids, run_id


BOX = [det("square", 0.9, 40, 40, 120, 120)]


def test_reviewed_accepted_and_corrected_labels_are_used(env, scripted) -> None:
    cfg, ids, run_id = _raw_project(env, scripted, BOX)
    label = ws(cfg).review / "labels" / f"{ids[1]}.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("2 0.500000 0.500000 0.300000 0.300000\n")
    decide(
        cfg,
        [
            {"item_id": ids[0], "based_on_run_id": run_id, "decision": "accepted"},
            {"item_id": ids[1], "based_on_run_id": run_id, "decision": "corrected"},
            {"item_id": ids[2], "based_on_run_id": run_id, "decision": "rejected"},
        ],
    )
    result = run_augment(cfg)
    prov = {i.parent_item_id: i.label_provenance for i in result.items}
    assert prov == {ids[0]: "human_reviewed_accepted", ids[1]: "human_reviewed_corrected"}
    corrected = next(i for i in result.items if i.parent_item_id == ids[1])
    first = (
        ws(cfg).root / "augmentation" / "runs" / result.run_id / corrected.label_path
    ).read_text()
    assert first.startswith("2 ")  # the human-corrected class, not the machine one
    skipped = {s.item_id: (s.category, s.reason) for s in result.skipped}
    assert skipped[ids[2]] == (SkipCategory.LABEL_UNTRUSTED, "rejected_by_reviewer")
    assert skipped[ids[3]] == (SkipCategory.LABEL_UNTRUSTED, "machine_label_not_reviewed")


def test_machine_labels_only_with_explicit_policy(env, scripted) -> None:
    cfg, ids, _ = _raw_project(env, scripted, BOX)
    default = run_augment(cfg)
    assert default.items == [] and len(default.skipped) == 4
    assert default.report["train_items"]["skipped_by_category"] == {"label_untrusted": 4}

    cfg = env.config(
        deep_update(
            AUG,
            deep_update(
                ALL_TRAIN,
                {
                    "annotation": {
                        "enabled": True,
                        "backend": "scripted",
                        "review_sample_rate": 0.0,
                    },
                    "augmentation": {"label_policy": "allow_machine_unreviewed"},
                },
            ),
        ),
        imports=[],
    )
    opted = run_augment(cfg)
    assert {i.label_provenance for i in opted.items} == {"machine_unreviewed"}
    assert opted.report["parents"]["machine_unreviewed_used"] == 4
    assert any("UNREVIEWED" in w for w in opted.report["warnings"])
    assert opted.run_id != default.run_id


def test_machine_empty_is_not_a_negative_unless_explicit(env, scripted) -> None:
    policy = {"label_policy": "allow_machine_unreviewed"}
    cfg, ids, _ = _raw_project(env, scripted, [], policy)
    result = run_augment(cfg)
    assert result.items == []
    assert {s.reason for s in result.skipped} == {"machine_zero_detections_not_a_negative"}

    cfg = env.config(
        deep_update(
            AUG,
            deep_update(
                ALL_TRAIN,
                {
                    "annotation": {
                        "enabled": True,
                        "backend": "scripted",
                        "review_sample_rate": 0.0,
                    },
                    "augmentation": {**policy, "allow_machine_empty": True},
                },
            ),
        ),
        imports=[],
    )
    allowed = run_augment(cfg)
    assert allowed.items and all(i.parent_negative and i.boxes_after == 0 for i in allowed.items)


def test_annotation_failure_is_its_own_skip_category(env, scripted) -> None:
    cv2.imwrite(str(env.raw / "field_0.png"), export_image(300)[0])
    scripted.default = RuntimeError("boom")
    over = deep_update(ALL_TRAIN, {"annotation": {"enabled": True, "backend": "scripted"}})
    cfg = prepared(env, over, imports=[], annotate=True)
    result = run_augment(cfg)
    assert [s.category for s in result.skipped] == [SkipCategory.ANNOTATION_FAILED]


def test_human_confirmed_negative_is_augmented_box_free(env) -> None:
    make_export(env.export, splits={"train": [0, 1, 2]}, empty={1})
    cfg = prepared(env, ALL_TRAIN, imports=[{**IMPORT, "empty_labels_are_negatives": True}])
    result = run_augment(cfg)
    negatives = [i for i in result.items if i.parent_negative]
    assert len(negatives) == 2 and all(i.boxes_after == 0 for i in negatives)
    run_dir = ws(cfg).root / "augmentation" / "runs" / result.run_id
    assert all((run_dir / i.label_path).read_text() == "" for i in negatives)
    assert result.report["parents"]["trusted_negatives"] == 1


def test_ignored_only_and_unconfirmed_empty_are_skipped(env) -> None:
    make_export(
        env.export,
        splits={"train": [0, 1, 2]},
        empty={2},
        names=("square", "circle", "triangle", "person"),
        label_override={1: "3 0.3 0.3 0.1 0.1\n"},
    )
    cfg = prepared(env, ALL_TRAIN, imports=[{**IMPORT, "ignore_classes": ["person"]}])
    result = run_augment(cfg)
    reasons = sorted(s.reason for s in result.skipped)
    assert reasons == ["empty_label_not_confirmed_negative", "ignored_only_unconfirmed_negative"]
    assert {s.category for s in result.skipped} == {SkipCategory.LABEL_UNTRUSTED}
    assert len({i.parent_item_id for i in result.items}) == 1


# --------------------------------------------------------------------------------------
# Refusals (nothing written)
# --------------------------------------------------------------------------------------


def test_stale_split_refuses(env) -> None:
    make_export(env.export, splits={"train": [0, 1, 2]})
    cfg = prepared(env, ALL_TRAIN)
    cv2.imwrite(str(env.raw / "late.png"), export_image(99)[0])
    run_ingest(cfg)
    run_dedup(cfg)  # dataset changed, split not re-run
    with pytest.raises(AugmentationError, match="stale split.*Re-run `split`"):
        run_augment(cfg)
    assert not (ws(cfg).root / "augmentation").exists()


def test_missing_split_refuses(env) -> None:
    make_export(env.export, splits={"train": [0, 1]})
    cfg = env.build(AUG)
    with pytest.raises(AugmentationError, match="run `split`"):
        run_augment(cfg)


def test_changed_workspace_file_refuses(env) -> None:
    make_export(env.export, splits={"train": [0, 1, 2]})
    cfg = prepared(env, ALL_TRAIN)
    item = read_records(ws(cfg).items_manifest, ItemRecord)[0]
    (ws(cfg).root / item.path).write_bytes(b"changed")
    with pytest.raises(DatasetStateError, match="differ from items.csv"):
        run_augment(cfg)
    assert not (ws(cfg).root / "augmentation").exists()


def test_machine_policy_with_stale_annotation_refuses(env, scripted) -> None:
    cfg, ids, _ = _raw_project(env, scripted, BOX, {"label_policy": "allow_machine_unreviewed"})
    cv2.imwrite(str(env.raw / "late.png"), export_image(98)[0])
    run_ingest(cfg)
    run_dedup(cfg)
    run_split(cfg)  # split current, annotation now stale
    with pytest.raises(AugmentationError, match="re-run\\s+`annotate`|Re-run `annotate`"):
        run_augment(cfg)


# --------------------------------------------------------------------------------------
# Boxes
# --------------------------------------------------------------------------------------


def _all_labels_valid(cfg, result) -> None:
    run_dir = ws(cfg).root / "augmentation" / "runs" / result.run_id
    for i in result.items:
        boxes, err = parse_label((run_dir / i.label_path).read_text(), len(cfg.classes))
        assert err is None, (i.aug_item_id, err)
        assert len(boxes) == i.boxes_after
        img = cv2.imread(str(run_dir / i.image_path))
        assert img.shape[:2] == (i.height, i.width)
        assert sha256_file(run_dir / i.image_path) == i.output_sha256


def test_transformed_boxes_remain_valid(env) -> None:
    make_export(env.export, splits={"train": list(range(10))})
    cfg = prepared(env, ALL_TRAIN)
    result = run_augment(cfg)
    _all_labels_valid(cfg, result)
    assert all(i.boxes_after == i.boxes_before for i in result.items)  # no partial loss


def test_clipped_boxes_handled_and_partial_loss_policy(env) -> None:
    make_export(env.export, splits={"train": list(range(12))})
    over = deep_update(
        ALL_TRAIN,
        {
            "augmentation": {
                "variants_per_item": 3,
                "min_visibility": 0.3,
                "allow_partial_box_loss": True,
                "transforms": {"affine": {"p": 1.0, "translate_percent": [-0.3, 0.3]}},
            }
        },
    )
    cfg = prepared(env, over)
    result = run_augment(cfg)
    _all_labels_valid(cfg, result)
    assert any(i.boxes_after < i.boxes_before for i in result.items)  # clipped away
    assert all(i.boxes_after > 0 for i in result.items)  # never all of a positive's boxes


def test_positive_losing_all_boxes_is_rejected_negatives_kept(env) -> None:
    make_export(env.export, splits={"train": [0, 1, 2]}, empty={2})
    over = deep_update(ALL_TRAIN, {"augmentation": {"min_box_area": 10_000_000, "max_attempts": 3}})
    cfg = prepared(env, over, imports=[{**IMPORT, "empty_labels_are_negatives": True}])
    result = run_augment(cfg)
    assert len(result.rejected) == 4  # 2 positives x 2 variants, after 3 attempts each
    assert {r.attempts for r in result.rejected} == {3}
    assert all("lost" in r.reason for r in result.rejected)
    assert [i.parent_negative for i in result.items] == [True, True]  # negatives still fine
    assert result.report["variants"]["rejected"] == 4


def test_deterministic_retry(env) -> None:
    make_export(env.export, splits={"train": list(range(20))})
    over = deep_update(
        ALL_TRAIN,
        {
            "augmentation": {
                "variants_per_item": 3,
                "min_visibility": 0.95,
                "transforms": {"affine": {"p": 1.0, "translate_percent": [-0.2, 0.2]}},
            }
        },
    )
    cfg = prepared(env, over)
    first = run_augment(cfg)
    assert any(i.attempt > 0 for i in first.items)  # some variants needed a retry
    cfg2 = prepared(env, deep_update(over, {"project": {"workspace": "work_retry"}}))
    second = run_augment(cfg2)
    assert [(i.aug_item_id, i.attempt, i.seed) for i in first.items] == [
        (i.aug_item_id, i.attempt, i.seed) for i in second.items
    ]


def test_every_registered_transform_builds_and_runs(env) -> None:
    make_export(env.export, splits={"train": [0, 1, 2]})
    over = deep_update(
        ALL_TRAIN,
        {
            "augmentation": {
                "allow_partial_box_loss": True,
                "min_visibility": 0.1,
                "transforms": {name: {"enabled": True, "p": 1.0} for name in TRANSFORMS},
            }
        },
    )
    cfg = prepared(env, over)
    result = run_augment(cfg)
    _all_labels_valid(cfg, result)
    assert set(result.report["transforms_enabled"]) == set(TRANSFORMS)


# --------------------------------------------------------------------------------------
# Determinism, identity, immutability, lineage
# --------------------------------------------------------------------------------------


def test_deterministic_reruns_and_fresh_workspace(env) -> None:
    make_export(env.export, splits={"train": list(range(8))})
    cfg = prepared(env, ALL_TRAIN)
    first = run_augment(cfg)
    files = run_files(first.run_dir)
    again = run_augment(cfg)
    assert again.run_id == first.run_id and not again.created
    assert run_files(first.run_dir) == files
    cfg2 = prepared(env, deep_update(ALL_TRAIN, {"project": {"workspace": "work2"}}))
    fresh = run_augment(cfg2)
    assert fresh.run_id == first.run_id and run_files(fresh.run_dir) == files


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1234},
        {"variants_per_item": 3},
        {"transforms": {"horizontal_flip": {"enabled": True}}},
        {"min_visibility": 0.6},
        {"max_attempts": 2},
    ],
)
def test_config_changes_create_new_run_and_keep_old(env, change) -> None:
    make_export(env.export, splits={"train": list(range(6))})
    cfg = prepared(env, ALL_TRAIN)
    first = run_augment(cfg)
    before = run_files(first.run_dir)
    second = run_augment(
        env.config(deep_update(AUG, deep_update(ALL_TRAIN, {"augmentation": change})))
    )
    assert second.run_id != first.run_id and second.created
    assert run_files(first.run_dir) == before


def test_changed_resolved_labels_create_new_run(env, scripted) -> None:
    cfg, ids, run_id = _raw_project(env, scripted, BOX)
    decide(cfg, [{"item_id": ids[0], "based_on_run_id": run_id, "decision": "accepted"}])
    first = run_augment(cfg)
    before = run_files(first.run_dir)
    decide(
        cfg,
        [
            {"item_id": ids[0], "based_on_run_id": run_id, "decision": "accepted"},
            {"item_id": ids[1], "based_on_run_id": run_id, "decision": "accepted"},
        ],
    )
    second = run_augment(cfg)
    assert second.run_id != first.run_id
    assert {i.parent_item_id for i in second.items} == {ids[0], ids[1]}
    assert run_files(first.run_dir) == before


def test_new_split_revision_creates_new_run(env) -> None:
    make_export(env.export, splits={"train": list(range(4))})
    cfg = prepared(env, ALL_TRAIN)
    first = run_augment(cfg)
    cv2.imwrite(str(env.raw / "late.png"), export_image(97)[0])
    run_ingest(cfg)
    run_dedup(cfg)
    run_split(cfg)
    second = run_augment(cfg)
    assert second.run_id != first.run_id
    assert first.run_dir.is_dir()


def test_tampered_run_dir_is_not_overwritten(env) -> None:
    make_export(env.export, splits={"train": [0, 1]})
    cfg = prepared(env, ALL_TRAIN)
    first = run_augment(cfg)
    run_json = first.run_dir / "run.json"
    data = json.loads(run_json.read_text())
    data["identity"]["labels_digest"] = "tampered"
    run_json.write_text(json.dumps(data))
    with pytest.raises(AugmentationError, match="refusing to overwrite"):
        run_augment(cfg)


def test_lineage_is_preserved(env) -> None:
    make_export(env.export, splits={"train": list(range(6))})
    cfg = prepared(env, ALL_TRAIN)
    result = run_augment(cfg)
    items = {i.item_id: i for i in read_records(ws(cfg).items_manifest, ItemRecord)}
    splits = {r.item_id: r for r in read_records(ws(cfg).split_items_manifest, SplitItemRecord)}
    on_disk = read_records(result.run_dir / "items.csv", AugmentedItemRecord)
    assert on_disk == result.items
    for i in result.items:
        parent = items[i.parent_item_id]
        assert i.source_id == parent.source_id and i.origin == parent.origin
        assert i.input_sha256 == parent.sha256
        assert i.split_group == splits[i.parent_item_id].split_group
        assert i.split == "train" and i.run_id == result.run_id
        steps = json.loads(i.recipe)
        assert all({"transform", "params"} <= set(s) for s in steps)


@pytest.mark.parametrize(
    "fractions, expected",
    [
        ({"train": 0.8, "val": 0.1, "test": 0.1}, "train"),
        ({"train": 0.2, "val": 0.4, "test": 0.4}, "val"),  # family allocated to a hold-out
    ],
)
def test_roboflow_family_stays_train_only(env, fractions, expected) -> None:
    make_export(env.export, splits={"train": list(range(12))})
    add_family(env.export, "valve_001", 500)
    cfg = prepared(env, {"split": fractions})
    splits = split_of(cfg)
    family = [
        r.source_id
        for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
        if r.family_key
    ]
    assert len(family) == 3
    (family_split,) = {splits[i] for i in family}  # one split for the whole family
    assert family_split == expected
    result = run_augment(cfg)
    augmented = {i.parent_item_id for i in result.items if i.family_key}
    if family_split == "train":
        assert augmented == set(family)
        assert result.report["parents"]["roboflow_family_members"] == 3
    else:
        assert augmented == set()  # a held-out family is never augmented
    assert not {i.aug_item_id for i in result.items} & set(splits)


def test_sources_untouched(env) -> None:
    make_export(env.export, splits={"train": list(range(6))})
    cv2.imwrite(str(env.raw / "field.png"), export_image(77)[0])
    cfg = prepared(env, ALL_TRAIN)
    snaps = [
        snapshot_tree(env.export),
        snapshot_tree(env.tmp / "raw"),
        snapshot_tree(ws(cfg).root / "interim"),
        snapshot_tree(ws(cfg).imports),
    ]
    run_augment(cfg)
    assert [
        snapshot_tree(env.export),
        snapshot_tree(env.tmp / "raw"),
        snapshot_tree(ws(cfg).root / "interim"),
        snapshot_tree(ws(cfg).imports),
    ] == snaps


# --------------------------------------------------------------------------------------
# Config and CLI
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aug, message",
    [
        ({"transforms": {"mosaic": {}}}, "unknown transform"),
        ({"transforms": {"affine": {"shear": 5}}}, "unknown key"),
        ({"transforms": {"affine": {"p": 1.5}}}, "must be in \\[0, 1\\]"),
        ({"allow_machine_empty": True}, "requires label_policy"),
        ({"variants_per_item": 0}, "variants_per_item"),
        ({"label_policy": "anything"}, "must be one of"),
    ],
)
def test_config_validation(env, aug, message) -> None:
    with pytest.raises(ConfigError, match=message):
        env.config({"augmentation": aug})


def test_cli_dry_run_and_prepare_opt_in(env) -> None:
    make_export(env.export, splits={"train": list(range(4))})
    cfg = env.config(ALL_TRAIN)
    path = cfg.root / "config.yaml"
    assert (
        main(
            ["import-dataset", "--config", str(path), "--name", "legacy", "--log-level", "CRITICAL"]
        )
        == 0
    )
    assert main(["prepare", "--config", str(path), "--log-level", "CRITICAL"]) == 0
    assert not (ws(cfg).root / "augmentation").exists()  # disabled by default
    assert main(["augment", "--config", str(path), "--dry-run", "--log-level", "CRITICAL"]) == 0
    assert not (ws(cfg).root / "augmentation").exists()
    assert main(["augment", "--config", str(path), "--log-level", "CRITICAL"]) == 0
    assert (ws(cfg).reports / "augmentation.md").is_file()

    enabled = env.config(deep_update(ALL_TRAIN, {"augmentation": {"enabled": True, "seed": 5}}))
    assert (
        main(["prepare", "--config", str(enabled.root / "config.yaml"), "--log-level", "CRITICAL"])
        == 0
    )
    runs = list((ws(cfg).root / "augmentation" / "runs").iterdir())
    assert len(runs) == 2
