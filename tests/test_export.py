"""Phase 6: validated, immutable YOLO export and the deliberate review queue."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import pytest
import yaml

from cv_pipeline.augmentation import run_augment
from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.dedup import run_dedup
from cv_pipeline.export import (
    EvaluationCoverageError,
    ExcludedRecord,
    ExportError,
    ExportLeakageError,
    ExportRecord,
    run_export,
)
from cv_pipeline.export.export import _assert_no_leakage, _Row, _validate_label
from cv_pipeline.ingest import run_ingest
from cv_pipeline.labels import LabelBox, LabelPolicy, LabelResolver, run_labels
from cv_pipeline.labels.review_queue import run_review_queue
from cv_pipeline.labels.usage import resolution_digest
from cv_pipeline.manifest import ItemRecord, SourceRefRecord, read_records
from cv_pipeline.splitting import run_split
from cv_pipeline.splitting.history import SplitItemRecord
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes import det
from .test_import_lineage import add_family
from .test_imports import IMPORT, Env
from .test_split import force_history_split
from .yolo_export import export_image, make_export

TRUSTED_IMPORT = {**IMPORT, "empty_labels_are_negatives": True}
NO_AUG = {"export": {"augmentation_run": "none"}}


@pytest.fixture
def env(tmp_path, write_config) -> Env:
    return Env(tmp_path, write_config)


def ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def setup(
    env: Env,
    n: int = 12,
    raw: int = 0,
    empty=frozenset(),
    imports=None,
    overrides: dict[str, Any] | None = None,
):
    make_export(env.export, splits={"train": list(range(n))}, empty=set(empty))
    for s in range(raw):
        cv2.imwrite(str(env.raw / f"field_{s}.png"), export_image(400 + s)[0])
    cfg = env.build(overrides, imports=imports or [TRUSTED_IMPORT])
    assert run_split(cfg).exit_code == 0
    return cfg


def item_of(cfg, origin: str, rel: str) -> str:
    for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord):
        if r.origin == origin and r.source_path == rel:
            return r.source_id
    raise KeyError(rel)


def splits(cfg) -> dict[str, str]:
    return {r.item_id: r.split for r in read_records(ws(cfg).split_items_manifest, SplitItemRecord)}


def place(cfg, assignments: dict[str, str]) -> None:
    """Deterministically put items (image item_id == source_id) into chosen splits."""
    for iid, split in assignments.items():
        force_history_split(cfg, {iid}, split)


def raw_ids(cfg) -> list[str]:
    return sorted(
        i.item_id for i in read_records(ws(cfg).items_manifest, ItemRecord) if i.origin == "raw"
    )


def imported_ids(cfg) -> list[str]:
    return sorted(
        i.item_id for i in read_records(ws(cfg).items_manifest, ItemRecord) if i.origin != "raw"
    )


def balanced(cfg, train_extra: list[str] = ()) -> dict[str, str]:
    """2 imported items each in val and test, everything else (incl. raw) in train."""
    imp = imported_ids(cfg)
    plan = {i: "train" for i in imp[4:]} | {
        imp[0]: "val",
        imp[1]: "val",
        imp[2]: "test",
        imp[3]: "test",
    }
    plan |= {i: "train" for i in train_extra}
    place(cfg, plan)
    return plan


def manifest(result) -> list[ExportRecord]:
    return read_records(result.run_dir / "manifest.csv", ExportRecord)


def files(run_dir: Path) -> dict[str, bytes]:
    return {
        p.relative_to(run_dir).as_posix(): p.read_bytes()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }


def labelled(env, cfg, overrides=None, policy=None):
    cfg = env.config(deep_update(NO_AUG, overrides or {}), imports=[TRUSTED_IMPORT])
    run_labels(cfg, policy)
    return cfg


# --------------------------------------------------------------------------------------
# Train inclusion
# --------------------------------------------------------------------------------------


def test_train_positives_negatives_and_unlabeled(env) -> None:
    paths = make_export(env.export, splits={"train": list(range(10))}, empty={9})
    cv2.imwrite(str(env.raw / "field_0.png"), export_image(400)[0])
    cfg = env.build(imports=[TRUSTED_IMPORT])
    run_split(cfg)
    neg = item_of(cfg, "import:legacy", paths[9])
    balanced(cfg, train_extra=[neg, *raw_ids(cfg)])
    cfg = labelled(env, cfg)
    result = run_export(cfg)

    rows = {r.item_id: r for r in manifest(result)}
    unlabeled = raw_ids(cfg)[0]
    assert unlabeled not in rows  # never exported as background
    excl = read_records(result.run_dir / "excluded.csv", ExcludedRecord)
    assert [(x.item_id, x.split, x.category) for x in excl] == [
        (unlabeled, "train", "label_unavailable")
    ]
    assert rows[neg].negative and rows[neg].split == "train"
    assert (result.run_dir / rows[neg].export_label).read_text() == ""
    positives = [r for r in rows.values() if not r.negative]
    assert all((result.run_dir / r.export_label).read_text().strip() for r in positives)
    t = result.report["train"]
    assert (t["positives"], t["trusted_negatives"]) == (5, 1)
    assert t["skipped_by_category"] == {"label_unavailable": 1}


def test_untrusted_machine_labels_skipped(env, scripted) -> None:
    make_export(env.export, splits={"train": list(range(8))})
    cv2.imwrite(str(env.raw / "field_0.png"), export_image(400)[0])
    scripted.default = [det("square", 0.9, 30, 30, 100, 100)]
    ann = {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}}
    cfg = env.build(ann, imports=[TRUSTED_IMPORT], annotate=True)
    run_split(cfg)
    balanced(cfg, train_extra=raw_ids(cfg))
    cfg = labelled(env, cfg, ann)
    result = run_export(cfg)
    excluded = {x.item_id: x for x in result.excluded}
    raw = raw_ids(cfg)[0]
    assert excluded[raw].category == "label_untrusted"
    assert raw not in {r.item_id for r in result.records}


# --------------------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------------------


def _with_aug(env, cfg):
    cfg = env.config({"augmentation": {"variants_per_item": 2}}, imports=[TRUSTED_IMPORT])
    run_labels(cfg)
    aug = run_augment(cfg)
    return cfg, aug


def test_augmentation_only_in_train(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg, aug = _with_aug(env, cfg)
    result = run_export(cfg)
    rows = manifest(result)
    augmented = [r for r in rows if r.kind == "augmented"]
    assert len(augmented) == len(aug.items) > 0
    assert {r.split for r in augmented} == {"train"}
    assert all(r.export_image.startswith("images/train/aug_") for r in augmented)
    assert all(r.aug_run_id == aug.run_id for r in augmented)
    for s in ("val", "test"):
        names = [p.name for p in (result.run_dir / "images" / s).iterdir()]
        assert names and not any(n.startswith("aug_") for n in names)
    assert result.report["train"]["augmented_included"] == len(aug.items)


def test_augmentation_run_not_found(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg, aug = _with_aug(env, cfg)
    explicit = env.config(
        {
            "augmentation": {"variants_per_item": 2},
            "export": {"augmentation_run": "augr_doesnotexist"},
        },
        imports=[TRUSTED_IMPORT],
    )
    with pytest.raises(ExportError, match="not found"):
        run_export(explicit)
    assert not (ws(cfg).root / "export").exists()


def _annotated_with_aug(env, scripted):
    make_export(env.export, splits={"train": list(range(10))})
    for s in range(3):
        cv2.imwrite(str(env.raw / f"field_{s}.png"), export_image(400 + s)[0])
    scripted.default = [det("square", 0.9, 30, 30, 100, 100)]
    ann = {
        "annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0},
        "augmentation": {"variants_per_item": 1},
    }
    cfg = env.build(ann, imports=[TRUSTED_IMPORT], annotate=True)
    run_split(cfg)
    balanced(cfg, train_extra=raw_ids(cfg))
    run_labels(cfg)
    aug = run_augment(cfg)
    return cfg, ann, aug


def test_train_labels_changed_after_augment_fails(env, scripted) -> None:
    cfg, ann, aug = _annotated_with_aug(env, scripted)
    run_id = json.loads(ws(cfg).active_annotation_run.read_text())["run_id"]
    with (ws(cfg).review / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "based_on_run_id", "decision"])
        w.writerow([raw_ids(cfg)[0], run_id, "accepted"])  # a train label became trusted
    run_labels(cfg)
    with pytest.raises(ExportError, match="Train labels changed since augmentation run"):
        run_export(cfg)


def test_augmentation_label_policy_mismatch_fails(env, scripted) -> None:
    cfg, ann, aug = _annotated_with_aug(env, scripted)
    machine = env.config(
        deep_update(ann, {"export": {"label_policy": "allow_machine_unreviewed"}}),
        imports=[TRUSTED_IMPORT],
    )
    run_labels(machine, LabelPolicy(allow_machine_unreviewed=True))
    with pytest.raises(ExportError, match="used label policy"):
        run_export(machine)


def test_augmentation_from_older_split_fails(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg, aug = _with_aug(env, cfg)
    place(cfg, {imported_ids(cfg)[5]: "val"})  # new split revision state
    run_labels(cfg)
    with pytest.raises(ExportError, match="different split revision — re-run `augment`"):
        run_export(cfg)


def test_changed_augmentation_files_fail(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg, aug = _with_aug(env, cfg)
    item = aug.items[0]
    label = aug.run_dir / item.label_path
    original = label.read_bytes()
    label.write_bytes(b"0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ExportError, match="missing or changed — re-run `augment`"):
        run_export(cfg)
    label.write_bytes(original)
    image = aug.run_dir / item.image_path
    image.write_bytes(image.read_bytes() + b"x")
    with pytest.raises(ExportError, match="changed or is missing"):
        run_export(cfg, dry_run=True)
    with pytest.raises(ExportError, match="changed during export"):
        run_export(cfg)
    runs = ws(cfg).root / "export" / "runs"
    assert not runs.exists() or not any(p.name.startswith("exp_") for p in runs.iterdir())


# --------------------------------------------------------------------------------------
# Stale state and integrity
# --------------------------------------------------------------------------------------


def test_stale_split_fails(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    cv2.imwrite(str(env.raw / "late.png"), export_image(99)[0])
    run_ingest(cfg)
    run_dedup(cfg)
    with pytest.raises(ExportError, match="re-run `split`"):
        run_export(cfg)


def test_stale_or_missing_labels_fail(env, scripted) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = env.config(NO_AUG, imports=[TRUSTED_IMPORT])
    with pytest.raises(ExportError, match="run `labels`"):
        run_export(cfg)
    run_labels(cfg)
    state = ws(cfg).labels_dir / "state.json"
    data = json.loads(state.read_text())
    data["resolution_digest"] = "0" * 64  # e.g. a review decision added after `labels`
    state.write_text(json.dumps(data))
    with pytest.raises(ExportError, match="re-run `labels`"):
        run_export(cfg)
    run_labels(cfg, LabelPolicy(allow_machine_unreviewed=True))
    with pytest.raises(ExportError, match="matching flags"):
        run_export(cfg)


def test_changed_canonical_file_fails(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    item = read_records(ws(cfg).items_manifest, ItemRecord)[0]
    (ws(cfg).root / item.path).write_bytes(b"changed")
    with pytest.raises(DatasetStateError, match="differ from items.csv"):
        run_export(cfg)
    assert not (ws(cfg).root / "export").exists()


# --------------------------------------------------------------------------------------
# Held-out policy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("heldout", ["val", "test"])
def test_strict_refuses_missing_heldout_label(env, heldout) -> None:
    cfg = setup(env, raw=1)
    balanced(cfg)
    place(cfg, {raw_ids(cfg)[0]: heldout})
    cfg = labelled(env, cfg)
    with pytest.raises(EvaluationCoverageError) as exc:
        run_export(cfg)
    msg = str(exc.value)
    assert f"{heldout}: 3 items, 2 with usable trusted labels, 1 missing" in msg
    assert "coverage 66.67%" in msg
    assert not (ws(cfg).root / "export").exists()
    assert json.loads((ws(cfg).reports / "export.json").read_text())["status"] == "refused"
    path = cfg.root / "config.yaml"
    assert main(["export", "--config", str(path), "--log-level", "CRITICAL"]) == 5


def test_non_strict_marks_incomplete_heldout(env) -> None:
    cfg = setup(env, raw=1)
    balanced(cfg)
    place(cfg, {raw_ids(cfg)[0]: "val"})
    cfg = labelled(env, cfg, {"export": {"evaluation_policy": "non_strict"}})
    result = run_export(cfg)
    ds = yaml.safe_load((result.run_dir / "dataset.yaml").read_text())
    assert ds["incomplete_evaluation_set"] is True
    assert "INCOMPLETE EVALUATION SET" in (result.run_dir / "dataset.yaml").read_text()
    assert json.loads((result.run_dir / "run.json").read_text())["incomplete_evaluation_set"]
    val = result.report["heldout"]["val"]
    assert (val["total"], val["exported"], val["unlabeled"], val["coverage_pct"]) == (
        3,
        2,
        1,
        66.67,
    )
    assert any("INCOMPLETE" in w for w in result.report["warnings"])
    assert raw_ids(cfg)[0] not in {r.item_id for r in manifest(result)}


def test_complete_heldout_strict_passes_with_coverage(env) -> None:
    cfg = setup(env, empty={11})
    neg = item_of(cfg, "import:legacy", "train/images/img_011.jpg")
    others = [i for i in imported_ids(cfg) if i != neg]
    place(
        cfg,
        {i: "train" for i in others[5:]}
        | {
            others[0]: "val",
            others[1]: "val",
            others[2]: "test",
            others[3]: "test",
            others[4]: "train",
            neg: "test",
        },
    )
    cfg = labelled(env, cfg)
    result = run_export(cfg)
    assert result.report["incomplete_evaluation_set"] is False
    test = result.report["heldout"]["test"]
    assert (test["total"], test["exported"], test["negatives"], test["coverage_pct"]) == (
        3,
        3,
        1,
        100.0,
    )


def test_empty_val_refused_in_strict(env) -> None:
    cfg = setup(env)
    place(cfg, {i: "train" for i in imported_ids(cfg)})
    cfg = labelled(env, cfg)
    with pytest.raises(EvaluationCoverageError, match="training needs a validation set"):
        run_export(cfg)


# --------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------


def test_class_mapping_is_config_ordered(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg, {"export": {"classes": ["triangle", "circle", "square"]}})
    result = run_export(cfg)
    ds = yaml.safe_load((result.run_dir / "dataset.yaml").read_text())
    assert ds["names"] == {0: "triangle", 1: "circle", 2: "square"} and ds["nc"] == 3
    assert result.report["class_mapping"] == {"square": 2, "circle": 1, "triangle": 0}
    resolver = LabelResolver.load(cfg)
    row = next(r for r in manifest(result) if not r.negative and r.kind == "original")
    project = {(b.cx, b.cy): b.class_id for b in resolver.resolve(row.item_id).label.boxes}
    remap = {0: 2, 1: 1, 2: 0}
    for line in (result.run_dir / row.export_label).read_text().splitlines():
        cid, cx, cy = line.split()[:3]
        assert int(cid) == remap[project[(float(cx), float(cy))]]


def test_non_target_classes_policy(env) -> None:
    cfg = setup(env, n=16)
    balanced(cfg)
    cfg = labelled(env, cfg, {"export": {"classes": ["square", "circle"]}})
    with pytest.raises(ExportError, match="outside export.classes"):
        run_export(cfg)
    cfg = labelled(
        env,
        cfg,
        {
            "export": {
                "classes": ["square", "circle"],
                "non_target_classes": "drop_boxes",
                "evaluation_policy": "non_strict",
            }
        },
    )
    result = run_export(cfg)
    for r in manifest(result):
        labels = (result.run_dir / r.export_label).read_text().splitlines()
        assert {line.split()[0] for line in labels} <= {"0", "1"}
        assert labels or r.negative
    dropped = [w for w in result.report["warnings"] if "Dropped non-target boxes" in w]
    assert dropped and "triangle" in dropped[0]
    only_triangles = [x for x in result.excluded if x.category == "non_target_only"]
    assert all(x.item_id not in {r.item_id for r in manifest(result)} for x in only_triangles)
    assert not any(r.negative for r in manifest(result))  # dropped-to-empty is not a negative


def test_invalid_class_ids_and_boxes_rejected() -> None:
    with pytest.raises(ExportError, match="outside 0..1"):
        _validate_label([LabelBox.make(2, 0.5, 0.5, 0.1, 0.1)], nc=2)
    with pytest.raises(ExportError, match="outside image"):
        _validate_label([LabelBox.make(0, 0.99, 0.5, 0.2, 0.1)], nc=2)


def test_export_config_validation(env) -> None:
    with pytest.raises(ConfigError, match="not in the project"):
        env.config({"export": {"classes": ["hexagon"]}})
    with pytest.raises(ConfigError, match="duplicates"):
        env.config({"export": {"classes": ["square", "square"]}})
    with pytest.raises(ConfigError, match="must be one of"):
        env.config({"export": {"evaluation_policy": "lenient"}})


# --------------------------------------------------------------------------------------
# Names, lineage, leakage
# --------------------------------------------------------------------------------------


def test_duplicate_basenames_do_not_collide(env) -> None:
    other = env.tmp / "exports" / "other"
    make_export(env.export, splits={"train": list(range(8))})
    make_export(other, splits={"train": [50, 51, 52, 53]})
    # rename the second export's files to collide with the first export's basenames
    for i, p in enumerate(sorted((other / "train" / "images").iterdir())):
        p.rename(p.with_name(f"img_{i:03d}.jpg"))
    for i, p in enumerate(sorted((other / "train" / "labels").iterdir())):
        p.rename(p.with_name(f"img_{i:03d}.txt"))
    imports = [TRUSTED_IMPORT, {**TRUSTED_IMPORT, "name": "other", "path": "exports/other"}]
    cfg = env.build(imports=imports)
    run_split(cfg)
    balanced(cfg)
    cfg = env.config(NO_AUG, imports=imports)
    run_labels(cfg)
    result = run_export(cfg)
    rows = manifest(result)
    same_name = [r for r in rows if r.source_path == "train/images/img_000.jpg"]
    assert len(same_name) == 2
    assert len({r.export_image for r in same_name}) == 2
    assert len({r.export_image for r in rows}) == len(rows)


def test_manifest_lineage(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg, aug = _with_aug(env, cfg)
    result = run_export(cfg)
    items = {i.item_id: i for i in read_records(ws(cfg).items_manifest, ItemRecord)}
    split_rows = {r.item_id: r for r in read_records(ws(cfg).split_items_manifest, SplitItemRecord)}
    for r in manifest(result):
        item = items[r.item_id]
        assert r.source_id == item.source_id and r.origin == item.origin
        assert r.source_path == item.source_path
        assert r.split == split_rows[r.item_id].split
        assert r.split_group == split_rows[r.item_id].split_group
        assert r.label_provenance == "human_imported"
        if r.kind == "original":
            assert r.image_sha256 == item.sha256
            assert Path(r.export_image).stem == r.item_id
        else:
            assert Path(r.export_image).stem == r.aug_item_id


def _row(split, item, group, source, kind="original", family=""):
    name = f"aug_{item}" if kind == "augmented" else item
    rec = ExportRecord(
        split,
        kind,
        f"images/{split}/{name}.jpg",
        f"labels/{split}/{name}.txt",
        item,
        name if kind == "augmented" else "",
        "",
        source,
        group,
        "raw",
        "",
        family,
        "human_imported",
        "",
        False,
        1,
        "",
        "",
    )
    return _Row(rec, Path("x"), "", [])


def test_leakage_assertions_fail_hard() -> None:
    split_rows = {
        i: SplitItemRecord(i, i, "image", g, s, "allocated", 1)
        for i, g, s in [("a", "g1", "train"), ("b", "g1", "val"), ("c", "g2", "test")]
    }
    with pytest.raises(ExportLeakageError, match="group g1 appears in several splits"):
        _assert_no_leakage(
            [_row("train", "a", "g1", "a"), _row("val", "b", "g1", "b")],
            split_rows,
            {"a", "b", "c"},
        )
    with pytest.raises(ExportLeakageError, match="outside train"):
        _assert_no_leakage(
            [_row("test", "c", "g2", "c", kind="augmented")], split_rows, {"a", "b", "c"}
        )
    with pytest.raises(ExportLeakageError, match="family fam appears"):
        _assert_no_leakage(
            [
                _row("train", "a", "ga", "a", family="fam"),
                _row("test", "c", "gc", "c", family="fam"),
            ],
            {
                **split_rows,
                "a": SplitItemRecord("a", "a", "image", "ga", "train", "", 1),
                "c": SplitItemRecord("c", "c", "image", "gc", "test", "", 1),
            },
            {"a", "c"},
        )
    with pytest.raises(ExportLeakageError, match="not dedup-eligible"):
        _assert_no_leakage([_row("train", "a", "g1", "a")], split_rows, {"b"})


def test_roboflow_family_exported_in_one_split(env) -> None:
    make_export(env.export, splits={"train": list(range(12))})
    family = add_family(env.export, "valve_001", 500)
    cfg = env.build(imports=[TRUSTED_IMPORT])
    run_split(cfg)
    balanced(cfg)
    cfg = labelled(env, cfg)
    result = run_export(cfg)
    fam = [r for r in manifest(result) if r.family_key]
    assert len(fam) == len(family) == 3 and len({r.split for r in fam}) == 1
    expected = 3 if fam[0].split == "train" else 0
    assert result.report["train"]["roboflow_family_members"] == expected


# --------------------------------------------------------------------------------------
# Determinism, immutability, dry run, Ultralytics
# --------------------------------------------------------------------------------------


def _full_flow(env, workspace: str):
    cfg = env.build({"project": {"workspace": workspace}}, imports=[TRUSTED_IMPORT])
    run_split(cfg)
    balanced(cfg)
    cfg = env.config(
        {"project": {"workspace": workspace}, "augmentation": {"variants_per_item": 2}},
        imports=[TRUSTED_IMPORT],
    )
    run_labels(cfg)
    run_augment(cfg)
    return cfg, run_export(cfg)


def test_deterministic_rerun_and_fresh_workspace(env) -> None:
    make_export(env.export, splits={"train": list(range(12))})
    cfg, first = _full_flow(env, "work_a")
    content = files(first.run_dir)
    again = run_export(cfg)
    assert again.run_id == first.run_id and not again.created
    assert files(first.run_dir) == content
    _, fresh = _full_flow(env, "work_b")
    assert fresh.run_id == first.run_id
    assert files(fresh.run_dir) == content  # byte-identical images, labels, yaml, manifests


def test_changed_config_creates_new_run_old_untouched(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    first = run_export(cfg)
    content = files(first.run_dir)
    cfg = labelled(env, cfg, {"export": {"classes": ["circle", "square", "triangle"]}})
    second = run_export(cfg)
    assert second.run_id != first.run_id and second.created
    assert files(first.run_dir) == content


def test_dry_run_writes_no_dataset(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    result = run_export(cfg, dry_run=True)
    assert result.records and result.run_dir is None
    assert not (ws(cfg).root / "export").exists()
    assert not (ws(cfg).reports / "export.json").exists()
    path = cfg.root / "config.yaml"
    assert main(["export", "--config", str(path), "--dry-run", "--log-level", "CRITICAL"]) == 0
    assert not (ws(cfg).root / "export").exists()


def test_tampered_export_run_not_overwritten(env) -> None:
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    first = run_export(cfg)
    run_json = first.run_dir / "run.json"
    data = json.loads(run_json.read_text())
    data["identity"]["labels_fp"] = "tampered"
    run_json.write_text(json.dumps(data))
    with pytest.raises(ExportError, match="refusing to overwrite"):
        run_export(cfg)


def test_ultralytics_can_load_the_export(env) -> None:
    utils = pytest.importorskip("ultralytics.data.utils")
    cfg = setup(env)
    balanced(cfg)
    cfg = labelled(env, cfg)
    result = run_export(cfg)
    data = utils.check_det_dataset(str(result.run_dir / "dataset.yaml"))
    assert data["nc"] == 3 and Path(data["train"]).is_dir() and Path(data["val"]).is_dir()
    assert Path(data["train"]) == (result.run_dir / "images" / "train").resolve()


def test_sources_untouched(env) -> None:
    cfg = setup(env, raw=1)
    balanced(cfg, train_extra=raw_ids(cfg))
    cfg, aug = _with_aug(env, cfg)
    snaps = [
        snapshot_tree(env.export),
        snapshot_tree(env.tmp / "raw"),
        snapshot_tree(ws(cfg).root / "interim"),
        snapshot_tree(ws(cfg).root / "augmentation"),
    ]
    run_export(cfg)
    assert [
        snapshot_tree(env.export),
        snapshot_tree(env.tmp / "raw"),
        snapshot_tree(ws(cfg).root / "interim"),
        snapshot_tree(ws(cfg).root / "augmentation"),
    ] == snaps


# --------------------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------------------


def test_review_queue_selects_everything_needing_confirmation(env, scripted) -> None:
    make_export(env.export, splits={"train": list(range(4))})
    for s in range(3):
        cv2.imwrite(str(env.raw / f"field_{s}.png"), export_image(400 + s)[0])
    scripted.default = [det("square", 0.95, 30, 30, 100, 100)]  # confident: NOT flagged
    ann = {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}}
    cfg = env.build(ann, imports=[TRUSTED_IMPORT], annotate=True)
    assert not list((ws(cfg).review / "labels").glob("*.txt"))  # nothing flagged at annotate

    before = resolution_digest(LabelResolver.load(cfg).resolve_all())
    result = run_review_queue(cfg)
    assert sorted(result.queued) == raw_ids(cfg)  # all 3 raw items, no imported item
    assert result.with_machine_label == 3
    assert resolution_digest(LabelResolver.load(cfg).resolve_all()) == before  # trust unchanged
    for iid in raw_ids(cfg):
        assert (ws(cfg).review / "labels" / f"{iid}.txt").read_text().startswith("0 ")


def test_review_queue_blank_items_and_manual_decision(env) -> None:
    cfg = setup(env, raw=2)
    queued = run_review_queue(cfg)
    assert sorted(queued.queued) == raw_ids(cfg) and queued.blank == 2
    with (ws(cfg).review / "review_manifest.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["based_on_run_id"] for r in rows} == {"manual"}
    target = raw_ids(cfg)[0]
    (ws(cfg).review / "labels" / f"{target}.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    with (ws(cfg).review / "decisions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "based_on_run_id", "decision"])
        w.writerow([target, "manual", "corrected"])
    res = LabelResolver.load(cfg).resolve(target)
    assert res.status == "trusted" and res.label.provenance == "human_reviewed_corrected"
    other = raw_ids(cfg)[1]
    assert LabelResolver.load(cfg).resolve(other).status == "unavailable"
    assert run_review_queue(cfg).queued == [other]  # trusted item no longer queued


def test_review_queue_split_filter_and_unreviewed_mode(env) -> None:
    cfg = setup(env, raw=2)
    place(cfg, {raw_ids(cfg)[0]: "val", raw_ids(cfg)[1]: "train"})
    assert run_review_queue(cfg, split="val").queued == [raw_ids(cfg)[0]]
    assert run_review_queue(cfg, mode="unreviewed_machine").queued == []  # no machine labels
    path = cfg.root / "config.yaml"
    assert (
        main(["review-queue", "--config", str(path), "--split", "train", "--log-level", "CRITICAL"])
        == 0
    )
