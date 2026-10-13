"""Legacy YOLO/Roboflow import, provenance, review decisions and label resolution."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import pytest

from cv_pipeline.annotation import run_annotate
from cv_pipeline.cli import main
from cv_pipeline.config import ConfigError, load_config
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.dedup import run_dedup
from cv_pipeline.imports import ImportError_, run_import
from cv_pipeline.imports.importer import ImportImageRecord
from cv_pipeline.ingest import run_ingest
from cv_pipeline.labels import (
    LabelPolicy,
    LabelProvenance,
    LabelResolver,
    ResolutionStatus,
    UnavailableReason,
    run_labels,
)
from cv_pipeline.manifest import ItemRecord, RejectedRecord, SourceRefRecord, read_records
from cv_pipeline.splitting import run_split
from cv_pipeline.splitting.history import SplitItemRecord, load_history
from cv_pipeline.workspace import Workspace

from .conftest import deep_update, snapshot_tree
from .fakes import det
from .yolo_export import export_image, make_export

IMPORT = {"name": "legacy", "path": "exports/legacy", "labels_trusted": True}


class Env:
    def __init__(self, tmp: Path, write_config) -> None:
        self.tmp, self._write = tmp, write_config
        self.raw = tmp / "raw" / "images"
        self.raw.mkdir(parents=True)
        self.export = tmp / "exports" / "legacy"

    def config(self, overrides: dict[str, Any] | None = None, imports: list[dict] | None = None):
        data = deep_update(
            {"split": {"train": 0.8, "val": 0.1, "test": 0.1}, "imports": imports or [IMPORT]},
            overrides or {},
        )
        return load_config(self._write(data))

    def build(self, overrides=None, imports=None, annotate=False):
        cfg = self.config(overrides, imports)
        for imp in cfg.imports:
            if (imp.path / "data.yaml").is_file():
                run_import(cfg, imp.name)
        run_ingest(cfg)
        run_dedup(cfg)
        if annotate:
            run_annotate(cfg)
        return cfg


@pytest.fixture
def env(tmp_path, write_config) -> Env:
    return Env(tmp_path, write_config)


def ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def item_by_ref(cfg, origin: str, rel: str) -> str:
    for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord):
        if r.origin == origin and r.source_path == rel:
            return r.source_id
    raise KeyError(rel)


def resolve(cfg, item_id: str, **policy):
    return LabelResolver.load(cfg, LabelPolicy(**policy)).resolve(item_id)


# --------------------------------------------------------------------------------------
# Import basics
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", ["roboflow", "ultralytics"])
def test_import_layouts_provenance_and_legacy_splits(env, layout) -> None:
    paths = make_export(env.export, layout=layout)
    cfg = env.build()
    imported = read_records(ws(cfg).imports / "legacy" / "images.csv", ImportImageRecord)
    assert len(imported) == 10
    assert {i.legacy_split for i in imported} == {"train", "val", "test"}
    assert {i.label_status for i in imported} == {"labeled"}

    items = read_records(ws(cfg).items_manifest, ItemRecord)
    assert len(items) == 10 and {i.origin for i in items} == {"import:legacy"}
    refs = {r.source_path: r for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)}
    assert refs[paths[6]].legacy_split == "val" and refs[paths[0]].legacy_split == "train"

    resolver = LabelResolver.load(cfg)
    resolutions = resolver.resolve_all()
    assert {r.status for r in resolutions} == {ResolutionStatus.TRUSTED}
    assert {r.label.provenance for r in resolutions} == {LabelProvenance.HUMAN_IMPORTED}
    first = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[0]))
    gt = (env.export / paths[0].replace("images", "labels").replace(".jpg", ".txt")).read_text()
    assert len(first.label.boxes) == len(gt.strip().splitlines())


def test_names_as_mapping_in_data_yaml(env) -> None:
    make_export(env.export, names_as_dict=True)
    cfg = env.build()
    assert all(r.status == "trusted" for r in LabelResolver.load(cfg).resolve_all())


def test_source_export_and_raw_untouched(env) -> None:
    make_export(env.export)
    before_export = snapshot_tree(env.export)
    before_raw = snapshot_tree(env.tmp / "raw")
    cfg = env.build()
    run_split(cfg)
    run_labels(cfg)
    assert snapshot_tree(env.export) == before_export
    assert snapshot_tree(env.tmp / "raw") == before_raw


def test_dry_run_writes_nothing(env) -> None:
    make_export(env.export)
    cfg = env.config()
    result = run_import(cfg, "legacy", dry_run=True)
    assert result.record["counts"]["images"] == 10 and not result.created
    assert not (ws(cfg).imports / "legacy").exists()


def test_reimport_is_idempotent_and_conflicts_are_refused(env) -> None:
    make_export(env.export)
    cfg = env.config()
    assert run_import(cfg, "legacy").created
    snap = snapshot_tree(ws(cfg).imports / "legacy")
    assert not run_import(cfg, "legacy").created
    assert snapshot_tree(ws(cfg).imports / "legacy") == snap

    make_export(env.export, splits={"train": [42]})  # export changed
    with pytest.raises(ImportError_, match="already exists with different content"):
        run_import(cfg, "legacy")
    assert snapshot_tree(ws(cfg).imports / "legacy") == snap


def test_settings_change_after_import_requires_new_name(env) -> None:
    make_export(env.export)
    env.build()
    changed = env.config(imports=[{**IMPORT, "labels_trusted": False}])
    with pytest.raises(ImportError_, match="re-import under a new name"):
        run_ingest(changed)


def test_deterministic_import_and_resolution(env, write_config) -> None:
    make_export(env.export)
    cfg = env.build()
    run_labels(cfg)
    files = ("images.csv", "boxes.csv", "import.json")
    first = {f: (ws(cfg).imports / "legacy" / f).read_bytes() for f in files}
    resolved = (ws(cfg).labels_dir / "resolved.csv").read_bytes()
    run_labels(cfg)
    assert (ws(cfg).labels_dir / "resolved.csv").read_bytes() == resolved

    cfg2 = env.build({"project": {"workspace": "work2"}})
    run_labels(cfg2)
    assert {f: (ws(cfg2).imports / "legacy" / f).read_bytes() for f in files} == first
    assert (ws(cfg2).labels_dir / "resolved.csv").read_bytes() == resolved


# --------------------------------------------------------------------------------------
# Validation and taxonomy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_label, problem",
    [
        ("0 0.5 0.5 0.2\n", "expected 5 values"),
        ("0 0.5 0.5 0.2 0.2 0.1 0.1\n", "expected 5 values"),
        ("7 0.5 0.5 0.2 0.2\n", "class id 7 outside"),
        ("x 0.5 0.5 0.2 0.2\n", "not an integer"),
        ("0 nan 0.5 0.2 0.2\n", "non-finite"),
        ("0 1.5 0.5 0.2 0.2\n", "outside [0, 1]"),
        ("0 0.95 0.5 0.2 0.2\n", "extends outside"),
        ("0 0.5 0.5 0 0.2\n", "width/height"),
    ],
)
def test_malformed_labels_are_recorded_and_untrusted(env, bad_label, problem) -> None:
    paths = make_export(env.export, label_override={3: bad_label})
    cfg = env.build()
    rec = next(
        i
        for i in read_records(ws(cfg).imports / "legacy" / "images.csv", ImportImageRecord)
        if i.rel_path == paths[3]
    )
    assert rec.label_status == "invalid" and problem in rec.label_error
    res = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[3]))
    assert res.status == "unavailable" and res.reason == UnavailableReason.INVALID_LABEL


def test_unknown_class_fails_clearly_and_writes_nothing(env) -> None:
    make_export(env.export, names=("square", "circle", "hexagon"))
    cfg = env.config()
    with pytest.raises(ImportError_, match=r"source classes \['hexagon'\]"):
        run_import(cfg, "legacy")
    assert not (ws(cfg).imports / "legacy").exists()


IGNORED_EXPORT = {
    "names": ("Square", "circle", "triangle", "person"),
    "label_override": {0: "0 0.5 0.5 0.2 0.2\n3 0.3 0.3 0.1 0.1\n", 1: "3 0.3 0.3 0.1 0.1\n"},
}
IGNORED_IMPORT = {**IMPORT, "class_map": {"Square": "square"}, "ignore_classes": ["person"]}


def test_class_mapping_and_ignored_classes(env) -> None:
    paths = make_export(env.export, **IGNORED_EXPORT)
    cfg = env.build(imports=[IGNORED_IMPORT])
    res = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[0]))
    assert [b.class_id for b in res.label.boxes] == [0]
    import_json = (ws(cfg).imports / "legacy" / "import.json").read_text()
    assert '"person": null' in import_json


def test_ignored_only_label_is_not_a_negative_by_default(env) -> None:
    paths = make_export(env.export, **IGNORED_EXPORT)
    cfg = env.build(imports=[IGNORED_IMPORT])
    only_person = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[1]))
    assert only_person.status == "unavailable"
    assert only_person.reason == UnavailableReason.IGNORED_ONLY
    report = run_labels(cfg).report
    assert report["unavailable_reasons"]["ignored_only_unconfirmed_negative"] == 1
    imported = (ws(cfg).reports / "import_legacy.json").read_text()
    assert '"ignored_only_labels": 1' in imported


def test_ignored_only_label_is_a_negative_when_explicitly_allowed(env) -> None:
    paths = make_export(env.export, **IGNORED_EXPORT)
    cfg = env.build(imports=[{**IGNORED_IMPORT, "ignored_only_labels_are_negatives": True}])
    neg = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[1]))
    assert neg.status == "trusted" and neg.label.negative and neg.label.boxes == ()
    assert neg.label.provenance == LabelProvenance.HUMAN_IMPORTED
    assert "ignored_only_source_annotation" in neg.label.notes
    report = run_labels(cfg).report
    assert report["negatives_from_ignored_only_annotations"] == 1
    with (ws(cfg).labels_dir / "resolved.csv").open(newline="") as fh:
        rows = {r["item_id"]: r for r in csv.DictReader(fh)}
    assert "ignored_only_source_annotation" in rows[neg.item_id]["label_notes"]


def test_ignored_only_policy_does_not_change_empty_file_policy(env) -> None:
    paths = make_export(env.export, empty={2}, **IGNORED_EXPORT)
    cfg = env.build(imports=[{**IGNORED_IMPORT, "ignored_only_labels_are_negatives": True}])
    empty = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[2]))
    assert empty.status == "unavailable"
    assert empty.reason == UnavailableReason.EMPTY_NOT_CONFIRMED


def test_duplicate_class_names_rejected(env) -> None:
    make_export(env.export, names=("square", "circle", "square"))
    with pytest.raises(ImportError_, match="duplicate class names"):
        run_import(env.config(), "legacy")


def test_import_config_validation(env) -> None:
    with pytest.raises(ConfigError, match="same project class"):
        env.config(imports=[{**IMPORT, "class_map": {"a": "square", "b": "square"}}])
    with pytest.raises(ConfigError, match="must not overlap ingest.raw_dir"):
        env.config(imports=[{**IMPORT, "path": "raw/legacy"}])
    with pytest.raises(ConfigError, match="requires labels_trusted"):
        env.config(
            imports=[{**IMPORT, "labels_trusted": False, "empty_labels_are_negatives": True}]
        )
    with pytest.raises(ConfigError, match="name must match"):
        env.config(imports=[{**IMPORT, "name": "Bad Name"}])
    with pytest.raises(ConfigError, match="duplicate import names"):
        env.config(imports=[IMPORT, IMPORT])
    with pytest.raises(ConfigError, match="ignored_only_labels_are_negatives requires"):
        env.config(
            imports=[{**IMPORT, "labels_trusted": False, "ignored_only_labels_are_negatives": True}]
        )


# --------------------------------------------------------------------------------------
# Negatives and trust
# --------------------------------------------------------------------------------------


def test_empty_human_label_is_negative_only_when_configured(env) -> None:
    paths = make_export(env.export, empty={2}, missing={3})
    cfg = env.build(imports=[{**IMPORT, "empty_labels_are_negatives": True}])
    neg = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[2]))
    assert neg.status == "trusted" and neg.label.negative and neg.label.boxes == ()
    miss = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[3]))
    assert miss.status == "unavailable" and miss.reason == UnavailableReason.MISSING_LABEL

    # Same export imported without confirming negatives (separate workspace).
    cfg2 = env.build({"project": {"workspace": "work_unconfirmed"}}, imports=[IMPORT])
    empty = resolve(cfg2, item_by_ref(cfg2, "import:legacy", paths[2]))
    assert empty.status == "unavailable" and empty.reason == UnavailableReason.EMPTY_NOT_CONFIRMED


def test_unverified_import_requires_opt_in(env) -> None:
    paths = make_export(env.export)
    cfg = env.build(imports=[{**IMPORT, "labels_trusted": False}])
    iid = item_by_ref(cfg, "import:legacy", paths[0])
    assert resolve(cfg, iid).reason == UnavailableReason.IMPORTED_UNVERIFIED
    opt = resolve(cfg, iid, allow_imported_unverified=True)
    assert opt.status == ResolutionStatus.OPT_IN
    assert opt.label.provenance == LabelProvenance.IMPORTED_UNVERIFIED


# --------------------------------------------------------------------------------------
# Machine labels and review decisions
# --------------------------------------------------------------------------------------


def _raw_annotated(env, scripted, response, overrides=None):
    cv2.imwrite(str(env.raw / "field.png"), export_image(77)[0])
    scripted.default = response
    ann = {"annotation": {"enabled": True, "backend": "scripted", "review_sample_rate": 0.0}}
    cfg = env.build(deep_update(ann, overrides or {}), imports=[], annotate=True)
    w = ws(cfg)
    run_id = json.loads(w.active_annotation_run.read_text())["run_id"]
    iid = read_records(w.items_manifest, ItemRecord)[0].item_id
    return cfg, iid, run_id


def _decide(cfg, rows: list[dict[str, str]]) -> None:
    path = ws(cfg).review / "decisions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "item_id",
                "based_on_run_id",
                "decision",
                "reviewer",
                "notes",
                "label_sha256",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_machine_zero_detections_never_a_trusted_negative(env, scripted) -> None:
    cfg, iid, _ = _raw_annotated(env, scripted, [])
    assert resolve(cfg, iid).reason == UnavailableReason.NOT_REVIEWED
    opt = resolve(cfg, iid, allow_machine_unreviewed=True)
    assert opt.status == "unavailable" and opt.reason == UnavailableReason.MACHINE_EMPTY
    forced = resolve(cfg, iid, allow_machine_unreviewed=True, allow_machine_empty=True)
    assert forced.status == ResolutionStatus.OPT_IN and forced.label.negative


def test_machine_labels_require_opt_in(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.9, 10, 10, 50, 50)])
    assert resolve(cfg, iid).status == "unavailable"
    opt = resolve(cfg, iid, allow_machine_unreviewed=True)
    assert opt.status == ResolutionStatus.OPT_IN
    assert opt.label.provenance == LabelProvenance.MACHINE_UNREVIEWED
    assert opt.label.source_ref == f"run:{run_id}"


def test_accepted_decision_makes_machine_label_trusted(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.9, 10, 10, 50, 50)])
    _decide(
        cfg, [{"item_id": iid, "based_on_run_id": run_id, "decision": "accepted", "reviewer": "jb"}]
    )
    res = resolve(cfg, iid)
    assert res.status == "trusted"
    assert res.label.provenance == LabelProvenance.HUMAN_REVIEWED_ACCEPTED
    assert len(res.label.boxes) == 1 and res.label.boxes[0].class_id == 0


def test_accepted_zero_detections_is_a_trusted_negative(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [])
    _decide(cfg, [{"item_id": iid, "based_on_run_id": run_id, "decision": "accepted"}])
    res = resolve(cfg, iid)
    assert res.status == "trusted" and res.label.negative


def test_corrected_decision_uses_edited_label(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.3, 10, 10, 50, 50)])
    label = ws(cfg).review / "labels" / f"{iid}.txt"
    assert label.is_file()  # exported because of low confidence
    label.write_text("2 0.500000 0.500000 0.250000 0.250000\n")
    _decide(cfg, [{"item_id": iid, "based_on_run_id": run_id, "decision": "corrected"}])
    res = resolve(cfg, iid)
    assert res.label.provenance == LabelProvenance.HUMAN_REVIEWED_CORRECTED
    assert [(b.class_id, b.cx, b.w) for b in res.label.boxes] == [(2, 0.5, 0.25)]

    _decide(
        cfg,
        [
            {
                "item_id": iid,
                "based_on_run_id": run_id,
                "decision": "corrected",
                "label_sha256": "0" * 64,
            }
        ],
    )
    resolver = LabelResolver.load(cfg)
    assert resolver.resolve(iid).status == "unavailable"
    assert "label_sha256 mismatch" in resolver.invalid_decisions[0].problem


def test_accepting_an_edited_label_is_invalid(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.3, 10, 10, 50, 50)])
    (ws(cfg).review / "labels" / f"{iid}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    _decide(cfg, [{"item_id": iid, "based_on_run_id": run_id, "decision": "accepted"}])
    resolver = LabelResolver.load(cfg)
    assert resolver.resolve(iid).status == "unavailable"
    assert "use 'corrected'" in resolver.invalid_decisions[0].problem


def test_rejected_is_unavailable_under_every_policy(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.9, 10, 10, 50, 50)])
    _decide(cfg, [{"item_id": iid, "based_on_run_id": run_id, "decision": "rejected"}])
    for policy in ({}, {"allow_machine_unreviewed": True, "allow_machine_empty": True}):
        res = resolve(cfg, iid, **policy)
        assert res.status == "unavailable" and res.reason == UnavailableReason.REJECTED


def test_review_is_not_inferred_from_label_files(env, scripted) -> None:
    cfg, iid, _ = _raw_annotated(env, scripted, [det("square", 0.3, 10, 10, 50, 50)])
    (ws(cfg).review / "labels" / f"{iid}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    assert resolve(cfg, iid).reason == UnavailableReason.NOT_REVIEWED


def test_invalid_decisions_are_reported(env, scripted) -> None:
    cfg, iid, run_id = _raw_annotated(env, scripted, [det("square", 0.9, 10, 10, 50, 50)])
    _decide(
        cfg,
        [
            {"item_id": "img_unknown", "based_on_run_id": run_id, "decision": "accepted"},
            {"item_id": iid, "based_on_run_id": "ann_doesnotexist", "decision": "accepted"},
            {"item_id": iid, "based_on_run_id": run_id, "decision": "maybe"},
        ],
    )
    result = run_labels(cfg)
    problems = " ".join(d["problem"] for d in result.report["invalid_decisions"])
    assert "unknown item_id" in problems and "not found" in problems and "not in" in problems


# --------------------------------------------------------------------------------------
# Duplicates across sources
# --------------------------------------------------------------------------------------


def test_byte_identical_raw_and_import_are_one_item_with_two_refs(env) -> None:
    paths = make_export(env.export)
    shutil.copyfile(env.export / paths[4], env.raw / "field_copy.jpg")
    cfg = env.build()
    items = read_records(ws(cfg).items_manifest, ItemRecord)
    assert len(items) == 10  # not 11
    refs = [
        r
        for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
        if r.source_path in {"images/field_copy.jpg", paths[4]}
    ]
    assert {(r.origin, r.role) for r in refs} == {
        ("raw", "canonical"),
        ("import:legacy", "duplicate"),
    }
    assert {r.source_id for r in refs} == {refs[0].source_id}
    rejected = read_records(ws(cfg).rejected_manifest, RejectedRecord)
    assert any(r.origin == "import:legacy" and r.reason == "exact_duplicate" for r in rejected)
    res = resolve(cfg, refs[0].source_id)
    assert res.status == "trusted" and res.label.provenance == LabelProvenance.HUMAN_IMPORTED


def test_pixel_identical_copy_is_one_example_and_label_transfers(env) -> None:
    paths = make_export(env.export)
    pixels = cv2.imread(str(env.export / paths[5]))
    cv2.imwrite(str(env.raw / "field_resaved.png"), pixels)  # same pixels, different bytes
    cfg = env.build()
    split_input_items = LabelResolver.load(cfg).eligible_ids
    raw_id = item_by_ref(cfg, "raw", "images/field_resaved.png")
    imp_id = item_by_ref(cfg, "import:legacy", paths[5])
    assert raw_id != imp_id
    assert (raw_id in split_input_items) != (imp_id in split_input_items)  # one example
    kept = raw_id if raw_id in split_input_items else imp_id
    res = resolve(cfg, kept)
    assert res.status == "trusted" and res.label.provenance == LabelProvenance.HUMAN_IMPORTED
    assert len(res.label.boxes) > 0


def test_conflicting_labels_for_the_same_image(env) -> None:
    paths = make_export(env.export)
    other = env.tmp / "exports" / "other"
    make_export(other, splits={"train": [0]}, label_override={0: "1 0.5 0.5 0.1 0.1\n"})
    imports = [IMPORT, {"name": "other", "path": "exports/other", "labels_trusted": True}]
    cfg = env.build(imports=imports)
    res = resolve(cfg, item_by_ref(cfg, "import:legacy", paths[0]))
    assert res.status == ResolutionStatus.CONFLICT and res.label is None
    assert len(res.candidates) == 2


# --------------------------------------------------------------------------------------
# Corpus integration, legacy splits and integrity
# --------------------------------------------------------------------------------------


def test_imported_and_new_data_form_one_grouped_split(env) -> None:
    make_export(env.export, splits={"test": list(range(12))})  # legacy: everything is test
    for s in range(8):
        cv2.imwrite(str(env.raw / f"new_{s}.png"), export_image(200 + s)[0])
    cfg = env.build()
    result = run_split(cfg)
    split_items = read_records(ws(cfg).split_items_manifest, SplitItemRecord)
    origins = {i.item_id: i.origin for i in read_records(ws(cfg).items_manifest, ItemRecord)}
    imported = [s for s in split_items if origins[s.item_id] == "import:legacy"]
    assert len(imported) == 12 and len(split_items) == 20
    assert {s.split for s in imported} >= {"train"}  # legacy 'test' did not carry over
    assert {r.reason for r in load_history(ws(cfg)).groups.values()} == {"allocated"}
    assert result.report["splits"]["train"]["items"] == 16

    labels = run_labels(cfg)
    assert labels.report["legacy_vs_current_split"]["test"]["train"] > 0
    assert labels.report["split"].startswith("split revision")


def test_tampered_imported_file_is_rejected_at_ingest(env) -> None:
    paths = make_export(env.export)
    cfg = env.build()
    target = ws(cfg).imports / "legacy" / "files" / paths[1]
    target.write_bytes(target.read_bytes() + b"tampered")
    run_ingest(cfg)
    rejected = read_records(ws(cfg).rejected_manifest, RejectedRecord)
    assert [r.source_path for r in rejected if r.reason == "import_file_modified"] == [paths[1]]


def test_tampered_workspace_file_blocks_label_output(env) -> None:
    make_export(env.export)
    cfg = env.build()
    item = read_records(ws(cfg).items_manifest, ItemRecord)[0]
    (ws(cfg).root / item.path).write_bytes(b"not the ingested image")
    with pytest.raises(DatasetStateError, match="differ from items.csv"):
        run_labels(cfg)


def test_cli_import_and_labels(env, capsys) -> None:
    make_export(env.export)
    cfg = env.config()
    path = cfg.root / "config.yaml"
    assert (
        main(
            ["import-dataset", "--config", str(path), "--name", "legacy", "--log-level", "CRITICAL"]
        )
        == 0
    )
    assert "imported" in capsys.readouterr().out
    assert (
        main(["import-dataset", "--config", str(path), "--name", "nope", "--log-level", "CRITICAL"])
        == 1
    )
    assert main(["prepare", "--config", str(path), "--log-level", "CRITICAL"]) == 0
    assert main(["labels", "--config", str(path), "--log-level", "CRITICAL"]) == 0
    assert (ws(cfg).reports / "labels.md").is_file()


def test_prepare_never_imports(env) -> None:
    make_export(env.export)
    cfg = env.config()
    assert (
        main(["prepare", "--config", str(cfg.root / "config.yaml"), "--log-level", "CRITICAL"]) == 0
    )
    assert not (ws(cfg).imports / "legacy").exists()
    assert read_records(ws(cfg).items_manifest, ItemRecord) == []
