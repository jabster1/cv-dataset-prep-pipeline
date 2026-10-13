"""Roboflow derivative families (leakage-prevention lineage) and EXIF reporting."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_pipeline.dataset_state import DatasetStateError, load_dataset_state
from cv_pipeline.dedup import run_dedup
from cv_pipeline.imports import run_import
from cv_pipeline.imports.exif import exif_orientation
from cv_pipeline.imports.importer import ImportImageRecord
from cv_pipeline.imports.yolo import family_key, roboflow_key
from cv_pipeline.ingest import run_ingest
from cv_pipeline.manifest import DedupRecord, SourceRefRecord, read_records, write_records
from cv_pipeline.splitting import run_split
from cv_pipeline.splitting.history import SplitItemRecord
from cv_pipeline.workspace import Workspace

from .conftest import snapshot_tree
from .test_imports import IMPORT, Env
from .yolo_export import export_image, make_export

LABEL = "0 0.500000 0.500000 0.200000 0.200000\n"


def rf_name(stem: str, variant: str) -> str:
    h = hashlib.md5(f"{stem}:{variant}".encode()).hexdigest()  # 32 hex, like Roboflow
    return f"{stem}_jpg.rf.{h}"


def variants(seed: int) -> list[np.ndarray]:
    """Transformations perceptual hashing does not reliably link (flip, heavy crop)."""
    base = export_image(seed)[0]
    h, w = base.shape[:2]
    crop = cv2.resize(base[h // 4 : h, 0 : w // 2], (w, h))
    return [base, cv2.flip(base, 1), crop]


def add_family(export: Path, stem: str, seed: int, split: str = "train") -> list[str]:
    img_dir, lbl_dir = export / split / "images", export / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rels = []
    for i, img in enumerate(variants(seed)):
        name = rf_name(stem, str(i))
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
        (lbl_dir / f"{name}.txt").write_text(LABEL)
        rels.append(f"{split}/images/{name}.jpg")
    return rels


@pytest.fixture
def env(tmp_path, write_config) -> Env:
    return Env(tmp_path, write_config)


def ws(cfg) -> Workspace:
    return Workspace(cfg.workspace_dir)


def by_rel(cfg, origin: str) -> dict[str, str]:
    return {
        r.source_path: r.source_id
        for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
        if r.origin == origin
    }


def dedup_rows(cfg) -> dict[str, DedupRecord]:
    return {d.item_id: d for d in read_records(ws(cfg).dedup_manifest, DedupRecord)}


def splits(cfg) -> dict[str, str]:
    return {r.item_id: r.split for r in read_records(ws(cfg).split_items_manifest, SplitItemRecord)}


# --------------------------------------------------------------------------------------


def test_roboflow_key_rule_is_conservative() -> None:
    good = rf_name("valve_001", "a") + ".jpg"
    assert roboflow_key(f"train/images/{good}") == "valve_001"
    assert family_key("loto_v3", good) == "import:loto_v3:roboflow:valve_001"
    for bad in (
        "valve_001.jpg",
        "valve_001_jpg.rf.abc123.jpg",
        "valve_001_jpg.rf.zz.jpg",
        "_jpg.rf." + "a" * 32 + ".jpg",
        "valve_001_foo.rf." + "a" * 32 + ".jpg",
    ):
        assert roboflow_key(bad) == "", bad


def test_family_members_share_one_split_without_exclusion(env) -> None:
    make_export(env.export, splits={"train": list(range(20)), "val": [20, 21], "test": [22, 23]})
    family = add_family(env.export, "valve_001", 500)
    cfg = env.build()
    ids = by_rel(cfg, "import:legacy")
    fam_ids = [ids[r] for r in family]
    rows = dedup_rows(cfg)
    assert len({rows[i].split_group for i in fam_ids}) == 1
    assert rows[fam_ids[0]].split_group.startswith("sg_")
    assert not any(rows[i].excluded for i in fam_ids)  # nothing discarded
    assert rows[fam_ids[1]].dup_group is None  # the flip is NOT treated as a duplicate

    run_split(cfg)
    assert len({splits(cfg)[i] for i in fam_ids}) == 1

    imported = read_records(ws(cfg).imports / "legacy" / "images.csv", ImportImageRecord)
    keys = {i.family_key for i in imported if i.rel_path in family}
    assert keys == {"import:legacy:roboflow:valve_001"}
    counts = json.loads((ws(cfg).imports / "legacy" / "import.json").read_text())
    assert counts["counts"]["lineage_families"] == 1
    assert counts["counts"]["lineage_family_images"] == 3
    dup_report = json.loads((ws(cfg).reports / "duplicates.json").read_text())
    assert dup_report["lineage"]["families"] == 1 and dup_report["lineage"]["sources"] == 3


def test_different_stems_stay_independent(env) -> None:
    make_export(env.export, splits={"train": [0, 1]})
    a = add_family(env.export, "valve_001", 500)
    b = add_family(env.export, "valve_002", 600)
    cfg = env.build()
    ids, rows = by_rel(cfg, "import:legacy"), dedup_rows(cfg)
    ga = {rows[ids[r]].split_group for r in a}
    gb = {rows[ids[r]].split_group for r in b}
    assert len(ga) == 1 and len(gb) == 1 and ga != gb


def test_same_stem_in_different_imports_does_not_collide(env) -> None:
    other = env.tmp / "exports" / "other"
    make_export(env.export, splits={"train": [0]})
    make_export(other, splits={"train": [1]})
    a = add_family(env.export, "valve_001", 500)
    b = add_family(other, "valve_001", 700)  # same stem, different images
    imports = [IMPORT, {"name": "other", "path": "exports/other", "labels_trusted": True}]
    cfg = env.build(imports=imports)
    rows = dedup_rows(cfg)
    ga = {rows[by_rel(cfg, "import:legacy")[r]].split_group for r in a}
    gb = {rows[by_rel(cfg, "import:other")[r]].split_group for r in b}
    assert len(ga) == 1 and len(gb) == 1 and ga != gb
    keys = {
        r.family_key
        for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
        if r.family_key
    }
    assert keys == {"import:legacy:roboflow:valve_001", "import:other:roboflow:valve_001"}


def test_family_grouping_is_deterministic(env) -> None:
    make_export(env.export, splits={"train": list(range(8))})
    add_family(env.export, "valve_001", 500)
    add_family(env.export, "valve_002", 600)
    cfg1 = env.build()
    run_split(cfg1)
    cfg2 = env.build({"project": {"workspace": "work2"}})
    run_split(cfg2)
    for name in ("dedup.csv", "source_refs.csv"):
        assert (ws(cfg1).manifests / name).read_bytes() == (ws(cfg2).manifests / name).read_bytes()
    assert ws(cfg1).split_items_manifest.read_bytes() == ws(cfg2).split_items_manifest.read_bytes()


def test_families_combine_with_raw_duplicates_and_near_duplicates(env) -> None:
    make_export(env.export, splits={"train": list(range(10))})
    family = add_family(env.export, "valve_001", 500)
    # raw near-duplicate of one member (brightened) and a byte-identical raw copy of another
    member0 = cv2.imread(str(env.export / family[0]))
    cv2.imwrite(str(env.raw / "field_bright.png"), cv2.convertScaleAbs(member0, beta=12))
    shutil.copyfile(env.export / family[2], env.raw / "field_copy.jpg")
    cfg = env.build()
    rows = dedup_rows(cfg)
    imp, raw = by_rel(cfg, "import:legacy"), by_rel(cfg, "raw")
    group = {
        rows[imp[family[0]]].split_group,
        rows[imp[family[1]]].split_group,
        rows[raw["images/field_bright.png"]].split_group,
        rows[raw["images/field_copy.jpg"]].split_group,
    }
    assert len(group) == 1
    # the byte-identical copy is one item whose import reference carries the family key
    refs = [
        r
        for r in read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
        if r.source_id == raw["images/field_copy.jpg"]
    ]
    assert {(r.origin, r.family_key) for r in refs} == {
        ("raw", ""),
        ("import:legacy", "import:legacy:roboflow:valve_001"),
    }
    run_split(cfg)
    s = splits(cfg)
    assert (
        len(
            {
                s[i]
                for i in (
                    imp[family[0]],
                    raw["images/field_bright.png"],
                    raw["images/field_copy.jpg"],
                )
            }
        )
        == 1
    )


def test_families_apply_even_with_dedup_disabled(env) -> None:
    make_export(env.export, splits={"train": [0, 1]})
    family = add_family(env.export, "valve_001", 500)
    cfg = env.build({"dedup": {"enabled": False}})
    rows, ids = dedup_rows(cfg), by_rel(cfg, "import:legacy")
    assert len({rows[ids[r]].split_group for r in family}) == 1
    load_dataset_state(cfg)  # disabled-dedup consistency check accepts lineage groups
    assert run_split(cfg).exit_code == 0


def test_stale_lineage_is_detected(env) -> None:
    make_export(env.export, splits={"train": [0, 1]})
    add_family(env.export, "valve_001", 500)
    cfg = env.build()
    refs = read_records(ws(cfg).source_refs_manifest, SourceRefRecord)
    for r in refs:
        r.family_key = ""
    write_records(ws(cfg).source_refs_manifest, refs, SourceRefRecord)
    with pytest.raises(DatasetStateError, match="Lineage families"):
        load_dataset_state(cfg)


# --------------------------------------------------------------------------------------
# EXIF
# --------------------------------------------------------------------------------------


def with_exif_orientation(jpeg: bytes, value: int) -> bytes:
    tiff = (
        b"MM\x00\x2a\x00\x00\x00\x08"
        + b"\x00\x01"
        + struct.pack(">HHIHH", 0x0112, 3, 1, value, 0)
        + b"\x00\x00\x00\x00"
    )
    app1 = b"\xff\xe1" + struct.pack(">H", 2 + 6 + len(tiff)) + b"Exif\x00\x00" + tiff
    return jpeg[:2] + app1 + jpeg[2:]


def test_exif_reader() -> None:
    jpeg = cv2.imencode(".jpg", export_image(1)[0])[1].tobytes()
    assert exif_orientation(jpeg) == 0
    assert exif_orientation(with_exif_orientation(jpeg, 6)) == 6
    assert exif_orientation(cv2.imencode(".png", export_image(1)[0])[1].tobytes()) == 0
    assert exif_orientation(b"garbage") == 0


def test_exif_orientation_is_reported_not_blocking(env) -> None:
    paths = make_export(env.export, splits={"train": [0, 1, 2]})
    target = env.export / paths[1]
    target.write_bytes(with_exif_orientation(target.read_bytes(), 6))
    before = snapshot_tree(env.export)
    cfg = env.config()
    result = run_import(cfg, "legacy")
    assert result.created and result.record["counts"]["exif_rotated_images"] == 1
    rec = next(i for i in result.images if i.rel_path == paths[1])
    assert rec.exif_orientation == 6 and rec.label_status == "labeled"
    assert "EXIF orientation" in (ws(cfg).reports / "import_legacy.md").read_text()
    assert snapshot_tree(env.export) == before  # source not normalised or modified
    run_ingest(cfg)
    run_dedup(cfg)
