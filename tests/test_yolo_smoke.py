"""Optional Ultralytics integration smoke test — skipped by default.

    CV_PIPELINE_SMOKE_MODEL=~/.cache/cv-pipeline-smoke/yolo11n.pt pytest tests/test_yolo_smoke.py

This only checks that the generic YOLO backend integrates correctly (loading, schema,
class mapping, caching, CPU determinism) using a stock COCO model on Ultralytics' bundled
sample images. It says NOTHING about annotation quality for LOTO or any other project.
Weights are never downloaded by this test or by the pipeline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from cv_pipeline.annotation import run_annotate
from cv_pipeline.annotation.schema import BoxDecision, InferenceStatus
from cv_pipeline.config import load_config
from cv_pipeline.dedup import run_dedup
from cv_pipeline.ingest import run_ingest
from cv_pipeline.utils.hashing import sha256_file
from cv_pipeline.workspace import Workspace

MODEL = Path(os.environ.get("CV_PIPELINE_SMOKE_MODEL", "")).expanduser()

pytestmark = pytest.mark.skipif(
    not os.environ.get("CV_PIPELINE_SMOKE_MODEL") or not MODEL.is_file(),
    reason="set CV_PIPELINE_SMOKE_MODEL to a local Ultralytics .pt file to run",
)


def test_yolo_backend_integration(tmp_path, write_config) -> None:
    import ultralytics

    assets = Path(ultralytics.__file__).parent / "assets"
    raw = tmp_path / "raw" / "images"
    raw.mkdir(parents=True)
    for name in ("bus.jpg", "zidane.jpg"):
        shutil.copyfile(assets / name, raw / name)

    cfg = load_config(
        write_config(
            {
                "classes": ["person", "vehicle"],
                "annotation": {
                    "enabled": True,
                    "backend": "yolo",
                    "model": str(MODEL),
                    "device": "cpu",
                    "class_map": {"bus": "vehicle", "car": "vehicle"},
                    "review_sample_rate": 0.0,
                },
            }
        )
    )
    run_ingest(cfg)
    run_dedup(cfg)

    first = run_annotate(cfg)
    assert first.exit_code == 0
    assert {r.inference_status for r in first.items} == {InferenceStatus.OK}
    assert {r.backend for r in first.items} == {"ultralytics_yolo"}
    assert {r.model_sha256 for r in first.items} == {sha256_file(MODEL)}
    assert {r.model_id for r in first.items} == {f"external:{MODEL.name}"}  # no abs path
    candidates = [p for p in first.predictions if p.decision == BoxDecision.CANDIDATE]
    assert {p.class_name for p in candidates} <= {"person", "vehicle"}
    assert candidates, "expected at least one mapped detection on the sample images"
    for p in candidates:
        assert 0 <= p.x1 < p.x2 and 0 <= p.y1 < p.y2  # type: ignore[operator]

    model_meta = Workspace(cfg.workspace_dir).model_cache / f"{sha256_file(MODEL)}.json"
    assert model_meta.is_file()

    cached = run_annotate(cfg)
    assert cached.run_id == first.run_id and cached.inferred == 0

    forced = run_annotate(cfg, force=True)
    assert forced.inferred == len(first.items)
    # On CPU we expect bit-for-bit reproducibility; if this ever fails, the new run is
    # preserved alongside the old one and the differences are reported.
    assert forced.forced_changes == [] and forced.run_id == first.run_id
