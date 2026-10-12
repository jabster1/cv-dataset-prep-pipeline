from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from cv_pipeline.config import PipelineConfig, load_config
from cv_pipeline.synthetic import SYNTHETIC_CLASSES, write_synthetic_raw
from cv_pipeline.utils.logging import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _quiet_logging() -> None:
    setup_logging("WARNING")


def base_config_dict() -> dict[str, Any]:
    return {
        "root": ".",
        "project": {"name": "test", "workspace": "work", "seed": 7},
        "classes": list(SYNTHETIC_CLASSES),
        "ingest": {
            "raw_dir": "raw",
            "video": {"mode": "every_n_frames", "every_n_frames": 5, "max_frames": None},
        },
        "annotation": {"enabled": False},
    }


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Write a config YAML into ``tmp_path`` (with overrides) and return its path."""

    def _write(overrides: dict[str, Any] | None = None, name: str = "config.yaml") -> Path:
        data = deep_update(base_config_dict(), overrides or {})
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def synthetic_project(tmp_path: Path, write_config: Callable[..., Path]) -> Callable[..., PipelineConfig]:
    """Synthetic raw data (12 images, 2 videos x 40 frames) + loaded config."""

    def _make(overrides: dict[str, Any] | None = None) -> PipelineConfig:
        raw = tmp_path / "raw"
        if not raw.exists():
            write_synthetic_raw(raw, n_images=12, n_videos=2, video_frames=40, video_fps=10.0, seed=3)
        return load_config(write_config(overrides))

    return _make


def snapshot_tree(root: Path) -> dict[str, tuple[int, int, str]]:
    """(size, mtime_ns, sha256) for every file under ``root`` — to prove immutability."""
    snap = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            snap[p.relative_to(root).as_posix()] = (
                st.st_size,
                st.st_mtime_ns,
                hashlib.sha256(p.read_bytes()).hexdigest(),
            )
    return snap


@pytest.fixture
def scripted():
    """Register the deterministic ``scripted`` annotation backend for one test."""
    from cv_pipeline.annotation.backend import register_backend, unregister_backend

    from .fakes import Script, ScriptedBackend

    script = Script()
    register_backend("scripted")(lambda cfg, ctx: ScriptedBackend(script, cfg, ctx))
    yield script
    unregister_backend("scripted")
