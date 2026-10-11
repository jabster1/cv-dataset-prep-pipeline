"""Stage registry and orchestration for ``prepare``.

Each stage is an independent function ``(cfg, force) -> None`` that reads the previous
stage's manifest from the workspace and writes its own, so stages can also be run one
at a time from the CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cv_pipeline.config import PipelineConfig
from cv_pipeline.ingest import run_ingest
from cv_pipeline.utils.logging import get_logger, log_event

logger = get_logger("pipeline")


@dataclass(frozen=True)
class Stage:
    name: str
    help: str
    run: Callable[[PipelineConfig, bool], object]


# Order matters: ``prepare`` runs these top to bottom. Later phases append
# dedup -> annotate -> split -> augment -> export.
STAGES: list[Stage] = [
    Stage("ingest", "Copy images and sample video frames into the workspace", run_ingest),
]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; available: {[s.name for s in STAGES]}")


def run_prepare(cfg: PipelineConfig, force: bool = False) -> None:
    log_event(logger, "prepare_started", project=cfg.project.name, stages=[s.name for s in STAGES])
    for stage in STAGES:
        log_event(logger, "stage_started", stage=stage.name)
        stage.run(cfg, force)
        log_event(logger, "stage_finished", stage=stage.name)
    log_event(logger, "prepare_finished", project=cfg.project.name, workspace=str(cfg.workspace_dir))
