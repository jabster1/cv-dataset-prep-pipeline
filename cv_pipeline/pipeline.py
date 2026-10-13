"""Stage registry and orchestration for ``prepare``.

Each stage is an independent function ``(cfg, force) -> None`` that reads the previous
stage's manifest from the workspace and writes its own, so stages can also be run one
at a time from the CLI.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cv_pipeline.annotation import run_annotate_stage
from cv_pipeline.augmentation import run_augment_stage
from cv_pipeline.config import PipelineConfig
from cv_pipeline.dedup import run_dedup
from cv_pipeline.ingest import run_ingest
from cv_pipeline.splitting import run_split_stage
from cv_pipeline.utils.logging import get_logger, log_event

logger = get_logger("pipeline")


def _always(cfg: PipelineConfig) -> bool:
    return True


@dataclass(frozen=True)
class Stage:
    name: str
    help: str
    run: Callable[[PipelineConfig, bool], object]
    # Whether ``prepare`` includes the stage. Invoking a stage directly always runs it.
    enabled: Callable[[PipelineConfig], bool] = _always


def exit_code_of(result: object) -> int:
    """Stages may return a result with an ``exit_code`` (e.g. 3 = partial failures)."""
    return int(getattr(result, "exit_code", 0) or 0)


# Order matters: ``prepare`` runs these top to bottom. Later phases append export.
STAGES: list[Stage] = [
    Stage("ingest", "Copy images and sample video frames into the workspace", run_ingest),
    Stage("dedup", "Find exact and near-duplicate images; assign split groups", run_dedup),
    Stage(
        "annotate",
        "Pre-label eligible images with a detector; build the review queue",
        run_annotate_stage,
        enabled=lambda cfg: cfg.annotation.enabled,
    ),
    Stage(
        "split",
        "Assign split groups to train/val/test, preserving history (never resplits)",
        run_split_stage,
    ),
    Stage(
        "augment",
        "Augment trusted-labelled TRAIN items only (never val/test)",
        run_augment_stage,
        enabled=lambda cfg: cfg.augmentation.enabled,
    ),
]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; available: {[s.name for s in STAGES]}")


def run_prepare(cfg: PipelineConfig, force: bool = False) -> int:
    """Run enabled stages in order; returns the highest stage exit code (0 = success)."""
    stages = [s for s in STAGES if s.enabled(cfg)]
    skipped = [s.name for s in STAGES if not s.enabled(cfg)]
    log_event(
        logger, "prepare_started", project=cfg.project.name,
        stages=[s.name for s in stages], skipped=skipped,
    )
    code = 0
    for stage in stages:
        log_event(logger, "stage_started", stage=stage.name)
        code = max(code, exit_code_of(stage.run(cfg, force)))
        log_event(logger, "stage_finished", stage=stage.name)
    log_event(
        logger, "prepare_finished", project=cfg.project.name, workspace=str(cfg.workspace_dir)
    )
    return code
