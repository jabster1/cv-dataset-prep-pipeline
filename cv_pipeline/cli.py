"""Command-line interface: ``python -m cv_pipeline <command> --config <file>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from cv_pipeline import __version__
from cv_pipeline.config import ConfigError, PipelineConfig, load_config
from cv_pipeline.pipeline import STAGES, get_stage, run_prepare
from cv_pipeline.utils.io import atomic_write_text
from cv_pipeline.utils.logging import get_logger, setup_logging
from cv_pipeline.workspace import Workspace

logger = get_logger("cli")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="Path to the pipeline YAML config")
    p.add_argument("--log-level", default=None, help="Override console log level (DEBUG, INFO, ...)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv_pipeline",
        description="Turn raw images/videos into a clean, leak-free YOLO dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Run every dataset-preparation stage in order")
    _add_common(p)
    p.add_argument("--force", action="store_true", help="Recompute outputs instead of reusing them")
    p.set_defaults(handler=_cmd_prepare)

    for stage in STAGES:
        p = sub.add_parser(stage.name, help=stage.help)
        _add_common(p)
        p.add_argument("--force", action="store_true", help="Recompute outputs instead of reusing them")
        p.set_defaults(handler=_cmd_stage, stage=stage.name)

    p = sub.add_parser("show-config", help="Print the fully resolved configuration")
    _add_common(p)
    p.set_defaults(handler=_cmd_show_config)

    return parser


def _cmd_prepare(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    run_prepare(cfg, force=args.force)
    return 0


def _cmd_stage(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    get_stage(args.stage).run(cfg, args.force)
    return 0


def _cmd_show_config(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    sys.stdout.write(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.handler is _cmd_show_config:
        return _cmd_show_config(cfg, args)

    ws = Workspace(cfg.workspace_dir).ensure()
    setup_logging(args.log_level or cfg.log_level, log_file=ws.logs / "pipeline.jsonl")
    # Snapshot of exactly what this run used, for lineage/reproducibility.
    atomic_write_text(ws.resolved_config, yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    try:
        return args.handler(cfg, args)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
