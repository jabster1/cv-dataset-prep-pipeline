"""Command-line interface: ``python -m cv_pipeline <command> --config <file>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import yaml

from cv_pipeline import __version__
from cv_pipeline.annotation import AnnotationSetupError, run_annotate
from cv_pipeline.augmentation import AugmentationError, run_augment
from cv_pipeline.config import ConfigError, PipelineConfig, load_config
from cv_pipeline.dataset_state import DatasetStateError
from cv_pipeline.imports import ImportError_, run_import
from cv_pipeline.labels import LabelPolicy, run_labels
from cv_pipeline.pipeline import STAGES, exit_code_of, get_stage, run_prepare
from cv_pipeline.splitting import SplitError, SplitHistoryError, run_split
from cv_pipeline.utils.io import atomic_write_text
from cv_pipeline.utils.logging import get_logger, setup_logging
from cv_pipeline.workspace import Workspace

logger = get_logger("cli")

EPILOG = """exit codes:
  0  success
  1  stage could not run (missing inputs, missing/invalid model, ...)
  2  invalid configuration
  3  annotation completed and outputs were written, but one or more items failed inference
  4  split-history conflict: nothing in splits/ was changed (see reports/split.md)
"""


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="Path to the pipeline YAML config")
    p.add_argument("--log-level", default=None, help="Override console log level (DEBUG, INFO, ...)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cv_pipeline",
        description="Turn raw images/videos into a clean, leak-free YOLO dataset.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        if stage.name != "split":  # split never recomputes history; see --resplit
            p.add_argument(
                "--force", action="store_true", help="Recompute outputs instead of reusing them"
            )
        p.set_defaults(handler=_cmd_stage, stage=stage.name, force=False)
        if stage.name == "annotate":
            p.add_argument(
                "--limit", type=int, default=None,
                help="Only annotate the first N eligible items (by item_id), e.g. for smoke tests",
            )
        if stage.name == "augment":
            p.add_argument(
                "--dry-run", action="store_true",
                help="Check eligibility and compute the run identity; generate nothing",
            )
        if stage.name == "split":
            p.add_argument(
                "--dry-run", action="store_true", help="Compute and report; write nothing"
            )
            p.add_argument(
                "--resplit",
                action="store_true",
                help="Deliberately rebuild ALL assignments from scratch (previews unless "
                "--confirm matches the current history token). Never used by prepare.",
            )
            p.add_argument("--reason", default=None, help="Required with --resplit")
            p.add_argument("--confirm", default=None, metavar="TOKEN", help="Resplit token")

    p = sub.add_parser(
        "import-dataset",
        help="Explicitly import an existing labeled YOLO/Roboflow dataset listed under `imports`",
    )
    _add_common(p)
    p.add_argument("--name", required=True, help="Name of the entry in the config's imports list")
    p.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    p.set_defaults(handler=_cmd_import)

    p = sub.add_parser(
        "labels",
        help="Resolve the label for every eligible item (trusted human labels only by default)",
    )
    _add_common(p)
    p.add_argument("--allow-machine-unreviewed", action="store_true",
                   help="Opt in: use unreviewed machine labels (development/POC only)")
    p.add_argument("--allow-imported-unverified", action="store_true",
                   help="Opt in: use imported labels whose import has labels_trusted: false")
    p.add_argument("--allow-machine-empty", action="store_true",
                   help="Opt in: treat machine zero-detection results as negatives")
    p.set_defaults(handler=_cmd_labels)

    p = sub.add_parser("show-config", help="Print the fully resolved configuration")
    _add_common(p)
    p.set_defaults(handler=_cmd_show_config)

    return parser


def _cmd_prepare(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    return run_prepare(cfg, force=args.force)


def _cmd_stage(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    if args.stage == "annotate":
        if args.limit is not None and args.limit < 1:
            raise SystemExit("error: --limit must be >= 1")
        return exit_code_of(run_annotate(cfg, force=args.force, limit=args.limit))
    if args.stage == "augment":
        return run_augment(cfg, dry_run=args.dry_run).exit_code
    if args.stage == "split":
        if (args.reason or args.confirm) and not args.resplit:
            raise SystemExit("error: --reason/--confirm are only valid with --resplit")
        result = run_split(
            cfg,
            dry_run=args.dry_run,
            resplit=args.resplit,
            reason=args.reason,
            confirm=args.confirm,
        )
        if result.resplit_token is not None and not result.written:
            _print_resplit_preview(result)
        return result.exit_code
    return exit_code_of(get_stage(args.stage).run(cfg, args.force))


def _cmd_import(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    result = run_import(cfg, args.name, dry_run=args.dry_run)
    c = result.record["counts"]
    state = "validated (dry run)" if result.dry_run else ("imported" if result.created else "unchanged")
    print(f"{args.name}: {state} — {c['images']} images, {c['boxes']} boxes, "
          f"labels {c['label_status']}, legacy splits {c['legacy_split']}")
    return 0


def _cmd_labels(cfg: PipelineConfig, args: argparse.Namespace) -> int:
    policy = LabelPolicy(
        allow_machine_unreviewed=args.allow_machine_unreviewed,
        allow_imported_unverified=args.allow_imported_unverified,
        allow_machine_empty=args.allow_machine_empty,
    )
    return run_labels(cfg, policy).exit_code


def _print_resplit_preview(result) -> None:  # type: ignore[no-untyped-def]
    report = result.report
    print("RESPLIT PREVIEW — nothing has been written.")
    for key, value in (report.get("resplit_movement") or {}).items():
        print(f"  {key}: {value} items")
    for w in report.get("warnings", []):
        print(f"  WARNING: {w}")
    print(f"To apply, re-run with: --resplit --reason \"...\" --confirm {result.resplit_token}")


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
    except ConfigError as exc:
        logger.error(str(exc))
        return 2
    except (
        FileNotFoundError,
        AnnotationSetupError,
        DatasetStateError,
        SplitHistoryError,
        SplitError,
        ImportError_,
        AugmentationError,
    ) as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
