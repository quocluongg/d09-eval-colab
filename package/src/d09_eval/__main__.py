"""CLI entry point for d09_eval."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from d09_eval.pack import pack_outputs
from d09_eval.runner import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_REFERENCE_PATH,
    run_models,
)
from d09_eval.score import score_predictions
from d09_eval.selfcheck import run_selfcheck


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d09_eval", description="D09 model evaluation runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_self = sub.add_parser("selfcheck", help="Retrain baseline LR and compare against reference")
    p_self.add_argument("--data", required=True, type=Path, help="Input data directory")
    p_self.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path, help="Experiment config JSON")
    p_self.add_argument("--reference", default=DEFAULT_REFERENCE_PATH, type=Path, help="Reference JSON")

    p_run = sub.add_parser("run", help="Train models on train split and write validation predictions")
    p_run.add_argument("--data", required=True, type=Path, help="Input data directory")
    p_run.add_argument("--config", default=DEFAULT_CONFIG_PATH, type=Path, help="Experiment config JSON")
    p_run.add_argument("--out", required=True, type=Path, help="Output directory")
    p_run.add_argument("--split", default="validation", choices=["validation", "test"], help="Target evaluation split")
    p_run.add_argument("--allow-test", action="store_true", help="Unlock test split evaluation when TEST_UNLOCK.txt is present")
    p_run.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to run")
    p_run.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional subset of seeds to run")

    p_pack = sub.add_parser("pack", help="Package output folder with a MANIFEST of sha256 digests")
    p_pack.add_argument("--out", required=True, type=Path, help="Output directory to package")
    p_pack.add_argument("--zip", default=None, type=Path, help="Optional output zip file path")

    p_score = sub.add_parser("score", help="Designer scoring command against truth file")
    p_score.add_argument("--pred", required=True, type=Path, help="Predictions root directory")
    p_score.add_argument("--truth", required=True, type=Path, help="Truth CSV file path")
    p_score.add_argument("--extra-truth", default=None, type=Path, help="Optional secondary truth/decisions CSV for target column")
    p_score.add_argument("--bootstrap", default=1000, type=int, help="Number of grouped bootstrap resamples")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    try:
        if args.command == "selfcheck":
            return run_selfcheck(args.data, args.config, args.reference)
        if args.command == "run":
            return run_models(
                args.data,
                args.config,
                args.out,
                target_split=args.split,
                allow_test=bool(args.allow_test),
                only_models=args.models,
                only_seeds=args.seeds,
            )
        if args.command == "pack":
            pack_outputs(args.out, args.zip)
            return 0
        if args.command == "score":
            score_predictions(
                args.pred,
                args.truth,
                extra_truth_path=args.extra_truth,
                n_bootstrap=int(args.bootstrap),
            )
            return 0
    except ImportError:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
