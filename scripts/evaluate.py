#!/usr/bin/env python3
"""Score a structured Prediction Model inference artifact."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.evaluation.config import load_evaluation_config
from explain_then_adapt.evaluation.scoring import evaluate_prediction_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_prediction_artifact(
        config=load_evaluation_config(args.config),
        prediction_path=args.predictions,
        tasks_directory=args.tasks_dir,
        output_directory=args.output_dir,
    )
    metrics = summary["metrics"]
    print(
        json.dumps(
            {
                "output_directory": str(args.output_dir),
                "tasks": summary["counts"]["tasks"],
                "candidates": summary["counts"]["candidates"],
                "thesis_solve": metrics["thesis_solve"],
                "all_test_inputs_solve": metrics["all_test_inputs_solve"],
                "sample_accuracy": metrics["sample_accuracy"],
                "parse_success": metrics["parse_success"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
