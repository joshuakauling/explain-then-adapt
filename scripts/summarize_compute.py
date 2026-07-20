#!/usr/bin/env python3
"""Calculate the thesis seconds-equivalent cost of one inference run."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.evaluation.artifacts import atomic_write_json
from explain_then_adapt.evaluation.config import load_evaluation_config
from explain_then_adapt.evaluation.cost import summarize_compute


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--reasoning",
        type=Path,
        action="append",
        default=[],
        help=(
            "RM JSONL artifact used by PM inference or TTT; repeat when the "
            "two stages used different artifacts."
        ),
    )
    parser.add_argument(
        "--ttt-run",
        type=Path,
        help="TTT run directory containing summary.json and resolved_config.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_compute(
        config=load_evaluation_config(args.config),
        prediction_path=args.predictions,
        reasoning_paths=args.reasoning,
        ttt_run_directory=args.ttt_run,
    )
    atomic_write_json(args.output, summary)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hours_equivalent": summary["hours_equivalent"],
                "seconds_equivalent": summary["seconds_equivalent"]["total"],
                "tokens": summary["tokens"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
