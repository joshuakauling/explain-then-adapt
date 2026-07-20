#!/usr/bin/env python3
"""Train one thesis-aligned Prediction Model QLoRA profile."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.training.config import load_prediction_training_config
from explain_then_adapt.training.prediction_trainer import train_prediction_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="Merged Unguided-ReARC model required by guided_rearc.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train_prediction_model(
        config=load_prediction_training_config(args.config, args.profile),
        cache_path=args.cache,
        output_directory=args.output_dir,
        run_name=args.run_name,
        use_wandb=not args.no_wandb,
        initial_model_path=args.initial_model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
