#!/usr/bin/env python3
"""Train one independent Test-Time Training adapter per selected ARC task."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.training.config import (
    TTT_GUIDANCE_BUDGETS,
    load_ttt_training_config,
)
from explain_then_adapt.training.ttt_data import load_ttt_task_ids
from explain_then_adapt.training.ttt_trainer import train_ttt_adapters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("guided", "unguided"), required=True)
    parser.add_argument(
        "--guidance-budget",
        type=int,
        choices=TTT_GUIDANCE_BUDGETS,
        help="Override the profile default with 0, 8, 16, 32, or 64 traces.",
    )
    parser.add_argument(
        "--missing-guidance-policy",
        choices=("error", "omit_system"),
        help="Use omit_system only to reproduce historical missing RM outputs.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="JSON task-ID list or JSONL task manifest.",
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-model",
        type=Path,
        required=True,
        help="Standalone Prediction Model with its offline adapter already merged.",
    )
    parser.add_argument(
        "--guidance",
        type=Path,
        help="RM JSONL artifact or legacy JSON mapping with augmented guidance.",
    )
    parser.add_argument(
        "--augmentation-plan",
        type=Path,
        help="Optional structured or historical augmentation plan to replay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete task adapters in an existing matching run.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_ttt_training_config(
        args.config,
        args.profile,
        guidance_budget=args.guidance_budget,
        missing_guidance_policy=args.missing_guidance_policy,
    )
    summary = train_ttt_adapters(
        config=config,
        tasks_directory=args.tasks_dir,
        task_ids=load_ttt_task_ids(args.tasks),
        prediction_model_path=args.prediction_model,
        output_directory=args.output_dir,
        run_name=args.run_name,
        guidance_path=args.guidance,
        augmentation_plan_path=args.augmentation_plan,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
