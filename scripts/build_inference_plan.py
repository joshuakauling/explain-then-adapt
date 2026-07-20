#!/usr/bin/env python3
"""Build the deterministic 64-variant plan shared by RM, TTT, and PM."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.inference.config import load_inference_config
from explain_then_adapt.inference.planning import (
    create_augmentation_plan,
    save_augmentation_plan,
)
from explain_then_adapt.training.ttt_data import load_ttt_task_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="JSON task-ID list or JSONL task manifest.",
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_inference_config(args.config)
    task_ids = load_ttt_task_ids(args.tasks)
    plan = create_augmentation_plan(
        config=config,
        task_ids=task_ids,
        tasks_directory=args.tasks_dir,
    )
    save_augmentation_plan(
        config=config,
        plan=plan,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "task_count": len(plan),
                "variants_per_task": 64,
                "augmentation_seed": config.augmentation_seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
