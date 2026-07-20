#!/usr/bin/env python3
"""Build an external token cache for one Prediction Model data profile."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.training.config import load_prediction_training_config
from explain_then_adapt.training.prediction_data import (
    build_prediction_token_cache,
    load_prediction_tokenizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        required=True,
        help="Directory with labelled ARC train and validation task JSON files.",
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=Path("resources/data_generation/task_manifest.jsonl"),
    )
    parser.add_argument("--rewrite-requests", type=Path, nargs="+")
    parser.add_argument("--rewrite-results", type=Path, nargs="+")
    parser.add_argument(
        "--rearc-tasks-dir",
        type=Path,
        help="External ReARC export; required only by the unguided_rearc profile.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("resources/training/reasoning_validation.jsonl"),
    )
    parser.add_argument(
        "--validation-augmented",
        type=Path,
        default=Path("resources/training/reasoning_validation_augmented.jsonl"),
    )
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_prediction_training_config(args.config, args.profile)
    tokenizer = load_prediction_tokenizer(config)
    manifest = build_prediction_token_cache(
        config=config,
        tokenizer=tokenizer,
        tasks_directory=args.tasks_dir,
        task_manifest_path=args.task_manifest,
        rewrite_request_paths=args.rewrite_requests or (),
        rewrite_result_paths=args.rewrite_results or (),
        rearc_tasks_directory=args.rearc_tasks_dir,
        validation_path=args.validation,
        augmented_validation_path=args.validation_augmented,
        output_cache_path=args.output_cache,
        output_manifest_path=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "profile": config.profile.name,
                "counts": manifest["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
