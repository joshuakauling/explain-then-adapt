#!/usr/bin/env python3
"""Migrate the final Reasoning Model validation resources."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.training.resource_migration import (
    migrate_reasoning_validation_resources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-augmented", type=Path, required=True)
    parser.add_argument("--training-task-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    counts = migrate_reasoning_validation_resources(
        validation_path=args.validation,
        augmented_validation_path=args.validation_augmented,
        training_task_manifest_path=args.training_task_manifest,
        output_directory=args.output_dir,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
