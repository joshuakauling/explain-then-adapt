#!/usr/bin/env python3
"""Migrate canonical reasoning-generation metadata from legacy inputs."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.data_generation.resource_migration import (
    migrate_reasoning_resources,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", type=Path, required=True)
    parser.add_argument("--legacy-task-ids", type=Path, required=True)
    parser.add_argument("--hints-dir", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    counts = migrate_reasoning_resources(
        task_ids_path=args.task_ids,
        legacy_task_ids_path=args.legacy_task_ids,
        hints_directory=args.hints_dir,
        traces_path=args.traces,
        output_directory=args.output_dir,
    )
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
