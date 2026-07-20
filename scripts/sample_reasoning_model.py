#!/usr/bin/env python3
"""Sample one Reasoning Model trace for every required inference task view."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.inference.config import (
    GUIDANCE_BUDGETS,
    INFERENCE_PROTOCOLS,
    load_inference_config,
)
from explain_then_adapt.inference.planning import (
    load_augmentation_plan,
    protocol_needs_augmentation_plan,
)
from explain_then_adapt.inference.vllm_runner import run_reasoning_inference
from explain_then_adapt.training.ttt_data import load_ttt_task_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", choices=INFERENCE_PROTOCOLS, required=True)
    parser.add_argument(
        "--guidance-budget",
        type=int,
        choices=GUIDANCE_BUDGETS,
        help="Required only for budgeted64.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        required=True,
        help="JSON task-ID list or JSONL task manifest.",
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument(
        "--augmentation-plan",
        type=Path,
        help="Required by augmented64 and budgeted64 with k > 0.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Merged Reasoning Model path or Hugging Face identifier.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--request-batch-size", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_inference_config(args.config)
    task_ids = load_ttt_task_ids(args.tasks)
    needs_plan = protocol_needs_augmentation_plan(
        args.protocol,
        args.guidance_budget,
    )
    if needs_plan and args.augmentation_plan is None:
        raise ValueError(
            f"{args.protocol} requires --augmentation-plan for this budget."
        )
    if not needs_plan and args.augmentation_plan is not None:
        raise ValueError(
            f"{args.protocol} does not use an augmentation plan for this budget."
        )
    plan = None
    if args.augmentation_plan is not None:
        plan = load_augmentation_plan(
            config=config,
            path=args.augmentation_plan,
            task_ids=task_ids,
            tasks_directory=args.tasks_dir,
        )

    summary = run_reasoning_inference(
        config=config,
        task_ids=task_ids,
        tasks_directory=args.tasks_dir,
        protocol=args.protocol,
        guidance_budget=args.guidance_budget,
        augmentation_plan=plan,
        augmentation_plan_path=args.augmentation_plan,
        model=args.model,
        output_path=args.output,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        request_batch_size=args.request_batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol": summary["protocol"],
                "request_count": summary["request_count"],
                "valid_guidance_count": summary["valid_guidance_count"],
                "total_prompt_tokens": summary["total_prompt_tokens"],
                "total_generated_tokens": summary["total_generated_tokens"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
