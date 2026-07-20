#!/usr/bin/env python3
"""Merge a LoRA checkpoint into a standalone base model directory."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from explain_then_adapt.training.model_merge import merge_lora_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = merge_lora_adapter(
        base_model=args.base_model,
        adapter_path=args.adapter,
        output_directory=args.output_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
