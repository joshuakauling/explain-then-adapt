import tempfile
import unittest
from pathlib import Path

import torch

from explain_then_adapt.training.model_merge import (
    merge_lora_adapter,
    torch_dtype,
)


class ModelMergeTests(unittest.TestCase):
    def test_dtype_mapping(self) -> None:
        self.assertIs(torch_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(torch_dtype("float16"), torch.float16)
        self.assertIs(torch_dtype("float32"), torch.float32)
        with self.assertRaisesRegex(ValueError, "dtype must be"):
            torch_dtype("invalid")

    def test_missing_adapter_fails_before_loading_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            with self.assertRaisesRegex(FileNotFoundError, "adapter directory"):
                merge_lora_adapter(
                    base_model="Qwen/example",
                    adapter_path=directory / "missing",
                    output_directory=directory / "merged",
                    dtype="bfloat16",
                    device_map="cpu",
                    max_shard_size="2GB",
                    trust_remote_code=False,
                )


if __name__ == "__main__":
    unittest.main()
