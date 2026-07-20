import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.arc.augmented_keys import make_augmented_key
from explain_then_adapt.arc.formatting import format_grid_to_string
from explain_then_adapt.arc.transforms import transform_individual_grid
from explain_then_adapt.evaluation.config import load_evaluation_config
from explain_then_adapt.evaluation.scoring import evaluate_prediction_artifact
from explain_then_adapt.inference.artifacts import (
    JsonlArtifactWriter,
    read_jsonl,
    task_sources_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "evaluation" / "evaluation.yaml"
IDENTITY_MAPPING = "0123456789"


def _output(sample_index, text):
    return {
        "sample_index": sample_index,
        "text": text,
        "generated_token_count": len(text),
        "finish_reason": "stop",
        "stop_reason": None,
    }


def _variant(
    task_id,
    *,
    key=None,
    code="ID",
    mapping=IDENTITY_MAPPING,
    order_mapping=None,
    variant_index=None,
):
    return {
        "key": key or task_id,
        "task_id": task_id,
        "is_augmented": variant_index is not None,
        "transformation_code": code,
        "value_mapping": mapping,
        "order_mapping": order_mapping,
        "variant_index": variant_index,
    }


def _record(task_id, test_index, variant, outputs, protocol="standard32"):
    return {
        "schema_version": 1,
        "kind": "prediction_candidates",
        "request_id": f"{variant['key']}__{test_index}",
        "task_id": task_id,
        "test_index": test_index,
        "variant": variant,
        "protocol": protocol,
        "guidance_mode": "unguided",
        "guidance_key": None,
        "adapter_used": False,
        "prompt_sha256": "0" * 64,
        "prompt_token_count": 10,
        "sample_count": len(outputs),
        "outputs": outputs,
    }


def _write_prediction_artifact(path, tasks_directory, task_id, records, protocol):
    with JsonlArtifactWriter(path) as writer:
        for record in records:
            writer.write(record)
        writer.complete(
            {
                "schema_version": 1,
                "kind": "prediction_inference_run",
                "protocol": protocol,
                "guidance_mode": "unguided",
                "guidance_budget": None,
                "model": "fake-pm",
                "task_count": 1,
                "task_ids": [task_id],
                "task_sources_sha256": task_sources_sha256(
                    tasks_directory,
                    [task_id],
                ),
                "augmentation_plan": {"path": None, "sha256": None},
                "guidance_artifact": {"path": None, "sha256": None},
                "ttt_enabled": False,
                "ttt_adapter_root": None,
                "request_count": len(records),
                "samples_per_prompt": len(records[0]["outputs"]),
                "candidates_per_test_input": sum(
                    len(record["outputs"])
                    for record in records
                    if record["test_index"] == 0
                ),
                "total_candidates": sum(
                    len(record["outputs"]) for record in records
                ),
                "total_prompt_tokens": 10 * len(records),
                "total_generated_tokens": 10,
                "outputs_are_in_variant_space": True,
            }
        )


class PredictionScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_evaluation_config(CONFIG_PATH)

    def test_reports_thesis_and_all_test_input_solve_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = root / "tasks"
            tasks.mkdir()
            task_id = "abc123"
            (tasks / f"{task_id}.json").write_text(
                json.dumps(
                    {
                        "train": [{"input": [[0]], "output": [[1]]}],
                        "test": [
                            {"input": [[0]], "output": [[1, 2]]},
                            {"input": [[0]], "output": [[3], [4]]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            variant = _variant(task_id)
            records = [
                _record(
                    task_id,
                    0,
                    variant,
                    [_output(0, "12"), _output(1, "00")],
                ),
                _record(
                    task_id,
                    1,
                    variant,
                    [_output(0, "0"), _output(1, "not a grid")],
                ),
            ]
            predictions = root / "predictions.jsonl"
            _write_prediction_artifact(
                predictions,
                tasks,
                task_id,
                records,
                "standard32",
            )

            summary = evaluate_prediction_artifact(
                config=self.config,
                prediction_path=predictions,
                tasks_directory=tasks,
                output_directory=root / "evaluation",
            )

            self.assertEqual(summary["metrics"]["thesis_solve"]["rate"], 1.0)
            self.assertEqual(
                summary["metrics"]["all_test_inputs_solve"]["rate"],
                0.0,
            )
            self.assertEqual(summary["metrics"]["sample_accuracy"]["rate"], 0.25)
            self.assertEqual(summary["metrics"]["parse_success"]["rate"], 0.75)
            report = (root / "evaluation" / "report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("| Thesis Solve | 1 | 1 | 100.00% |", report)
            self.assertIn("| All-Test-Inputs Solve | 0 | 1 | 0.00% |", report)

    def test_inverse_transforms_augmented_predictions_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tasks = root / "tasks"
            tasks.mkdir()
            task_id = "def456"
            expected = [[1, 2, 3], [4, 5, 6]]
            (tasks / f"{task_id}.json").write_text(
                json.dumps(
                    {
                        "train": [
                            {"input": [[0]], "output": [[0]]},
                            {"input": [[1]], "output": [[1]]},
                        ],
                        "test": [{"input": [[0]], "output": expected}],
                    }
                ),
                encoding="utf-8",
            )
            code = "R90"
            mapping = "2479185036"
            order_mapping = "10"
            key = make_augmented_key(task_id, code, mapping, order_mapping)
            variant = _variant(
                task_id,
                key=key,
                code=code,
                mapping=mapping,
                order_mapping=order_mapping,
                variant_index=0,
            )
            transformed = transform_individual_grid(expected, code, mapping)
            records = [
                _record(
                    task_id,
                    0,
                    variant,
                    [_output(0, format_grid_to_string(transformed, delimiter=""))],
                    protocol="augmented64",
                )
            ]
            predictions = root / "predictions.jsonl"
            _write_prediction_artifact(
                predictions,
                tasks,
                task_id,
                records,
                "augmented64",
            )

            summary = evaluate_prediction_artifact(
                config=self.config,
                prediction_path=predictions,
                tasks_directory=tasks,
                output_directory=root / "evaluation",
            )
            candidates = list(read_jsonl(root / "evaluation" / "candidates.jsonl"))

            self.assertEqual(summary["metrics"]["sample_accuracy"]["rate"], 1.0)
            self.assertEqual(candidates[0]["prediction_original_space"], expected)
            self.assertTrue(candidates[0]["is_correct"])


if __name__ == "__main__":
    unittest.main()
