import json
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.evaluation.config import load_evaluation_config
from explain_then_adapt.evaluation.cost import summarize_compute
from explain_then_adapt.inference.artifacts import JsonlArtifactWriter

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "evaluation" / "evaluation.yaml"
TASK_HASH = "a" * 64


def _write_reasoning(path):
    with JsonlArtifactWriter(path) as writer:
        writer.write({"kind": "dummy"})
        writer.complete(
            {
                "schema_version": 1,
                "kind": "reasoning_inference_run",
                "task_count": 1,
                "task_ids": ["abc123"],
                "task_sources_sha256": TASK_HASH,
                "total_prompt_tokens": 100,
                "total_generated_tokens": 200,
            }
        )
    return json.loads(path.with_name(f"{path.stem}.manifest.json").read_text())


def _write_predictions(path, guidance_sha):
    with JsonlArtifactWriter(path) as writer:
        writer.write({"kind": "dummy"})
        writer.complete(
            {
                "schema_version": 1,
                "kind": "prediction_inference_run",
                "protocol": "budgeted64",
                "guidance_mode": "guided",
                "guidance_budget": 8,
                "task_count": 1,
                "task_ids": ["abc123"],
                "task_sources_sha256": TASK_HASH,
                "guidance_artifact": {
                    "path": "reasoning.jsonl",
                    "sha256": guidance_sha,
                },
                "ttt_enabled": True,
                "total_prompt_tokens": 300,
                "total_generated_tokens": 400,
            }
        )


class ComputeSummaryTests(unittest.TestCase):
    def test_combines_rm_ttt_and_pm_tokens_without_double_counting(self) -> None:
        config = load_evaluation_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reasoning = root / "reasoning.jsonl"
            reasoning_manifest = _write_reasoning(reasoning)
            predictions = root / "predictions.jsonl"
            _write_predictions(predictions, reasoning_manifest["output_sha256"])
            ttt_run = root / "ttt"
            ttt_run.mkdir()
            (ttt_run / "summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "ttt_run",
                        "guidance_mode": "guided",
                        "guidance_budget": 8,
                        "task_count": 1,
                        "completed_task_count": 1,
                        "completed_tasks": ["abc123"],
                        "processed_tokens": 300,
                    }
                ),
                encoding="utf-8",
            )
            (ttt_run / "resolved_config.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "tasks_sha256": TASK_HASH,
                            "guidance_sha256": reasoning_manifest["output_sha256"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_compute(
                config=config,
                prediction_path=predictions,
                reasoning_paths=[reasoning],
                ttt_run_directory=ttt_run,
            )

            self.assertEqual(summary["tokens"]["prefill_total"], 400)
            self.assertEqual(summary["tokens"]["generated_total"], 600)
            self.assertEqual(summary["tokens"]["ttt_train"], 300)
            self.assertAlmostEqual(summary["seconds_equivalent"]["prefill"], 0.08)
            self.assertAlmostEqual(summary["seconds_equivalent"]["training"], 0.18)
            self.assertAlmostEqual(summary["seconds_equivalent"]["decode"], 8.0)
            self.assertAlmostEqual(summary["seconds_equivalent"]["total"], 8.26)

    def test_requires_every_provenance_linked_reasoning_artifact(self) -> None:
        config = load_evaluation_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reasoning = root / "reasoning.jsonl"
            reasoning_manifest = _write_reasoning(reasoning)
            predictions = root / "predictions.jsonl"
            _write_predictions(predictions, reasoning_manifest["output_sha256"])

            with self.assertRaisesRegex(ValueError, "requires --ttt-run"):
                summarize_compute(
                    config=config,
                    prediction_path=predictions,
                    reasoning_paths=[reasoning],
                )


if __name__ == "__main__":
    unittest.main()
