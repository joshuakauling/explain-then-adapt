import json
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.data_generation.cli import main
from explain_then_adapt.data_generation.pipeline import build_judge_requests
from explain_then_adapt.data_generation.records import (
    GenerationResult,
    GenerationStage,
    HintMode,
    read_requests,
    read_results,
    write_jsonl,
    write_requests,
    write_results,
)


TRACE = """<think>
1) INPUT ANALYSIS
x
2) OUTPUT ANALYSIS
x
3) TRANSFORMATION ANALYSIS
x
4) STEPS FOR THE TRANSFORMATION
x
</think>
General natural language description:
x
General steps:
x"""


def write_task(directory: Path, task_id: str) -> None:
    task = {
        "train": [
            {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
            {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
        ],
        "test": [{"input": [[3, 0], [0, 0]]}],
    }
    (directory / f"{task_id}.json").write_text(json.dumps(task), encoding="utf-8")


class CliTests(unittest.TestCase):
    def test_prepare_initial_uses_explicit_inputs_and_records_hint_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            for task_id in ("target", "few_a", "few_b", "few_c"):
                write_task(tasks, task_id)
            task_ids = root / "task_ids.json"
            task_ids.write_text(json.dumps(["target"]), encoding="utf-8")
            traces = root / "traces.json"
            traces.write_text(
                json.dumps(
                    {task_id: TRACE for task_id in ("few_a", "few_b", "few_c")}
                ),
                encoding="utf-8",
            )
            manifest = root / "few_shots.json"
            manifest.write_text(
                json.dumps(
                    [
                        {"task_id": task_id, "trace_path": "traces.json"}
                        for task_id in ("few_c", "few_a", "few_b")
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "requests.jsonl"

            exit_code = main(
                [
                    "prepare-initial",
                    "--tasks-dir",
                    str(tasks),
                    "--task-ids",
                    str(task_ids),
                    "--few-shot-manifest",
                    str(manifest),
                    "--output",
                    str(output),
                ]
            )
            requests = read_requests(output)

            rendered = root / "rendered"
            render_exit_code = main(
                [
                    "render-requests",
                    "--input",
                    str(output),
                    "--output-dir",
                    str(rendered),
                ]
            )
            rendered_files = list(rendered.glob("*.txt"))
            rendered_text = rendered_files[0].read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(render_exit_code, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(rendered_files), 1)
        self.assertIn("## SYSTEM", rendered_text)
        self.assertEqual(len(requests[0].few_shot_task_ids), 2)
        target_hash = int.from_bytes(
            hashlib.sha256(b"target").digest()[:4],
            "big",
        )
        expected = random.Random(target_hash).sample(
            ["few_c", "few_a", "few_b"],
            2,
        )
        self.assertEqual(list(requests[0].few_shot_task_ids), expected)
        self.assertEqual(requests[0].metadata["hint_status"], "missing")

    def test_prepare_initial_reads_versioned_jsonl_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            for task_id in ("target", "few_a", "few_b"):
                write_task(tasks, task_id)

            task_manifest = root / "task_manifest.jsonl"
            write_jsonl(
                [{"schema_version": 1, "task_id": "target"}],
                task_manifest,
            )
            hints = root / "hints.jsonl"
            write_jsonl(
                [
                    {
                        "schema_version": 1,
                        "task_id": task_id,
                        "general": "General observation",
                        "inputs": "Input structures",
                        "outputs": "Output structures",
                        "transformation": "Transformation rule",
                        "transformation_steps": "Apply the rule",
                    }
                    for task_id in ("target", "few_a", "few_b")
                ],
                hints,
            )
            few_shots = root / "few_shot_traces.jsonl"
            write_jsonl(
                [
                    {"schema_version": 1, "task_id": task_id, "trace": TRACE}
                    for task_id in ("few_a", "few_b")
                ],
                few_shots,
            )
            output = root / "requests.jsonl"

            exit_code = main(
                [
                    "prepare-initial",
                    "--tasks-dir",
                    str(tasks),
                    "--task-ids",
                    str(task_manifest),
                    "--hints",
                    str(hints),
                    "--few-shot-manifest",
                    str(few_shots),
                    "--candidate-start",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            requests = read_requests(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].hint_mode, HintMode.PROVIDED)
        self.assertEqual(requests[0].metadata["hint_status"], "complete")
        self.assertEqual(requests[0].metadata["candidate_index"], 3)
        self.assertEqual(set(requests[0].few_shot_task_ids), {"few_a", "few_b"})

    def test_prepare_initial_rejects_invalid_candidate_ranges(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "prepare-initial",
                    "--tasks-dir",
                    "tasks",
                    "--task-ids",
                    "tasks.json",
                    "--few-shot-manifest",
                    "few_shots.json",
                    "--candidate-start",
                    "-1",
                    "--output",
                    "requests.jsonl",
                ]
            )

        self.assertEqual(caught.exception.code, 2)

    def test_prepare_rewrite_plans_the_requested_accepted_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            write_task(tasks, "target")
            sources = root / "sources.jsonl"
            write_results(
                [
                    GenerationResult(
                        request_id="source-trace",
                        task_id="target",
                        stage=GenerationStage.INITIAL,
                        backend="test",
                        model="test",
                        raw_output=TRACE,
                        normalized_output=TRACE,
                        validation={
                            "static": {"accepted": True},
                            "quality": {"accepted": True, "route": "manual_review"},
                        },
                    )
                ],
                sources,
            )
            output = root / "rewrite.requests.jsonl"

            exit_code = main(
                [
                    "prepare-rewrite",
                    "--sources",
                    str(sources),
                    "--tasks-dir",
                    str(tasks),
                    "--target-count",
                    "3",
                    "--output",
                    str(output),
                ]
            )
            requests = read_requests(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(requests), 3)
        self.assertTrue(all(request.augmentation is not None for request in requests))

    def test_prepare_rewrite_accepts_selected_base_trace_resource(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks"
            tasks.mkdir()
            write_task(tasks, "target")
            base_traces = root / "base_traces.jsonl"
            write_jsonl(
                [{"schema_version": 1, "task_id": "target", "trace": TRACE}],
                base_traces,
            )
            output = root / "rewrite.requests.jsonl"

            exit_code = main(
                [
                    "prepare-rewrite",
                    "--base-traces",
                    str(base_traces),
                    "--tasks-dir",
                    str(tasks),
                    "--target-count",
                    "2",
                    "--output",
                    str(output),
                ]
            )
            requests = read_requests(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            {request.augmentation.source_trace_id for request in requests},
            {"base-trace-target"},
        )

    def test_evaluate_judges_attaches_strict_five_of_five_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = GenerationResult(
                request_id="source-trace",
                task_id="target",
                stage=GenerationStage.INITIAL,
                backend="test",
                model="test",
                raw_output=TRACE,
                normalized_output=TRACE,
                validation={"static": {"accepted": True}},
            )
            judge_requests = build_judge_requests(
                task_id="target",
                puzzle=[
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                candidate_trace=TRACE,
                source_request_id=source.request_id,
                hint=None,
            )
            judge_results = [
                GenerationResult(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    stage=GenerationStage.JUDGE,
                    backend="test",
                    model="test",
                    raw_output='{"verdict": "pass"}',
                )
                for request in judge_requests
            ]
            sources_path = root / "sources.jsonl"
            requests_path = root / "judges.requests.jsonl"
            results_path = root / "judges.results.jsonl"
            output = root / "evaluated.jsonl"
            write_results([source], sources_path)
            write_requests(judge_requests, requests_path)
            write_results(judge_results, results_path)

            exit_code = main(
                [
                    "evaluate-judges",
                    "--sources",
                    str(sources_path),
                    "--judge-requests",
                    str(requests_path),
                    "--judge-results",
                    str(results_path),
                    "--output",
                    str(output),
                ]
            )
            evaluated = read_results(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(evaluated), 1)
        self.assertTrue(evaluated[0].validation["quality"]["accepted"])
        self.assertEqual(
            evaluated[0].validation["quality"]["route"],
            "judge_5_of_5",
        )


if __name__ == "__main__":
    unittest.main()
