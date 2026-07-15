import tempfile
import unittest
from pathlib import Path

from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    GenerationStage,
    HintMode,
    SamplingParameters,
    make_request_id,
    read_requests,
    read_results,
    write_requests,
    write_results,
)


class RecordTests(unittest.TestCase):
    def test_request_and_result_round_trip_through_jsonl(self) -> None:
        spec = AugmentationSpec(
            source_trace_id="trace-1",
            transformation_code="R90",
            value_mapping="1023456789",
            order_mapping="10",
            style="concise",
            variant_index=4,
        )
        request = GenerationRequest(
            request_id="rewrite-task-1",
            task_id="task",
            stage=GenerationStage.REWRITE,
            messages=(ChatMessage("user", "Rewrite this trace."),),
            prompt_version="rewrite-v1",
            hint_mode=HintMode.NONE,
            sampling=SamplingParameters(temperature=0.5, seed=7),
            augmentation=spec,
        )
        result = GenerationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            stage=request.stage,
            backend="test",
            model="model",
            raw_output="output",
        )

        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "requests.jsonl"
            result_path = Path(directory) / "results.jsonl"
            self.assertEqual(write_requests([request], request_path), 1)
            self.assertEqual(write_results([result], result_path), 1)
            self.assertEqual(read_requests(request_path), [request])
            self.assertEqual(read_results(result_path), [result])

    def test_request_identifier_is_stable_and_semantic(self) -> None:
        identity = {"candidate": 0, "few_shots": ["a", "b"]}
        first = make_request_id(GenerationStage.INITIAL, "task", identity)
        second = make_request_id(
            GenerationStage.INITIAL,
            "task",
            {"few_shots": ["a", "b"], "candidate": 0},
        )
        changed = make_request_id(
            GenerationStage.INITIAL,
            "task",
            {"candidate": 1, "few_shots": ["a", "b"]},
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_provider_default_sampling_round_trip_is_explicit(self) -> None:
        parameters = SamplingParameters(use_provider_defaults=True)

        self.assertEqual(parameters.to_dict(), {"use_provider_defaults": True})
        self.assertEqual(
            SamplingParameters.from_dict(parameters.to_dict()),
            parameters,
        )


if __name__ == "__main__":
    unittest.main()
