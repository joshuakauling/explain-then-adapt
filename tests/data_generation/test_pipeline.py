import unittest

from explain_then_adapt.data_generation.pipeline import (
    apply_static_validation,
    build_initial_request,
    build_judge_requests,
    build_rewrite_request,
)
from explain_then_adapt.data_generation.records import (
    AugmentationSpec,
    GenerationResult,
    GenerationStage,
    HintMode,
    SamplingParameters,
)


PUZZLE = [
    {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
    {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
]
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


class PipelineTests(unittest.TestCase):
    def test_initial_and_five_judge_requests_preserve_provenance(self) -> None:
        initial = build_initial_request(
            task_id="task",
            puzzle=PUZZLE,
            hint=None,
            few_shots=(),
        )
        judges = build_judge_requests(
            task_id="task",
            puzzle=PUZZLE,
            candidate_trace=TRACE,
            source_request_id=initial.request_id,
            hint=None,
            sampling=SamplingParameters(temperature=0.7, seed=10),
        )

        self.assertEqual(initial.hint_mode, HintMode.NONE)
        self.assertEqual(len(judges), 5)
        self.assertEqual(len({request.request_id for request in judges}), 5)
        self.assertEqual([request.sampling.seed for request in judges], [10, 11, 12, 13, 14])
        self.assertTrue(
            all(
                request.metadata["source_request_id"] == initial.request_id
                for request in judges
            )
        )

    def test_rewrite_request_contains_structured_augmentation(self) -> None:
        spec = AugmentationSpec(
            source_trace_id="source",
            transformation_code="ID",
            value_mapping="0123456789",
            order_mapping="10",
        )

        request = build_rewrite_request(
            task_id="task",
            puzzle=PUZZLE,
            accepted_trace=TRACE,
            spec=spec,
        )

        self.assertEqual(request.stage, GenerationStage.REWRITE)
        self.assertEqual(request.augmentation, spec)

    def test_static_validation_is_attached_to_result(self) -> None:
        result = GenerationResult(
            request_id="request",
            task_id="task",
            stage=GenerationStage.INITIAL,
            backend="test",
            model="test",
            raw_output=TRACE,
        )

        validated = apply_static_validation(result)

        self.assertEqual(validated.normalized_output, TRACE)
        self.assertTrue(validated.validation["static"]["accepted"])


if __name__ == "__main__":
    unittest.main()
