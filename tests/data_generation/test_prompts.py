import unittest

from explain_then_adapt.data_generation.hints import Hint
from explain_then_adapt.data_generation.prompts import (
    FewShotExample,
    build_initial_prompt,
    build_judge_prompt,
    build_rewrite_prompt,
)
from explain_then_adapt.data_generation.records import AugmentationSpec


PUZZLE = [
    {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
    {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
]
HINT = Hint("general", "inputs", "outputs", "rule", "steps")
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


class PromptTests(unittest.TestCase):
    def test_initial_prompt_distinguishes_hint_free_generation(self) -> None:
        prompt = build_initial_prompt(PUZZLE, hint=None, few_shots=())

        self.assertIn("no task-specific hint", prompt)
        self.assertNotIn("German", prompt)
        self.assertNotIn("## Hints:", prompt)
        self.assertIn("3) TRANSFORMATION ANALYSIS", prompt)

    def test_initial_prompt_contains_few_shot_trace_and_hint(self) -> None:
        few_shot = FewShotExample("few", PUZZLE, TRACE, HINT)

        prompt = build_initial_prompt(PUZZLE, hint=HINT, few_shots=(few_shot,))

        self.assertIn("### Few-shot task: few", prompt)
        self.assertIn("Transformation Steps:\nsteps", prompt)
        self.assertIn(TRACE, prompt)

    def test_judge_prompt_requires_json_and_includes_candidate(self) -> None:
        prompt = build_judge_prompt(PUZZLE, TRACE, hint=HINT)

        self.assertIn('"verdict": "pass" | "fail"', prompt)
        self.assertIn("No hint leakage", prompt)
        self.assertIn(TRACE, prompt)

    def test_rewrite_prompt_describes_all_augmentation_levels(self) -> None:
        spec = AugmentationSpec(
            source_trace_id="source",
            transformation_code="R90",
            value_mapping="1023456789",
            order_mapping="10",
            style="analytical",
        )

        prompt = build_rewrite_prompt(PUZZLE, TRACE, list(reversed(PUZZLE)), spec)

        self.assertIn("Rotate 90 degrees clockwise", prompt)
        self.assertIn("0 -> 1", prompt)
        self.assertIn("new example 1 = old example 2", prompt)
        self.assertIn("Output only the adapted reasoning trace", prompt)


if __name__ == "__main__":
    unittest.main()
