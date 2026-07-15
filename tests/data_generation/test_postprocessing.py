import unittest

from explain_then_adapt.data_generation.postprocessing import normalize_trace
from explain_then_adapt.data_generation.validation import validate_trace_format


VALID_TRACE = """<think>
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


class PostprocessingTests(unittest.TestCase):
    def test_markdown_wrapping_is_normalized(self) -> None:
        wrapped = "```text\n" + VALID_TRACE.replace(
            "1) INPUT ANALYSIS", "**1) INPUT ANALYSIS**"
        ) + "\n```"

        normalized = normalize_trace(wrapped)
        result = validate_trace_format(wrapped)

        self.assertTrue(normalized.startswith("<think>"))
        self.assertIn("\n1) INPUT ANALYSIS\n", normalized)
        self.assertTrue(result.accepted)

    def test_missing_reordered_and_duplicate_sections_are_rejected(self) -> None:
        self.assertFalse(
            validate_trace_format(VALID_TRACE.replace("General steps:", "Steps:"))
            .accepted
        )
        reordered = VALID_TRACE.replace(
            "2) OUTPUT ANALYSIS\nx\n3) TRANSFORMATION ANALYSIS",
            "3) TRANSFORMATION ANALYSIS\nx\n2) OUTPUT ANALYSIS",
        )
        self.assertFalse(validate_trace_format(reordered).accepted)
        duplicate = VALID_TRACE + "\nGeneral steps:\nextra"
        self.assertFalse(validate_trace_format(duplicate).accepted)


if __name__ == "__main__":
    unittest.main()
