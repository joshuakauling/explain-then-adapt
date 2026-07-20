import unittest

from explain_then_adapt.data_generation.resource_migration import (
    _repair_legacy_trace_format,
)
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


class LegacyTraceRepairTests(unittest.TestCase):
    def test_valid_trace_is_unchanged(self) -> None:
        repaired, repair = _repair_legacy_trace_format(VALID_TRACE)

        self.assertEqual(repaired, VALID_TRACE)
        self.assertIsNone(repair)

    def test_inserts_missing_think_close(self) -> None:
        malformed = VALID_TRACE.replace("</think>\n", "")

        repaired, repair = _repair_legacy_trace_format(malformed)

        self.assertEqual(repair, "insert_missing_think_close")
        self.assertTrue(validate_trace_format(repaired).accepted)

    def test_removes_duplicate_summary_inside_think(self) -> None:
        summary = "General natural language description:\nx\nGeneral steps:\nx\n"
        malformed = VALID_TRACE.replace("</think>", f"{summary}</think>")

        repaired, repair = _repair_legacy_trace_format(malformed)

        self.assertEqual(repair, "remove_duplicate_summary_inside_think")
        self.assertEqual(repaired.count("General natural language description:"), 1)
        self.assertTrue(validate_trace_format(repaired).accepted)

    def test_rejects_unknown_format_defect(self) -> None:
        malformed = VALID_TRACE.replace("General steps:", "Unexpected heading:")

        with self.assertRaisesRegex(ValueError, "unsupported legacy format defect"):
            _repair_legacy_trace_format(malformed)


if __name__ == "__main__":
    unittest.main()
