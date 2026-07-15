import unittest

from explain_then_adapt.data_generation.validation import (
    ValidationRoute,
    evaluate_judge_responses,
    parse_judge_verdict,
    record_manual_review,
)


class ValidationTests(unittest.TestCase):
    def test_five_passes_accept_a_trace(self) -> None:
        result = evaluate_judge_responses(
            [{"verdict": "pass"} for _ in range(5)]
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.route, ValidationRoute.JUDGE_5_OF_5)
        self.assertEqual(result.pass_count, 5)
        self.assertEqual(result.vote_count, 5)

    def test_one_failed_vote_rejects_a_trace(self) -> None:
        responses = ['{"verdict": "pass"}'] * 4 + ['{"verdict": "fail"}']

        result = evaluate_judge_responses(responses)

        self.assertFalse(result.accepted)
        self.assertEqual(result.pass_count, 4)

    def test_exactly_five_votes_are_required(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_judge_responses([{"verdict": "pass"}] * 4)

    def test_verdict_parser_rejects_malformed_responses(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_verdict("not JSON")
        with self.assertRaises(ValueError):
            parse_judge_verdict({"verdict": "maybe"})

    def test_manual_review_is_recorded_as_a_separate_route(self) -> None:
        result = record_manual_review(
            accepted=True,
            reviewer_note="Trace checked against every demonstration pair.",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.route, ValidationRoute.MANUAL_REVIEW)
        self.assertEqual(
            result.reviewer_note,
            "Trace checked against every demonstration pair.",
        )

    def test_manual_review_requires_a_note(self) -> None:
        with self.assertRaises(ValueError):
            record_manual_review(accepted=True, reviewer_note="  ")


if __name__ == "__main__":
    unittest.main()
