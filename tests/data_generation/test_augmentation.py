import unittest

from explain_then_adapt.data_generation.augmentation import (
    apply_augmentation,
    augmentation_signature,
    plan_augmentation_specs,
    remaining_augmentation_count,
)


PUZZLE = [
    {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
    {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
]


class AugmentationTests(unittest.TestCase):
    def test_planning_fills_only_the_missing_accepted_count(self) -> None:
        first = plan_augmentation_specs(
            source_trace_id="trace",
            example_count=2,
            target_count=3,
            seed=12,
        )
        second = plan_augmentation_specs(
            source_trace_id="trace",
            example_count=2,
            accepted_specs=first[:1],
            attempted_specs=first,
            target_count=3,
            seed=12,
        )

        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 2)
        self.assertTrue(
            set(map(augmentation_signature, first)).isdisjoint(
                map(augmentation_signature, second)
            )
        )
        self.assertEqual(remaining_augmentation_count(97), 3)

    def test_planning_is_deterministic_and_transform_is_non_mutating(self) -> None:
        first = plan_augmentation_specs(
            source_trace_id="trace",
            example_count=2,
            target_count=1,
            seed=3,
        )
        second = plan_augmentation_specs(
            source_trace_id="trace",
            example_count=2,
            target_count=1,
            seed=3,
        )
        original = [[row[:] for row in pair["input"]] for pair in PUZZLE]

        transformed = apply_augmentation(PUZZLE, first[0])

        self.assertEqual(first, second)
        self.assertEqual([pair["input"] for pair in PUZZLE], original)
        self.assertEqual(len(transformed), len(PUZZLE))


if __name__ == "__main__":
    unittest.main()
