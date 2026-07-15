import unittest

from explain_then_adapt.arc.augmented_keys import (
    apply_order_mapping,
    is_augmented_key,
    make_augmented_key,
    parse_augmented_key,
    parse_order_mapping,
)


class AugmentedKeyTests(unittest.TestCase):
    def test_plain_key_has_no_augmentation_components(self) -> None:
        parsed = parse_augmented_key("cc9053aa")

        self.assertEqual(parsed.original_key, "cc9053aa")
        self.assertIsNone(parsed.transformation_id)
        self.assertFalse(is_augmented_key("cc9053aa"))

    def test_augmented_key_round_trip(self) -> None:
        key = make_augmented_key(
            "cc9053aa",
            "fd2",
            "6384521079",
            "201",
        )

        self.assertEqual(key, "cc9053aa_FD2_6384521079_201")
        self.assertEqual(
            tuple(parse_augmented_key(key)),
            ("cc9053aa", "FD2", "6384521079", "201"),
        )
        self.assertTrue(is_augmented_key(key))

    def test_order_mapping_supports_common_formats(self) -> None:
        self.assertEqual(parse_order_mapping("201"), [2, 0, 1])
        self.assertEqual(parse_order_mapping("2, 0, 1"), [2, 0, 1])
        self.assertEqual(parse_order_mapping("2 0 1"), [2, 0, 1])

    def test_order_mapping_reorders_without_mutating_input(self) -> None:
        pairs = ["first", "second", "third"]

        reordered = apply_order_mapping(pairs, "201")

        self.assertEqual(reordered, ["third", "first", "second"])
        self.assertEqual(pairs, ["first", "second", "third"])

    def test_invalid_augmented_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_augmented_key("cc9053aa_R45_6384521079_012")
        with self.assertRaises(ValueError):
            parse_order_mapping("001")


if __name__ == "__main__":
    unittest.main()
