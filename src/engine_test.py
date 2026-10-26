from absl.testing import absltest
from absl.testing import parameterized

from engine import Value


class ValueTest(parameterized.TestCase):
    """Tests the Value class."""

    @parameterized.named_parameters(
        ("positive_integer", 1),
        ("negative_integer", -2),
        ("zero_integer", 0),
        ("positive_float", 1.0),
        ("negative_float", -2.0),
    )
    def test_initialization_success(self, data: int | float) -> None:
        """Tests that initializing `Value` succeeds."""
        v = Value(data)
        self.assertEqual(v.data, data)
        self.assertIsInstance(v.data, float)

    @parameterized.named_parameters(("string", "1"), ("boolean", False))
    def test_initialization_invalid_data_failure(self, data) -> None:
        """Tests that initializing `Value` with invalid data raises an error."""
        with self.assertRaisesRegex(
            ValueError, r"Expected data to be `int` or `float`"
        ):
            Value(data)

    @parameterized.named_parameters(
        ("all_values", Value(1.0), Value(2.0), 3.0),
        ("value_and_int", Value(1.0), 2, 3.0),
        ("int_and_value", 1, Value(2.0), 3.0),
        ("value_and_float", Value(1.0), 2.0, 3.0),
        ("float_and_value", 1.0, Value(2.0), 3.0),
    )
    def test_add_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: float,
    ) -> None:
        """Tests that adding `Value`s and scalars works."""
        result = left + right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)  # type: ignore

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_add_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that adding `Value`s and scalars works."""
        with self.assertRaisesRegex(
            ValueError,
            r"Expected `other` to be of type `Value`, `int`, or, `float`",
        ):
            _ = left + right

    @parameterized.named_parameters(
        ("all_values", Value(1.0), Value(2.0), 2.0),
        ("value_and_int", Value(1.0), 2, 2.0),
        ("int_and_value", 1, Value(2.0), 2.0),
        ("value_and_float", Value(1.0), 2.0, 2.0),
        ("float_and_value", 1.0, Value(2.0), 2.0),
    )
    def test_mul_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: float,
    ) -> None:
        """Tests that adding `Value`s and scalars works."""
        result = left * right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)  # type: ignore

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_mul_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that adding `Value`s and scalars works."""
        with self.assertRaisesRegex(
            ValueError,
            r"Expected `other` to be of type `Value`, `int`, or, `float`",
        ):
            _ = left * right


if __name__ == "__main__":
    absltest.main()
