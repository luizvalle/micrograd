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
        ("negative_float", -2.0))
    def test_initialization_success(self, data: int | float) -> None:
        """Tests that initializing `Value` succeeds."""
        v = Value(data)
        self.assertEqual(v.data, data)

    @parameterized.named_parameters(
        ("string", "1"),
        ("boolean", False))
    def test_initialization_invalid_data_failure(self, data) -> None:
        """Tests that initializing `Value` with invalid data raises an error."""
        with self.assertRaisesRegex(ValueError, r"Expected data to be `int` or `float`"):
            Value(data)


    @parameterized.named_parameters(
        ("all_values", Value(1.0), Value(2.0), 3.0))
    def test_add_success(self, left: Value | int | float,
                         right: Value | int | float, expected: float) -> None:
        """Tests that adding `Value`s and scalars works."""
        result = left + right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)  # type: ignore


if __name__ == "__main__":
    absltest.main()
