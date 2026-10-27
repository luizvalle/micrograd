"""Tests the `Value` class."""

import math

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
        ("value", Value(1.0))
    )
    def test_initialization_success(self, value: Value | int | float) -> None:
        """Tests that initializing `Value` succeeds."""
        v = Value(value)
        self.assertEqual(v, value)
        self.assertIsInstance(v.data, float)

    def test_initialization_with_label(self) -> None:
        """Tests initialization with the optional label."""
        a = Value(0.0, "a")
        b = Value(1.0, "b")
        self.assertEqual(a.label, "a")
        self.assertEqual(b.label, "b")

    @parameterized.named_parameters(("string", "1"), ("boolean", False))
    def test_initialization_invalid_data_failure(self, data) -> None:
        """Tests that initializing `Value` with invalid data raises an error."""
        with self.assertRaisesRegex(
            ValueError, r"Expected `value` to be `int` or `float`"
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
        self.assertAlmostEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_add_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that adding `Value`s and other objects raises and error."""
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
        """Tests that multiplying `Value`s and scalars works."""
        result = left * right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_mul_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that multiplying `Value`s and other types raises an error."""
        with self.assertRaisesRegex(
            ValueError,
            r"Expected `other` to be of type `Value`, `int`, or, `float`",
        ):
            _ = left * right

    @parameterized.named_parameters(
        ("int", Value(2.0), 3, 8.0),
        ("float", Value(2.0), 3.0, 8.0),
    )
    def test_pow_success(
        self,
        base: Value,
        power: int | float,
        expected: float,
    ) -> None:
        """Tests that raising `Value`s to a power works."""
        result = base ** power
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("value_and_string", Value(1.0), "a"),
        ("value_and_value", Value(1.0), Value(2.0))
    )
    def test_pow_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that raising `Value` anything but ints and floats raises an error."""
        with self.assertRaisesRegex(
            ValueError,
            r"Expected `other` to be of type `int`, or `float`",
        ):
            _ = left ** right  # type: ignore[operator, call-overload]  # pyright: ignore[reportOperatorIssue]

    @parameterized.named_parameters(
        ("all_values", Value(1.0), Value(2.0), -1.0),
        ("value_and_int", Value(1.0), 2, -1.0),
        ("int_and_value", 1, Value(2.0), -1.0),
        ("value_and_float", Value(1.0), 2.0, -1.0),
        ("float_and_value", 1.0, Value(2.0), -1.0),
    )
    def test_sub_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: float,
    ) -> None:
        """Tests that adding `Value`s and scalars works."""
        result = left - right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected)  # type: ignore   

    @parameterized.named_parameters(
        ("value", Value(6.0), Value(2.0), 3.0),
        ("int", Value(6.0), 2, 3.0),
        ("float", Value(6.0), 2.0, 3.0),
    )
    def test_truediv_success(
        self,
        left: Value,
        right: Value | int | float,
        expected: float,
    ) -> None:
        """Tests that dividing `Value`s by `Value`s and scalars works."""
        result = left / right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("value", Value(6.0), Value(2.0), 3.0),
        ("int", 6, Value(2.0), 3.0),
        ("float", 6.0, Value(2.0), 3.0),
    )
    def test_rtruediv_success(
        self,
        left: Value | int | float,
        right: Value,
        expected: float,
    ) -> None:
        """Tests reflected division of `Value`s and scalars by a `Value`."""
        result = left / right
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected)  # type: ignore

    def test_neg_success(self) -> None:
        """Tests that negating `Value`s works."""
        v = Value(1.0)
        neg_v = -v
        self.assertIsInstance(neg_v, Value)
        self.assertAlmostEqual(neg_v.data, -v.data)  # type: ignore

    
    @parameterized.named_parameters(
        ("positive", Value(1.0), math.e),
        ("zero", Value(0), 1.0),
        ("negative", Value(-2.0), math.e ** -2),
    )
    def test_exp_success(self, value: Value, expected: float) -> None:
        """Tests that exponentiating the value works."""
        result = value.exp()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected) # type: ignore

    @parameterized.named_parameters(
        ("positive", Value(1.0), 1.0),
        ("zero", Value(0), 0.0),
        ("negative", Value(-2.0), 0.0),
    )
    def test_relu_success(self, value: Value, expected: float) -> None:
        """Tests that applying relu to the value works."""
        result = value.relu()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected) # type: ignore

    @parameterized.named_parameters(
        ("positive", Value(1.0), math.tanh(1.0)),
        ("zero", Value(0), math.tanh(0.0)),
        ("negative", Value(-2.0), math.tanh(-2.0)),
    )
    def test_tanh_success(self, value: Value, expected: float) -> None:
        """Tests that applying tanh to the value works."""
        result = value.tanh()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result, expected) # type: ignore

    @parameterized.named_parameters(
        ("all_values_true", Value(1.0), Value(1.0), True),
        ("all_values_false", Value(1.0), Value(2.0), False),
        ("value_and_int_true", Value(1.0), 1, True),
        ("value_and_int_false", Value(1.0), 2.0, False),
        ("int_and_value_true", 1, Value(1.0), True),
        ("int_and_value_false", 1, Value(2.0), False),
        ("value_and_float_true", Value(1.0), 1.0, True),
        ("value_and_float_false", Value(1.0), 2.0, False),
        ("float_and_value_true", 1.0, Value(1.0), True),
        ("float_and_value_false", 1.0, Value(2.0), False),
    )
    def test_eq_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that comparing equality of `Value`s and scalars works."""
        result = left == right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore


    @parameterized.named_parameters(
        ("all_values_true", Value(1.0), Value(2.0), True),
        ("all_values_false", Value(1.0), Value(1.0), False),
        ("value_and_int_true", Value(1.0), 2, True),
        ("value_and_int_false", Value(1.0), 1, False),
        ("int_and_value_true", 1, Value(2.0), True),
        ("int_and_value_false", 1, Value(1.0), False),
        ("value_and_float_true", Value(1.0), 2.0, True),
        ("value_and_float_false", Value(1.0), 1.0, False),
        ("float_and_value_true", 1.0, Value(2.0), True),
        ("float_and_value_false", 1.0, Value(1.0), False),
    )
    def test_ne_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that comparing inequality of `Value`s and scalars works."""
        result = left != right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("all_values_less", Value(1.0), Value(2.0), True),
        ("all_values_equal", Value(1.0), Value(1.0), False),
        ("all_values_greater", Value(2.0), Value(1.0), False),
        ("value_and_int_less", Value(1.0), 2, True),
        ("value_and_int_equal", Value(1.0), 1, False),
        ("value_and_int_greater", Value(2.0), 1, False),
        ("int_and_value_less", 1, Value(2.0), True),
        ("int_and_value_equal", 1, Value(1.0), False),
        ("int_and_value_greater", 2, Value(1.0), False),
        ("value_and_float_less", Value(1.0), 2.0, True),
        ("value_and_float_equal", Value(1.0), 1.0, False),
        ("value_and_float_greater", Value(2.0), 1.0, False),
        ("float_and_value_less", 1.0, Value(2.0), True),
        ("float_and_value_equal", 1.0, Value(1.0), False),
        ("float_and_value_greater", 2.0, Value(1.0), False),
    )
    def test_lt_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that less-than comparisons of `Value`s and scalars work."""
        result = left < right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("all_values_less", Value(1.0), Value(2.0), True),
        ("all_values_equal", Value(1.0), Value(1.0), True),
        ("all_values_greater", Value(2.0), Value(1.0), False),
        ("value_and_int_less", Value(1.0), 2, True),
        ("value_and_int_equal", Value(1.0), 1, True),
        ("value_and_int_greater", Value(2.0), 1, False),
        ("int_and_value_less", 1, Value(2.0), True),
        ("int_and_value_equal", 1, Value(1.0), True),
        ("int_and_value_greater", 2, Value(1.0), False),
        ("value_and_float_less", Value(1.0), 2.0, True),
        ("value_and_float_equal", Value(1.0), 1.0, True),
        ("value_and_float_greater", Value(2.0), 1.0, False),
        ("float_and_value_less", 1.0, Value(2.0), True),
        ("float_and_value_equal", 1.0, Value(1.0), True),
        ("float_and_value_greater", 2.0, Value(1.0), False),
    )
    def test_le_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that less-than-or-equal comparisons of `Value`s and scalars work."""
        result = left <= right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("all_values_less", Value(1.0), Value(2.0), False),
        ("all_values_equal", Value(1.0), Value(1.0), False),
        ("all_values_greater", Value(2.0), Value(1.0), True),
        ("value_and_int_less", Value(1.0), 2, False),
        ("value_and_int_equal", Value(1.0), 1, False),
        ("value_and_int_greater", Value(2.0), 1, True),
        ("int_and_value_less", 1, Value(2.0), False),
        ("int_and_value_equal", 1, Value(1.0), False),
        ("int_and_value_greater", 2, Value(1.0), True),
        ("value_and_float_less", Value(1.0), 2.0, False),
        ("value_and_float_equal", Value(1.0), 1.0, False),
        ("value_and_float_greater", Value(2.0), 1.0, True),
        ("float_and_value_less", 1.0, Value(2.0), False),
        ("float_and_value_equal", 1.0, Value(1.0), False),
        ("float_and_value_greater", 2.0, Value(1.0), True),
    )
    def test_gt_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that greater-than comparisons of `Value`s and scalars work."""
        result = left > right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore

    @parameterized.named_parameters(
        ("all_values_less", Value(1.0), Value(2.0), False),
        ("all_values_equal", Value(1.0), Value(1.0), True),
        ("all_values_greater", Value(2.0), Value(1.0), True),
        ("value_and_int_less", Value(1.0), 2, False),
        ("value_and_int_equal", Value(1.0), 1, True),
        ("value_and_int_greater", Value(2.0), 1, True),
        ("int_and_value_less", 1, Value(2.0), False),
        ("int_and_value_equal", 1, Value(1.0), True),
        ("int_and_value_greater", 2, Value(1.0), True),
        ("value_and_float_less", Value(1.0), 2.0, False),
        ("value_and_float_equal", Value(1.0), 1.0, True),
        ("value_and_float_greater", Value(2.0), 1.0, True),
        ("float_and_value_less", 1.0, Value(2.0), False),
        ("float_and_value_equal", 1.0, Value(1.0), True),
        ("float_and_value_greater", 2.0, Value(1.0), True),
    )
    def test_ge_success(
        self,
        left: Value | int | float,
        right: Value | int | float,
        expected: bool,
    ) -> None:
        """Tests that greater-than-or-equal comparisons of `Value`s and scalars work."""
        result = left >= right
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)  # type: ignore


if __name__ == "__main__":
    absltest.main()
