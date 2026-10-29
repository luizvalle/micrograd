"""Tests the `Value` class."""

import math

from absl.testing import absltest
from absl.testing import parameterized

from engine import Value


class ValueTest(parameterized.TestCase):
    """Tests the Value class."""

    @parameterized.named_parameters(
        ("positive_integer", 1, 1.0),
        ("negative_integer", -2, -2.0),
        ("zero_integer", 0, 0.0),
        ("positive_float", 1.0, 1.0),
        ("negative_float", -2.0, -2.0),
        ("value", Value(1.0), 1.0)
    )
    def test_initialization_success(
        self, value: Value | int | float, expected: float
    ) -> None:
        """Tests that initializing `Value` succeeds."""
        v = Value(value)
        self.assertEqual(v.data, expected)
        self.assertIsInstance(v.data, float)

    def test_initialization_with_label(self) -> None:
        """Tests initialization with the optional label."""
        a = Value(0.0, label="a")
        b = Value(1.0, label="b")
        self.assertEqual(a.label, "a")
        self.assertEqual(b.label, "b")

    def test_initialization_with_op(self) -> None:
        """Tests initialization with the optional operation."""
        a = Value(0.0, op="+", parents=(Value(1), Value(2)))
        self.assertEqual(a.op, "+")

    @parameterized.named_parameters(("string", "1"), ("boolean", False))
    def test_initialization_invalid_data_failure(self, data) -> None:
        """Tests that initializing `Value` with invalid data raises an error."""
        with self.assertRaisesRegex(
            ValueError, r"Expected `value` to be `int` or `float`"
        ):
            Value(data)

    @parameterized.named_parameters(
        ("missing_op", "", (Value(1), Value(2))),
        ("missing_parents", "+", tuple())
    )
    def test_initialization_fails_with_wrong_op_and_parents(self,
                                                            op: str,
                                                            parents: tuple[Value]) -> None:
        """Tests that initializating `Value` when one of `op` or `parents` is empty fails."""
        with self.assertRaisesRegex(ValueError,
                                    ("Either both `op` and `parents` should be "
                                     "specified or none.")):
            _ = Value(1, op=op, parents=parents)

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
        self.assertAlmostEqual(result.data, expected)

    def test_add_builds_graph(self) -> None:
        """Tests that adding two `Value`s records both as parents."""
        a = Value(1.0, label="a")
        b = Value(2.0, label="b")
        result = a + b
        self.assertEqual(result.op, "+")
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 2)
        self.assertIs(result.parents[0], a)
        self.assertIs(result.parents[1], b)

    @parameterized.named_parameters(("right", False), ("left", True))
    def test_add_wraps_constant(self, reflected: bool) -> None:
        """Tests that adding a scalar wraps it in a `Value` parent."""
        a = Value(1.0, label="a")
        result = 2 + a if reflected else a + 2
        self.assertEqual(result.op, "+")
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 2)
        self.assertIs(result.parents[0], a)
        constant = result.parents[1]
        self.assertIsInstance(constant, Value)
        self.assertEqual(constant.data, 2.0)
        self.assertEqual(constant.label, "")

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_add_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that adding `Value`s and other objects raises and error."""
        with self.assertRaises(TypeError):
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
        self.assertAlmostEqual(result.data, expected)

    def test_mul_builds_graph(self) -> None:
        """Tests that multiplying two `Value`s records both as parents."""
        a = Value(1.0, label="a")
        b = Value(2.0, label="b")
        result = a * b
        self.assertEqual(result.op, "*")
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 2)
        self.assertIs(result.parents[0], a)
        self.assertIs(result.parents[1], b)

    @parameterized.named_parameters(("right", False), ("left", True))
    def test_mul_wraps_constant(self, reflected: bool) -> None:
        """Tests that multiplying by a scalar wraps it in a `Value` parent."""
        a = Value(1.0, label="a")
        result = 2 * a if reflected else a * 2
        self.assertEqual(result.op, "*")
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 2)
        self.assertIs(result.parents[0], a)
        constant = result.parents[1]
        self.assertIsInstance(constant, Value)
        self.assertEqual(constant.data, 2.0)
        self.assertEqual(constant.label, "")

    @parameterized.named_parameters(
        ("string_right", Value(1.0), "a"), ("string_left", "a", Value(1.0))
    )
    def test_mul_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that multiplying `Value`s and other types raises an error."""
        with self.assertRaises(TypeError):
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
        self.assertAlmostEqual(result.data, expected)

    def test_pow_builds_graph(self) -> None:
        """Tests that raising to a power wraps the exponent in a `Value` parent."""
        a = Value(2.0, label="a")
        result = a ** 3
        self.assertEqual(result.op, "**")
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 2)
        self.assertIs(result.parents[0], a)
        exponent = result.parents[1]
        self.assertIsInstance(exponent, Value)
        self.assertEqual(exponent.data, 3.0)
        self.assertEqual(exponent.label, "")

    @parameterized.named_parameters(
        ("value_and_string", Value(1.0), "a"),
        ("value_and_value", Value(1.0), Value(2.0))
    )
    def test_pow_failure(
        self, left: Value | int | float, right: Value | int | float
    ) -> None:
        """Tests that raising `Value` anything but ints and floats raises an error."""
        with self.assertRaises(TypeError):
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
        self.assertAlmostEqual(result.data, expected)

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
        self.assertAlmostEqual(result.data, expected)

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
        self.assertAlmostEqual(result.data, expected)

    def test_neg_success(self) -> None:
        """Tests that negating `Value`s works."""
        v = Value(1.0)
        neg_v = -v
        self.assertIsInstance(neg_v, Value)
        self.assertAlmostEqual(neg_v.data, -v.data)  # type: ignore

    @parameterized.named_parameters(
        ("positive", Value(6.0), 6.0),
        ("zero", Value(0.0), 0.0),
        ("negative", Value(-2.0), 2.0),
    )
    def test_abs_success(self, value: Value, expected: float) -> None:
        """Tests that taking the absolute value works."""
        result = abs(value)
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)

    @parameterized.named_parameters(
        ("positive", Value(1.0), math.e),
        ("zero", Value(0), 1.0),
        ("negative", Value(-2.0), math.e ** -2),
    )
    def test_exp_success(self, value: Value, expected: float) -> None:
        """Tests that exponentiating the value works."""
        result = value.exp()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)

    @parameterized.named_parameters(
        ("positive", Value(1.0), 1.0),
        ("zero", Value(0), 0.0),
        ("negative", Value(-2.0), 0.0),
    )
    def test_relu_success(self, value: Value, expected: float) -> None:
        """Tests that applying relu to the value works."""
        result = value.relu()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)

    @parameterized.named_parameters(
        ("positive", Value(1.0), math.tanh(1.0)),
        ("zero", Value(0), math.tanh(0.0)),
        ("negative", Value(-2.0), math.tanh(-2.0)),
    )
    def test_tanh_success(self, value: Value, expected: float) -> None:
        """Tests that applying tanh to the value works."""
        result = value.tanh()
        self.assertIsInstance(result, Value)
        self.assertAlmostEqual(result.data, expected)

    @parameterized.named_parameters(
        ("exp", lambda v: v.exp(), "exp"),
        ("relu", lambda v: v.relu(), "relu"),
        ("abs", abs, "abs"),
    )
    def test_unary_builds_graph(self, fn, op: str) -> None:
        """Tests that unary operations record the input as the only parent."""
        a = Value(1.0, label="a")
        result = fn(a)
        self.assertEqual(result.op, op)
        self.assertEqual(result.label, "")
        self.assertLen(result.parents, 1)
        self.assertIs(result.parents[0], a)


if __name__ == "__main__":
    absltest.main()
