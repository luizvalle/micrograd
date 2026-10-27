"""Implementation of the Value class"""

from __future__ import annotations

import math


class Value:
    """Scalar value that keeps track of dependencies and calculates gradients.

    Attributes:
        data: The scalar value held by this object.
    """

    def __init__(self, value: Value | int | float) -> None:
        if isinstance(value, bool) or not isinstance(
            value, (Value, int, float)
        ):
            raise ValueError(
                "Expected `value` to be `int` or `float` but got"
                f" {type(value).__name__}"
            )
        self._data: float = (
            value.data if isinstance(value, Value) else float(value)
        )

    @property
    def data(self) -> float:
        """Returns the data held by this instance."""
        return self._data

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            return Value(self._data + other._data)

        if isinstance(other, (int, float)):
            return Value(self._data + other)

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}"
        )

    def __radd__(self, other: Value | int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            return Value(self._data * other._data)

        if isinstance(other, (int, float)):
            return Value(self._data * other)

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}"
        )

    def __rmul__(self, other: Value | int | float) -> Value:
        return self * other

    def __pow__(self, other: int | float) -> Value:
        if isinstance(other, (int, float)):
            return Value(self._data**other)

        raise ValueError(
            "Expected `other` to be of type "
            f"`int`, or `float`, but got {type(other).__name__}"
        )

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * (other**-1)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return other * (self**-1)

    def __neg__(self) -> Value:
        return Value(self.data) * -1

    def __abs__(self) -> Value:
        return Value(abs(self.data))

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:
        return other + (-self)

    def exp(self) -> Value:
        """Raises `e` to this value."""
        return Value(math.exp(self.data))

    def relu(self) -> Value:
        """Calculates the ReLU of this value."""
        return Value(0 if self <= 0 else self.data)

    def tanh(self) -> Value:
        """Applies tanh to this value."""
        c = (2 * self).exp()
        return (c - 1) / (c + 1)

    def __eq__(self, other) -> bool:
        if isinstance(other, Value):
            return self.data == other.data

        if isinstance(other, (int, float)):
            return self.data == other

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}"
        )

    def __lt__(self, other):
        if isinstance(other, Value):
            return self.data < other.data

        if isinstance(other, (int, float)):
            return self.data < other

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}"
        )

    def __le__(self, other):
        return (self == other) or (self < other)

    def __gt__(self, other):
        if isinstance(other, Value):
            return self.data > other.data

        if isinstance(other, (int, float)):
            return self.data > other

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}"
        )

    def __ge__(self, other):
        return (self == other) or (self > other)
