"""Implementation of the Value class"""

from __future__ import annotations

import math

from typing import Iterable


class Value:
    """Scalar value that keeps track of dependencies and calculates gradients.

    Attributes:
        data: The scalar value held by this object.
        label: The label used to identify this object.
        op: The operation that generated this data.
        parents: The parent nodes of this value.
    """

    def __init__(self,
                 value: Value | int | float,
                 *,
                 label: str = "",
                 op: str = "",
                 parents: Iterable[Value] = ()) -> None:
        if isinstance(value, bool) or not isinstance(value,
                                                     (Value, int, float)):
            raise ValueError("Expected `value` to be `int` or `float` but got"
                             f" {type(value).__name__}")
        self._data: float = (value.data
                             if isinstance(value, Value) else float(value))
        self._label: str = label
        self._op: str = op
        self._parents: tuple[Value, ...] = tuple(parents)

    @property
    def data(self) -> float:
        """Returns the data held by this instance."""
        return self._data

    @property
    def label(self) -> str:
        """Returns the label."""
        return self._label

    @property
    def op(self) -> str:
        """Returns the operation that generated this value."""
        return self._op

    @property
    def parents(self) -> tuple[Value, ...]:
        """Returns the parent values."""
        return self._parents

    def __repr__(self) -> str:
        return f"Value({self.data}, label={self.label}, op={self.op})"

    def __add__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            data = self.data + other.data
            label = f"{self.label} + {other.label}"
            parents = (self, other)
        elif isinstance(other, (int, float)):
            data = self.data + other
            other_v = Value(other, label=str(other))
            label = f"{self.label} + {other_v.label}"
            parents = (self, other_v)
        else:
            return NotImplemented

        return Value(data, label=label, op="+", parents=parents)

    def __radd__(self, other: Value | int | float) -> Value:
        return self.__add__(other)

    def __mul__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            data = self.data * other.data
            label = f"{self.label} * {other.label}"
            parents = (self, other)
        elif isinstance(other, (int, float)):
            data = self.data * other
            other_v = Value(other, label=str(other))
            label = f"{self.label} * {other_v.label}"
            parents = (self, other_v)
        else:
            return NotImplemented

        return Value(data, label=label, op="*", parents=parents)

    def __rmul__(self, other: Value | int | float) -> Value:
        return self.__mul__(other)

    def __pow__(self, other: int | float) -> Value:
        if isinstance(other, (int, float)):
            data = self.data ** other
            other_v = Value(other, label=str(other))
            label = f"{self.label}^{other_v.label}"
            parents = (self, other_v)
        else:
            return NotImplemented

        return Value(data, label=label, op="**", parents=parents)

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * (other**-1)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return other * (self**-1)

    def __neg__(self) -> Value:
        return self * -1

    def __abs__(self) -> Value:
        return Value(abs(self.data),
                     label=f"|{self.label}|",
                     op="abs",
                     parents=(self,))

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:
        return other + (-self)

    def exp(self) -> Value:
        """Raises `e` to this value."""
        return Value(math.exp(self.data),
                     label=f"exp({self.label})",
                     op="exp",
                     parents=(self,))

    def relu(self) -> Value:
        """Calculates the ReLU of this value."""
        return Value(0 if self.data <= 0 else self.data,
                     label=f"relu({self.label})",
                     op="relu",
                     parents=(self,))

    def tanh(self) -> Value:
        """Applies tanh to this value."""
        c = (2 * self).exp()
        return (c - 1) / (c + 1)
