"""Implementation of the Value class"""

from __future__ import annotations

import math

from collections import defaultdict
from collections.abc import Iterable, Callable


def _get_backprop_order(root: Value) -> list[Value]:
    num_children_from_value: dict[Value, int] = defaultdict(int)
    visited: set[Value] = set()
    frontier: list[Value] = [root]
    while frontier:
        n = frontier.pop()
        if n in visited:
            continue
        for p in n.parents:
            num_children_from_value[p] += 1
            if p not in visited:
                frontier.append(p)
        visited.add(n)

    out: list[Value] = []
    frontier = [root]
    while frontier:
        n = frontier.pop()
        out.append(n)
        for p in n.parents:
            num_children_from_value[p] -= 1
            if num_children_from_value[p] == 0:
                frontier.append(p)

    return out


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
        parents = tuple(parents)
        if bool(op) != bool(parents):
            raise ValueError("Either both `op` and `parents` should be "
                             "specified or none.")
        self.data: float = (value.data
                            if isinstance(value, Value) else float(value))
        self.label: str = label
        self._op: str = op
        self._parents: tuple[Value, ...] = parents
        self._grad = 0.0
        self._backward: Callable[[], None] = lambda: None

    @property
    def op(self) -> str:
        """Returns the operation that generated this value."""
        return self._op

    @property
    def parents(self) -> tuple[Value, ...]:
        """Returns the parent values."""
        return self._parents

    @property
    def grad(self) -> float:
        """Returns the gradient of this value with respect to the output."""
        return self._grad

    def zero_grad(self) -> None:
        """Resets the gradient to 0.0."""
        self._grad = 0.0

    def __repr__(self) -> str:
        return f"Value({self.data}, label={self.label}, op={self.op})"

    def __add__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            data = self.data + other.data
        elif isinstance(other, (int, float)):
            data = self.data + other
            other = Value(other)
        else:
            return NotImplemented
            
        parents = (self, other)

        out = Value(data, op="+", parents=parents)

        def _backward() -> None:
            self._grad += out.grad
            other._grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Value | int | float) -> Value:
        return self.__add__(other)

    def __mul__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            data = self.data * other.data
        elif isinstance(other, (int, float)):
            data = self.data * other
            other = Value(other)
        else:
            return NotImplemented
            
        parents = (self, other)

        out = Value(data, op="*", parents=parents)

        def _backward() -> None:
            self._grad += out.grad * other.data
            other._grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other: Value | int | float) -> Value:
        return self.__mul__(other)

    def __pow__(self, other: int | float) -> Value:
        if isinstance(other, (int, float)):
            data = self.data ** other
        else:
            return NotImplemented

        parents = (self,)
        out = Value(data, op=f"**{other}", parents=parents)

        def _backward() -> None:
            self._grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward

        return out

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * (other**-1)

    def __rtruediv__(self, other: Value | int | float) -> Value:
        return other * (self**-1)

    def __neg__(self) -> Value:
        return self * -1

    def __abs__(self) -> Value:
        out = Value(abs(self.data), op="abs", parents=(self,))

        def _backward() -> None:
            c = 0.0 if self.data == 0 else (1 if self.data > 0 else -1)
            self._grad += c * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:
        return other + (-self)

    def exp(self) -> Value:
        """Raises `e` to this value."""
        out =  Value(math.exp(self.data), op="exp", parents=(self,))

        def _backward() -> None:
            self._grad += out.grad * math.exp(self.data)

        out._backward = _backward

        return out

    def relu(self) -> Value:
        """Calculates the ReLU of this value."""
        out = Value(0 if self.data <= 0 else self.data,
                     op="relu", parents=(self,))

        def _backward() -> None:
            self._grad += out.grad * bool(self.data > 0)

        out._backward = _backward

        return out

    def tanh(self) -> Value:
        """Applies tanh to this value."""
        c = (2 * self).exp()
        return (c - 1) / (c + 1)

    def backward(self) -> None:
        """Runs backpropagation with this node as the output."""
        self._grad = 1.0
        for n in _get_backprop_order(self):
            n._backward()

