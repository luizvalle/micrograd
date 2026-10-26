"""Implementation of the Value class"""

from __future__ import annotations



class Value:
    """Scalar value that keeps track of dependencies and calculates gradients.

    Attributes:
        data: The scalar value held by this object.
    """

    def __init__(self, data: int | float) -> None:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ValueError("Expected data to be `int` or `float` but got"
                             f"{type(data).__name__}")
        self._data: float = float(data)

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
            f"`int`, or, `float`, but got {type(other).__name__}")

    def __radd__(self, other: Value | int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        if isinstance(other, Value):
            return Value(self._data * other._data)

        if isinstance(other, (int, float)):
            return Value(self._data * other)

        raise ValueError(
            "Expected `other` to be of type `Value`, "
            f"`int`, or, `float`, but got {type(other).__name__}")



if __name__ == "__main__":
    a = Value(1.0)
    b = Value(2.0)

    print(f"{a + b=}")
    print(f"{a + 3}")
    print(f"{3 + a}")
    print(f"")
