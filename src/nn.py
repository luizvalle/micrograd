"""Neural network building blocks backed by automatic differentiation.

Provides a base module interface and neurons, with layers and multilayer
perceptrons planned for future implementation.
"""

import random

from collections.abc import Iterable, Callable

from engine import Value


class Module:
    """Base class for modules."""
    def parameters(self) -> list[Value]:
        """Returns the parameters of this module."""
        return []

    def zero_grad(self) -> None:
        """Zeros the gradients of all the parameters in this module."""
        for p in self.parameters():
            p.zero_grad()


class Neuron(Module):
    """Represents a neuron."""
    def __init__(self,
                 n_in: int,
                 actv_fn: Callable[[Value], Value] = lambda x: x) -> None:
        """Initializes the neuron.

        Args:
            n_in: Number of inputs to the neuron. Must be at least 1.
            actv_fn: Activation function applied to the weighted sum plus bias.
                Defaults to the identity function.

        Raises:
            ValueError: If n_in is less than 1.
        """
        if n_in < 1:
            raise ValueError(f"`n_in` must be at least 1, got {n_in=}")
        self._ws = tuple(Value(random.uniform(-1,1)) for _ in range(n_in))
        self._b = Value(random.uniform(-1,1))
        self._actv_fn = actv_fn

    def __call__(self, xs: Iterable[Value | float | int]) -> Value:
        xs = tuple(xs)
        num_xs = len(xs)
        if num_xs != len(self._ws):
            raise ValueError("Expected number of weights and number of "
                             f"inputs to be the same, but got {num_xs} inputs "
                             f"and {len(self._ws)} weights.")
        s: Value = sum((w * x for w, x in zip(self._ws, xs)), start=self._b)
        return self._actv_fn(s)

    def parameters(self) -> list[Value]:
        return [*self._ws, self._b]
