"""Neural network building blocks backed by automatic differentiation.

Provides a base module interface and neurons, with layers and multilayer
perceptrons.
"""

import random

from collections.abc import Iterable, Callable

from engine import Value


class Module:
    """Base class for modules."""
    def parameters(self) -> tuple[Value, ...]:
        """Returns the parameters of this module."""
        return ()

    def zero_grad(self) -> None:
        """Zeros the gradients of all the parameters in this module."""
        for p in self.parameters():
            p.zero_grad()


class Neuron(Module):
    """Represents a neuron."""
    def __init__(self,
                 n_in: int,
                 actv_fn: Callable[[Value], Value]) -> None:
        """Initializes the neuron.

        Args:
            n_in: Number of inputs to the neuron. Must be at least 1.
            actv_fn: Activation function applied to the weighted sum plus bias.

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

    def parameters(self) -> tuple[Value, ...]:
        return (*self._ws, self._b)


class Layer(Module):
    """Represents a layer of neurons."""
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 actv_fn: Callable[[Value], Value] = lambda x: x) -> None:
        """Initializes the layer.

        Args:
            n_in: Number of inputs to each neuron. Must be at least 1.
            n_out: Number of neurons in the layer. Must be at least 1.
            actv_fn: Activation function applied to each neuron's weighted
                sum plus bias. Defaults to the identity function.

        Raises:
            ValueError: If n_in or n_out is less than 1.
        """
        if n_in < 1:
            raise ValueError(f"`n_in` must be at least 1, got {n_in=}")
        if n_out < 1:
            raise ValueError(f"`n_out` must be at least 1, got {n_out=}")
        self._neurons: tuple[Neuron, ...] = tuple(
            Neuron(n_in, actv_fn) for _ in range(n_out))

    def __call__(self, xs: Iterable[Value | float | int]) -> tuple[Value, ...]:
        """Returns the outputs of the layer's neurons for the given inputs.

        Args:
            xs: Inputs to each neuron in the layer.

        Returns:
            A tuple containing the output of each neuron.

        Raises:
            ValueError: If the number of inputs does not match the number
                of weights in a neuron.
        """
        xs = tuple(xs)
        return tuple(n(xs) for n in self._neurons)

    def parameters(self) -> tuple[Value, ...]:
        """Returns the parameters of all neurons in this layer."""
        return tuple(p for n in self._neurons for p in n.parameters())
