import os
from typing import Callable


class Value:
    """A number that also remembers how it was made.

    data is the number.
    grad will later hold how much the final answer changes because of this number.
    """

    def __init__(
        self,
        data: float,
        _children: tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:
        """Store the number, its starting grad, and the Values that made it."""
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Make print(value) easy to read."""
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: "Value") -> "Value":
        """Make a new Value for self + other."""
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            """Move this result's grad back to both inputs."""
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        """Make a new Value for self * other."""
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            """Move this result's grad back using the other input's number."""
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power: int) -> "Value":
        """Make a new Value for self ** 2."""
        if power != 2:
            raise ValueError(f"Only powers of 2 are supported, not {power}.")

        out = Value(
            data=self.data * self.data,
            _children=(self,),
            _op=f"**{power}",
        )

        def _backward() -> None:
            """Move this result's grad back using the other input's number.

            backward rule: near x = 3.0, changing x changes y about 2 * x times as much.
            """
            self.grad += 2 * self.data * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Fill in grads, starting from this final Value.

        First find all earlier Values.
        Then work backward so each Value passes its grad to the Values that made it.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def _build_topo(value: Value) -> None:
            """Put earlier Values before later Values."""
            if value in visited:
                return

            visited.add(value)

            for child in value._prev:
                _build_topo(child)

            topo.append(value)

        _build_topo(self)

        if os.environ.get("DEBUG"):
            print(f"Order: {topo}")

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
