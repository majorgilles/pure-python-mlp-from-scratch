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
            """Move this result's grad back to both inputs.

            If out = self + other, increasing either input by 1 increases out by 1.
            Each input therefore receives +1 times the result's gradient.
            """
            # The left input affects the sum positively: d_out/d_self = 1.
            self.grad += out.grad
            # The right input also affects the sum positively: d_out/d_other = 1.
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: "Value") -> "Value":
        """Make a new Value for self - other."""
        out = Value(data=self.data - other.data, _children=(self, other), _op="-")

        def _backward() -> None:
            """Move this result's grad back to both inputs.

            If out = self - other, increasing self by 1 increases out by 1.
            Increasing other by 1 decreases out by 1, so the right input gets a minus sign.
            """
            # The left input affects the difference positively: d_out/d_self = 1.
            self.grad += out.grad
            # The right input is subtracted: d_out/d_other = -1, so subtract out.grad.
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        """Make a new Value for self * other."""
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            """Move this result's grad back using the other input's number.

            If out = self * other, changing self by 1 changes out by other.data.
            Changing other by 1 changes out by self.data.
            """
            # The left input's slope is the right input's value: d_out/d_self = other.data.
            self.grad += other.data * out.grad
            # The right input's slope is the left input's value: d_out/d_other = self.data.
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
            """Move this result's grad back through squaring.

            If out = self ** 2, changing self by 1 changes out by about 2 * self.data.
            Multiply that local slope by out.grad to pass the final sensitivity backward.
            """
            # The slope of x squared is 2*x, so scale out.grad by 2 * self.data.
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

        # The final Value changes one-for-one with itself, so its starting grad is 1.
        self.grad = 1.0

        # Walk from the final Value back to the leaves so every operation can
        # pass its gradient contribution to the Values that created it.
        for node in reversed(topo):
            node._backward()
