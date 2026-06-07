import os
from typing import Callable


class Tensor:
    """A number that also remembers how it was made.

    data is the number.
    grad will later hold how much the final answer changes because of this number.
    """

    def __init__(
        self,
        data: float,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """Store the number, its starting grad, and the Tensors that made it."""
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Make print(Tensor) easy to read."""
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self + other."""
        out = Tensor(data=self.data + other.data, _children=(self, other), _op="+")

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

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self - other."""
        out = Tensor(data=self.data - other.data, _children=(self, other), _op="-")

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

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self * other."""
        out = Tensor(data=self.data * other.data, _children=(self, other), _op="*")

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

    def __pow__(self, power: int) -> "Tensor":
        """Make a new Tensor for self ** 2."""
        if power != 2:
            raise ValueError(f"Only powers of 2 are supported, not {power}.")

        out = Tensor(
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
        """Fill in grads, starting from this final Tensor.

        First find all earlier Tensors.
        Then work backward so each Tensor passes its grad to the Tensors that made it.
        """
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def _build_topo(tensor: Tensor) -> None:
            """Put earlier Tensors before later Tensors."""
            if tensor in visited:
                return

            visited.add(tensor)

            for child in tensor._prev:
                _build_topo(child)

            topo.append(tensor)

        _build_topo(self)

        if os.environ.get("DEBUG"):
            print(f"Order: {topo}")

        # The final Tensor changes one-for-one with itself, so its starting grad is 1.
        self.grad = 1.0

        # Walk from the final Tensor back to the leaves so every operation can
        # pass its gradient contribution to the Tensors that created it.
        for node in reversed(topo):
            node._backward()


__all__ = ["Tensor"]