import os
from typing import Callable, TypeAlias

TensorData: TypeAlias = float | list["TensorData"]
Shape: TypeAlias = tuple[int, ...]
TensorGrad: TypeAlias = TensorData


def _shape_of(data: TensorData) -> Shape:
    if isinstance(data, float):
        return ()

    if isinstance(data, list):
        if not data:
            return (0,)

        first_shape = _shape_of(data[0])

        for item in data[1:]:
            if _shape_of(item) != first_shape:
                raise ValueError("Tensor data must be rectangular.")

        return len(data), *first_shape

    raise TypeError("Tensor data must be a float or nested lists of floats.")


def _init_gradients(data: TensorData) -> TensorGrad:
    if isinstance(data, float):
        return 0.0

    if isinstance(data, list):
        return [_init_gradients(item) for item in data]

    raise TypeError("Tensor data must be a float or nested lists of floats.")


def _ensure_same_shape(left: "Tensor", right: "Tensor", operation: str) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"Tensor {operation} requires tensors with the same shape. "
            f"Got {left.shape} and {right.shape}."
        )


def _elementwise_binary(
    left: TensorData,
    right: TensorData,
    operation: Callable[[float, float], float],
) -> TensorData:
    if isinstance(left, float) and isinstance(right, float):
        return operation(left, right)

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Tensor operation requires matching nested structure.")

        return [
            _elementwise_binary(left_item, right_item, operation)
            for left_item, right_item in zip(left, right)
        ]

    raise ValueError("Tensor operation requires matching nested structure.")


def _elementwise_unary(
    data: TensorData,
    operation: Callable[[float], float],
) -> TensorData:
    if isinstance(data, float):
        return operation(data)

    if isinstance(data, list):
        return [_elementwise_unary(item, operation) for item in data]

    raise ValueError("Tensor operation requires matching nested structure.")


def _elementwise_add(left: TensorData, right: TensorData) -> TensorData:
    return _elementwise_binary(left, right, lambda left_value, right_value: left_value + right_value)


def _elementwise_subtract(left: TensorData, right: TensorData) -> TensorData:
    return _elementwise_binary(left, right, lambda left_value, right_value: left_value - right_value)


def _elementwise_multiply(left: TensorData, right: TensorData) -> TensorData:
    return _elementwise_binary(left, right, lambda left_value, right_value: left_value * right_value)


def _elementwise_negate(data: TensorData) -> TensorData:
    return _elementwise_unary(data, lambda value: -value)


def _elementwise_square(data: TensorData) -> TensorData:
    return _elementwise_unary(data, lambda value: value * value)


def _elementwise_square_slope(data: TensorData) -> TensorData:
    return _elementwise_unary(data, lambda value: 2 * value)


def _accumulate_gradients(current: TensorGrad, contribution: TensorGrad) -> TensorGrad:
    return _elementwise_add(current, contribution)


class Tensor:
    """A number or nested list of numbers that remembers how it was made.

    data stores the forward value.
    grad stores how much the final answer changes because of each data value.
    """

    def __init__(
        self,
        data: TensorData,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """Store the data, its starting grad, and the Tensors that made it."""
        self.data = data
        self.shape: Shape = _shape_of(data)
        self.grad: TensorGrad = _init_gradients(data)
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        """Make print(Tensor) easy to read."""
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self + other."""
        _ensure_same_shape(self, other, "addition")

        out = Tensor(
            data=_elementwise_add(self.data, other.data),
            _children=(self, other),
            _op="+",
        )

        def _backward() -> None:
            """Move this result's grad back to both inputs.

            If out = self + other, increasing either input by 1 increases out by 1.
            Each input therefore receives +1 times the result's gradient.
            """
            # The left input affects the sum positively: d_out/d_self = 1.
            self.grad = _accumulate_gradients(self.grad, out.grad)
            # The right input also affects the sum positively: d_out/d_other = 1.
            other.grad = _accumulate_gradients(other.grad, out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self - other."""
        _ensure_same_shape(self, other, "subtraction")

        out = Tensor(
            data=_elementwise_subtract(self.data, other.data),
            _children=(self, other),
            _op="-",
        )

        def _backward() -> None:
            """Move this result's grad back to both inputs.

            If out = self - other, increasing self by 1 increases out by 1.
            Increasing other by 1 decreases out by 1, so the right input gets a minus sign.
            """
            # The left input affects the difference positively: d_out/d_self = 1.
            self.grad = _accumulate_gradients(self.grad, out.grad)
            # The right input is subtracted: d_out/d_other = -1, so subtract out.grad.
            other.grad = _accumulate_gradients(other.grad, _elementwise_negate(out.grad))

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self * other."""
        _ensure_same_shape(self, other, "multiplication")

        out = Tensor(
            data=_elementwise_multiply(self.data, other.data),
            _children=(self, other),
            _op="*",
        )

        def _backward() -> None:
            """Move this result's grad back using the other input's number.

            If out = self * other, changing self by 1 changes out by other.data.
            Changing other by 1 changes out by self.data.
            """
            # The left input's slope is the right input's value: d_out/d_self = other.data.
            self.grad = _accumulate_gradients(
                self.grad,
                _elementwise_multiply(other.data, out.grad),
            )
            # The right input's slope is the left input's value: d_out/d_other = self.data.
            other.grad = _accumulate_gradients(
                other.grad,
                _elementwise_multiply(self.data, out.grad),
            )

        out._backward = _backward
        return out

    def __pow__(self, power: int) -> "Tensor":
        """Make a new Tensor for self ** 2."""
        if power != 2:
            raise ValueError(f"Only powers of 2 are supported, not {power}.")

        out = Tensor(
            data=_elementwise_square(self.data),
            _children=(self,),
            _op=f"**{power}",
        )

        def _backward() -> None:
            """Move this result's grad back through squaring.

            If out = self ** 2, changing self by 1 changes out by about 2 * self.data.
            Multiply that local slope by out.grad to pass the final sensitivity backward.
            """
            # The slope of x squared is 2*x, so scale out.grad by 2 * self.data.
            self.grad = _accumulate_gradients(
                self.grad,
                _elementwise_multiply(_elementwise_square_slope(self.data), out.grad),
            )

        out._backward = _backward
        return out

    def backward(self, gradient: TensorGrad | None = None) -> None:
        """Fill in grads, starting from this final Tensor.

        Scalar final Tensors start with grad 1.0.
        Non-scalar final Tensors need an initial gradient with the same shape.
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

        if gradient is None:
            if self.shape != ():
                raise ValueError(
                    "backward() on a non-scalar Tensor requires an initial gradient."
                )

            # The final scalar Tensor changes one-for-one with itself.
            self.grad = 1.0
        else:
            gradient_shape = _shape_of(gradient)
            if gradient_shape != self.shape:
                raise ValueError(
                    "Initial gradient must have the same shape as this Tensor. "
                    f"Got {gradient_shape} and {self.shape}."
                )

            self.grad = gradient

        # Walk from the final Tensor back to the leaves so every operation can
        # pass its gradient contribution to the Tensors that created it.
        for node in reversed(topo):
            node._backward()


__all__ = ["Tensor"]
