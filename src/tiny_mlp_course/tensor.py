"""Tiny tensor-shaped autograd helper for teaching.

The goal is to keep the scalar gradient rules visible while allowing data to be
stored as nested Python lists. Helpers such as `_add`, `_sub`, and `_mul` apply
simple scalar operations element by element; `_apply_to_pair` is the recursive
list walker that makes those helpers work for 1D, 2D, or deeper tensors.

Example: 2D addition, step by step.

```python
left = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
right = Tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
out = left + right
```

The forward call is:

```text
Tensor.__add__
  _ensure_same_shape(left, right, "addition")
  _add(left.data, right.data)
    _apply_to_pair(left_matrix, right_matrix)
      _apply_to_pair(left_row_0, right_row_0)
        scalar_operation(1.0, 10.0) -> 11.0
        scalar_operation(2.0, 20.0) -> 22.0
        scalar_operation(3.0, 30.0) -> 33.0
      _apply_to_pair(left_row_1, right_row_1)
        scalar_operation(4.0, 40.0) -> 44.0
        scalar_operation(5.0, 50.0) -> 55.0
        scalar_operation(6.0, 60.0) -> 66.0
  Tensor.__init__ for out
  store out._backward from the addition operation
```

So `out.data` becomes:

```python
[[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]
```

Because `out` is not scalar, backward needs one starting gradient per output
position:

```python
out.backward([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

The backward call is:

```text
Tensor.backward
  _build_topo(out)  # puts left and right before out
  _shape_of(initial_gradient) -> (2, 3)
  out.grad = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  out._backward()
```

For addition, `out._backward()` uses the same scalar rule at every position:

```text
left.grad += out.grad
right.grad += out.grad
```

In code, the scalar-looking rule is written with `_add` so it also works on
nested lists:

```python
left.grad = _add(left.grad, out.grad)
right.grad = _add(right.grad, out.grad)
```

That means the gradient recursion is:

```text
_add(left.grad, out.grad)
  _apply_to_pair(left_grad_matrix, out_grad_matrix)
    _apply_to_pair(left_grad_row_0, out_grad_row_0)
      scalar_operation(0.0, 1.0) -> 1.0
      scalar_operation(0.0, 2.0) -> 2.0
      scalar_operation(0.0, 3.0) -> 3.0
    _apply_to_pair(left_grad_row_1, out_grad_row_1)
      scalar_operation(0.0, 4.0) -> 4.0
      scalar_operation(0.0, 5.0) -> 5.0
      scalar_operation(0.0, 6.0) -> 6.0
```

The same recursion updates `right.grad`. The important teaching point is that
recursion handles the list shape, while `_backward()` still shows the local
scalar gradient rule.
"""

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


def _apply_to_pair(
    left: TensorData,
    right: TensorData,
    scalar_operation: Callable[[float, float], float],
) -> TensorData:
    """Apply one scalar operation to matching positions in nested data.

    For scalar data, this calls scalar_operation(left, right) once.
    For list-shaped data, it walks both lists in lockstep and recursively applies
    the same scalar operation to each matching pair. For a 2D tensor, that means
    it first pairs row 0 with row 0, row 1 with row 1, and so on; inside each row,
    it pairs the matching scalar positions. This keeps the visible operation
    rules scalar-like while still supporting non-scalar tensors.
    """
    if isinstance(left, float) and isinstance(right, float):
        return scalar_operation(left, right)

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Tensor operation requires matching nested structure.")

        return [
            _apply_to_pair(left_item, right_item, scalar_operation)
            for left_item, right_item in zip(left, right)
        ]

    raise ValueError("Tensor operation requires matching nested structure.")


def _apply_to_each(
    data: TensorData,
    scalar_operation: Callable[[float], float],
) -> TensorData:
    """Apply one scalar operation to every position in nested data.

    For scalar data, this calls scalar_operation(data) once.
    For list-shaped data, it recursively walks into each list item and applies
    the same scalar operation to every float it finds. The output keeps the same
    nested-list shape as the input.
    """
    if isinstance(data, float):
        return scalar_operation(data)

    if isinstance(data, list):
        return [_apply_to_each(item, scalar_operation) for item in data]

    raise ValueError("Tensor operation requires matching nested structure.")


def _add(left: TensorData, right: TensorData) -> TensorData:
    """Return left + right, element by element for nested data."""
    return _apply_to_pair(
        left,
        right,
        lambda left_value, right_value: left_value + right_value,
    )


def _sub(left: TensorData, right: TensorData) -> TensorData:
    """Return left - right, element by element for nested data."""
    return _apply_to_pair(
        left,
        right,
        lambda left_value, right_value: left_value - right_value,
    )


def _mul(left: TensorData, right: TensorData) -> TensorData:
    """Return left * right, element by element for nested data."""
    return _apply_to_pair(
        left,
        right,
        lambda left_value, right_value: left_value * right_value,
    )


def _square(data: TensorData) -> TensorData:
    """Return data ** 2, element by element for nested data."""
    return _apply_to_each(data, lambda value: value * value)


def _double(data: TensorData) -> TensorData:
    """Return 2 * data, element by element for nested data."""
    return _apply_to_each(data, lambda value: 2 * value)


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
            data=_add(self.data, other.data),
            _children=(self, other),
            _op="+",
        )

        def _backward() -> None:
            """Move this result's grad back to both inputs.

            If out = self + other, increasing either input by 1 increases out by 1.
            Each input therefore receives +1 times the result's gradient.
            For nested data, the same scalar rule is applied element by element.
            """
            # Scalar rule: self.grad += out.grad
            self.grad = _add(self.grad, out.grad)
            # Scalar rule: other.grad += out.grad
            other.grad = _add(other.grad, out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self - other."""
        _ensure_same_shape(self, other, "subtraction")

        out = Tensor(
            data=_sub(self.data, other.data),
            _children=(self, other),
            _op="-",
        )

        def _backward() -> None:
            """Move this result's grad back to both inputs.

            If out = self - other, increasing self by 1 increases out by 1.
            Increasing other by 1 decreases out by 1, so the right input gets a minus sign.
            For nested data, the same scalar rule is applied element by element.
            """
            # Scalar rule: self.grad += out.grad
            self.grad = _add(self.grad, out.grad)
            # Scalar rule: other.grad -= out.grad
            other.grad = _sub(other.grad, out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Make a new Tensor for self * other."""
        _ensure_same_shape(self, other, "multiplication")

        out = Tensor(
            data=_mul(self.data, other.data),
            _children=(self, other),
            _op="*",
        )

        def _backward() -> None:
            """Move this result's grad back using the other input's number.

            If out = self * other, changing self by 1 changes out by other.data.
            Changing other by 1 changes out by self.data.
            For nested data, the same scalar rule is applied element by element.
            """
            # Scalar rule: self.grad += other.data * out.grad
            self.grad = _add(self.grad, _mul(other.data, out.grad))
            # Scalar rule: other.grad += self.data * out.grad
            other.grad = _add(other.grad, _mul(self.data, out.grad))

        out._backward = _backward
        return out

    def __pow__(self, power: int) -> "Tensor":
        """Make a new Tensor for self ** 2."""
        if power != 2:
            raise ValueError(f"Only powers of 2 are supported, not {power}.")

        out = Tensor(
            data=_square(self.data),
            _children=(self,),
            _op=f"**{power}",
        )

        def _backward() -> None:
            """Move this result's grad back through squaring.

            If out = self ** 2, changing self by 1 changes out by about 2 * self.data.
            Multiply that local slope by out.grad to pass the final sensitivity backward.
            For nested data, the same scalar rule is applied element by element.
            """
            slope = _double(self.data)

            # Scalar rule: self.grad += (2 * self.data) * out.grad
            self.grad = _add(self.grad, _mul(slope, out.grad))

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
