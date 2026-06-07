from tiny_mlp_course.tensor import Tensor
import pytest


def test_tensor_keeps_scalar_forward_expression() -> None:
    a = Tensor(2.0)
    b = Tensor(-3.0)
    c = Tensor(10.0)

    d = a * b
    e = d + c
    f = e * e

    assert f.data == 16.0


def test_tensor_keeps_scalar_backward_expression() -> None:
    a = Tensor(2.0)
    b = Tensor(-3.0)
    c = Tensor(10.0)

    d = a * b
    e = d + c
    f = e * e

    f.backward()

    assert a.grad == -24.0
    assert b.grad == 16.0
    assert c.grad == 8.0


def test_tensor_records_scalar_shape() -> None:
    tensor = Tensor(1.0)

    assert tensor.shape == ()
    assert tensor.grad == 0.0


def test_tensor_records_1d_shape_and_zero_grad() -> None:
    tensor = Tensor([1.0, 2.0, 3.0])

    assert tensor.shape == (3,)
    assert tensor.grad == [0.0, 0.0, 0.0]


def test_tensor_records_2d_shape_and_zero_grad() -> None:
    tensor = Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tensor.shape == (2, 2)
    assert tensor.grad == [[0.0, 0.0], [0.0, 0.0]]


def test_tensor_records_3d_shape_and_zero_grad() -> None:
    tensor = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]])

    assert tensor.shape == (2, 2, 1)
    assert tensor.grad == [[[0.0], [0.0]], [[0.0], [0.0]]]


def test_tensor_add_rejects_non_scalar_data_for_now() -> None:
    with pytest.raises(NotImplementedError, match="scalar"):
        Tensor([1.0]) + Tensor([2.0])