import pytest

from tiny_mlp_course.tensor import Tensor


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


def test_tensor_rejects_ragged_nested_lists() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        Tensor([[1.0], [2.0, 3.0]])


def test_tensor_adds_1d_data_elementwise() -> None:
    result = Tensor([1.0, 2.0]) + Tensor([3.0, 4.0])

    assert result.data == [4.0, 6.0]
    assert result.shape == (2,)


def test_tensor_adds_2d_data_elementwise() -> None:
    result = Tensor([[1.0, 2.0], [3.0, 4.0]]) + Tensor(
        [[10.0, 20.0], [30.0, 40.0]]
    )

    assert result.data == [[11.0, 22.0], [33.0, 44.0]]
    assert result.shape == (2, 2)


def test_tensor_rejects_add_with_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        Tensor([1.0, 2.0]) + Tensor([3.0])


def test_tensor_adds_1d_gradients_elementwise() -> None:
    left = Tensor([1.0, 2.0])
    right = Tensor([3.0, 4.0])

    result = left + right
    result.backward([10.0, 20.0])

    assert left.grad == [10.0, 20.0]
    assert right.grad == [10.0, 20.0]


def test_tensor_subtracts_1d_data_and_gradients_elementwise() -> None:
    left = Tensor([5.0, 7.0])
    right = Tensor([2.0, 3.0])

    result = left - right
    result.backward([10.0, 20.0])

    assert result.data == [3.0, 4.0]
    assert left.grad == [10.0, 20.0]
    assert right.grad == [-10.0, -20.0]


def test_tensor_multiplies_1d_data_and_gradients_elementwise() -> None:
    left = Tensor([2.0, 3.0])
    right = Tensor([4.0, 5.0])

    result = left * right
    result.backward([10.0, 20.0])

    assert result.data == [8.0, 15.0]
    assert left.grad == [40.0, 100.0]
    assert right.grad == [20.0, 60.0]


def test_tensor_squares_1d_data_and_gradients_elementwise() -> None:
    tensor = Tensor([2.0, -3.0])

    result = tensor**2
    result.backward([10.0, 20.0])

    assert result.data == [4.0, 9.0]
    assert tensor.grad == [40.0, -120.0]


def test_tensor_backward_on_non_scalar_requires_initial_gradient() -> None:
    result = Tensor([1.0, 2.0]) + Tensor([3.0, 4.0])

    with pytest.raises(ValueError, match="requires an initial gradient"):
        result.backward()


def test_tensor_backward_rejects_wrong_initial_gradient_shape() -> None:
    result = Tensor([1.0, 2.0]) + Tensor([3.0, 4.0])

    with pytest.raises(ValueError, match="same shape"):
        result.backward([1.0])
