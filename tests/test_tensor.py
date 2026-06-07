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