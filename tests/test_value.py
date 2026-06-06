from tiny_mlp_course.value import Value


def test_value_forward_expression() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b
    e = d + c
    f = e * e

    assert f.data == 16.0


def test_value_backward_expression() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    d = a * b
    e = d + c
    f = e * e

    f.backward()

    assert a.grad == -24.0
    assert b.grad == 16.0
    assert c.grad == 8.0
