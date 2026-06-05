from tiny_mlp_course.loss import squared_error


def test_squared_error_for_one_example() -> None:
    assert squared_error(3.0, 1.0) == 4.0
