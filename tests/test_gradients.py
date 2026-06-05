def loss_for_weight(weight: float) -> float:
    prediction = weight * 2
    target = 10.0
    error = prediction - target
    return error * error


def test_scalar_gradient_matches_finite_difference() -> None:
    weight = 3.0
    epsilon = 1e-6

    loss_before = loss_for_weight(weight)
    loss_after = loss_for_weight(weight + epsilon)

    numerical_gradient = (loss_after - loss_before) / epsilon
    hand_written_gradient = 2.0 * ((2.0 * weight) - 10.0) * 2.0

    assert abs(numerical_gradient - hand_written_gradient) < 1e-4