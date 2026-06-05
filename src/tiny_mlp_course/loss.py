def squared_error(prediction: float, target: float) -> float:
    """Return the squared wrongness for one prediction and one target."""
    error = prediction - target
    return error * error
