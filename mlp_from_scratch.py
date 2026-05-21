"""Train the Week 05 one-hidden-layer MLP using only the Python standard library.

No PyTorch, NumPy, pandas, scikit-learn, or autograd is used here. The point of
this file is to make every moving part visible:

1. Load the existing Week 05 two-moons CSV.
2. Split it into train/test rows.
3. Store weights and biases as Python lists of floats.
4. Run the forward pass by hand.
5. Compute BCE-with-logits loss by hand.
6. Backpropagate gradients by hand.
7. Update parameters with Adam by hand.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

FeatureVector = tuple[float, float]
Vector = list[float]
Matrix = list[list[float]]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "week-05-moons.csv"
REQUIRED_DATASET_COLUMNS = ("x_position", "y_position", "class_label")


@dataclass(frozen=True)
class Example:
    """One labeled point in the moon-shaped dataset."""

    x_position: float
    y_position: float
    class_label: int

    def features(self) -> FeatureVector:
        return (self.x_position, self.y_position)


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split for labeled examples."""

    train: list[Example]
    test: list[Example]


@dataclass(frozen=True)
class ForwardCache:
    """Values saved during the forward pass because backprop needs them."""

    features: FeatureVector
    hidden_raw: Vector
    hidden_after_relu: Vector
    logit: float


@dataclass(frozen=True)
class TrainingResult:
    """Final metrics and the full loss curve."""

    losses: list[float]
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float


@dataclass(frozen=True)
class CliOptions:
    """Command-line options with concrete types."""

    data_path: Path
    seed: int
    test_size: float
    hidden_units: int
    epochs: int
    learning_rate: float
    report_every: int
    output_dir: Path


def zeros_vector(length: int) -> Vector:
    return [0.0 for _ in range(length)]


def zeros_matrix(rows: int, columns: int) -> Matrix:
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


def zeros_like_matrix(matrix: Matrix) -> Matrix:
    return [[0.0 for _ in row] for row in matrix]


def resolve_data_path(data_path: Path) -> Path:
    """Resolve CLI data paths while allowing `--data-path ~/file.csv`."""

    expanded_path = data_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return (Path.cwd() / expanded_path).resolve()


def parse_float(raw_value: str | None, column_name: str, row_number: int) -> float:
    if raw_value is None:
        raise ValueError(f"Missing {column_name!r} in CSV row {row_number}.")

    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Could not parse {column_name!r} as float in CSV row {row_number}: "
            f"{raw_value!r}"
        ) from error


def parse_class_label(raw_value: str | None, row_number: int) -> int:
    if raw_value is None:
        raise ValueError(f"Missing 'class_label' in CSV row {row_number}.")

    try:
        label = int(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Could not parse 'class_label' as int in CSV row {row_number}: "
            f"{raw_value!r}"
        ) from error

    if label not in (0, 1):
        raise ValueError(
            f"Expected 'class_label' to be 0 or 1 in CSV row {row_number}, "
            f"got {label}."
        )

    return label


def load_examples_csv(path: Path) -> list[Example]:
    """Load the checked-in Week 05 moons dataset with only `csv` and `Path`."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Dataset CSV has no header row: {path}")

        missing_columns = [
            column for column in REQUIRED_DATASET_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing required columns {missing_columns}: {path}"
            )

        examples: list[Example] = []
        for row_number, row in enumerate(reader, start=2):
            examples.append(
                Example(
                    x_position=parse_float(
                        row.get("x_position"), "x_position", row_number
                    ),
                    y_position=parse_float(
                        row.get("y_position"), "y_position", row_number
                    ),
                    class_label=parse_class_label(
                        row.get("class_label"), row_number
                    ),
                )
            )

    if len(examples) == 0:
        raise ValueError(f"Dataset CSV has no data rows: {path}")

    return examples


def stratified_train_test_split(
    examples: list[Example], test_size: float, seed: int
) -> DatasetSplit:
    """Split examples while keeping roughly the same label mix in each split."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = random.Random(seed)
    examples_by_label: dict[int, list[Example]] = {}

    for example in examples:
        if example.class_label not in examples_by_label:
            examples_by_label[example.class_label] = []
        examples_by_label[example.class_label].append(example)

    train_examples: list[Example] = []
    test_examples: list[Example] = []

    for label in sorted(examples_by_label):
        label_examples = list(examples_by_label[label])
        rng.shuffle(label_examples)
        test_count = int(round(float(len(label_examples)) * test_size))
        test_examples.extend(label_examples[:test_count])
        train_examples.extend(label_examples[test_count:])

    rng.shuffle(train_examples)
    rng.shuffle(test_examples)

    return DatasetSplit(train=train_examples, test=test_examples)


def sigmoid(logit: float) -> float:
    """Convert one logit into a probability in a numerically stable way."""

    if logit >= 0.0:
        scaled = math.exp(-logit)
        return 1.0 / (1.0 + scaled)

    scaled = math.exp(logit)
    return scaled / (1.0 + scaled)


def binary_cross_entropy_with_logit(logit: float, label: int) -> float:
    """BCEWithLogitsLoss for one example.

    This stable formula is equivalent to:

        -label * log(sigmoid(logit))
        -(1 - label) * log(1 - sigmoid(logit))
    """

    label_float = float(label)
    return max(logit, 0.0) - (logit * label_float) + math.log1p(math.exp(-abs(logit)))


@dataclass
class OneHiddenLayerMLP:
    """A 2-input -> ReLU hidden layer -> 1-logit MLP.

    The fields are intentionally concrete. For 16 hidden units:

    - input_to_hidden_weights has shape 16 x 2
    - hidden_biases has shape 16
    - hidden_to_output_weights has shape 16
    - output_bias is one float
    """

    input_to_hidden_weights: Matrix
    hidden_biases: Vector
    hidden_to_output_weights: Vector
    output_bias: float
    grad_input_to_hidden_weights: Matrix
    grad_hidden_biases: Vector
    grad_hidden_to_output_weights: Vector
    grad_output_bias: float

    @classmethod
    def random(cls, hidden_units: int, seed: int) -> OneHiddenLayerMLP:
        """Initialize weights similarly to torch.nn.Linear's default range."""

        if hidden_units <= 0:
            raise ValueError("hidden_units must be positive.")

        rng = random.Random(seed)
        input_limit = 1.0 / math.sqrt(2.0)
        output_limit = 1.0 / math.sqrt(float(hidden_units))

        input_to_hidden_weights = [
            [rng.uniform(-input_limit, input_limit) for _ in range(2)]
            for _ in range(hidden_units)
        ]
        hidden_biases = [rng.uniform(-input_limit, input_limit) for _ in range(hidden_units)]
        hidden_to_output_weights = [
            rng.uniform(-output_limit, output_limit) for _ in range(hidden_units)
        ]
        output_bias = rng.uniform(-output_limit, output_limit)

        return cls(
            input_to_hidden_weights=input_to_hidden_weights,
            hidden_biases=hidden_biases,
            hidden_to_output_weights=hidden_to_output_weights,
            output_bias=output_bias,
            grad_input_to_hidden_weights=zeros_matrix(hidden_units, 2),
            grad_hidden_biases=zeros_vector(hidden_units),
            grad_hidden_to_output_weights=zeros_vector(hidden_units),
            grad_output_bias=0.0,
        )

    @property
    def hidden_units(self) -> int:
        return len(self.hidden_biases)

    def parameter_count(self) -> int:
        first_layer_weights = self.hidden_units * 2
        first_layer_biases = self.hidden_units
        output_layer_weights = self.hidden_units
        output_layer_biases = 1
        return (
            first_layer_weights
            + first_layer_biases
            + output_layer_weights
            + output_layer_biases
        )

    def zero_gradients(self) -> None:
        self.grad_input_to_hidden_weights = zeros_matrix(self.hidden_units, 2)
        self.grad_hidden_biases = zeros_vector(self.hidden_units)
        self.grad_hidden_to_output_weights = zeros_vector(self.hidden_units)
        self.grad_output_bias = 0.0

    def forward(self, features: FeatureVector) -> ForwardCache:
        """Run one point through linear_2(ReLU(linear_1(point)))."""

        hidden_raw: Vector = []
        hidden_after_relu: Vector = []

        for hidden_index in range(self.hidden_units):
            weights = self.input_to_hidden_weights[hidden_index]
            raw_value = (
                weights[0] * features[0]
                + weights[1] * features[1]
                + self.hidden_biases[hidden_index]
            )
            hidden_raw.append(raw_value)
            hidden_after_relu.append(max(0.0, raw_value))

        logit = self.output_bias
        for hidden_index in range(self.hidden_units):
            logit += (
                self.hidden_to_output_weights[hidden_index]
                * hidden_after_relu[hidden_index]
            )

        return ForwardCache(
            features=features,
            hidden_raw=hidden_raw,
            hidden_after_relu=hidden_after_relu,
            logit=logit,
        )

    def logit(self, features: FeatureVector) -> float:
        return self.forward(features).logit

    def probability(self, features: FeatureVector) -> float:
        return sigmoid(self.logit(features))

    def accumulate_gradients(self, example: Example, loss_scale: float) -> float:
        """Add this example's gradients to the model's gradient buffers.

        The derivative facts used here are:

        - d(BCEWithLogitsLoss)/d(logit) = sigmoid(logit) - label
        - d(ReLU(raw))/d(raw) = 1 when raw > 0, else 0
        - d(weight * input + bias)/d(weight) = input
        - d(weight * input + bias)/d(bias) = 1
        """

        cache = self.forward(example.features())
        label_float = float(example.class_label)
        loss = binary_cross_entropy_with_logit(cache.logit, example.class_label)
        d_loss_d_logit = (sigmoid(cache.logit) - label_float) * loss_scale

        for hidden_index in range(self.hidden_units):
            self.grad_hidden_to_output_weights[hidden_index] += (
                d_loss_d_logit * cache.hidden_after_relu[hidden_index]
            )
        self.grad_output_bias += d_loss_d_logit

        for hidden_index in range(self.hidden_units):
            d_loss_d_hidden_after_relu = (
                d_loss_d_logit * self.hidden_to_output_weights[hidden_index]
            )
            d_hidden_after_relu_d_hidden_raw = (
                1.0 if cache.hidden_raw[hidden_index] > 0.0 else 0.0
            )
            d_loss_d_hidden_raw = (
                d_loss_d_hidden_after_relu * d_hidden_after_relu_d_hidden_raw
            )

            self.grad_input_to_hidden_weights[hidden_index][0] += (
                d_loss_d_hidden_raw * cache.features[0]
            )
            self.grad_input_to_hidden_weights[hidden_index][1] += (
                d_loss_d_hidden_raw * cache.features[1]
            )
            self.grad_hidden_biases[hidden_index] += d_loss_d_hidden_raw

        return loss


@dataclass
class AdamOptimizer:
    """Adam optimizer implemented directly over OneHiddenLayerMLP's fields."""

    model: OneHiddenLayerMLP
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    step_count: int
    first_moment_input_to_hidden_weights: Matrix
    second_moment_input_to_hidden_weights: Matrix
    first_moment_hidden_biases: Vector
    second_moment_hidden_biases: Vector
    first_moment_hidden_to_output_weights: Vector
    second_moment_hidden_to_output_weights: Vector
    first_moment_output_bias: float
    second_moment_output_bias: float

    @classmethod
    def for_model(
        cls,
        model: OneHiddenLayerMLP,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> AdamOptimizer:
        return cls(
            model=model,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            step_count=0,
            first_moment_input_to_hidden_weights=zeros_like_matrix(
                model.input_to_hidden_weights
            ),
            second_moment_input_to_hidden_weights=zeros_like_matrix(
                model.input_to_hidden_weights
            ),
            first_moment_hidden_biases=zeros_vector(model.hidden_units),
            second_moment_hidden_biases=zeros_vector(model.hidden_units),
            first_moment_hidden_to_output_weights=zeros_vector(model.hidden_units),
            second_moment_hidden_to_output_weights=zeros_vector(model.hidden_units),
            first_moment_output_bias=0.0,
            second_moment_output_bias=0.0,
        )

    def adam_delta(
        self, gradient: float, first_moment: float, second_moment: float
    ) -> tuple[float, float, float]:
        """Return parameter delta plus updated Adam moments for one float."""

        new_first_moment = self.beta1 * first_moment + (1.0 - self.beta1) * gradient
        new_second_moment = (
            self.beta2 * second_moment + (1.0 - self.beta2) * gradient * gradient
        )

        first_unbiased = new_first_moment / (1.0 - (self.beta1**self.step_count))
        second_unbiased = new_second_moment / (1.0 - (self.beta2**self.step_count))
        delta = self.learning_rate * first_unbiased / (
            math.sqrt(second_unbiased) + self.epsilon
        )

        return delta, new_first_moment, new_second_moment

    def step(self) -> None:
        """Move every parameter opposite its gradient."""

        self.step_count += 1

        for hidden_index in range(self.model.hidden_units):
            for input_index in range(2):
                delta, new_first, new_second = self.adam_delta(
                    gradient=self.model.grad_input_to_hidden_weights[hidden_index][
                        input_index
                    ],
                    first_moment=self.first_moment_input_to_hidden_weights[
                        hidden_index
                    ][input_index],
                    second_moment=self.second_moment_input_to_hidden_weights[
                        hidden_index
                    ][input_index],
                )
                self.model.input_to_hidden_weights[hidden_index][input_index] -= delta
                self.first_moment_input_to_hidden_weights[hidden_index][
                    input_index
                ] = new_first
                self.second_moment_input_to_hidden_weights[hidden_index][
                    input_index
                ] = new_second

            delta, new_first, new_second = self.adam_delta(
                gradient=self.model.grad_hidden_biases[hidden_index],
                first_moment=self.first_moment_hidden_biases[hidden_index],
                second_moment=self.second_moment_hidden_biases[hidden_index],
            )
            self.model.hidden_biases[hidden_index] -= delta
            self.first_moment_hidden_biases[hidden_index] = new_first
            self.second_moment_hidden_biases[hidden_index] = new_second

            delta, new_first, new_second = self.adam_delta(
                gradient=self.model.grad_hidden_to_output_weights[hidden_index],
                first_moment=self.first_moment_hidden_to_output_weights[hidden_index],
                second_moment=self.second_moment_hidden_to_output_weights[hidden_index],
            )
            self.model.hidden_to_output_weights[hidden_index] -= delta
            self.first_moment_hidden_to_output_weights[hidden_index] = new_first
            self.second_moment_hidden_to_output_weights[hidden_index] = new_second

        delta, new_first, new_second = self.adam_delta(
            gradient=self.model.grad_output_bias,
            first_moment=self.first_moment_output_bias,
            second_moment=self.second_moment_output_bias,
        )
        self.model.output_bias -= delta
        self.first_moment_output_bias = new_first
        self.second_moment_output_bias = new_second


def average_loss(model: OneHiddenLayerMLP, examples: list[Example]) -> float:
    if len(examples) == 0:
        raise ValueError("examples must not be empty.")

    total_loss = 0.0
    for example in examples:
        total_loss += binary_cross_entropy_with_logit(
            model.logit(example.features()), example.class_label
        )
    return total_loss / float(len(examples))


def accuracy(model: OneHiddenLayerMLP, examples: list[Example]) -> float:
    if len(examples) == 0:
        raise ValueError("examples must not be empty.")

    correct_count = 0
    for example in examples:
        predicted_label = 1 if model.probability(example.features()) >= 0.5 else 0
        if predicted_label == example.class_label:
            correct_count += 1

    return float(correct_count) / float(len(examples))


def train_mlp(
    model: OneHiddenLayerMLP,
    train_examples: list[Example],
    test_examples: list[Example],
    epochs: int,
    learning_rate: float,
    report_every: int,
) -> TrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if len(train_examples) == 0:
        raise ValueError("train_examples must not be empty.")

    optimizer = AdamOptimizer.for_model(model=model, learning_rate=learning_rate)
    losses: list[float] = []
    loss_scale = 1.0 / float(len(train_examples))

    for epoch_index in range(epochs):
        model.zero_gradients()
        total_loss = 0.0

        for example in train_examples:
            total_loss += model.accumulate_gradients(
                example=example,
                loss_scale=loss_scale,
            )

        epoch_loss = total_loss / float(len(train_examples))
        losses.append(epoch_loss)
        optimizer.step()

        is_report_epoch = report_every > 0 and (
            epoch_index == 0 or (epoch_index + 1) % report_every == 0
        )
        if is_report_epoch:
            print(f"epoch {epoch_index + 1:4d} | train loss before update {epoch_loss:.4f}")

    return TrainingResult(
        losses=losses,
        train_loss=average_loss(model, train_examples),
        test_loss=average_loss(model, test_examples),
        train_accuracy=accuracy(model, train_examples),
        test_accuracy=accuracy(model, test_examples),
    )


def write_loss_history_csv(path: Path, losses: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss_before_update"])
        for index, loss in enumerate(losses, start=1):
            writer.writerow([index, loss])


def render_ascii_decision_boundary(
    model: OneHiddenLayerMLP,
    examples: list[Example],
    width: int = 80,
    height: int = 30,
    padding: float = 0.5,
) -> str:
    """Draw the learned classifier as plain text.

    Dots are class-0 regions, hashes are class-1 regions, and asterisks mark
    places where the model is near probability 0.5.
    """

    if width < 2 or height < 2:
        raise ValueError("width and height must be at least 2.")

    x_values = [example.x_position for example in examples]
    y_values = [example.y_position for example in examples]
    x_min = min(x_values) - padding
    x_max = max(x_values) + padding
    y_min = min(y_values) - padding
    y_max = max(y_values) + padding

    lines: list[str] = []
    lines.append("ASCII decision boundary")
    lines.append(".: predicted class 0   #: predicted class 1   *: near p=0.5")

    for row_index in range(height):
        y_position = y_max - (y_max - y_min) * float(row_index) / float(height - 1)
        characters: list[str] = []

        for column_index in range(width):
            x_position = x_min + (x_max - x_min) * float(column_index) / float(
                width - 1
            )
            probability = model.probability((x_position, y_position))

            if abs(probability - 0.5) <= 0.03:
                characters.append("*")
            elif probability >= 0.5:
                characters.append("#")
            else:
                characters.append(".")

        lines.append("".join(characters))

    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(
        description="Train a one-hidden-layer MLP from scratch in pure Python."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    namespace = parser.parse_args()

    return CliOptions(
        data_path=namespace.data_path,
        seed=namespace.seed,
        test_size=namespace.test_size,
        hidden_units=namespace.hidden_units,
        epochs=namespace.epochs,
        learning_rate=namespace.learning_rate,
        report_every=namespace.report_every,
        output_dir=namespace.output_dir,
    )


def main() -> None:
    options = parse_args()

    data_path = resolve_data_path(options.data_path)
    examples = load_examples_csv(data_path)
    split = stratified_train_test_split(
        examples=examples,
        test_size=options.test_size,
        seed=options.seed,
    )
    model = OneHiddenLayerMLP.random(
        hidden_units=options.hidden_units,
        seed=options.seed,
    )

    print("Pure Python Week 05 MLP")
    print("No torch, numpy, pandas, sklearn, or autograd.")
    print(f"dataset: {len(examples)} examples from {data_path}")
    print(f"split: {len(split.train)} train / {len(split.test)} test")
    print(f"model: 2 inputs -> {model.hidden_units} ReLU hidden values -> 1 logit")
    print(f"learned numbers: {model.parameter_count()}")
    print()

    result = train_mlp(
        model=model,
        train_examples=split.train,
        test_examples=split.test,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        report_every=options.report_every,
    )

    history_path = options.output_dir / "training_history.csv"
    boundary_path = options.output_dir / "ascii_decision_boundary.txt"

    write_loss_history_csv(history_path, result.losses)
    write_text(
        boundary_path,
        render_ascii_decision_boundary(model=model, examples=examples),
    )

    print()
    print("Final metrics after the last update:")
    print(f"train loss:     {result.train_loss:.4f}")
    print(f"test loss:      {result.test_loss:.4f}")
    print(f"train accuracy: {result.train_accuracy:.3f}")
    print(f"test accuracy:  {result.test_accuracy:.3f}")
    print()
    print("Wrote artifacts:")
    print(f"- {history_path}")
    print(f"- {boundary_path}")


if __name__ == "__main__":
    main()
