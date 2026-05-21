from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlp_from_scratch import (
    DEFAULT_DATA_PATH,
    OneHiddenLayerMLP,
    accuracy,
    average_loss,
    load_examples_csv,
    stratified_train_test_split,
    train_mlp,
)


class MLPFromScratchTests(unittest.TestCase):
    def test_load_examples_csv_returns_week05_dataset(self) -> None:
        examples = load_examples_csv(DEFAULT_DATA_PATH)

        zero_count = sum(1 for example in examples if example.class_label == 0)
        one_count = sum(1 for example in examples if example.class_label == 1)

        self.assertEqual(500, len(examples))
        self.assertEqual(250, zero_count)
        self.assertEqual(250, one_count)

    def test_load_examples_csv_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            missing_column_path = Path(temporary_directory_name) / "missing_columns.csv"
            missing_column_path.write_text(
                "x_position,y_position\n0.0,1.0\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_examples_csv(missing_column_path)

    def test_stratified_split_keeps_expected_sizes(self) -> None:
        examples = load_examples_csv(DEFAULT_DATA_PATH)
        split = stratified_train_test_split(examples=examples, test_size=0.2, seed=42)

        self.assertEqual(400, len(split.train))
        self.assertEqual(100, len(split.test))
        self.assertEqual(50, sum(1 for example in split.test if example.class_label == 0))
        self.assertEqual(50, sum(1 for example in split.test if example.class_label == 1))

    def test_forward_pass_returns_probability(self) -> None:
        model = OneHiddenLayerMLP.random(hidden_units=4, seed=42)

        probability = model.probability((0.25, -0.10))

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_training_reduces_loss_and_learns_signal(self) -> None:
        examples = load_examples_csv(DEFAULT_DATA_PATH)
        split = stratified_train_test_split(examples=examples, test_size=0.2, seed=42)
        model = OneHiddenLayerMLP.random(hidden_units=8, seed=42)
        initial_loss = average_loss(model=model, examples=split.train)

        result = train_mlp(
            model=model,
            train_examples=split.train,
            test_examples=split.test,
            epochs=300,
            learning_rate=0.01,
            report_every=0,
        )

        self.assertLess(result.train_loss, initial_loss)
        self.assertGreater(accuracy(model=model, examples=split.train), 0.80)


if __name__ == "__main__":
    unittest.main()
