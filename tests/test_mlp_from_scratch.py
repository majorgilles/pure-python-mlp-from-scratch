from __future__ import annotations

import unittest

from mlp_from_scratch import (
    OneHiddenLayerMLP,
    accuracy,
    average_loss,
    make_moons,
    stratified_train_test_split,
    train_mlp,
)


class MLPFromScratchTests(unittest.TestCase):
    def test_make_moons_returns_balanced_labels(self) -> None:
        examples = make_moons(n_samples=100, noise=0.0, seed=42)

        zero_count = sum(1 for example in examples if example.class_label == 0)
        one_count = sum(1 for example in examples if example.class_label == 1)

        self.assertEqual(50, zero_count)
        self.assertEqual(50, one_count)

    def test_stratified_split_keeps_expected_sizes(self) -> None:
        examples = make_moons(n_samples=100, noise=0.0, seed=42)
        split = stratified_train_test_split(examples=examples, test_size=0.2, seed=42)

        self.assertEqual(80, len(split.train))
        self.assertEqual(20, len(split.test))
        self.assertEqual(10, sum(1 for example in split.test if example.class_label == 0))
        self.assertEqual(10, sum(1 for example in split.test if example.class_label == 1))

    def test_forward_pass_returns_probability(self) -> None:
        model = OneHiddenLayerMLP.random(hidden_units=4, seed=42)

        probability = model.probability((0.25, -0.10))

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_training_reduces_loss_and_learns_signal(self) -> None:
        examples = make_moons(n_samples=200, noise=0.20, seed=42)
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
