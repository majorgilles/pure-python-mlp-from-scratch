# Pure Python MLP From Scratch

This is a separate learning repo for re-implementing the Week 05 small neural network without PyTorch or ML helper libraries.

The goal is not maximum performance. The goal is to see every moving part.

## Time estimate

Estimated learning/build time: **6-10 hours total**.

- Repo setup and scope check: **15-30 minutes**
- Pure-Python moons dataset and split: **45-90 minutes**
- Forward pass, sigmoid, and BCE-with-logits loss: **1-2 hours**
- Hand-written backpropagation: **2-3 hours**
- Adam optimizer and training metrics: **1-2 hours**
- README/artifacts/tests polish: **1-2 hours**

If you only want to run the finished script and read through it, expect about **30-60 minutes**.

## What this implements

The model matches the Week 05 MLP shape:

```text
input point -> Linear(2 -> 16) -> ReLU -> Linear(16 -> 1) -> logit
```

Then:

```text
sigmoid(logit) -> probability
probability >= 0.5 -> class 1
probability < 0.5 -> class 0
```

Everything is implemented with Python standard-library code:

- two-moons dataset generation
- stratified train/test split
- weights and biases as lists of floats
- forward pass
- ReLU
- sigmoid
- BCE-with-logits loss
- backpropagation
- Adam optimizer
- accuracy measurement
- ASCII decision-boundary rendering

No `torch`, `numpy`, `pandas`, `sklearn`, or autograd is used.

## Run it

```bash
python mlp_from_scratch.py
```

Default settings mirror the Week 05 notebook:

- 500 moon-shaped examples
- noise `0.25`
- seed `42`
- 400 train examples / 100 test examples
- 16 hidden units
- 2000 epochs
- Adam learning rate `0.01`

Useful options:

```bash
python mlp_from_scratch.py --hidden-units 4
python mlp_from_scratch.py --hidden-units 64
python mlp_from_scratch.py --epochs 500 --report-every 100
```

## Generated artifacts

Running the script writes:

```text
artifacts/moons.csv
artifacts/training_history.csv
artifacts/ascii_decision_boundary.txt
```

The ASCII boundary uses:

```text
. predicted class 0
# predicted class 1
* near probability 0.5
```

## The main learning loop

For each epoch:

1. Run each training point through the model.
2. Measure wrongness with BCE-with-logits loss.
3. Use hand-written derivatives to fill gradient buffers.
4. Use hand-written Adam updates to change each weight and bias.
5. Repeat.

The important backprop derivative is:

```text
d_loss/d_logit = sigmoid(logit) - label
```

From there, the script walks backward through:

```text
output layer -> ReLU -> input layer
```
