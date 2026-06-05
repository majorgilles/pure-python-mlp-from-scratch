# Tiny MLP From Scratch: Backpropagation Notebook Course

A low-level, notebook-first course for building a tiny neural network from scratch in pure Python.

The goal is not maximum performance. The goal is to understand every moving part: numbers go forward through a model, a loss measures wrongness, derivatives flow backward, and gradient descent updates the learned parameters. By the end, the project should make PyTorch-style `.backward()` feel much less magical.

## Project status

This repository is being redesigned as a beginner-friendly course. The planned course is notebook-first, uses `uv`, and builds toward a real regression task: predicting bike rental demand from a tiny curated dataset.

The previous single-script direction is no longer the baseline. Treat this as a clean course/project design centered on staged notebooks and transparent pure-Python code.

## What you will build

By the end of the course, you should have:

- a tiny scalar `.backward()` engine for simple expressions;
- a hand-written one-hidden-layer MLP using Python lists and floats;
- manual derivatives for MSE loss, ReLU, linear layers, and gradient descent;
- finite-difference gradient checks proving the hand-written gradients are correct;
- a trained bike-rental demand regression model that beats a simple average-prediction baseline;
- an optional PyTorch comparison showing how the same gradient idea maps to `.backward()`.

## Learning promises

This project should stay low-level and explainable:

- no NumPy, pandas, scikit-learn, PyTorch, JAX, or autograd in the core MLP/backprop implementation;
- formulas and tiny numeric examples before code;
- small helpers only when repeated code becomes distracting;
- notebook outputs and tests that prove the model is learning;
- human-readable predictions converted back into bike rental counts.

PyTorch may appear only near the end as an optional comparison, not as part of the from-scratch implementation.

## Planned task: bike rental demand regression

The final useful task will predict bike rental demand from a small real dataset.

Planned data source:

- UCI Bike Sharing Dataset hourly data: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>

Planned curated CSV:

- about 240 fixed rows sampled from the hourly dataset;
- balanced across different hours, seasons/months, and working-day/weekend cases;
- only the teaching columns needed for the course.

Planned inputs per row:

```text
hour of day
temperature
humidity
wind speed
working-day/weekend flag
```

Target:

```text
bike rental count
```

Training will use small scaled numbers so gradient descent is easier to explain:

```text
hour_scaled = hour / 23
target_scaled = rental_count / 1000
predicted_rentals = predicted_scaled * 1000
```

The original rental count should remain visible in the CSV or notebook output so predictions can be interpreted by humans.

## Planned model

The final model will be a configurable one-hidden-layer MLP:

```text
5 inputs -> N ReLU hidden neurons -> 1 rental-demand output
```

Core choices:

- one hidden layer;
- ReLU nonlinearity;
- MSE loss;
- plain gradient descent;
- configurable hidden size;
- final default chosen by evidence, likely 6 or 8 hidden neurons, so the model can learn meaningful patterns while staying inspectable.

Adam is intentionally not part of the main course path. It can be mentioned later as an advanced optimizer idea.

## Planned repository shape

The course should be mostly notebook-based:

```text
notebooks/        Main course lessons, in order
src/              Tiny shared helpers only when necessary
data/             Curated mini bike-rental CSV and source notes
scripts/          Optional data-prep or comparison scripts
tests/            Gradient checks, data checks, and training checks
pyproject.toml    uv project configuration
README.md         Course overview and setup
```

The `src/` folder should not hide the learning. If a helper is introduced, the notebook should first show the idea directly with tiny numbers.

## Setup with uv

This must be a `uv` project. Exact dependency grouping will be decided as the course is implemented, but the core rule is:

> The neural network and backpropagation code stays standard-library-only.

Expected workflow shape:

```bash
uv sync
uv run pytest
uv run jupyter lab
```

If PyTorch is added, it should be isolated for the optional end-of-course comparison.

## Planned course outline

The exact notebook names may change, but the course should have about 10-12 small lessons.

1. **First bike-rental prediction**
   Start from one real bike-rental row, make a simple prediction, measure squared-error loss, and use tiny weight nudges to introduce gradient intuition.

2. **Derivatives from tiny examples**  
   Slopes, nudging a number, and reading a derivative as “how much the output changes.”

3. **The chain rule by hand**  
   Follow a few simple operations forward, then send sensitivity backward step by step.

4. **Build a tiny `.backward()` engine**  
   Implement a small scalar `Value` object for expressions such as `a * b + c`, then call `.backward()`.

5. **MSE loss and one trainable number**  
   Predict a number, compute squared error, derive the gradient, and update one parameter.

6. **A single neuron with several inputs**  
   Dot products, weights, bias, shapes, and gradients for a linear prediction.

7. **ReLU and why nonlinearity matters**  
   Show what ReLU does, derive its simple derivative, and explain why a hidden layer needs nonlinearity.

8. **One hidden layer forward pass**  
   Build `5 inputs -> hidden ReLU neurons -> 1 output` with Python lists and clear cached values.

9. **Backpropagate through the MLP**  
   Derive and implement gradients for output weights, hidden weights, and biases.

10. **Train with plain gradient descent**  
    Load the curated bike-rental CSV, scale values, train the MLP, and print human-readable predictions.

11. **Prove the gradients and model work**  
    Add finite-difference gradient checks, loss-decreasing checks, and a baseline comparison.

12. **Optional PyTorch `.backward()` comparison**  
    Recreate a tiny example or matching model in PyTorch and compare gradients/predictions.

## Validation plan

The project should prove two things:

1. **The gradients are correct.**  
   Use finite differences: nudge one parameter a tiny amount, recompute loss, and compare the numerical slope to the hand-written gradient.

2. **The model learned something useful.**  
   Require the trained model to beat a simple baseline, such as always predicting the average rental count.

Useful checks include:

- curated CSV has expected columns and row count;
- scaling constants are documented;
- forward pass returns one numeric prediction;
- MSE loss decreases during training;
- finite-difference gradients match hand-written gradients within tolerance;
- trained model beats the average-prediction baseline;
- optional PyTorch comparison matches the from-scratch result on a tiny example.

## GitHub issue strategy

Implementation should be split into well-detailed GitHub issues:

- one issue per lesson/notebook;
- a few non-uv foundation issues for dataset creation, shared helpers, validation/test scaffolding, and optional PyTorch comparison support;
- no separate issue for uv setup — uv setup is a direct project prerequisite / first implementation step.

Each issue should include:

- clear learning goal;
- files or notebooks to create/update;
- acceptance criteria;
- dependencies/blockers;
- proof that the lesson is runnable or testable.

## Guardrails

To protect the learning goal:

- avoid hidden magic;
- avoid notebook/src drift;
- avoid accepting gradients on faith;
- keep helpers transparent;
- show formulas before code;
- run notebooks and tests reproducibly with uv;
- keep the core implementation pure Python;
- make final predictions understandable as real bike rental counts.
