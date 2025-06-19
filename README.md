# Logistic Regression Implementation

This repository contains a minimal implementation of logistic regression from scratch in Python. The goal is to demonstrate how logistic regression works under the hood without relying on external machine learning libraries.

## What is Logistic Regression?

Logistic regression is a supervised learning algorithm used for binary classification tasks. Unlike linear regression, which predicts continuous outputs, logistic regression predicts the probability that a given input belongs to a particular class. The model uses the logistic (sigmoid) function to map any real-valued input to a value between 0 and 1.

Mathematically, the model computes:

```
P(y=1|x) = sigmoid(w \* x + b)
```

where `w` is the weight vector, `b` is the bias term and `sigmoid(z) = 1 / (1 + exp(-z))`.

## Repository Structure

- `logistic_regression.py` – A small Python script implementing logistic regression with gradient descent.
- `README.md` – This guide.

## Using the Code

1. Ensure you have Python 3 installed.
2. Run the script directly to see a simple example using the OR logical function:

```bash
python3 logistic_regression.py
```

The script will train the model on the OR dataset and output predictions. You can modify the training data or the hyperparameters (`lr` and `epochs`) to experiment with the model's behaviour.

## Code Overview

The key parts of `logistic_regression.py` are:

- **Initialization** – Sets the learning rate and number of training epochs.
- **Sigmoid Function** – Converts linear outputs to probabilities.
- **`fit` Method** – Performs gradient descent to learn the weights.
- **`predict_proba` Method** – Returns predicted probabilities for new data.
- **`predict` Method** – Converts probabilities to binary class labels using a threshold (default 0.5).

The example at the bottom of the script trains the model on a simple dataset and prints the resulting predictions.

## When to Use Logistic Regression

Logistic regression works well for linearly separable data and is often a strong baseline for binary classification problems. It is easy to interpret and computationally efficient compared to more complex models.

## License

This project is released into the public domain for educational purposes.
