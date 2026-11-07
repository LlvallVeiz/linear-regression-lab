"""Cost and quality metrics for linear regression.

The cost is the ordinary-least-squares objective in the CS229
convention,

    J(beta) = (1 / 2m) * sum_i (x_i . beta - y_i)^2 ,

whose 1/2 cancels the square's factor 2 in the gradient, giving
grad J = (1/m) X^T (X beta - y). Every cost reported by this project
(train curves, solver comparison, tests) is this J on the *original*
feature scale, so numbers from the two solvers are directly
comparable.
"""

from __future__ import annotations

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones (the intercept term) to a design matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(len(X)), X])


def cost(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """OLS cost J(beta) = ||X beta - y||^2 / (2m). X includes intercept."""
    r = X @ beta - y
    return float(r @ r / (2.0 * len(y)))


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Linear prediction; X includes the intercept column."""
    return X @ beta


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot
