"""Metric tests with hand-computed values."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linear_regression import metrics  # noqa: E402


def test_cost_hand_computed():
    # X beta = [1*1 + 2*1, 1*1 + 4*1] = [3, 5]; residuals [3-4, 5-4] = [-1, 1]
    # J = (1 + 1) / (2 * 2) = 0.5
    X = np.array([[1.0, 2.0], [1.0, 4.0]])
    y = np.array([4.0, 4.0])
    beta = np.array([1.0, 1.0])
    assert metrics.cost(X, y, beta) == 0.5
    # beta = 0 -> residuals are -y: J = (16 + 16) / 4 = 8
    assert metrics.cost(X, y, np.zeros(2)) == 8.0


def test_rmse_hand_computed():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, y) == 0.0
    # errors [0, 0, 3] -> sqrt(9/3) = sqrt(3)
    assert abs(metrics.rmse(y, np.array([1.0, 2.0, 6.0]))
               - np.sqrt(3.0)) < 1e-12


def test_r2_reference_cases():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.r2_score(y, y) == 1.0
    # predicting the mean gives exactly 0
    assert abs(metrics.r2_score(y, np.full(4, 2.5))) < 1e-12
    # worse than the mean gives a negative R^2
    assert metrics.r2_score(y, np.zeros(4)) < 0.0


def test_add_intercept():
    X = metrics.add_intercept(np.array([5.0, 6.0, 7.0]))
    assert X.shape == (3, 2)
    assert np.all(X[:, 0] == 1.0) and np.all(X[:, 1] == [5.0, 6.0, 7.0])


if __name__ == "__main__":
    test_cost_hand_computed()
    test_rmse_hand_computed()
    test_r2_reference_cases()
    test_add_intercept()
    print("test_metrics: all passed")
