"""Solver tests: exact recovery, optimality conditions, agreement.

The assertions are of three kinds, on purpose:

* **exact** — noiseless data must be recovered to rounding error;
* **optimality** — the closed form must (numerically) satisfy the
  normal equations it claims to solve, ||X^T X beta - X^T y|| ~ 0;
* **agreement** — the iterative solver, run to its tolerance stop,
  must land on the closed-form solution, and its train cost must be
  monotone decreasing (guaranteed by the alpha = 1/L step, so any
  increase is a bug, not bad luck).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linear_regression import data, metrics, solvers  # noqa: E402


def _design(X):
    return metrics.add_intercept(X)


def test_noiseless_exact_recovery():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)) * np.array([10.0, 1.0, 100.0])
    beta_true = np.array([4.0, -2.0, 0.5, 3.0])
    y = _design(X) @ beta_true          # no noise at all
    ne = solvers.fit_normal_equation(_design(X), y)
    assert ne["path"] == "solve"
    assert np.max(np.abs(ne["beta"] - beta_true)) < 1e-6
    gd = solvers.fit_gradient_descent(_design(X), y, tol=1e-14)
    assert np.max(np.abs(gd["beta"] - beta_true)) < 1e-4


def test_normal_equation_residual():
    ds = data.make_multivariate()
    X_tr, _, y_tr, _ = data.train_test_split(ds["X"], ds["y"])
    Xd = _design(X_tr)
    ne = solvers.fit_normal_equation(Xd, y_tr)
    res = solvers.normal_equation_residual(Xd, y_tr, ne["beta"])
    scale = float(np.linalg.norm(Xd.T @ y_tr))
    assert res / scale < 1e-10, f"relative residual {res / scale:.2e}"


def test_gd_agrees_with_normal_equation():
    for make in (data.make_univariate, data.make_multivariate):
        ds = make()
        X_tr, _, y_tr, _ = data.train_test_split(ds["X"], ds["y"])
        Xd = _design(X_tr)
        ne = solvers.fit_normal_equation(Xd, y_tr)
        gd = solvers.fit_gradient_descent(Xd, y_tr)
        assert gd["converged"]
        diff = np.max(np.abs(gd["beta"] - ne["beta"]))
        scale = np.maximum(1.0, np.max(np.abs(ne["beta"])))
        assert diff / scale < 1e-4, f"max |GD-NE| {diff:.2e}"
        j_ne = metrics.cost(Xd, y_tr, ne["beta"])
        j_gd = metrics.cost(Xd, y_tr, gd["beta"])
        assert abs(j_gd - j_ne) / j_ne < 1e-6


def test_gd_cost_is_monotone():
    ds = data.make_multivariate()
    X_tr, _, y_tr, _ = data.train_test_split(ds["X"], ds["y"])
    gd = solvers.fit_gradient_descent(_design(X_tr), y_tr)
    h = np.asarray(gd["cost_history"])
    # alpha = 1/L on a quadratic: mathematically non-increasing.
    assert np.all(np.diff(h) <= 1e-9 * np.maximum(1.0, h[:-1]))


def test_singular_design_uses_pinv_fallback():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 5, size=40)
    X = np.column_stack([x, 2.0 * x])   # second column duplicates first
    y = 1.0 + 3.0 * x                   # noiseless, exactly representable
    Xd = _design(X)
    XtX = Xd.T @ Xd
    assert np.linalg.matrix_rank(XtX) < XtX.shape[0]  # really singular
    ne = solvers.fit_normal_equation(Xd, y)
    assert ne["path"] == "pinv"
    pred = metrics.predict(Xd, ne["beta"])
    assert np.max(np.abs(pred - y)) < 1e-6
    # the fallback solution still satisfies least squares optimality
    res = solvers.normal_equation_residual(Xd, y, ne["beta"])
    assert res < 1e-6 * np.linalg.norm(Xd.T @ y)


def test_standardize_roundtrip():
    rng = np.random.default_rng(2)
    X = _design(rng.normal(size=(30, 2)) * np.array([50.0, 0.1])
                + np.array([100.0, 3.0]))
    Xs, mean, std = solvers.standardize(X)
    assert np.allclose(Xs[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Xs[:, 1:].std(axis=0), 1.0, atol=1e-12)
    beta_s = np.array([7.0, 2.0, -1.0])
    beta = solvers.unstandardize_beta(beta_s, mean, std)
    assert np.allclose(metrics.predict(Xs, beta_s),
                       metrics.predict(X, beta))


if __name__ == "__main__":
    test_noiseless_exact_recovery()
    test_normal_equation_residual()
    test_gd_agrees_with_normal_equation()
    test_gd_cost_is_monotone()
    test_singular_design_uses_pinv_fallback()
    test_standardize_roundtrip()
    print("test_solvers: all passed")
