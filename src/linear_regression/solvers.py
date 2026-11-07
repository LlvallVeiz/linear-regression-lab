"""Two solvers for the same least-squares problem, built from scratch.

1. **Normal equation** — the closed form. Setting grad J = 0 gives
   the normal equations X^T X beta = X^T y. Solved directly with
   ``numpy.linalg.solve``; if X^T X is singular (e.g. a duplicated
   feature column), it falls back to the Moore-Penrose
   pseudo-inverse, which still returns *a* least-squares solution
   (the minimum-norm one).

2. **Batch gradient descent** — the iterative route, on
   *standardised* features. The features here span wildly different
   scales, which makes the raw Hessian X^T X / m extremely
   ill-conditioned and fixed-step descent crawl; standardising each
   feature to zero mean / unit variance is the standard fix. The
   step size needs no tuning: for the quadratic cost, the gradient
   is Lipschitz with constant L = lambda_max(X^T X / m), and any
   fixed step alpha in (0, 2/L) converges (descent lemma; Boyd &
   Vandenberghe, *Convex Optimization*, sec. 9.3). We use
   alpha = 1/L, which also guarantees a monotone cost decrease, and
   an optional exponential decay alpha_t = alpha / (1 + decay * t).
   The fitted coefficients are mapped back to the original scale
   before being returned, so both solvers report the same units.

Nothing here calls numpy's ready-made regression routines
(``linalg.lstsq`` is deliberately *not* used); the only linear
algebra primitives are solve / pinv / eigvalsh.
"""

from __future__ import annotations

import numpy as np

from . import metrics


def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> dict:
    """Closed-form least squares via the normal equations.

    X must already include the intercept column. Returns the
    coefficients plus which path produced them (``"solve"`` or, for
    a singular X^T X, the ``"pinv"`` fallback).
    """
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
        path = "solve"
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ Xty
        path = "pinv"
    return {"beta": beta, "path": path}


def normal_equation_residual(X: np.ndarray, y: np.ndarray,
                             beta: np.ndarray) -> float:
    """||X^T X beta - X^T y||_2 — zero (to rounding) at the exact optimum."""
    return float(np.linalg.norm(X.T @ (X @ beta - y)))


def standardize(X: np.ndarray):
    """Column-wise (mean, std) of the non-intercept columns.

    Returns (X_standardised, mean, std); a zero-variance column gets
    std 1.0 so it passes through unchanged instead of dividing by 0.
    """
    mean = X[:, 1:].mean(axis=0)
    std = X[:, 1:].std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - mean) / std
    return Xs, mean, std


def unstandardize_beta(beta_s: np.ndarray, mean: np.ndarray,
                       std: np.ndarray) -> np.ndarray:
    """Map coefficients fitted on standardised features back to raw units.

    If y = b0 + sum_j bj * (xj - muj)/sigmaj, then in raw units the
    slopes are bj/sigmaj and the intercept absorbs -sum bj*muj/sigmaj.
    """
    beta = np.empty_like(beta_s)
    beta[1:] = beta_s[1:] / std
    beta[0] = beta_s[0] - float(np.sum(beta_s[1:] * mean / std))
    return beta


def fit_gradient_descent(X: np.ndarray, y: np.ndarray,
                         max_iter: int = 10000, tol: float = 1e-12,
                         decay: float = 0.0) -> dict:
    """Batch gradient descent on standardised features.

    X must already include the intercept column. The step size is
    alpha = 1/L with L the largest eigenvalue of the standardised
    Hessian X^T X / m (optionally decayed as alpha/(1 + decay*t)).
    Stops when the relative cost decrease falls below ``tol``.

    Returns a dict with the coefficients on the *original* scale,
    the full cost history (original scale, one entry per iteration
    including the starting point), the iteration count, whether the
    tolerance stop fired, and conditioning diagnostics.
    """
    Xs, mean, std = standardize(X)
    m = len(y)
    hessian = Xs.T @ Xs / m
    eigs = np.linalg.eigvalsh(hessian)
    L = float(eigs[-1])
    kappa = float(eigs[-1] / eigs[0]) if eigs[0] > 0 else float("inf")
    alpha0 = 1.0 / L

    beta_s = np.zeros(X.shape[1])
    beta = unstandardize_beta(beta_s, mean, std)
    history = [metrics.cost(X, y, beta)]
    converged = False
    iterations = 0
    for t in range(1, max_iter + 1):
        alpha = alpha0 / (1.0 + decay * t)
        grad = Xs.T @ (Xs @ beta_s - y) / m
        beta_s = beta_s - alpha * grad
        beta = unstandardize_beta(beta_s, mean, std)
        j = metrics.cost(X, y, beta)
        history.append(j)
        iterations = t
        prev = history[-2]
        if prev - j <= tol * max(prev, 1e-300):
            converged = True
            break
    return {
        "beta": beta,
        "cost_history": history,
        "iterations": iterations,
        "converged": converged,
        "learning_rate": alpha0,
        "lipschitz_L": L,
        "condition_number": kappa,
    }
