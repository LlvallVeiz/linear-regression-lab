"""CLI: run the gradient-descent vs normal-equation comparison.

For each dataset (univariate, then multivariate) the same seeded
split is fitted by both solvers; the report prints the recovered
coefficients next to the ground truth, the final train cost, the
iteration count and wall time, the held-out RMSE / R^2, and two
independent cross-checks (the normal-equation residual of the
closed-form solution, and the gradient-descent solution's distance
to it).

Outputs (into --outdir, default: current directory):
    dataset.csv        the multivariate dataset (synthetic, labelled)
    fit_and_cost.png   univariate fit line + both cost curves
and the report on stdout.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from . import data, metrics, solvers


def _run_experiment(name: str, dataset: dict) -> dict:
    X_tr, X_te, y_tr, y_te = data.train_test_split(
        dataset["X"], dataset["y"])
    Xtr = metrics.add_intercept(X_tr)
    Xte = metrics.add_intercept(X_te)

    t0 = time.perf_counter()
    ne = solvers.fit_normal_equation(Xtr, y_tr)
    t_ne = time.perf_counter() - t0

    t0 = time.perf_counter()
    gd = solvers.fit_gradient_descent(Xtr, y_tr)
    t_gd = time.perf_counter() - t0

    beta_true = dataset["beta_true"]
    out = {
        "name": name,
        "n_train": len(y_tr), "n_test": len(y_te),
        "beta_true": beta_true,
        "ne": ne, "gd": gd, "t_ne": t_ne, "t_gd": t_gd,
        "j_ne": metrics.cost(Xtr, y_tr, ne["beta"]),
        "j_gd": metrics.cost(Xtr, y_tr, gd["beta"]),
        "rmse_ne": metrics.rmse(y_te, metrics.predict(Xte, ne["beta"])),
        "rmse_gd": metrics.rmse(y_te, metrics.predict(Xte, gd["beta"])),
        "r2_ne": metrics.r2_score(y_te, metrics.predict(Xte, ne["beta"])),
        "r2_gd": metrics.r2_score(y_te, metrics.predict(Xte, gd["beta"])),
        "ne_residual": solvers.normal_equation_residual(Xtr, y_tr, ne["beta"]),
        "agree": float(np.max(np.abs(gd["beta"] - ne["beta"]))),
        "err_ne": float(np.max(np.abs(ne["beta"] - beta_true))),
        "err_gd": float(np.max(np.abs(gd["beta"] - beta_true))),
        "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te,
    }
    return out


def _fmt_beta(beta: np.ndarray) -> str:
    return "[" + ", ".join(f"{b:+.4f}" for b in beta) + "]"


def _report(r: dict) -> str:
    gd = r["gd"]
    lines = [
        f"=== {r['name']} (train {r['n_train']} / test {r['n_test']}) ===",
        f"ground truth beta   : {_fmt_beta(r['beta_true'])}",
        f"normal equation     : {_fmt_beta(r['ne']['beta'])}"
        f"  (path: {r['ne']['path']})",
        f"gradient descent    : {_fmt_beta(r['gd']['beta'])}"
        f"  ({gd['iterations']} iters, converged: {gd['converged']})",
        f"max |beta - truth|  : NE {r['err_ne']:.4f}   GD {r['err_gd']:.4f}",
        f"final train cost J  : NE {r['j_ne']:.6f}   GD {r['j_gd']:.6f}",
        f"test RMSE / R^2     : NE {r['rmse_ne']:.4f} / {r['r2_ne']:.4f}"
        f"   GD {r['rmse_gd']:.4f} / {r['r2_gd']:.4f}",
        f"wall time           : NE {r['t_ne'] * 1e3:.2f} ms"
        f"   GD {r['t_gd'] * 1e3:.2f} ms",
        f"GD diagnostics      : alpha = 1/L = {gd['learning_rate']:.4f}"
        f" (L = {gd['lipschitz_L']:.3f}), standardised Hessian"
        f" condition number {gd['condition_number']:.1f}",
        f"cross-checks        : ||X^T X beta_NE - X^T y|| ="
        f" {r['ne_residual']:.3e};  max |beta_GD - beta_NE| = {r['agree']:.3e}",
        "",
    ]
    return "\n".join(lines)


def _plot(uni: dict, multi: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = uni["X_train"][:, 0]
    ax1.scatter(x, uni["y_train"], s=14, alpha=0.6, label="train data")
    xs = np.linspace(0, 10, 100)
    beta = uni["ne"]["beta"]
    ax1.plot(xs, beta[0] + beta[1] * xs, "r-", lw=1.6,
             label=f"fit: y = {beta[0]:.2f} + {beta[1]:.2f} x")
    ax1.plot(xs, 2.0 + 3.0 * xs, "k--", lw=1.0, label="ground truth")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Univariate fit (normal equation)")
    ax1.legend(fontsize=8)

    ax2.semilogy(uni["gd"]["cost_history"], label="univariate GD cost")
    ax2.semilogy(multi["gd"]["cost_history"], label="multivariate GD cost")
    ax2.axhline(uni["j_ne"], color="C0", ls=":", lw=1.0)
    ax2.axhline(multi["j_ne"], color="C1", ls=":", lw=1.0,
                label="normal-equation cost (dotted)")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("cost J(beta)  (log scale)")
    ax2.set_title("Gradient-descent cost history")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gradient descent vs normal equation, from scratch.")
    ap.add_argument("--outdir", default=".",
                    help="where dataset.csv and fit_and_cost.png go")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    uni = _run_experiment("univariate: y = 2 + 3x + noise",
                          data.make_univariate())
    multi_ds = data.make_multivariate()
    multi = _run_experiment(
        "multivariate: price ~ size + rooms + age (synthetic)",
        multi_ds)

    print(_report(uni), end="")
    print(_report(multi), end="")

    data.write_dataset_csv(multi_ds, os.path.join(args.outdir,
                                                  "dataset.csv"))
    _plot(uni, multi, os.path.join(args.outdir, "fit_and_cost.png"))
    print(f"wrote {os.path.join(args.outdir, 'dataset.csv')}")
    print(f"wrote {os.path.join(args.outdir, 'fit_and_cost.png')}")
    return 0
