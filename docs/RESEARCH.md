# Background & Research — Linear Regression Lab

This document records the sources the two solvers are built on,
the exact formulas implemented, and the places where the code makes
a documented choice. Every source below was consulted (and its URL
checked) while writing the code.

## 1. Least squares and the normal equations: CS229

The formulation follows the Stanford CS229 lecture notes (Ng,
updates by Ma) [1], Part I: hypothesis `h_theta(x) = theta^T x`,
cost

```text
J(theta) = (1 / 2m) * sum_i (h_theta(x_i) - y_i)^2
```

and two minimisation routes presented side by side — exactly the
pair this project implements. Setting the gradient to zero using
the notes' matrix-derivative identities gives the **normal
equations**

```text
X^T X theta = X^T y ,        theta = (X^T X)^{-1} X^T y
```

when `X^T X` is invertible. The notes make the two points the
project's experiments illustrate: batch gradient descent "looks at
every example in the entire training set on every step", and for
linear regression the problem "has only one global, and no other
local, optima; thus gradient descent always converges (assuming
the learning rate alpha is not too large) to the global minimum".
Section 3 below removes the "assuming" by deriving the step size.

The optimality cross-check in the CLI and tests — the residual
`||X^T X beta - X^T y||` — is a direct numerical audit of this
section: a vector that claims to solve the normal equations must
annihilate that residual, and ours does (2.3e-12 univariate,
1.8e-9 absolute on the multivariate problem, ~1e-12 relative).

## 2. Gradient descent on a quadratic: Boyd & Vandenberghe

With `J(beta) = (1/2m)||X beta - y||^2` the gradient is
`(1/m) X^T (X beta - y)` and the Hessian is the constant matrix
`H = X^T X / m`. The gradient is therefore Lipschitz with constant
`L = lambda_max(H)`, and the classical analysis of the gradient
method (Boyd & Vandenberghe, *Convex Optimization*, sec. 9.3 [2])
gives, for a fixed step `alpha`:

* any `alpha in (0, 2/L)` converges to the optimum;
* `alpha = 1/L` additionally guarantees `J` decreases at *every*
  step (the descent lemma), with a linear (geometric) rate whose
  constant worsens with the condition number
  `kappa = lambda_max / lambda_min` of `H`.

The implementation uses `alpha = 1/L` exactly, computed from
`numpy.linalg.eigvalsh` of the standardised Hessian, and the test
suite asserts the promised monotonicity of the recorded cost
history — the theory is thus checked, not just cited. An optional
exponential decay `alpha_t = alpha / (1 + decay * t)` is provided
for experimentation; on a quadratic it can only slow the tail, so
it defaults to 0 and the README says why.

**Why standardise.** `kappa` of the raw multivariate design is
dominated by the features' units (m^2 vs rooms vs years), which
says nothing about the problem; standardising to zero mean / unit
variance removes the unit artefact. What remains — `kappa = 37.4`
on this project's multivariate data — is real geometry:
`rooms` was generated from `size`, and correlated columns make an
elongated valley that fixed-step descent traverses slowly
(471 iterations here, vs an exactly-one-step univariate problem
whose standardised Hessian is the identity, `kappa = 1`). Both
numbers are printed by the CLI as "GD diagnostics" so the claim is
auditable from the output alone.

## 3. The pseudo-inverse fallback: NumPy / Strang

When `X^T X` is singular there is no unique solution, but the
least-squares *predictions* are still well defined, and the
Moore–Penrose pseudo-inverse picks the minimum-norm coefficient
vector among all minimisers. NumPy's documentation [3] defines
`pinv` through the SVD — singular values at or below
`rcond * sigma_max` are treated as zero — and gives the defining
property used here: `A^+ b` is "the matrix that 'solves' [the
least-squares problem] Ax = b", with Strang's *Linear Algebra and
Its Applications* (pp. 139–142) as its reference [4].

`fit_normal_equation` therefore tries `numpy.linalg.solve` first
and falls back to `pinv(X^T X) @ X^T y` on `LinAlgError`, reporting
the path taken. The test builds a deliberately duplicated-column
design, asserts the rank deficiency is real, and verifies the
fallback: exact predictions on noiseless data, and a normal-
equation residual at rounding level (the minimum-norm solution
still satisfies `X^T X beta = X^T y` whenever a solution exists,
because `X^T y` is always in the column space of `X^T X`).

## 4. Metrics

RMSE and R^2 are the standard definitions
(`R^2 = 1 - SS_res / SS_tot`, with R^2 = 1 for a perfect fit and 0
for predicting the mean); both are pinned by hand-computed cases
in `tests/test_metrics.py`, including the reference behaviours
(perfect prediction, mean prediction, worse-than-mean) rather
than by snapshotting the implementation's own output.

## 5. The data is synthetic — and labelled as such

Both datasets are generated in `data.py` from stated models
(univariate `y = 2 + 3x + N(0, 1.5^2)`; multivariate
`y = 50 + 2.5*size + 8*rooms - 1.2*age + N(0, 12^2)`, with `rooms`
drawn from `size` to plant the correlation discussed in sec. 2).
Known ground truth is the point of the exercise: solver error can
be separated from the noise floor, which no real dataset allows.
The house-price framing of the multivariate set is presentation
only; the CSV in `examples/` is that generated data, and it is
described as synthetic in the README, the module docstring and
this document.

## References

1. Ng A (updates by Ma T). *CS229 Lecture Notes*, Part I —
   Supervised learning / Linear regression (LMS algorithm; the
   normal equations). Stanford University —
   https://cs229.stanford.edu/notes2019fall/cs229-notes1.pdf
2. Boyd S, Vandenberghe L (2004). *Convex Optimization*,
   sec. 9.3 (gradient method: step-size rules, convergence rates).
   Cambridge University Press —
   https://web.stanford.edu/~boyd/cvxbook/
3. NumPy documentation. `numpy.linalg.pinv` — Moore–Penrose
   pseudo-inverse via SVD, rcond cutoff —
   https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
4. Strang G (1980). *Linear Algebra and Its Applications*,
   2nd ed., Academic Press, pp. 139–142 (least squares and the
   pseudo-inverse; cited as reference [1] of the NumPy `pinv`
   documentation above).
