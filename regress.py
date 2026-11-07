#!/usr/bin/env python3
"""Entry point: gradient descent vs normal equation comparison.

Usage:
    python3 regress.py [--outdir examples]

Runs both solvers on a seeded univariate and a seeded multivariate
problem (known ground truth), prints the comparison report, and
writes dataset.csv + fit_and_cost.png into the output directory.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from linear_regression import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
