"""Engle-Granger cointegration test (guide, Part 2). Write these yourself.

Step 1: regress y on x by least squares on the formation window:
        y_t = alpha + beta * x_t + X_t           -> alpha_hat, beta_hat, residual spread X_hat_t
Step 2: ADF regression on the residual spread:
        dX_t = gamma * X_{t-1} + sum_{j=1}^{n_lag} a_j * dX_{t-j} + noise
        (no constant: the residuals already have mean zero)
Step 3: t-statistic of gamma. Reject "no cointegration" if t < about -3.34 (5%, MacKinnon,
        two variables). NOT the usual -1.96 and NOT the plain Dickey-Fuller -2.86.

Only numpy is needed. Tip: np.linalg.lstsq for the regressions; the standard error of a
coefficient is sqrt(s^2 * [(A^T A)^{-1}]_{kk}) with s^2 = RSS / (n - number of columns).
"""

import numpy as np

EG_CRIT_5PCT = -3.34


def ols(y, x):
    """Least squares fit of y = alpha + beta * x.
    Return alpha_hat, beta_hat, residuals."""
    raise NotImplementedError


def adf_tstat(X, n_lag=1):
    """t-statistic of gamma in the ADF regression on the series X (no constant)."""
    raise NotImplementedError


def engle_granger(y, x, n_lag=1):
    """Return dict(alpha=..., beta=..., tstat=..., cointegrated=bool, spread=residuals)."""
    raise NotImplementedError
