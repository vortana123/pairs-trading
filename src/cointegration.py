import numpy as np

EG_CRIT_5PCT = -3.34


def ols(y, x):
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta = coef
    resid = y - A @ coef
    return alpha, beta, resid


def adf_tstat(X, n_lag=1):
    dX = np.diff(X)
    target = dX[n_lag:]
    cols = [X[n_lag:-1]]
    for j in range(1, n_lag + 1):
        cols.append(dX[n_lag - j:-j])
    B = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(B, target, rcond=None)
    resid = target - B @ coef
    s2 = resid @ resid / (len(target) - B.shape[1])
    cov = s2 * np.linalg.inv(B.T @ B)
    return coef[0] / np.sqrt(cov[0, 0])


def engle_granger(y, x, n_lag=1):
    alpha, beta, spread = ols(y, x)
    tstat = adf_tstat(spread, n_lag)
    return dict(alpha=alpha, beta=beta, tstat=tstat,
                cointegrated=bool(tstat < EG_CRIT_5PCT), spread=spread)