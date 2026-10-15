import numpy as np


def fit_ar1(X):
    prev = X[:-1]
    curr = X[1:]
    phi = (prev @ curr) / (prev @ prev)
    resid = curr - phi * prev
    sigma = np.sqrt(resid @ resid / len(resid))
    half_life = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf
    sigma_X = sigma / np.sqrt(1 - phi**2) if abs(phi) < 1 else np.inf
    return dict(phi=phi, sigma=sigma, half_life=half_life, sigma_X=sigma_X, resid=resid)