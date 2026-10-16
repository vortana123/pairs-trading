"""Simulate log-price pairs for testing the pipeline.

Notation follows the guide:
    x_t  log price of stock x (random walk)
    y_t  log price of stock y
    y_t = alpha + beta_t * x_t + X_t
    X_t = phi * X_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)   (the spread)

Three datasets are written to this folder:
    sim_cointegrated.csv      beta fixed, spread AR(1)   -> test should say cointegrated
    sim_not_cointegrated.csv  two independent random walks -> test should NOT
    sim_drifting_beta.csv     beta_t random walk (variance q per day), for the Kalman filter later
The true parameters are saved in true_params.json so you can check your estimates.

Run:  python data/simulate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 2026

T = 750                 # about 3 years of trading days
FORMATION = 500         # first 500 days: formation window, last 250: trading window
X0 = np.log(50.0)       # starting log price of x
VOL_X = 0.015           # daily log-return volatility of x
ALPHA = 0.5
BETA = 1.2
PHI = 0.90              # half-life ln(0.5)/ln(0.9), about 6.6 days
SIGMA = 0.005           # daily shock to the spread
Q = 2.5e-7               # daily variance of beta's random walk (drifting-beta dataset only)


def random_walk(rng, x0, vol, n):
    return x0 + np.concatenate([[0.0], np.cumsum(rng.normal(0.0, vol, n - 1))])


def ar1(rng, phi, sigma, n):
    X = np.empty(n)
    X[0] = rng.normal(0.0, sigma / np.sqrt(1 - phi**2))   # start in the stationary distribution
    for t in range(1, n):
        X[t] = phi * X[t - 1] + rng.normal(0.0, sigma)
    return X


def save(name, x, y, extra=None):
    df = pd.DataFrame({"t": np.arange(len(x)), "x": x, "y": y})
    if extra:
        for k, v in extra.items():
            df[k] = v
    df["window"] = np.where(df["t"] < FORMATION, "formation", "trading")
    df.to_csv(HERE / name, index=False)


def main():
    rng = np.random.default_rng(SEED)

    # 1. Cointegrated pair, fixed beta
    x = random_walk(rng, X0, VOL_X, T)
    X = ar1(rng, PHI, SIGMA, T)
    y = ALPHA + BETA * x + X
    save("sim_cointegrated.csv", x, y, {"true_spread": X})

    # 2. Not cointegrated: two independent random walks
    x2 = random_walk(rng, X0, VOL_X, T)
    y2 = random_walk(rng, ALPHA + BETA * X0, VOL_X * BETA, T)
    save("sim_not_cointegrated.csv", x2, y2)

    # 3. Cointegrated with a slowly drifting hedge ratio
    x3 = random_walk(rng, X0, VOL_X, T)
    beta_t = BETA + np.concatenate([[0.0], np.cumsum(rng.normal(0.0, np.sqrt(Q), T - 1))])
    X3 = ar1(rng, PHI, SIGMA, T)
    y3 = ALPHA + BETA * x3 + (beta_t - BETA) * (x3 - X0) + X3
    save("sim_drifting_beta.csv", x3, y3, {"true_beta": beta_t, "true_spread": X3})

    params = dict(seed=SEED, T=T, formation=FORMATION, x0=float(X0), vol_x=VOL_X,
                  alpha=ALPHA, beta=BETA, phi=PHI, sigma=SIGMA, q=Q,
                  sigma_X=SIGMA / np.sqrt(1 - PHI**2),
                  half_life=float(np.log(0.5) / np.log(PHI)))
    (HERE / "true_params.json").write_text(json.dumps(params, indent=2))
    print("wrote 3 datasets and true_params.json to", HERE)


if __name__ == "__main__":
    main()
