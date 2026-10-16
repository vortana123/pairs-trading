import numpy as np
from scipy.optimize import minimize_scalar


class KalmanPair:
    def __init__(self, alpha, phi, sigma, q, m0, P0):
        self.alpha = alpha
        self.F = np.diag([1.0, phi])
        self.Q = np.diag([q, sigma**2])
        self.m = np.array(m0, dtype=float)
        self.P = np.array(P0, dtype=float)

    def predict(self):
        self.m = self.F @ self.m
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y, x):
        h = np.array([x, 1.0])
        self.e = y - self.alpha - h @ self.m
        g = self.P @ h
        self.S = h @ g
        K = g / self.S
        self.m = self.m + K * self.e
        self.P = self.P - np.outer(K, g)
        self.P = (self.P + self.P.T) / 2
        self.z = self.e / np.sqrt(self.S)


def run_filter(y, x, alpha, phi, sigma, q, m0, P0):
    kf = KalmanPair(alpha, phi, sigma, q, m0, P0)
    n = len(y)
    beta, beta_sd, X, e, S, z = (np.empty(n) for _ in range(6))
    for t in range(n):
        kf.predict()
        kf.update(y[t], x[t])
        beta[t], X[t] = kf.m
        beta_sd[t] = np.sqrt(max(kf.P[0, 0], 0.0))
        e[t], S[t], z[t] = kf.e, kf.S, kf.z
    return dict(beta=beta, beta_sd=beta_sd, X=X, e=e, S=S, z=z)


def loglik(e, S):
    return -0.5 * np.sum(np.log(2 * np.pi * S) + e**2 / S)


def fit_q(y, x, alpha, phi, sigma, m0, P0, burn=20):
    def neg_loglik(log_q):
        r = run_filter(y, x, alpha, phi, sigma, np.exp(log_q), m0, P0)
        return -loglik(r["e"][burn:], r["S"][burn:])

    res = minimize_scalar(neg_loglik, bounds=(np.log(1e-12), np.log(1e-3)), method="bounded")
    return np.exp(res.x)