import numpy as np

from ar1 import fit_ar1
from bellman import solve_repeated, solve_thresholds
from cointegration import engle_granger
from cusum import gate
from kalman import run_filter


def prepare(y, x, n_form, cost, delta):
    xc = x - x[:n_form].mean()
    eg = engle_granger(y[:n_form], xc[:n_form])
    ar = fit_ar1(eg["spread"])
    m0 = [eg["beta"], eg["spread"][0]]
    P0 = np.diag([1e-4, ar["sigma_X"] ** 2])
    kf = run_filter(y, xc, eg["alpha"], ar["phi"], ar["sigma"], 0.0, m0, P0)
    one = solve_thresholds(ar["phi"], ar["sigma"], cost, delta, n=401)
    rep = solve_repeated(ar["phi"], ar["sigma"], cost)
    return dict(X_hat=kf["X"], beta=kf["beta"], z=kf["z"], sigma_X=ar["sigma_X"],
                one=(one["u_in"], one["u_out"]), rep=(rep["u_in"], rep["u_out"]),
                cointegrated=eg["cointegrated"])


def run_strategy(y, x, prep, n_form, u_in, u_out, cost, use_gate, level=7.0):
    T = len(y)
    alarm = gate(prep["z"][n_form:], level=level)["alarm"] if use_gate else None
    stop_day = n_form + alarm if alarm is not None else T
    pnl = np.zeros(T)
    pos, target, beta_lock, trades = 0, 0, 0.0, 0
    for t in range(n_form, T):
        if pos != 0:
            pnl[t] += pos * ((y[t] - y[t - 1]) - beta_lock * (x[t] - x[t - 1]))
        if target != pos:
            pnl[t] -= cost * abs(target - pos)
            if target != 0:
                beta_lock = prep["beta"][t - 1]
                trades += 1
            pos = target
        X = prep["X_hat"][t]
        if t >= stop_day or t == T - 1:
            target = 0
            continue
        new = pos
        if pos == 1 and X >= u_out:
            new = 0
        elif pos == -1 and X <= -u_out:
            new = 0
        if new == 0:
            if X <= u_in:
                new = 1
            elif X >= -u_in:
                new = -1
        target = new
    if pos != 0:
        pnl[T - 1] -= cost * abs(pos)
    return pnl[n_form:], trades