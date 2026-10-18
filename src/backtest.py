import numpy as np

from ar1 import fit_ar1
from bellman import solve_repeated, solve_thresholds
from cointegration import engle_granger
from cusum import gate
from kalman import run_filter


def prepare(y, x, n_form, cost, delta, one_trip=True):
    xc = x - x[:n_form].mean()
    eg = engle_granger(y[:n_form], xc[:n_form])
    ar = fit_ar1(eg["spread"])
    info = dict(tstat=eg["tstat"], beta_EG=eg["beta"], phi=ar["phi"], half_life=ar["half_life"],
                sigma_X=ar["sigma_X"], cointegrated=eg["cointegrated"] and 0 < ar["phi"] < 1)
    if not info["cointegrated"]:
        return info
    m0 = [eg["beta"], eg["spread"][0]]
    P0 = np.diag([1e-4, ar["sigma_X"] ** 2])
    kf = run_filter(y, xc, eg["alpha"], ar["phi"], ar["sigma"], 0.0, m0, P0)
    rep = solve_repeated(ar["phi"], ar["sigma"], cost)
    info.update(X_hat=kf["X"], beta=kf["beta"], z=kf["z"], rep=(rep["u_in"], rep["u_out"]))
    if one_trip:
        one = solve_thresholds(ar["phi"], ar["sigma"], cost, delta, n=401)
        info["one"] = (one["u_in"], one["u_out"])
    return info


def vol_scale(z, window=60):
    sq = np.concatenate([[0.0], np.cumsum(z**2)])
    t = np.arange(1, len(z) + 1)
    lo = np.maximum(t - window, 0)
    return np.sqrt((sq[t] - sq[lo]) / (t - lo))


def run_strategy(y, x, prep, n_form, u_in, u_out, cost, use_gate,
                 level=7.0, gate_mode="close", scaled=False, stop=None, max_hold=None):
    T = len(y)
    alarm = gate(prep["z"][n_form:], level=level)["alarm"] if use_gate else None
    stop_day = n_form + alarm if alarm is not None else T
    k = vol_scale(prep["z"]) if scaled else np.ones(T)
    pnl = np.zeros(T)
    pos, target, beta_lock, trades, hold, blocked = 0, 0, 0.0, 0, 0, 0
    for t in range(n_form, T):
        if pos != 0:
            pnl[t] += pos * ((y[t] - y[t - 1]) - beta_lock * (x[t] - x[t - 1]))
        if target != pos:
            pnl[t] -= cost * abs(target - pos)
            if target != 0:
                beta_lock = prep["beta"][t - 1]
                trades += 1
                hold = 0
            pos = target
        if pos != 0:
            hold += 1
        X = prep["X_hat"][t]
        if t == T - 1 or (t >= stop_day and gate_mode == "close"):
            target = 0
            continue
        enter, leave = u_in * k[t], u_out * k[t]
        timed_out = max_hold is not None and hold >= max_hold
        new = pos
        if pos == 1:
            if X >= leave:
                new = 0
            elif timed_out or (stop is not None and X <= -stop * k[t]):
                new, blocked = 0, 1
        elif pos == -1:
            if X <= -leave:
                new = 0
            elif timed_out or (stop is not None and X >= stop * k[t]):
                new, blocked = 0, -1
        if (blocked == 1 and X >= 0) or (blocked == -1 and X <= 0):
            blocked = 0
        if new == 0 and t < stop_day:
            if X <= enter and blocked != 1:
                new = 1
            elif X >= -enter and blocked != -1:
                new = -1
        target = new
    if pos != 0:
        pnl[T - 1] -= cost * abs(pos)
    return pnl[n_form:], trades