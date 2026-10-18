import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

import simulate as sm
from backtest import prepare, run_strategy

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
COST = 0.001
DELTA = 0.9995
T_BREAK = 625


def make_pair(seed, broken):
    r = np.random.default_rng(seed)
    x = sm.random_walk(r, sm.X0, sm.VOL_X, sm.T)
    X = sm.ar1(r, sm.PHI, sm.SIGMA, sm.T)
    if broken:
        for t in range(T_BREAK, sm.T):
            X[t] = X[t - 1] + r.normal(0, sm.SIGMA)
    return x, sm.ALPHA + sm.BETA * x + X


def summarise(name, pnls, trades):
    P = np.array(pnls)
    totals = P.sum(axis=1) * 100
    sd = P.std(axis=1)
    sharpe = np.where(sd > 0, P.mean(axis=1) / np.where(sd > 0, sd, 1) * np.sqrt(252), 0.0)
    curves = np.cumsum(P, axis=1) * 100
    max_dd = np.max(np.maximum.accumulate(curves, axis=1) - curves, axis=1)
    print(f"   {name:18s} mean P&L {totals.mean():6.2f}%   profitable {np.mean(totals > 0):4.0%}   "
          f"trades {np.mean(trades):4.1f}   median Sharpe {np.median(sharpe):5.2f}   "
          f"median max DD {np.median(max_dd):5.2f}%")


start = time.time()
for broken in [False, True]:
    label = "pairs that break on day 625 (random walk)" if broken else "healthy pairs"
    names = ["one trip + gate", "one trip, no gate", "repeated + gate", "repeated, no gate",
             "2 sigma + gate", "2 sigma, no gate"]
    results = {k: ([], []) for k in names}
    passed, th_one, th_rep = 0, [], []
    for seed in range(N_PAIRS):
        x, y = make_pair(1000 + seed, broken)
        prep = prepare(y, x, sm.FORMATION, COST, DELTA)
        if not prep["cointegrated"]:
            continue
        passed += 1
        sx = prep["sigma_X"]
        th_one.append(np.array(prep["one"]) / sx)
        th_rep.append(np.array(prep["rep"]) / sx)
        rules = {
            "one trip + gate": (*prep["one"], True),
            "one trip, no gate": (*prep["one"], False),
            "repeated + gate": (*prep["rep"], True),
            "repeated, no gate": (*prep["rep"], False),
            "2 sigma + gate": (-2 * sx, 0.0, True),
            "2 sigma, no gate": (-2 * sx, 0.0, False),
        }
        for name, (u_in, u_out, use_gate) in rules.items():
            pnl, n = run_strategy(y, x, prep, sm.FORMATION, u_in, u_out, COST, use_gate)
            results[name][0].append(pnl)
            results[name][1].append(n)
    print(f"{label}: {passed}/{N_PAIRS} passed the cointegration test")
    a, b = np.mean(th_one, axis=0), np.mean(th_rep, axis=0)
    print(f"   thresholds in sigma_X units: one trip enter {a[0]:.2f} exit {a[1]:.2f}; "
          f"repeated enter {b[0]:.2f} exit {b[1]:.2f}")
    for name, (pnls, trades) in results.items():
        summarise(name, pnls, trades)
    tag = "broken" if broken else "healthy"
    np.savez(ROOT / f"backtest_{tag}.npz",
             **{name.replace(" ", "_").replace(",", ""): np.array(p) for name, (p, _) in results.items()})
    print()
print(f"took {time.time() - start:.0f} s")