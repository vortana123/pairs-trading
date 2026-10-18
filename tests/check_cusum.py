import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

import simulate as sm
from ar1 import fit_ar1
from cointegration import engle_granger
from cusum import cusum, gate, mean_scores, siegmund_arl0
from kalman import run_filter

rng = np.random.default_rng(0)

print("False alarms on a correct model (z iid N(0,1)), alarm level 5")
runs = []
for _ in range(2000):
    z = rng.standard_normal(20000)
    C = cusum(mean_scores(z, 1.0))
    runs.append(np.argmax(C >= 5) + 1)
print(f"   one-sided mean CUSUM: simulated ARL0 = {np.mean(runs):.0f} days, "
      f"Siegmund = {siegmund_arl0(0.5, 5.0):.0f} days")

runs = []
for _ in range(1000):
    z = rng.standard_normal(20000)
    runs.append(gate(z, level=5.0)["alarm"] + 1)
print(f"   full gate (up, down, variance): simulated ARL0 = {np.mean(runs):.0f} days\n")


def broken_pair(seed, kind, t_break=625):
    r = np.random.default_rng(seed)
    T = sm.T
    x = sm.random_walk(r, sm.X0, sm.VOL_X, T)
    X = sm.ar1(r, sm.PHI, sm.SIGMA, T)
    if kind == "random walk":
        for t in range(t_break, T):
            X[t] = X[t - 1] + r.normal(0, sm.SIGMA)
    elif kind == "volatility x2":
        for t in range(t_break, T):
            X[t] = sm.PHI * X[t - 1] + r.normal(0, 2 * sm.SIGMA)
    elif kind == "jump 3 sigma_X":
        X[t_break:] += 3 * sm.SIGMA / np.sqrt(1 - sm.PHI**2)
    return x, sm.ALPHA + sm.BETA * x + X


print(" Pair breaks on day 625, 100 simulated pairs each")
t_break = 625
results = {}
for kind in ["none", "random walk", "volatility x2", "jump"]:
    zs = []
    for seed in range(100):
        x, y = broken_pair(seed, kind, t_break)
        form = np.arange(sm.T) < sm.FORMATION
        xc = x - x[form].mean()
        eg = engle_granger(y[form], xc[form])
        ar = fit_ar1(eg["spread"])
        m0 = [eg["beta"], eg["spread"][0]]
        P0 = np.diag([1e-4, ar["sigma_X"] ** 2])
        r = run_filter(y, xc, eg["alpha"], ar["phi"], ar["sigma"], 0.0, m0, P0)
        zs.append(r["z"][sm.FORMATION:])
    results[kind] = zs

for level in [5.0, 7.0]:
    print(f"   alarm level {level:.0f}")
    for kind, zs in results.items():
        alarms = [gate(z, level=level)["alarm"] for z in zs]
        early = [a is not None and a + sm.FORMATION < t_break for a in alarms]
        clean = [a for a, e in zip(alarms, early) if not e]
        delays = [a + sm.FORMATION - t_break for a in clean if a is not None]
        line = f"     {kind:15s} false alarm before break: {np.mean(early):4.0%}"
        if kind != "none":
            line += (f"   detected (no false alarm): {len(delays) / len(clean):4.0%}   "
                     f"median delay: {np.median(delays):.0f} days")
        print(line)