import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ar1 import fit_ar1
from bellman import solve_thresholds
from cointegration import engle_granger


def simulate_exit_value(u0, threshold, phi, sigma, cost, delta, n_paths=20000, max_days=3000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.full(n_paths, u0, dtype=float)
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for t in range(max_days):
        stop = alive & (u >= threshold)
        value[stop] = delta**t * (u[stop] - cost)
        alive &= ~stop
        if not alive.any():
            break
        u[alive] = phi * u[alive] + sigma * rng.standard_normal(alive.sum())
    return value.mean()


print("On known data: phi=0.9, sigma=1, cost=0.1, delta=0.99")
phi, sigma, cost, delta = 0.9, 1.0, 0.1, 0.99
r = solve_thresholds(phi, sigma, cost, delta)
print(f"   u_out={r['u_out']:.3f}  u_in={r['u_in']:.3f}  lookahead u_my={r['u_my']:.4f}  "
      f"iterations={r['iterations']}")
print(f"   exit set is one half-line: {bool(np.all(r['exit_set'][r['u'] >= r['u_out']]) and not r['exit_set'][r['u'] < r['u_out']].any())}")

V0 = np.interp(0.0, r["u"], r["V"])
mc_opt = simulate_exit_value(0.0, r["u_out"], phi, sigma, cost, delta)
mc_my = simulate_exit_value(0.0, r["u_my"], phi, sigma, cost, delta)
print(f"   value of holding at u=0: V(0)={V0:.3f}  simulated optimal rule={mc_opt:.3f}  "
      f"simulated lookahead rule={mc_my:.3f}\n")

print("On Fitted spread from sim_cointegrated, cost=0.001 per trade")
df = pd.read_csv(ROOT / "data" / "sim_cointegrated.csv")
form = (df["window"] == "formation").to_numpy()
y, x = df["y"].to_numpy(), df["x"].to_numpy()
xc = x - x[form].mean()
ar = fit_ar1(engle_granger(y[form], xc[form])["spread"])
print(f"   phi={ar['phi']:.3f}  sigma={ar['sigma']:.5f}  sigma_X={ar['sigma_X']:.4f}")
for delta in [0.999, 0.9995, 0.9998]:
    r = solve_thresholds(ar["phi"], ar["sigma"], 0.001, delta)
    sx = r["sigma_X"]
    print(f"   delta={delta}: u_out={r['u_out']:.4f} ({r['u_out'] / sx:.2f} sigma_X)  "
          f"u_in={r['u_in']:.4f} ({r['u_in'] / sx:.2f} sigma_X)")