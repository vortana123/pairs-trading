import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

import simulate
from cointegration import engle_granger

NAMES = ["sim_cointegrated", "sim_not_cointegrated", "sim_drifting_beta"]
LAGS = [0, 1, 2, 5]

if len(sys.argv) > 1:
    simulate.SEED = int(sys.argv[1])
simulate.main()

truth = json.loads((ROOT / "data" / "true_params.json").read_text())
print(f"\nseed = {truth['seed']}, true alpha = {truth['alpha']}, true beta = {truth['beta']}\n")

data = {}
for name in NAMES:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    f = df[df["window"] == "formation"]
    data[name] = (f["y"].to_numpy(), f["x"].to_numpy())

for name in NAMES:
    y, x = data[name]
    r = engle_granger(y, x)
    print(f"{name:22s} alpha_hat={r['alpha']:7.3f}  beta_hat={r['beta']:6.3f}  "
          f"t={r['tstat']:6.2f}  cointegrated={r['cointegrated']}")

print("\nt-statistic for different n_lag (cutoff -3.34)")
print(f"{'':22s}" + "".join(f"  n_lag={p:<3d}" for p in LAGS))
for name in NAMES:
    y, x = data[name]
    ts = [engle_granger(y, x, n_lag=p)["tstat"] for p in LAGS]
    print(f"{name:22s}" + "".join(f"  {t:9.2f}" for t in ts))