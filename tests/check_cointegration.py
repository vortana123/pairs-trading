"""Run your Engle-Granger test on the simulated data and compare with the truth.

Run from the project folder:  python tests/check_cointegration.py
Expected once your code works:
    sim_cointegrated       -> cointegrated, beta_hat near 1.2, alpha_hat near 0.5
    sim_not_cointegrated   -> not cointegrated
    sim_drifting_beta      -> borderline: a fixed-beta test struggles when beta drifts,
                              which is the reason for the Kalman filter in Part 3
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cointegration import engle_granger  # noqa: E402

truth = json.loads((ROOT / "data" / "true_params.json").read_text())
print(f"true alpha = {truth['alpha']}, true beta = {truth['beta']}\n")

for name in ["sim_cointegrated", "sim_not_cointegrated", "sim_drifting_beta"]:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    f = df[df["window"] == "formation"]
    r = engle_granger(f["y"].to_numpy(), f["x"].to_numpy())
    print(f"{name:22s} alpha_hat={r['alpha']:7.3f}  beta_hat={r['beta']:6.3f}  "
          f"t={r['tstat']:6.2f}  cointegrated={r['cointegrated']}")
