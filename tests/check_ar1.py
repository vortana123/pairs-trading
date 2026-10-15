import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ar1 import fit_ar1
from cointegration import engle_granger

truth = json.loads((ROOT / "data" / "true_params.json").read_text())
print(f"true phi = {truth['phi']}, sigma = {truth['sigma']}, "
      f"half-life = {truth['half_life']:.2f}, sigma_X = {truth['sigma_X']:.4f}\n")

df = pd.read_csv(ROOT / "data" / "sim_cointegrated.csv")
f = df[df["window"] == "formation"]

eg = engle_granger(f["y"].to_numpy(), f["x"].to_numpy())
fits = {
    "true spread": fit_ar1(f["true_spread"].to_numpy()),
    "EG spread": fit_ar1(eg["spread"]),
}

for label, r in fits.items():
    print(f"{label:12s} phi={r['phi']:.4f}  sigma={r['sigma']:.5f}  "
          f"half-life={r['half_life']:5.2f}  sigma_X={r['sigma_X']:.4f}")