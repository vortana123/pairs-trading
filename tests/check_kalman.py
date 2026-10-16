import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ar1 import fit_ar1
from cointegration import engle_granger
from kalman import fit_q, run_filter

truth = json.loads((ROOT / "data" / "true_params.json").read_text())
print(f"true phi = {truth['phi']}, sigma = {truth['sigma']}, "
      f"q (drifting data) = {truth['q']:.1e}\n")

for name in ["sim_drifting_beta", "sim_cointegrated"]:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    form = (df["window"] == "formation").to_numpy()
    trade = ~form
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    xc = x - x[form].mean()

    eg = engle_granger(y[form], xc[form])
    ar = fit_ar1(eg["spread"])
    s2 = eg["spread"] @ eg["spread"] / (form.sum() - 2)
    se_beta = np.sqrt(s2 / np.sum(xc[form] ** 2))

    m0 = [eg["beta"], eg["spread"][0]]
    P0 = np.diag([se_beta**2, ar["sigma_X"] ** 2])
    q = fit_q(y[form], xc[form], eg["alpha"], ar["phi"], ar["sigma"], m0, P0)
    r = run_filter(y, xc, eg["alpha"], ar["phi"], ar["sigma"], q, m0, P0)

    print(name)
    print(f"  formation fit: beta_EG={eg['beta']:.3f}  phi={ar['phi']:.3f}  "
          f"sigma={ar['sigma']:.5f}  q={q:.1e}")

    zt = r["z"][trade]
    print(f"  trading window: z mean={zt.mean():.3f}  z var={zt.var():.3f}")

    if "true_beta" in df:
        tb = df["true_beta"].to_numpy()[trade]
        kb = r["beta"][trade]
        rmse_k = np.sqrt(np.mean((kb - tb) ** 2))
        rmse_eg = np.sqrt(np.mean((eg["beta"] - tb) ** 2))
        cover = np.mean(np.abs(kb - tb) < 2 * r["beta_sd"][trade])
        print(f"  beta error (RMSE): Kalman={rmse_k:.4f}  fixed EG beta={rmse_eg:.4f}")
        print(f"  true beta inside Kalman 2-sd band: {cover:.0%} of days")
    print()