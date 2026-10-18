import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import prepare, run_strategy

COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
DELTA = 0.9995
FORM, TRADE, STEP = 504, 252, 252
PAIRS = [("PEP", "KO"), ("CVX", "XOM"), ("MA", "V"), ("LOW", "HD"), ("MS", "GS"),
         ("FDX", "UPS"), ("YUM", "MCD"), ("EWC", "EWA"), ("RIO", "BHP"), ("MPC", "VLO")]
VARIANTS = {
    "plain": dict(use_gate=False),
    "gate closes": dict(use_gate=True, gate_mode="close"),
    "gate blocks": dict(use_gate=True, gate_mode="block"),
    "vol-scaled": dict(use_gate=False, scaled=True),
    "vol-scaled + blocks": dict(use_gate=True, gate_mode="block", scaled=True),
}

prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
rows = []
for y_name, x_name in PAIRS:
    pair = prices[[y_name, x_name]].dropna()
    logp = np.log(pair.to_numpy())
    dates = pair.index
    tested = passed = 0
    for start in range(0, len(pair) - FORM - TRADE + 1, STEP):
        sl = slice(start, start + FORM + TRADE)
        y, x = logp[sl, 0], logp[sl, 1]
        prep = prepare(y, x, FORM, COST, DELTA, one_trip=False)
        tested += 1
        row = dict(pair=f"{y_name}/{x_name}", trade_start=dates[start + FORM].date(),
                   tstat=prep["tstat"], phi=prep["phi"], half_life=prep["half_life"],
                   cointegrated=prep["cointegrated"])
        if prep["cointegrated"]:
            passed += 1
            sx = prep["sigma_X"]
            row.update(beta=prep["beta_EG"], u_in=prep["rep"][0] / sx, z_var=np.var(prep["z"][FORM:]))
            for rule, (u_in, u_out) in {"repeated": prep["rep"], "2 sigma": (-2 * sx, 0.0)}.items():
                for variant, opts in VARIANTS.items():
                    pnl, n = run_strategy(y, x, prep, FORM, u_in, u_out, COST, **opts)
                    sd = pnl.std()
                    name = f"{rule}, {variant}"
                    row[f"{name} P&L"] = pnl.sum() * 100
                    row[f"{name} Sharpe"] = pnl.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
                    row[f"{name} trades"] = n
        rows.append(row)
    print(f"{y_name}/{x_name}: {passed}/{tested} trading years passed the cointegration test")

df = pd.DataFrame(rows)
df.to_csv(ROOT / f"results_real_cost{COST}.csv", index=False)
traded = df[df["cointegrated"]]
print(f"\ncost {COST}: {len(traded)} pair-years traded out of {len(df)} tested\n")
for rule in ["repeated", "2 sigma"]:
    for variant in VARIANTS:
        name = f"{rule}, {variant}"
        p, s, n = traded[f"{name} P&L"], traded[f"{name} Sharpe"], traded[f"{name} trades"]
        print(f"   {name:30s} mean P&L {p.mean():6.2f}%   profitable {np.mean(p > 0):4.0%}   "
              f"trades {n.mean():4.1f}   median Sharpe {s.median():5.2f}   worst {p.min():7.2f}%")
    print()