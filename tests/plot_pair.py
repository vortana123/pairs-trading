import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import prepare, run_strategy
from cusum import gate

Y_NAME, X_NAME, TRADE_START = sys.argv[1], sys.argv[2], sys.argv[3]
COST = float(sys.argv[4]) if len(sys.argv) > 4 else 0.002
FORM, TRADE = 504, 252
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e6e5df"

prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
pair = prices[[Y_NAME, X_NAME]].dropna()
t0 = int(pair.index.searchsorted(pd.Timestamp(TRADE_START)))
window = pair.iloc[t0 - FORM: t0 + TRADE]
logp = np.log(window.to_numpy())
y, x, dates = logp[:, 0], logp[:, 1], window.index

prep = prepare(y, x, FORM, COST, 0.9995, one_trip=False)
if not prep["cointegrated"]:
    sys.exit(f"{Y_NAME}/{X_NAME} is not cointegrated in the formation window before {TRADE_START}")

sx = prep["sigma_X"]
u_in, u_out = prep["rep"]
spread = prep["X_hat"] / sx
z = prep["z"][FORM:]
g = gate(z)
pnl_rep, n_rep = run_strategy(y, x, prep, FORM, u_in, u_out, COST, use_gate=False)
pnl_2s, n_2s = run_strategy(y, x, prep, FORM, -2 * sx, 0.0, COST, use_gate=False)
trade_dates = dates[FORM:]
roll_sd = pd.Series(z, index=trade_dates).rolling(20).std()

print(f"{Y_NAME}/{X_NAME}, trading year from {trade_dates[0].date()}")
print(f"   formation: t = {prep['tstat']:.2f}  phi = {prep['phi']:.3f}  half-life = {prep['half_life']:.1f} days  "
      f"beta = {prep['beta_EG']:.3f}")
print(f"   spread in trading year: mean = {spread[FORM:].mean():+.2f} sigma_X  "
      f"days above 0 = {np.mean(spread[FORM:] > 0):.0%}  z variance = {np.var(z):.2f}")
if g["alarm"] is not None:
    which = [k for k in ["C_up", "C_down", "C_var"] if g[k][g["alarm"]] >= 7]
    print(f"   gate alarm on {trade_dates[g['alarm']].date()} from {', '.join(which)}")
else:
    print("   gate: no alarm")
print(f"   repeated rule: P&L {pnl_rep.sum() * 100:+.2f}%  trades {n_rep}   "
      f"2 sigma rule: P&L {pnl_2s.sum() * 100:+.2f}%  trades {n_2s}")

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 1, 1.4]))
for ax in axes:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.axvline(trade_dates[0], color=MUTED, linewidth=1, linestyle="--")
    if g["alarm"] is not None:
        ax.axvline(trade_dates[g["alarm"]], color=INK, linewidth=1, linestyle=":")

ax = axes[0]
ax.plot(dates, spread, color=BLUE, linewidth=1.5)
for level, text in [(2, "2 sigma"), (-2, ""), (-u_in / sx, "repeated rule"), (u_in / sx, ""), (0, "")]:
    ax.axhline(level, color=MUTED, linewidth=0.8, linestyle=":" if level else "-")
    if text:
        ax.text(dates[3], level + 0.1, text, color=MUTED, fontsize=8)
ax.text(trade_dates[2], ax.get_ylim()[1] * 0.92, "trading year", color=MUTED, fontsize=9)
ax.set_title(f"Filtered spread {Y_NAME}/{X_NAME}, in units of formation sigma_X",
             loc="left", fontsize=11, color=INK)

ax = axes[1]
ax.plot(trade_dates, roll_sd, color=BLUE, linewidth=1.5)
ax.axhline(1, color=MUTED, linewidth=0.8)
ax.set_title("20-day standard deviation of innovations z (model says 1)", loc="left", fontsize=11, color=INK)

ax = axes[2]
for pnl, color, label in [(pnl_rep, BLUE, "repeated rule"), (pnl_2s, ORANGE, "2 sigma rule")]:
    curve = np.cumsum(pnl) * 100
    ax.plot(trade_dates, curve, color=color, linewidth=1.5, label=label)
    ax.text(trade_dates[-1], curve[-1], f" {label}", color=INK, fontsize=8, va="center")
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Cumulative P&L, % (no gate)", loc="left", fontsize=11, color=INK)

fig.tight_layout()
out = ROOT / "plots"
out.mkdir(exist_ok=True)
path = out / f"{Y_NAME}_{X_NAME}_{trade_dates[0].date()}.png"
fig.savefig(path, dpi=130)
print(f"   saved {path}")