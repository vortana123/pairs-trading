import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import prepare, run_strategy
from screen import screen_pairs
from universe import SECTORS

BPS = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
N_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
FORM, TRADE, STEP = 504, 252, 252
DELTA = 0.9995
RULES = {
    "2 sigma": dict(kind="2 sigma"),
    "2 sigma + time stop": dict(kind="2 sigma", time_stop=True),
    "2 sigma + stop-loss": dict(kind="2 sigma", stop_loss=True),
    "2 sigma + both stops": dict(kind="2 sigma", time_stop=True, stop_loss=True),
    "repeated + both stops": dict(kind="repeated", time_stop=True, stop_loss=True),
}
STOP_SIGMAS = 4.0
HOLD_HALF_LIVES = 3.0

start_time = time.time()
prices = pd.read_csv(ROOT / "data" / "prices.csv", index_col=0, parse_dates=True)
daily = {name: [] for name in RULES}
dates_all, pair_rows = [], []

for s in range(0, len(prices) - FORM - TRADE + 1, STEP):
    window = prices.iloc[s: s + FORM + TRADE]
    window = window.loc[:, window.notna().all()]
    logp = np.log(window)
    picks = screen_pairs(logp.iloc[:FORM], SECTORS, n_top=N_PAIRS)
    trade_dates = window.index[FORM:]
    book = {name: np.zeros(TRADE) for name in RULES}
    for pick in picks:
        y, x = logp[pick["y"]].to_numpy(), logp[pick["x"]].to_numpy()
        gross = 1 + abs(pick["beta"])
        cost = BPS / 1e4 * gross
        prep = prepare(y, x, FORM, cost, DELTA, one_trip=False)
        if not prep["cointegrated"]:
            continue
        sx = prep["sigma_X"]
        row = dict(trade_start=trade_dates[0].date(), **pick)
        for name, rule in RULES.items():
            u_in, u_out = prep["rep"] if rule["kind"] == "repeated" else (-2 * sx, 0.0)
            stop = STOP_SIGMAS * sx if rule.get("stop_loss") else None
            max_hold = int(round(HOLD_HALF_LIVES * prep["half_life"])) if rule.get("time_stop") else None
            pnl, n = run_strategy(y, x, prep, FORM, u_in, u_out, cost, use_gate=False,
                                  stop=stop, max_hold=max_hold)
            ret = pnl / gross
            book[name] += ret / N_PAIRS
            row[f"{name} return %"] = ret.sum() * 100
            row[f"{name} trades"] = n
        pair_rows.append(row)
    for name in RULES:
        daily[name].append(book[name])
    dates_all.extend(trade_dates)
    line = "  ".join(f"{name} {book[name].sum() * 100:+6.2f}%" for name in RULES)
    print(f"{trade_dates[0].date()}: {len(picks):2d} pairs   {line}")

daily = pd.DataFrame({name: np.concatenate(v) for name, v in daily.items()}, index=dates_all)
daily.to_csv(ROOT / f"universe_daily_{BPS:g}bps.csv")
pd.DataFrame(pair_rows).to_csv(ROOT / f"universe_pairs_{BPS:g}bps.csv", index=False)

print(f"\ncost {BPS:g} bp per leg, {N_PAIRS} pair slots, {len(daily) / 252:.1f} years out of sample\n")
for name in RULES:
    r = daily[name]
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    curve = r.cumsum()
    max_dd = (curve.cummax() - curve).max()
    yearly = r.groupby(np.arange(len(r)) // TRADE).sum()
    t_stat = r.mean() / (r.std() / np.sqrt(len(r)))
    print(f"   {name:22s} return {ann * 100:5.2f}%/yr   vol {vol * 100:5.2f}%   Sharpe {ann / vol:5.2f}   "
          f"max DD {max_dd * 100:5.2f}%   up years {np.mean(yearly > 0):4.0%}   t-stat {t_stat:5.2f}")
print(f"\ntook {time.time() - start_time:.0f} s")