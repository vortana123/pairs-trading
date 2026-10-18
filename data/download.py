import sys
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
from universe import SECTORS

TICKERS = sorted({t for names in SECTORS.values() for t in names})
START = "2012-01-01"
END = "2026-01-01"

prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
prices = prices.dropna(how="all")
prices.to_csv(HERE / "prices.csv")
print(f"saved {prices.shape[0]} days x {prices.shape[1]} tickers to {HERE / 'prices.csv'}")
missing = [t for t in TICKERS if t not in prices or prices[t].notna().sum() < 1000]
print("missing or short:", missing if missing else "none")