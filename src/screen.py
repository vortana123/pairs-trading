from itertools import combinations

from ar1 import fit_ar1
from cointegration import engle_granger


def screen_pairs(logp, sectors, n_top=10, min_hl=2.0, max_hl=25.0, min_beta=0.0):
    candidates = []
    for sector, names in sectors.items():
        names = [n for n in names if n in logp.columns and logp[n].notna().all()]
        for a, b in combinations(names, 2):
            best = None
            for y_name, x_name in [(a, b), (b, a)]:
                y, x = logp[y_name].to_numpy(), logp[x_name].to_numpy()
                eg = engle_granger(y, x - x.mean())
                if best is None or eg["tstat"] < best[2]:
                    best = (y_name, x_name, eg["tstat"], eg["spread"], eg["beta"])
            y_name, x_name, tstat, spread, beta = best
            if tstat >= -3.34 or beta <= min_beta:
                continue
            hl = fit_ar1(spread)["half_life"]
            if min_hl <= hl <= max_hl:
                candidates.append(dict(sector=sector, y=y_name, x=x_name, beta=beta, tstat=tstat, half_life=hl))
    candidates.sort(key=lambda c: c["tstat"])
    return candidates[:n_top]