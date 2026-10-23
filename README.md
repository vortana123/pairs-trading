
Kalman-filtered hedge ratio, Bellman-optimal entry and exit thresholds, and a CUSUM gate that stops trading when the data stops following the model.

## Layout

```
data/simulate.py          generates the simulated pairs 
data/*.csv                columns: t, x, y, window, plus true values
data/true_params.json     the parameters the data were generated with
src/cointegration.py      Part 2: Engle-Granger test 
tests/check_cointegration.py  
```

## Plan

1. Cointegration test (Part 2)
2. AR(1) fit of the spread: phi, sigma, half-life (Part 1)
3. Kalman filter for beta_t and the spread (Part 3)
4. Value iteration for u_in and u_out (Part 4)
5. CUSUM on the innovations z_t (Part 5)
6. Backtest on simulated data, then on real pairs


