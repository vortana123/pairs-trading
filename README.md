Bayesian pairs trading

A pairs-trading system built in Python: an Engle–Granger cointegration screen, a Kalman filter for the hedge ratio and spread, entry and exit thresholds from a Bellman equation with trading costs, and a CUSUM test on the filter innovations to detect pairs that break. Every component was tested against simulated data to see if it worked.

Results: the tests worked as the theory predicted in simulation, but on real data the simple ±2σ rule beats the model-optimal rules. An out-of-sample universe study (88 US-listed stocks and ETFs, from 2014 to 2024) gives a Sharpe ratio of 0.51 with t = 1.69 which is a small edge that is not statistically significant.
