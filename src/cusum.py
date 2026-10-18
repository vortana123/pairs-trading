import numpy as np


def mean_scores(z, eta):
    return eta * z - eta**2 / 2


def var_scores(z, zeta):
    return -np.log(zeta) + z**2 / 2 * (1 - 1 / zeta**2)


def cusum(s):
    C = np.empty(len(s))
    c = 0.0
    for t, st in enumerate(s):
        c = max(0.0, c + st)
        C[t] = c
    return C


def gate(z, eta=1.0, zeta=1.5, level=7.0):
    C_up = cusum(mean_scores(z, eta))
    C_down = cusum(mean_scores(z, -eta))
    C_var = cusum(var_scores(z, zeta))
    hit = (C_up >= level) | (C_down >= level) | (C_var >= level)
    alarm = int(np.argmax(hit)) if hit.any() else None
    return dict(C_up=C_up, C_down=C_down, C_var=C_var, alarm=alarm)


def siegmund_arl0(k, level):
    b = level + 1.166
    return (np.exp(2 * k * b) - 2 * k * b - 1) / (2 * k**2)