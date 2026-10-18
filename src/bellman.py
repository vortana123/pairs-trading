import numpy as np


def make_expectation(u, phi, sigma, n_gh=41):
    eps, w = np.polynomial.hermite_e.hermegauss(n_gh)
    w = w / w.sum()
    u_next = phi * u[:, None] + sigma * eps[None, :]

    def E(V):
        return (np.interp(u_next, u, V) * w).sum(axis=1)

    return E


def value_iterate(G, E, delta, tol=1e-10, max_iter=200000):
    V = G.copy()
    for k in range(1, max_iter + 1):
        V_new = np.maximum(G, delta * E(V))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, k
        V = V_new
    raise RuntimeError("value iteration did not converge")


def solve_thresholds(phi, sigma, cost, delta, width=6.5, n=1201):
    sigma_X = sigma / np.sqrt(1 - phi**2)
    u = np.linspace(-width * sigma_X, width * sigma_X, n)
    E = make_expectation(u, phi, sigma)

    G = u - cost
    V, it_exit = value_iterate(G, E, delta)
    exit_set = G >= delta * E(V)
    u_out = u[exit_set].min()

    H = V - u - cost
    J, it_entry = value_iterate(H, E, delta)
    entry_set = H >= delta * E(J)
    u_in = u[entry_set].max()

    u_my = cost * (1 - delta) / (1 - delta * phi)
    return dict(u=u, V=V, J=J, exit_set=exit_set, entry_set=entry_set,
                u_out=u_out, u_in=u_in, u_my=u_my, sigma_X=sigma_X,
                iterations=(it_exit, it_entry))


def solve_repeated(phi, sigma, cost, delta=0.99, width=6.5, n=401, tol=1e-10, max_iter=200000):
    sigma_X = sigma / np.sqrt(1 - phi**2)
    u = np.linspace(-width * sigma_X, width * sigma_X, n)
    E = make_expectation(u, phi, sigma)

    J = np.zeros(n)
    L = u - cost
    S = -u - cost
    for k in range(1, max_iter + 1):
        L_new = np.maximum(u - cost + J, delta * E(L))
        S_new = np.maximum(-u - cost + J, delta * E(S))
        J_new = np.maximum.reduce([L_new - u - cost, S_new + u - cost, delta * E(J)])
        change = max(np.abs(L_new - L).max(), np.abs(S_new - S).max(), np.abs(J_new - J).max())
        L, S, J = L_new, S_new, J_new
        if change < tol:
            break
    else:
        raise RuntimeError("value iteration did not converge")

    exit_long = u - cost + J >= delta * E(L)
    enter_long = L - u - cost >= np.maximum(delta * E(J), S + u - cost)
    return dict(u=u, J=J, L=L, S=S, u_out=u[exit_long].min(), u_in=u[enter_long].max(),
                sigma_X=sigma_X, iterations=k)