# monte_carlo_pricer.py
import numpy as np

def simulate_price_paths(
    s0: float,
    mu: float,
    sigma: float,
    days: int,
    n_paths: int = 5000,
    dt: float = 1.0 / 252,
):
    # Geometric Brownian Motion
    steps = int(days)
    drift = (mu - 0.5 * sigma**2) * dt
    vol_step = sigma * np.sqrt(dt)

    paths = np.zeros((steps + 1, n_paths))
    paths[0] = s0
    for t in range(1, steps + 1):
        z = np.random.normal(size=n_paths)
        paths[t] = paths[t - 1] * np.exp(drift + vol_step * z)
    return paths

def option_mc_ev(
    s0: float,
    mu: float,
    sigma: float,
    days: int,
    premium: float,
    strike: float,
    n_paths: int = 5000,
    is_call: bool = True,
):
    paths = simulate_price_paths(s0, mu, sigma, days, n_paths=n_paths)
    st = paths[-1]

    if is_call:
        payoff = np.maximum(st - strike, 0.0)
    else:
        payoff = np.maximum(strike - st, 0.0)

    pnl = payoff - premium
    ev = float(pnl.mean())
    pop_gt0 = float((pnl > 0).mean())
    return {"mc_ev": ev, "mc_pop_gt0": pop_gt0}
