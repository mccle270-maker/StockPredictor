import pandas as pd
import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes price for European options.
    S: spot price
    K: strike
    T: time to expiry (years)
    r: risk-free rate
    sigma: implied volatility (annualized)
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def build_option_pricing_table(
    spot,
    strikes,
    expiries,
    ivs,
    r=0.02,
    option_type='call',
):
    """
    Build a DataFrame of option prices for a grid of strikes/expiries/IVs.
    Args:
        spot: underlying price
        strikes: list of strikes
        expiries: list of expiries (in years)
        ivs: list of implied vols (annualized)
        r: risk-free rate
        option_type: 'call' or 'put'
    Returns:
        DataFrame with columns: strike, expiry, iv, price
    """
    rows = []
    for K in strikes:
        for T in expiries:
            for sigma in ivs:
                price = black_scholes_price(spot, K, T, r, sigma, option_type)
                rows.append({'strike': K, 'expiry': T, 'iv': sigma, 'price': price})
    return pd.DataFrame(rows)
