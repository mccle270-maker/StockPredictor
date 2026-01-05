"""
Options pricing models.
Pure functions for Black-Scholes, Heston, and Greeks calculation.
"""
import numpy as np
from scipy.stats import norm
from typing import Optional, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum
import datetime as dt


class PricingModel(Enum):
    """Available pricing models."""
    BLACK_SCHOLES = "black_scholes"
    HESTON = "heston"


@dataclass
class OptionSpec:
    """Option contract specification."""
    spot: float
    strike: float
    maturity_date: dt.date
    valuation_date: dt.date
    rate: float = 0.05
    div_yield: float = 0.0
    vol: float = 0.25
    is_call: bool = True
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        days = (self.maturity_date - self.valuation_date).days
        return max(days / 365.0, 0.001)  # Minimum 1 day


@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float = 0.04      # Initial variance
    theta: float = 0.04   # Long-run variance
    kappa: float = 1.5    # Mean reversion speed
    sigma: float = 0.3    # Vol of vol
    rho: float = -0.6     # Correlation


class Greeks(NamedTuple):
    """Option Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# ============================================================================
# BLACK-SCHOLES
# ============================================================================

def black_scholes_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple:
    """Calculate d1 and d2 for Black-Scholes."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    q: float = 0.0,
) -> Optional[float]:
    """
    Calculate Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annual)
        sigma: Volatility (annual)
        is_call: True for call, False for put
        q: Dividend yield (annual)
    
    Returns:
        Option price or None if invalid inputs
    """
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    if d1 is None:
        return None
    
    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return float(price)


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
    q: float = 0.0,
) -> Optional[Greeks]:
    """
    Calculate Black-Scholes Greeks.
    
    Returns:
        Greeks namedtuple or None if invalid inputs
    """
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma, q)
    if d1 is None:
        return None
    
    # Common terms
    sqrt_T = np.sqrt(T)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)
    nd1 = norm.pdf(d1)
    
    # Delta
    if is_call:
        delta = exp_qT * norm.cdf(d1)
    else:
        delta = -exp_qT * norm.cdf(-d1)
    
    # Gamma (same for call and put)
    gamma = exp_qT * nd1 / (S * sigma * sqrt_T)
    
    # Vega (same for call and put) - per 1% vol change
    vega = S * exp_qT * nd1 * sqrt_T / 100
    
    # Theta (per day)
    if is_call:
        theta = (
            -S * exp_qT * nd1 * sigma / (2 * sqrt_T)
            - r * K * exp_rT * norm.cdf(d2)
            + q * S * exp_qT * norm.cdf(d1)
        ) / 365
    else:
        theta = (
            -S * exp_qT * nd1 * sigma / (2 * sqrt_T)
            + r * K * exp_rT * norm.cdf(-d2)
            - q * S * exp_qT * norm.cdf(-d1)
        ) / 365
    
    # Rho (per 1% rate change)
    if is_call:
        rho = K * T * exp_rT * norm.cdf(d2) / 100
    else:
        rho = -K * T * exp_rT * norm.cdf(-d2) / 100
    
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility_bs(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool = True,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson.
    
    Returns:
        Implied volatility or None if not convergent
    """
    sigma = 0.25  # Initial guess
    
    for _ in range(max_iter):
        calc_price = black_scholes_price(S, K, T, r, sigma, is_call, q)
        if calc_price is None:
            return None
        
        diff = calc_price - price
        if abs(diff) < tol:
            return sigma
        
        greeks = black_scholes_greeks(S, K, T, r, sigma, is_call, q)
        if greeks is None or greeks.vega == 0:
            return None
        
        # Newton-Raphson step (vega is per 1%, so multiply by 100)
        sigma = sigma - diff / (greeks.vega * 100)
        sigma = max(0.01, min(5.0, sigma))  # Clamp to reasonable range
    
    return None  # Did not converge


# ============================================================================
# HESTON MODEL (Simplified/Approximate)
# ============================================================================

def heston_price_approx(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    is_call: bool = True,
    q: float = 0.0,
) -> Optional[float]:
    """
    Approximate Heston price using moment-matching to BS.
    
    For a full Heston implementation, use FFT-based pricing.
    This approximation uses the average expected variance.
    """
    if T <= 0 or S <= 0 or K <= 0:
        return None
    
    v0, theta, kappa, sigma_v, rho = (
        params.v0, params.theta, params.kappa, params.sigma, params.rho
    )
    
    # Average variance over life of option
    if kappa * T < 0.01:
        avg_var = v0
    else:
        avg_var = theta + (v0 - theta) * (1 - np.exp(-kappa * T)) / (kappa * T)
    
    # Effective volatility
    sigma_eff = np.sqrt(max(avg_var, 0.0001))
    
    # Use BS with effective volatility
    return black_scholes_price(S, K, T, r, sigma_eff, is_call, q)


# ============================================================================
# UNIFIED PRICING INTERFACE
# ============================================================================

def price_option(
    spec: OptionSpec,
    model: PricingModel = PricingModel.BLACK_SCHOLES,
    heston_params: Optional[HestonParams] = None,
) -> Optional[float]:
    """
    Price an option using specified model.
    
    Args:
        spec: Option specification
        model: Pricing model to use
        heston_params: Required if model is HESTON
    
    Returns:
        Option price or None
    """
    S = spec.spot
    K = spec.strike
    T = spec.time_to_expiry
    r = spec.rate
    q = spec.div_yield
    sigma = spec.vol
    is_call = spec.is_call
    
    if model == PricingModel.BLACK_SCHOLES:
        return black_scholes_price(S, K, T, r, sigma, is_call, q)
    
    elif model == PricingModel.HESTON:
        if heston_params is None:
            # Fall back to BS
            return black_scholes_price(S, K, T, r, sigma, is_call, q)
        return heston_price_approx(S, K, T, r, heston_params, is_call, q)
    
    else:
        raise ValueError(f"Unknown pricing model: {model}")


def get_greeks(
    spec: OptionSpec,
) -> Optional[Greeks]:
    """
    Get option Greeks (Black-Scholes based).
    """
    return black_scholes_greeks(
        S=spec.spot,
        K=spec.strike,
        T=spec.time_to_expiry,
        r=spec.rate,
        sigma=spec.vol,
        is_call=spec.is_call,
        q=spec.div_yield,
    )


# ============================================================================
# MONTE CARLO PRICING
# ============================================================================

def monte_carlo_option_ev(
    s0: float,
    strike: float,
    mu: float,
    sigma: float,
    days: int,
    premium: float,
    is_call: bool = True,
    n_paths: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo expected value of option position.
    
    Args:
        s0: Initial stock price
        strike: Strike price
        mu: Expected daily return
        sigma: Daily volatility
        days: Days to expiration
        premium: Option premium paid
        is_call: True for call, False for put
        n_paths: Number of simulation paths
        seed: Random seed
    
    Returns:
        Dict with mc_ev, mc_pop_gt0, mc_pct_itm
    """
    np.random.seed(seed)
    
    # Simulate paths
    dt = 1.0  # Daily steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Generate random returns
    z = np.random.standard_normal((n_paths, days))
    log_returns = drift + diffusion * z
    
    # Final prices
    final_prices = s0 * np.exp(np.sum(log_returns, axis=1))
    
    # Option payoffs
    if is_call:
        payoffs = np.maximum(final_prices - strike, 0)
    else:
        payoffs = np.maximum(strike - final_prices, 0)
    
    # P&L (payoff minus premium)
    pnl = payoffs - premium
    
    return {
        "mc_ev": float(pnl.mean()),
        "mc_pop_gt0": float((pnl > 0).mean()),
        "mc_pct_itm": float((payoffs > 0).mean()),
        "mc_mean_payoff": float(payoffs.mean()),
        "mc_std_payoff": float(payoffs.std()),
    }


# ============================================================================
# STRATEGY SUGGESTION
# ============================================================================

# Horizon multipliers for options strategy thresholds
HORIZON_MULTIPLIERS = {1: 1.0, 2: 1.4, 3: 1.7, 4: 2.0, 5: 2.3}


def suggest_options_strategy(
    pred_ret: float,
    put_call_ratio: Optional[float] = None,
    atm_iv: Optional[float] = None,
    horizon: int = 1,
) -> tuple[str, str]:
    """
    Suggest an options strategy based on predicted return and market conditions.
    
    Args:
        pred_ret: Predicted return (as decimal, e.g., 0.02 for 2%)
        put_call_ratio: Put/Call open interest ratio
        atm_iv: At-the-money implied volatility
        horizon: Prediction horizon in days
    
    Returns:
        Tuple of (strategy_description, bias) where bias is "bullish", "bearish", or "neutral"
    """
    pred_pct = float(pred_ret or 0.0) * 100.0
    threshold_multiplier = HORIZON_MULTIPLIERS.get(int(horizon), 1.0)
    adjusted_threshold = 1.0 * threshold_multiplier
    
    # Strong directional signal
    if abs(pred_pct) > adjusted_threshold:
        if pred_pct > 0:
            if put_call_ratio and put_call_ratio > 1.2:
                return "BULLISH: Buy Calls (high put OI suggests potential squeeze)", "bullish"
            return "BULLISH: Buy Calls or Bull Call Spread", "bullish"
        else:
            if put_call_ratio and put_call_ratio < 0.8:
                return "BEARISH: Buy Puts (low protection)", "bearish"
            return "BEARISH: Buy Puts or Bear Put Spread", "bearish"
    
    # High IV, low directional conviction → Iron Condor
    if abs(pred_pct) < (0.5 * threshold_multiplier) and atm_iv and float(atm_iv) > 0.35:
        return "NEUTRAL: Sell Iron Condor (high IV, harvest premium)", "neutral"
    
    return "NEUTRAL: Wait / no-trade", "neutral"


def normalize_strategy(text: str, prefer_spreads: bool = True) -> Optional[str]:
    """
    Normalize strategy text to standard strategy code.
    
    Args:
        text: Strategy description from suggest_options_strategy
        prefer_spreads: Whether to prefer spread strategies over single-leg
    
    Returns:
        Strategy code (BUY_CALL, BUY_PUT, BULL_CALL_SPREAD, BEAR_PUT_SPREAD, IRON_CONDOR)
        or None if no trade suggested
    """
    s = (text or "").lower()
    
    if "iron condor" in s:
        return "IRON_CONDOR"
    if "bullish" in s and "call" in s and "spread" in s:
        return "BULL_CALL_SPREAD" if prefer_spreads else "BUY_CALL"
    if "bullish" in s and "buy call" in s:
        return "BUY_CALL"
    if "bearish" in s and "put" in s and "spread" in s:
        return "BEAR_PUT_SPREAD" if prefer_spreads else "BUY_PUT"
    if "bearish" in s and "buy put" in s:
        return "BUY_PUT"
    
    return None
