"""
Options data fetching and Greeks calculation.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

from ..core.pricing import black_scholes_price, black_scholes_greeks, Greeks

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ============================================================================
# OPTIONS CHAIN
# ============================================================================

def get_option_chain(
    ticker: str,
    expiration: Optional[str] = None,
    calls_only: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch options chain for a ticker.
    
    Args:
        ticker: Stock symbol
        expiration: Specific expiration date (YYYY-MM-DD), or None for nearest
        calls_only: If True, return only calls
    
    Returns:
        DataFrame with option chain data
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        
        if not expirations:
            return None
        
        if expiration is None:
            expiration = expirations[0]
        elif expiration not in expirations:
            # Find closest expiration
            exp_dates = pd.to_datetime(expirations)
            target = pd.to_datetime(expiration)
            closest_idx = (exp_dates - target).abs().argmin()
            expiration = expirations[closest_idx]
        
        chain = t.option_chain(expiration)
        
        if calls_only:
            return chain.calls
        
        # Combine calls and puts
        calls = chain.calls.copy()
        calls["optionType"] = "call"
        puts = chain.puts.copy()
        puts["optionType"] = "put"
        
        return pd.concat([calls, puts], ignore_index=True)
    
    except Exception as e:
        print(f"[get_option_chain] Failed for {ticker}: {e}")
        return None


def get_expirations(ticker: str) -> list:
    """Get available expiration dates for a ticker."""
    if not HAS_YFINANCE:
        return []
    
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception:
        return []


# ============================================================================
# ATM OPTIONS
# ============================================================================

def get_atm_options(
    ticker: str,
    moneyness_window: float = 0.05,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """
    Get near-ATM call and put options.
    
    Args:
        ticker: Stock symbol
        moneyness_window: Range around spot to consider ATM (0.05 = 5%)
    
    Returns:
        Tuple of (atm_calls_df, atm_puts_df, expiration_date)
    """
    if not HAS_YFINANCE:
        return None, None, None
    
    try:
        t = yf.Ticker(ticker)
        
        # Get current price
        hist = t.history(period="1d")
        if hist.empty:
            return None, None, None
        spot = float(hist["Close"].iloc[-1])
        
        # Get nearest expiration
        expirations = t.options
        if not expirations:
            return None, None, None
        
        expiration = expirations[0]
        chain = t.option_chain(expiration)
        
        # Filter to ATM
        strike_low = spot * (1 - moneyness_window)
        strike_high = spot * (1 + moneyness_window)
        
        atm_calls = chain.calls[
            (chain.calls["strike"] >= strike_low) & 
            (chain.calls["strike"] <= strike_high)
        ].copy()
        
        atm_puts = chain.puts[
            (chain.puts["strike"] >= strike_low) & 
            (chain.puts["strike"] <= strike_high)
        ].copy()
        
        return atm_calls, atm_puts, expiration
    
    except Exception as e:
        print(f"[get_atm_options] Failed for {ticker}: {e}")
        return None, None, None


# ============================================================================
# OPTIONS SNAPSHOT
# ============================================================================

def get_option_snapshot(
    ticker: str,
    moneyness_window: float = 0.05,
) -> Dict:
    """
    Get key options metrics for a ticker.
    
    Returns:
        Dict with atm_iv, put_call_oi_ratio, opt_exp, call_oi, put_oi
    """
    atm_calls, atm_puts, expiration = get_atm_options(ticker, moneyness_window)
    
    out = {
        "atm_iv": None,
        "put_call_oi_ratio": None,
        "opt_exp": expiration,
        "call_oi": None,
        "put_oi": None,
    }
    
    if atm_calls is None or atm_puts is None:
        return out
    
    # Average ATM IV (weighted by volume)
    if "impliedVolatility" in atm_calls.columns and not atm_calls.empty:
        call_iv = atm_calls["impliedVolatility"].mean()
        out["atm_iv"] = float(call_iv) if pd.notna(call_iv) else None
    
    # Open interest
    if "openInterest" in atm_calls.columns:
        call_oi = atm_calls["openInterest"].sum()
        out["call_oi"] = int(call_oi) if pd.notna(call_oi) else 0
    
    if "openInterest" in atm_puts.columns:
        put_oi = atm_puts["openInterest"].sum()
        out["put_oi"] = int(put_oi) if pd.notna(put_oi) else 0
    
    # Put/Call OI ratio
    if out["call_oi"] and out["call_oi"] > 0:
        out["put_call_oi_ratio"] = out["put_oi"] / out["call_oi"]
    
    return out


# ============================================================================
# GREEKS
# ============================================================================

def get_atm_greeks(
    ticker: str,
    risk_free_rate: float = 0.05,
) -> Dict:
    """
    Calculate Greeks for ATM options.
    
    Returns:
        Dict with call_greeks, put_greeks, spot, expiration
    """
    if not HAS_YFINANCE:
        return {}
    
    try:
        t = yf.Ticker(ticker)
        
        # Get spot price
        hist = t.history(period="1d")
        if hist.empty:
            return {}
        spot = float(hist["Close"].iloc[-1])
        
        # Get options
        expirations = t.options
        if not expirations:
            return {}
        
        expiration = expirations[0]
        exp_date = datetime.strptime(expiration, "%Y-%m-%d")
        days_to_exp = max((exp_date - datetime.now()).days, 1)
        T = days_to_exp / 365.0
        
        chain = t.option_chain(expiration)
        
        # Find ATM strike
        calls = chain.calls
        if calls.empty:
            return {}
        
        atm_idx = (calls["strike"] - spot).abs().argmin()
        atm_call = calls.iloc[atm_idx]
        atm_strike = float(atm_call["strike"])
        
        # Get IV
        iv = atm_call.get("impliedVolatility", 0.25)
        if pd.isna(iv) or iv <= 0:
            iv = 0.25
        
        # Calculate Greeks
        call_greeks = black_scholes_greeks(spot, atm_strike, T, risk_free_rate, iv, is_call=True)
        put_greeks = black_scholes_greeks(spot, atm_strike, T, risk_free_rate, iv, is_call=False)
        
        return {
            "spot": spot,
            "strike": atm_strike,
            "expiration": expiration,
            "days_to_exp": days_to_exp,
            "iv": iv,
            "call_greeks": call_greeks._asdict() if call_greeks else None,
            "put_greeks": put_greeks._asdict() if put_greeks else None,
        }
    
    except Exception as e:
        print(f"[get_atm_greeks] Failed for {ticker}: {e}")
        return {}


# ============================================================================
# IV ANALYSIS
# ============================================================================

def get_iv_percentile(
    ticker: str,
    lookback_days: int = 252,
) -> Optional[float]:
    """
    Calculate current IV percentile (rank over lookback period).
    
    Returns:
        IV percentile (0-100) or None
    """
    # This would require historical IV data which yfinance doesn't provide
    # For now, return None - could be implemented with CBOE data or similar
    return None


def iv_vs_realized(atm_iv: float, hist_vol: float) -> Optional[float]:
    """
    Calculate difference between implied and realized volatility.
    
    Returns:
        IV - HV (positive means IV is higher than realized)
    """
    if atm_iv is None or hist_vol is None:
        return None
    return atm_iv - hist_vol


# Alias for backward compatibility
def get_option_snapshot_features(ticker: str, moneyness_window: float = 0.05) -> Dict:
    """Alias for get_option_snapshot."""
    return get_option_snapshot(ticker, moneyness_window)
