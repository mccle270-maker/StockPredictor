"""
Macro data fetching from FRED API.
"""
import os
import json
import datetime as dt
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

import requests

from ..config import FRED_API_KEY, FRED_SERIES, MACRO_CACHE_DIR


# ============================================================================
# FRED API
# ============================================================================

def get_fred_series(
    series_id: str,
    start: dt.date,
    end: dt.date,
    api_key: Optional[str] = None,
) -> pd.Series:
    """
    Fetch a single FRED series.
    
    Args:
        series_id: FRED series ID (e.g., 'VIXCLS')
        start: Start date
        end: End date
        api_key: FRED API key (uses config if not provided)
    
    Returns:
        Series with date index
    """
    api_key = api_key or FRED_API_KEY
    
    if not api_key:
        return pd.Series(dtype=float)
    
    # Check cache first
    cache_key = f"{series_id}_{start.isoformat()}_{end.isoformat()}"
    cache_file = MACRO_CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
            return pd.Series(data["values"], index=pd.to_datetime(data["dates"]))
        except Exception:
            pass
    
    # Fetch from API
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        
        dates = []
        values = []
        for obs in observations:
            if obs["value"] != ".":
                dates.append(obs["date"])
                values.append(float(obs["value"]))
        
        series = pd.Series(values, index=pd.to_datetime(dates))
        
        # Cache result
        try:
            with open(cache_file, "w") as f:
                json.dump({"dates": [d.isoformat() for d in series.index], "values": list(series.values)}, f)
        except Exception:
            pass
        
        return series
    
    except Exception as e:
        print(f"[get_fred_series] Failed to fetch {series_id}: {e}")
        return pd.Series(dtype=float)


def get_vix(period: str = "5y") -> pd.Series:
    """Get VIX series for a period."""
    end = dt.date.today()
    days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}.get(period, 365)
    start = end - dt.timedelta(days=days)
    return get_fred_series(FRED_SERIES["vix"], start, end)


def get_treasury_rates(period: str = "5y") -> Dict[str, pd.Series]:
    """Get treasury rate series."""
    end = dt.date.today()
    days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}.get(period, 365)
    start = end - dt.timedelta(days=days)
    
    return {
        "t10y": get_fred_series(FRED_SERIES["t10y"], start, end),
        "t3m": get_fred_series(FRED_SERIES["t3m"], start, end),
    }


# ============================================================================
# MACRO DATAFRAME
# ============================================================================

def get_macro_df(
    period: str = "5y",
    spx_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with all macro features.
    
    Args:
        period: Time period to fetch
        spx_returns: Optional SPX returns series (if not provided, fetched)
    
    Returns:
        DataFrame with macro columns, indexed by date
    """
    end = dt.date.today()
    days = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}.get(period, 365)
    start = end - dt.timedelta(days=days)
    
    # Fetch all series
    vix = get_fred_series(FRED_SERIES["vix"], start, end)
    t10y = get_fred_series(FRED_SERIES["t10y"], start, end)
    t3m = get_fred_series(FRED_SERIES["t3m"], start, end)
    unrate = get_fred_series(FRED_SERIES["unrate"], start, end)
    cpi = get_fred_series(FRED_SERIES["cpi"], start, end)
    oas = get_fred_series(FRED_SERIES["oas"], start, end)
    fed_funds = get_fred_series(FRED_SERIES["fed_funds"], start, end)
    
    # Build DataFrame
    macro = pd.DataFrame({
        "vix": vix,
        "t10y": t10y,
        "unrate": unrate,
        "cpi": cpi,
        "oas": oas,
        "fed_funds": fed_funds,
    })
    
    # Calculate term spread
    if not t10y.empty and not t3m.empty:
        # Align indices
        aligned = pd.DataFrame({"t10y": t10y, "t3m": t3m}).ffill().bfill()
        macro["term_spread"] = aligned["t10y"] - aligned["t3m"]
    else:
        macro["term_spread"] = 0.0
    
    # Add market return if provided
    if spx_returns is not None and not spx_returns.empty:
        macro["mkt_ret_1d"] = spx_returns
    else:
        macro["mkt_ret_1d"] = 0.0
    
    # Forward-fill then backward-fill (macro data updates less frequently)
    macro = macro.ffill().bfill().fillna(0)
    
    return macro


def align_macro_to_index(
    macro_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align macro DataFrame to a target index.
    
    Args:
        macro_df: Macro data DataFrame
        target_index: Target DatetimeIndex to align to
    
    Returns:
        Aligned DataFrame
    """
    if macro_df.empty:
        return pd.DataFrame(index=target_index)
    
    # Forward-fill before reindex to prevent look-ahead
    macro_filled = macro_df.ffill().bfill()
    
    # Reindex to target
    aligned = macro_filled.reindex(target_index, method="ffill")
    
    return aligned.fillna(0)


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_macro_cache():
    """Clear all cached macro data."""
    for f in MACRO_CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
        except Exception:
            pass
