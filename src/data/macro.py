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

_FRED_SESSION = requests.Session()
_MACRO_MEMO: Dict[str, pd.DataFrame] = {}


def _load_cached_series(cache_file: Path) -> pd.Series:
    """Load a cached FRED series from disk."""
    with open(cache_file) as f:
        data = json.load(f)
    return pd.Series(data["values"], index=pd.to_datetime(data["dates"]))


def _find_stale_cache(series_id: str) -> Optional[Path]:
    """Return the newest cache file for a FRED series, if any."""
    matches = sorted(MACRO_CACHE_DIR.glob(f"{series_id}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


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
            return _load_cached_series(cache_file)
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
        resp = _FRED_SESSION.get(url, params=params, timeout=10)
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
        if cache_file.exists():
            try:
                return _load_cached_series(cache_file)
            except Exception:
                pass
        stale_cache = _find_stale_cache(series_id)
        if stale_cache is not None:
            try:
                print(f"[get_fred_series] Using stale cache for {series_id}: {stale_cache.name}")
                return _load_cached_series(stale_cache)
            except Exception:
                pass
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
    if spx_returns is None and period in _MACRO_MEMO:
        return _MACRO_MEMO[period].copy()

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
    
    # NEW: Credit spreads & VIX term structure
    baa_yield = get_fred_series(FRED_SERIES.get("baa_yield", "DBAA"), start, end)
    aaa_yield = get_fred_series(FRED_SERIES.get("aaa_yield", "DAAA"), start, end)
    t2y = get_fred_series(FRED_SERIES.get("t2y", "DGS2"), start, end)
    vix3m = get_fred_series(FRED_SERIES.get("vix3m", "VXVCLS"), start, end)
    
    # Build DataFrame
    macro = pd.DataFrame({
        "vix": vix,
        "t10y": t10y,
        "unrate": unrate,
        "cpi": cpi,
        "oas": oas,
        "fed_funds": fed_funds,
    })
    
    # Calculate term spread (10Y - 3M)
    if not t10y.empty and not t3m.empty:
        # Align indices
        aligned = pd.DataFrame({"t10y": t10y, "t3m": t3m}).ffill().bfill()
        macro["term_spread"] = aligned["t10y"] - aligned["t3m"]
    else:
        macro["term_spread"] = 0.0
    
    # NEW: Credit spread (BAA - AAA) — widens during stress/risk-off
    if not baa_yield.empty and not aaa_yield.empty:
        aligned_credit = pd.DataFrame({"baa": baa_yield, "aaa": aaa_yield}).ffill().bfill()
        macro["credit_spread"] = aligned_credit["baa"] - aligned_credit["aaa"]
        macro["credit_spread_chg_5d"] = macro["credit_spread"].diff(5)
    else:
        macro["credit_spread"] = 0.0
        macro["credit_spread_chg_5d"] = 0.0
    
    # NEW: 2s10s term spread — inversion is a recession signal
    if not t10y.empty and not t2y.empty:
        aligned_2s10s = pd.DataFrame({"t10y": t10y, "t2y": t2y}).ffill().bfill()
        macro["term_spread_2s10s"] = aligned_2s10s["t10y"] - aligned_2s10s["t2y"]
    else:
        macro["term_spread_2s10s"] = 0.0
    
    # NEW: VIX term structure — ratio of VIX to VIX3M
    # < 1.0 = contango (calm), > 1.0 = backwardation (fear/panic)
    if not vix.empty and not vix3m.empty:
        aligned_vts = pd.DataFrame({"vix": vix, "vix3m": vix3m}).ffill().bfill()
        macro["vix_term_structure"] = aligned_vts["vix"] / aligned_vts["vix3m"].replace(0, pd.NA)
        macro["vix_ts_chg_5d"] = macro["vix_term_structure"].diff(5)
    else:
        macro["vix_term_structure"] = 1.0
        macro["vix_ts_chg_5d"] = 0.0
    
    # Add market return if provided
    if spx_returns is not None and not spx_returns.empty:
        macro["mkt_ret_1d"] = spx_returns
    else:
        macro["mkt_ret_1d"] = 0.0
    
    # Forward-fill then backward-fill (macro data updates less frequently)
    macro = macro.ffill().bfill().fillna(0)
    
    if spx_returns is None:
        _MACRO_MEMO[period] = macro.copy()
    
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
    
    target_index = pd.DatetimeIndex(target_index)
    if target_index.tz is not None:
        target_index = target_index.tz_localize(None)
    
    macro_df = macro_df.copy()
    macro_index = pd.DatetimeIndex(macro_df.index)
    if macro_index.tz is not None:
        macro_df.index = macro_index.tz_localize(None)
    
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
