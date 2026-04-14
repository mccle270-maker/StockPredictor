"""
Market data fetching (prices, volume).

Data source priority:
1. yfinance Ticker.history() (fast, free, reliable)
2. yfinance download() (fallback)
3. Alpaca Markets API (demoted - under maintenance 2026-04)

Removed:
- Stooq CSV: returns empty data as of 2026-04
- Alpha Vantage: now premium-only, always rate-limited
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from functools import lru_cache
import warnings
from pathlib import Path

from .cache_manager import get_cache

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Alpaca import is LAZY — only loaded when actually needed
# (Alpaca SDK can hang on import if API is under maintenance)
HAS_ALPACA = False
get_price_history_alpaca = None

def _lazy_load_alpaca():
    """Load Alpaca module on demand to avoid blocking imports."""
    global HAS_ALPACA, get_price_history_alpaca
    if get_price_history_alpaca is not None:
        return  # Already loaded
    try:
        from .alpaca_market import get_price_history_alpaca as _fn, HAS_ALPACA as _has
        HAS_ALPACA = _has
        get_price_history_alpaca = _fn
    except ImportError:
        HAS_ALPACA = False
        get_price_history_alpaca = None

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# Handle yfinance version compatibility
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    class YFRateLimitError(Exception):
        pass


# ============================================================================
# SPX CACHE (avoid repeated downloads)
# ============================================================================

_SPX_CACHE: Dict[tuple, pd.DataFrame] = {}
_PRICE_CACHE = get_cache()
_PRICE_CACHE_DIR = Path(".cache/data/price")


def _period_to_days(period: str) -> int:
    """Convert period string to approximate days."""
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
        "6mo": 180, "1y": 365, "2y": 730, "3y": 1095,
        "5y": 1825, "10y": 3650, "max": 7300,
    }
    return mapping.get(period, 365)


def _days_to_period(days: int) -> str:
    """Convert days to yfinance period string."""
    if days >= 3650:
        return "10y"
    if days >= 1825:
        return "5y"
    if days >= 1095:
        return "3y"
    if days >= 730:
        return "2y"
    if days >= 365:
        return "1y"
    if days >= 180:
        return "6mo"
    if days >= 90:
        return "3mo"
    if days >= 30:
        return "1mo"
    return "5d"


def _get_stale_cached_price(
    ticker: str,
    interval: str = "1d",
    min_days: int = 0,
) -> Optional[pd.DataFrame]:
    """Return the longest cached history for a ticker when exact key misses."""
    if not _PRICE_CACHE_DIR.exists():
        return None
    
    best_df = None
    best_len = -1
    for cache_file in _PRICE_CACHE_DIR.glob("*.cache"):
        try:
            import pickle
            with open(cache_file, "rb") as f:
                entry = pickle.load(f)
            if entry.get("identifier") != ticker:
                continue
            if entry.get("params", {}).get("interval") != interval:
                continue
            data = entry.get("data")
            if data is None or data.empty:
                continue
            if min_days > 0:
                try:
                    idx = pd.DatetimeIndex(data.index)
                    if len(idx) < max(30, min_days // 5):
                        continue
                    covered_days = max(0, (idx.max() - idx.min()).days)
                    if covered_days < int(min_days * 0.7):
                        continue
                except Exception:
                    continue
            if len(data) > best_len:
                best_df = data
                best_len = len(data)
        except Exception:
            continue
    
    return best_df.copy() if best_df is not None else None


# ============================================================================
# PRICE HISTORY
# ============================================================================

def get_price_history(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch price history with fallback chain.
    
    Data source priority:
    1. yfinance Ticker.history() (fast, free, reliable)
    2. yfinance download() (fallback)
    3. Alpaca (demoted - under maintenance)
    
    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV data or None
    """
    cached = _PRICE_CACHE.get("price", ticker, period=period, interval=interval)
    if cached is not None:
        return cached.copy() if hasattr(cached, "copy") else cached
    
    stale_cached = _get_stale_cached_price(
        ticker,
        interval=interval,
        min_days=_period_to_days(period),
    )
    if stale_cached is not None:
        return stale_cached

    if not HAS_YFINANCE:
        # If yfinance is unavailable, try Alpaca as last resort
        _lazy_load_alpaca()
        if HAS_ALPACA and get_price_history_alpaca is not None:
            try:
                df = get_price_history_alpaca(ticker, period=period, interval=interval)
                if df is not None and not df.empty:
                    _PRICE_CACHE.set("price", ticker, df, period=period, interval=interval)
                    return df
            except Exception as e:
                print(f"[get_price_history] Alpaca failed for {ticker}: {e}")
        print("No data providers available (yfinance not installed, Alpaca failed)")
        return None
    
    # Try 1: yfinance Ticker.history() — fastest
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True, timeout=15)
        
        if df is not None and not df.empty:
            # Normalize MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _PRICE_CACHE.set("price", ticker, df, period=period, interval=interval)
            return df
    except Exception as e:
        print(f"[get_price_history] yfinance Ticker.history failed for {ticker}: {e}")
    
    # Try 2: yfinance download() — different code path, sometimes works when .history() doesn't
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=15)
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _PRICE_CACHE.set("price", ticker, df, period=period, interval=interval)
            return df
    except Exception as e:
        print(f"[get_price_history] yfinance download failed for {ticker}: {e}")
    
    # Try 3: Alpaca (last resort — may be slow or under maintenance)
    _lazy_load_alpaca()
    if HAS_ALPACA and get_price_history_alpaca is not None:
        try:
            df = get_price_history_alpaca(ticker, period=period, interval=interval)
            if df is not None and not df.empty:
                _PRICE_CACHE.set("price", ticker, df, period=period, interval=interval)
                return df
        except Exception as e:
            print(f"[get_price_history] Alpaca failed for {ticker}: {e}")
    
    return None


def get_price_history_range(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch price history for a specific date range.
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
        )
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        print(f"[get_price_history_range] Failed for {ticker}: {e}")
    
    return None


def get_intraday_history(
    ticker: str,
    period: str = "1d",
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday price data.
    """
    return get_price_history(ticker, period=period, interval=interval)


# ============================================================================
# SPX DATA (for relative strength)
# ============================================================================

def get_spx(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz=None,
) -> pd.DataFrame:
    """
    Fetch SPX data with caching and fallback to ETF proxies.
    
    Args:
        start: Start date
        end: End date
        tz: Timezone for index
    
    Returns:
        DataFrame with SPX price data
    """
    global _SPX_CACHE
    
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    
    key = (start_ts, end_ts, str(tz))
    if key in _SPX_CACHE:
        return _SPX_CACHE[key]
    
    # Calculate period needed
    days = max(1, (end_ts - start_ts).days + 5)
    period = _days_to_period(days)
    
    # Try multiple symbols
    # Prefer ETF proxies first because they are more likely to exist in local cache
    # and tend to be more reliable than index symbols across providers.
    candidates = ["SPY", "VOO", "^GSPC", "^SPX"]
    
    spx = pd.DataFrame()
    for sym in candidates:
        try:
            df = get_price_history(sym, period=period, interval="1d")
            if df is None or df.empty:
                continue
            
            # Filter to requested range
            df_tz_naive = df.copy()
            df_tz_naive.index = df_tz_naive.index.tz_localize(None)
            df_filtered = df_tz_naive.loc[
                (df_tz_naive.index >= start_ts) & (df_tz_naive.index <= end_ts)
            ]
            
            if df_filtered.empty:
                continue
            
            spx = df_filtered.copy()
            break
        except Exception as e:
            print(f"[get_spx] {sym} failed: {e}")
            continue
    
    if spx.empty:
        print("[get_spx] Warning: all SPX proxies failed")
    else:
        # Apply timezone if requested
        if tz is not None:
            spx.index = spx.index.tz_localize(tz)
    
    _SPX_CACHE[key] = spx
    return spx


def clear_spx_cache():
    """Clear the SPX cache."""
    global _SPX_CACHE
    _SPX_CACHE = {}


# ============================================================================
# TICKER INFO
# ============================================================================

def get_ticker_info(ticker: str) -> Dict:
    """
    Get basic ticker information.
    """
    if not HAS_YFINANCE:
        return {}
    
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange"),
        }
    except Exception:
        return {}


def get_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    """
    Get next earnings date for a ticker.
    """
    if not HAS_YFINANCE:
        return None
    
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is not None and "Earnings Date" in cal.index:
            return pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass
    
    return None


# ============================================================================
# LIVE PRICE
# ============================================================================

def get_live_price(ticker: str) -> Optional[float]:
    """
    Get current/latest price for a ticker.
    """
    df = get_intraday_history(ticker, period="1d", interval="1m")
    if df is not None and not df.empty and "Close" in df.columns:
        return float(df["Close"].iloc[-1])
    
    # Fallback to daily
    df = get_price_history(ticker, period="5d", interval="1d")
    if df is not None and not df.empty and "Close" in df.columns:
        return float(df["Close"].iloc[-1])
    
    return None
