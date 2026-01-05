"""
Market data fetching (prices, volume).
Handles Yahoo Finance with fallbacks to Stooq.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from functools import lru_cache
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    
    Args:
        ticker: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
        interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV data or None
    """
    if not HAS_YFINANCE:
        print("yfinance not installed")
        return None
    
    # Try 1: yfinance Ticker.history()
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        
        if df is not None and not df.empty:
            # Normalize MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        print(f"[get_price_history] yfinance Ticker.history failed for {ticker}: {e}")
    
    # Try 2: Stooq CSV (daily only)
    if interval == "1d":
        try:
            url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
            df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            df = df.sort_index()
            
            # Filter by period
            days = _period_to_days(period)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df.index >= cutoff]
            
            if not df.empty:
                return df
        except Exception as e:
            print(f"[get_price_history] Stooq failed for {ticker}: {e}")
    
    # Try 3: yfinance download
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception as e:
        print(f"[get_price_history] yfinance download failed for {ticker}: {e}")
    
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
    candidates = ["^GSPC", "^SPX", "SPY", "VOO"]
    
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
