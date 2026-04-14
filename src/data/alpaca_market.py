"""
Alpaca Markets API for historical stock data.

Alpaca provides up to 6 years of historical bar data with clean, reliable data.
Uses alpaca-py SDK for modern Python interface.

API Docs: https://docs.alpaca.markets/docs/market-data-api
"""
import os
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
import warnings

# Try to import alpaca-py
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False
    StockHistoricalDataClient = None
    StockBarsRequest = None
    TimeFrame = None
    TimeFrameUnit = None

# Load API keys from environment
def _get_alpaca_client() -> Optional["StockHistoricalDataClient"]:
    """
    Get authenticated Alpaca client.
    
    Reads keys from environment variables:
    - APCA_API_KEY_ID
    - APCA_API_SECRET_KEY
    """
    if not HAS_ALPACA:
        return None
    
    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")
    
    if not api_key or not secret_key:
        # Try loading from .env files
        try:
            from dotenv import load_dotenv
            load_dotenv()
            load_dotenv(".env.apis")
            api_key = os.environ.get("APCA_API_KEY_ID")
            secret_key = os.environ.get("APCA_API_SECRET_KEY")
        except ImportError:
            pass
    
    if not api_key or not secret_key:
        print("[Alpaca] API keys not found. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        return None
    
    try:
        import concurrent.futures
        def _create():
            return StockHistoricalDataClient(api_key, secret_key)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_create)
            client = future.result(timeout=10)  # 10s max — Alpaca may be under maintenance
        return client
    except concurrent.futures.TimeoutError:
        print("[Alpaca] Client creation timed out (API may be under maintenance)")
        return None
    except Exception as e:
        print(f"[Alpaca] Failed to create client: {e}")
        return None


def _period_to_start_date(period: str) -> datetime:
    """
    Convert period string to start date.
    
    Alpaca supports up to ~6 years of historical data.
    
    Args:
        period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 6y, max)
    
    Returns:
        Start datetime for the period
    """
    now = datetime.now()
    
    period_map = {
        "1d": timedelta(days=1),
        "5d": timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730),
        "3y": timedelta(days=1095),
        "5y": timedelta(days=1825),
        "6y": timedelta(days=2190),  # Alpaca max
        "max": timedelta(days=2190),  # ~6 years max
    }
    
    delta = period_map.get(period.lower(), timedelta(days=365))
    return now - delta


def _interval_to_timeframe(interval: str) -> Optional["TimeFrame"]:
    """
    Convert interval string to Alpaca TimeFrame.
    
    Args:
        interval: Interval string (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        Alpaca TimeFrame object
    """
    if not HAS_ALPACA:
        return None
    
    interval_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
        "1wk": TimeFrame.Week,
        "1mo": TimeFrame.Month,
    }
    
    return interval_map.get(interval.lower(), TimeFrame.Day)


def get_price_history_alpaca(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch historical price data from Alpaca Markets API.
    
    Alpaca provides clean, reliable data with up to 6 years of history.
    Data is adjusted for splits and dividends.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "MSFT")
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 6y, max)
        interval: Bar interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex
        Returns None on failure
    
    Example:
        >>> df = get_price_history_alpaca("AAPL", period="2y", interval="1d")
        >>> print(df.head())
                            Open    High     Low   Close    Volume
        2024-01-02  185.32  186.20  184.90  185.50  45678900
        ...
    """
    if not HAS_ALPACA:
        print("[Alpaca] alpaca-py not installed. Run: pip install alpaca-py")
        return None
    
    client = _get_alpaca_client()
    if client is None:
        return None
    
    # Get timeframe
    timeframe = _interval_to_timeframe(interval)
    if timeframe is None:
        print(f"[Alpaca] Unsupported interval: {interval}")
        return None
    
    # Calculate date range
    start_date = _period_to_start_date(period)
    end_date = datetime.now()
    
    try:
        # Create request
        request = StockBarsRequest(
            symbol_or_symbols=ticker.upper(),
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )
        
        # Fetch bars
        bars = client.get_stock_bars(request)
        
        if bars is None:
            print(f"[Alpaca] No data returned for {ticker}")
            return None
        
        # Convert to DataFrame
        # bars.df returns MultiIndex DataFrame with symbol as first level
        df = bars.df
        
        if df is None or df.empty:
            print(f"[Alpaca] Empty data for {ticker}")
            return None
        
        # Handle MultiIndex (symbol, timestamp)
        if isinstance(df.index, pd.MultiIndex):
            # Get data for our specific ticker
            if ticker.upper() in df.index.get_level_values(0):
                df = df.loc[ticker.upper()]
            else:
                print(f"[Alpaca] Ticker {ticker} not found in response")
                return None
        
        # Rename columns to match yfinance format (capitalized)
        column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "trade_count": "TradeCount",
            "vwap": "VWAP",
        }
        df = df.rename(columns=column_map)
        
        # Ensure we have required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                print(f"[Alpaca] Missing required column: {col}")
                return None
        
        # Keep only OHLCV columns for compatibility
        df = df[required].copy()
        
        # Ensure index is timezone-naive for compatibility with yfinance data
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        
        # Sort by date ascending
        df = df.sort_index()
        
        print(f"[Alpaca] Fetched {len(df)} bars for {ticker} ({period}, {interval})")
        return df
        
    except Exception as e:
        print(f"[Alpaca] Error fetching {ticker}: {e}")
        return None


def get_latest_quote_alpaca(ticker: str) -> Optional[dict]:
    """
    Get the latest quote for a ticker from Alpaca.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Dict with bid, ask, last price, etc. or None
    """
    if not HAS_ALPACA:
        return None
    
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        
        client = _get_alpaca_client()
        if client is None:
            return None
        
        request = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
        quote = client.get_stock_latest_quote(request)
        
        if quote and ticker.upper() in quote:
            q = quote[ticker.upper()]
            return {
                "bid": q.bid_price,
                "ask": q.ask_price,
                "bid_size": q.bid_size,
                "ask_size": q.ask_size,
                "timestamp": q.timestamp,
            }
    except Exception as e:
        print(f"[Alpaca] Error getting quote for {ticker}: {e}")
    
    return None


def get_latest_trade_alpaca(ticker: str) -> Optional[dict]:
    """
    Get the latest trade for a ticker from Alpaca.
    
    Args:
        ticker: Stock symbol
    
    Returns:
        Dict with price, size, timestamp or None
    """
    if not HAS_ALPACA:
        return None
    
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        
        client = _get_alpaca_client()
        if client is None:
            return None
        
        request = StockLatestTradeRequest(symbol_or_symbols=ticker.upper())
        trade = client.get_stock_latest_trade(request)
        
        if trade and ticker.upper() in trade:
            t = trade[ticker.upper()]
            return {
                "price": t.price,
                "size": t.size,
                "timestamp": t.timestamp,
            }
    except Exception as e:
        print(f"[Alpaca] Error getting trade for {ticker}: {e}")
    
    return None


# Quick test
if __name__ == "__main__":
    print("Testing Alpaca Market Data...")
    
    # Test historical data
    df = get_price_history_alpaca("AAPL", period="1y", interval="1d")
    if df is not None:
        print(f"\n✅ Historical data: {len(df)} bars")
        print(df.tail())
    else:
        print("❌ Failed to fetch historical data")
    
    # Test latest quote
    quote = get_latest_quote_alpaca("AAPL")
    if quote:
        print(f"\n✅ Latest quote: bid=${quote['bid']:.2f}, ask=${quote['ask']:.2f}")
    else:
        print("❌ Failed to get quote")
