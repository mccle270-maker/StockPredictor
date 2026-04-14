"""
Alpaca Markets provider - High quality historical data.
Requires API key (free tier available).
Up to 6 years of historical data.
"""
from __future__ import annotations

import os
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

from .base import BaseProvider, ProviderResponse


def get_alpaca_credentials() -> tuple[Optional[str], Optional[str]]:
    """
    Load Alpaca API credentials from multiple sources:
    1. Streamlit secrets (production)
    2. Environment variables (development)
    
    Returns: (api_key, secret_key) or (None, None) if not found
    """
    api_key = None
    secret_key = None
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        api_key = st.secrets.get("ALPACA_API_KEY") or st.secrets.get("APCA_API_KEY_ID")
        secret_key = st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("APCA_API_SECRET_KEY")
    except Exception:
        pass
    
    # Fall back to environment variables
    if not api_key:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    if not secret_key:
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    
    return api_key, secret_key


class AlpacaProvider(BaseProvider):
    """
    Alpaca Markets data provider for historical prices.
    
    Pros:
    - High quality, clean institutional data
    - Up to 6 years of historical data (free tier)
    - IEX + SIP data feeds available
    - No rate limiting issues like yfinance
    - Same credentials as trading API
    
    Cons:
    - Requires API key (but free tier available)
    - US stocks only (good for our use case)
    """
    
    name = "alpaca"
    requires_key = True
    rate_limit_per_minute = 200  # Generous limits
    
    def __init__(self):
        super().__init__()
        self._client = None
        self._api_key = None
        self._secret_key = None
        self._available = False
        self._initialized = False
        self._StockBarsRequest = None
        self._TimeFrame = None
        
        # Only check if SDK + credentials exist — do NOT connect yet
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # noqa: F401
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            self._api_key, self._secret_key = get_alpaca_credentials()
            self._StockBarsRequest = StockBarsRequest
            self._TimeFrame = TimeFrame
            
            if self._api_key and self._secret_key:
                self._available = True  # Credentials exist, but client not yet created
            else:
                self._available = False
                
        except ImportError:
            self._available = False
    
    def _ensure_client(self) -> bool:
        """Lazily create the Alpaca client on first use (avoids blocking imports)."""
        if self._initialized:
            return self._client is not None
        self._initialized = True
        
        if not self._available:
            return False
        
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            import concurrent.futures
            
            # Create client with a timeout — Alpaca may be under maintenance
            def _create():
                return StockHistoricalDataClient(
                    api_key=self._api_key,
                    secret_key=self._secret_key,
                )
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_create)
                self._client = future.result(timeout=10)
            
            return True
        except Exception as e:
            print(f"[AlpacaProvider] Client creation failed (may be under maintenance): {e}")
            self._client = None
            self._available = False
            return False
    
    def is_available(self) -> bool:
        """Check if Alpaca credentials exist (does NOT connect yet)."""
        return self._available
    
    def _period_to_dates(self, period: str) -> tuple[datetime, datetime]:
        """Convert period string to start/end dates."""
        end = datetime.now()
        
        period_map = {
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=365 * 2),
            "5y": timedelta(days=365 * 5),
            "6y": timedelta(days=365 * 6),  # Alpaca max
            "max": timedelta(days=365 * 6),  # Use 6 years for max
        }
        
        delta = period_map.get(period, timedelta(days=365 * 2))
        start = end - delta
        
        return start, end
    
    def _interval_to_timeframe(self, interval: str):
        """Convert interval string to Alpaca TimeFrame."""
        if not hasattr(self, '_TimeFrame'):
            return None
            
        TimeFrame = self._TimeFrame
        
        interval_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrame.Minute),
            "15m": TimeFrame(15, TimeFrame.Minute),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
            "1wk": TimeFrame.Week,
            "1mo": TimeFrame.Month,
        }
        
        return interval_map.get(interval, TimeFrame.Day)
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """
        Fetch OHLCV data from Alpaca.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, 6y, max)
            interval: Bar interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
            
        Returns:
            ProviderResponse with standardized OHLCV DataFrame
        """
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca not available (missing credentials or package)",
            )
        
        # Lazy-init client on first actual use
        if not self._ensure_client():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca client failed to initialize (may be under maintenance)",
            )
        
        try:
            self._log_request()
            
            start_date, end_date = self._period_to_dates(period)
            timeframe = self._interval_to_timeframe(interval)
            
            if timeframe is None:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"Invalid interval: {interval}",
                )
            
            # Create request
            request = self._StockBarsRequest(
                symbol_or_symbols=ticker.upper(),
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
            
            # Fetch bars (with timeout protection)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._client.get_stock_bars, request)
                try:
                    bars = future.result(timeout=15)  # 15 second hard limit
                except concurrent.futures.TimeoutError:
                    return ProviderResponse(
                        success=False,
                        data=None,
                        source=self.name,
                        error="Alpaca API timed out after 15s (may be under maintenance)",
                    )
            
            if bars is None or ticker.upper() not in bars.data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No data returned for {ticker}",
                )
            
            # Convert to DataFrame
            bar_list = bars.data[ticker.upper()]
            
            if not bar_list:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"Empty data for {ticker}",
                )
            
            # Build DataFrame from bar objects
            data = []
            for bar in bar_list:
                data.append({
                    "timestamp": bar.timestamp,
                    "Open": bar.open,
                    "High": bar.high,
                    "Low": bar.low,
                    "Close": bar.close,
                    "Volume": bar.volume,
                })
            
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            df.index = pd.to_datetime(df.index)
            
            # Ensure timezone-naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Ensure required columns
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    return ProviderResponse(
                        success=False,
                        data=None,
                        source=self.name,
                        error=f"Missing column: {col}",
                    )
            
            return ProviderResponse(
                success=True,
                data=df[required],
                source=self.name,
            )
            
        except Exception as e:
            error_str = str(e).lower()
            rate_limited = "429" in error_str or "rate" in error_str or "too many" in error_str
            
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
                rate_limited=rate_limited,
            )
    
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """
        Alpaca doesn't provide fundamentals - return not supported.
        Use yfinance or SEC EDGAR for fundamentals.
        """
        return ProviderResponse(
            success=False,
            data=None,
            source=self.name,
            error="Fundamentals not supported by Alpaca provider",
        )
    
    def get_latest_price(self, ticker: str) -> ProviderResponse:
        """
        Get the latest price/quote for a ticker.
        
        Returns:
            ProviderResponse with dict containing latest price info
        """
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca not available",
            )
        
        # Lazy-init client
        if not self._ensure_client():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca client failed to initialize",
            )
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            self._log_request()
            
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
            quotes = self._client.get_stock_latest_quote(request)
            
            if ticker.upper() not in quotes:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No quote for {ticker}",
                )
            
            quote = quotes[ticker.upper()]
            
            return ProviderResponse(
                success=True,
                data={
                    "bid": quote.bid_price,
                    "ask": quote.ask_price,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                    "mid": (quote.bid_price + quote.ask_price) / 2,
                    "timestamp": quote.timestamp,
                },
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_trades(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        limit: int = 10000,
    ) -> ProviderResponse:
        """
        Get historical trades for a ticker.
        
        Returns:
            ProviderResponse with DataFrame of trades
        """
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca not available",
            )
        
        # Lazy-init client
        if not self._ensure_client():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpaca client failed to initialize",
            )
        
        try:
            from alpaca.data.requests import StockTradesRequest
            
            self._log_request()
            
            request = StockTradesRequest(
                symbol_or_symbols=ticker.upper(),
                start=start,
                end=end,
                limit=limit,
            )
            
            trades = self._client.get_stock_trades(request)
            
            if ticker.upper() not in trades.data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No trades for {ticker}",
                )
            
            trade_list = trades.data[ticker.upper()]
            
            data = []
            for trade in trade_list:
                data.append({
                    "timestamp": trade.timestamp,
                    "price": trade.price,
                    "size": trade.size,
                    "exchange": trade.exchange,
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)
            
            return ProviderResponse(
                success=True,
                data=df,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
