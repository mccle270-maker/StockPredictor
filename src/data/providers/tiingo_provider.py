"""
Tiingo provider - High quality price data and fundamentals.
Requires API key: TIINGO_API_KEY
Free tier: 500 requests/hour
"""
from __future__ import annotations

import pandas as pd
import requests
from typing import Optional
from datetime import datetime, timedelta

from .base import BaseProvider, ProviderResponse


class TiingoProvider(BaseProvider):
    """
    Tiingo provider for price history and fundamentals.
    
    Pros:
    - High quality data
    - Good historical coverage
    - Free tier generous (500 req/hr)
    
    Cons:
    - Requires API key
    """
    
    name = "tiingo"
    requires_key = True
    rate_limit_per_minute = 8  # 500/hr ≈ 8/min
    
    BASE_URL = "https://api.tiingo.com"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            from ...config import TIINGO_API_KEY
            api_key = TIINGO_API_KEY
        
        self._api_key = api_key
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {api_key}" if api_key else "",
        }
    
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    def _period_to_dates(self, period: str) -> tuple:
        """Convert period string to start/end dates."""
        end = datetime.now()
        days_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
        }
        days = days_map.get(period, 730)
        start = end - timedelta(days=days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """Fetch OHLCV data from Tiingo."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Tiingo API key not set",
            )
        
        try:
            self._log_request()
            
            start_date, end_date = self._period_to_dates(period)
            
            url = f"{self.BASE_URL}/tiingo/daily/{ticker}/prices"
            params = {
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": "daily" if interval == "1d" else interval,
            }
            
            resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            
            if resp.status_code == 429:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Rate limited",
                    rate_limited=True,
                )
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}: {resp.text[:100]}",
                )
            
            data = resp.json()
            
            if not data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No data returned for {ticker}",
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            
            # Rename columns to standard format
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "adjOpen": "Adj Open",
                "adjHigh": "Adj High",
                "adjLow": "Adj Low",
                "adjClose": "Adj Close",
                "adjVolume": "Adj Volume",
            })
            
            required = ["Open", "High", "Low", "Close", "Volume"]
            available = [c for c in required if c in df.columns]
            
            if len(available) < 4:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Missing required columns",
                )
            
            return ProviderResponse(
                success=True,
                data=df[available],
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """Fetch fundamentals from Tiingo."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Tiingo API key not set",
            )
        
        try:
            self._log_request()
            
            # Get meta info
            url = f"{self.BASE_URL}/tiingo/daily/{ticker}"
            resp = requests.get(url, headers=self._headers, timeout=10)
            
            if resp.status_code == 429:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Rate limited",
                    rate_limited=True,
                )
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            data = resp.json()
            
            # Tiingo provides limited fundamentals in basic tier
            # Extract what we can
            fundamentals = {
                "fund_pe_trailing": None,  # Not in basic tier
                "fund_pb": None,
                "fund_marketcap": None,
                "description": data.get("description"),
                "exchange": data.get("exchangeCode"),
                "start_date": data.get("startDate"),
                "end_date": data.get("endDate"),
            }
            
            return ProviderResponse(
                success=True,
                data=fundamentals,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_news(self, ticker: str, days: int = 7) -> ProviderResponse:
        """Fetch news from Tiingo."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Tiingo API key not set",
            )
        
        try:
            self._log_request()
            
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            url = f"{self.BASE_URL}/tiingo/news"
            params = {
                "tickers": ticker,
                "startDate": start_date,
                "limit": 50,
            }
            
            resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            
            if resp.status_code == 429:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Rate limited",
                    rate_limited=True,
                )
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            news = resp.json()
            
            return ProviderResponse(
                success=True,
                data=news,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
