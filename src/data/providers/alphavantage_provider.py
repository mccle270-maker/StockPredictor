"""
Alpha Vantage provider - Last resort backup for price and fundamentals.
"""
from __future__ import annotations

import pandas as pd
import requests
from typing import Optional
from datetime import datetime

from .base import BaseProvider, ProviderResponse
from ...config import ALPHAVANTAGE_API_KEY


class AlphaVantageProvider(BaseProvider):
    """
    Alpha Vantage provider as last resort backup.
    
    Pros:
    - Reliable data quality
    - Good fundamental coverage
    - Free tier available
    
    Cons:
    - Very rate limited (5 calls/min free tier)
    - Premium tier expensive
    - Slower than other sources
    """
    
    name = "alphavantage"
    requires_key = True
    rate_limit_per_minute = 5  # Free tier limit
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        super().__init__()
        self.api_key = ALPHAVANTAGE_API_KEY
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """Fetch price history from Alpha Vantage."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpha Vantage API key not configured",
            )
        
        try:
            self._log_request()
            
            # Determine output size based on period
            if period in ("1mo", "3mo", "6mo"):
                outputsize = "compact"  # 100 data points
            else:
                outputsize = "full"  # 20+ years
            
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": outputsize,
                "apikey": self.api_key,
            }
            
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            data = resp.json()
            
            # Check for API errors
            if "Error Message" in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=data["Error Message"][:200],
                )
            
            if "Note" in data:
                # Rate limit warning
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Rate limit reached",
                )
            
            ts_data = data.get("Time Series (Daily)", {})
            if not ts_data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="No time series data returned",
                )
            
            # Convert to DataFrame
            records = []
            for date_str, values in ts_data.items():
                try:
                    records.append({
                        "Date": pd.Timestamp(date_str),
                        "Open": float(values["1. open"]),
                        "High": float(values["2. high"]),
                        "Low": float(values["3. low"]),
                        "Close": float(values["4. close"]),
                        "Adj Close": float(values["5. adjusted close"]),
                        "Volume": int(float(values["6. volume"])),
                    })
                except (KeyError, ValueError):
                    continue
            
            if not records:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Failed to parse time series data",
                )
            
            df = pd.DataFrame(records)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            
            # Filter by period
            df = self._filter_by_period(df, period)
            
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
    
    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter DataFrame by period string."""
        from datetime import timedelta
        
        period_days = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "max": 36500,
        }
        
        days = period_days.get(period, 365)
        cutoff = pd.Timestamp.now() - timedelta(days=days)
        
        if df.index.tz is not None:
            cutoff = cutoff.tz_localize(df.index.tz)
        
        return df[df.index >= cutoff]
    
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """Fetch fundamentals from Alpha Vantage."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpha Vantage API key not configured",
            )
        
        try:
            self._log_request()
            
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key,
            }
            
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            data = resp.json()
            
            if "Error Message" in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=data["Error Message"][:200],
                )
            
            if "Note" in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="Rate limit reached",
                )
            
            if not data or data == {}:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="No fundamental data returned",
                )
            
            # Map Alpha Vantage fields to our standard names
            fundamentals = {
                "fund_pe_trailing": self._safe_float(data.get("TrailingPE")),
                "fund_pe_forward": self._safe_float(data.get("ForwardPE")),
                "fund_pb": self._safe_float(data.get("PriceToBookRatio")),
                "fund_marketcap": self._safe_float(data.get("MarketCapitalization")),
                "fund_revenue": self._safe_float(data.get("RevenueTTM")),
                "fund_gross_profit": self._safe_float(data.get("GrossProfitTTM")),
                "fund_eps_diluted": self._safe_float(data.get("EPS")),
                "fund_dividend_yield": self._safe_float(data.get("DividendYield")),
                "fund_beta": self._safe_float(data.get("Beta")),
                "fund_52_week_high": self._safe_float(data.get("52WeekHigh")),
                "fund_52_week_low": self._safe_float(data.get("52WeekLow")),
                "fund_50_day_ma": self._safe_float(data.get("50DayMovingAverage")),
                "fund_200_day_ma": self._safe_float(data.get("200DayMovingAverage")),
                "fund_shares_outstanding": self._safe_float(data.get("SharesOutstanding")),
                "fund_profit_margin": self._safe_float(data.get("ProfitMargin")),
                "fund_roe": self._safe_float(data.get("ReturnOnEquityTTM")),
                "fund_roa": self._safe_float(data.get("ReturnOnAssetsTTM")),
                "fund_ev_to_revenue": self._safe_float(data.get("EVToRevenue")),
                "fund_ev_to_ebitda": self._safe_float(data.get("EVToEBITDA")),
                "fund_analyst_target": self._safe_float(data.get("AnalystTargetPrice")),
                # Raw fields for reference
                "company_name": data.get("Name"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "exchange": data.get("Exchange"),
            }
            
            # Filter out None values
            fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
            
            return ProviderResponse(
                success=bool(fundamentals),
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
    
    def _safe_float(self, val) -> Optional[float]:
        """Safely convert to float, returning None for invalid values."""
        if val is None or val == "" or val == "None" or val == "-":
            return None
        try:
            result = float(val)
            return result if result != 0 else None  # 0 often means "no data"
        except (ValueError, TypeError):
            return None
    
    def get_quote(self, ticker: str) -> ProviderResponse:
        """Get real-time quote from Alpha Vantage."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Alpha Vantage API key not configured",
            )
        
        try:
            self._log_request()
            
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": self.api_key,
            }
            
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            
            if resp.status_code != 200:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"HTTP {resp.status_code}",
                )
            
            data = resp.json()
            
            if "Error Message" in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=data["Error Message"][:200],
                )
            
            quote_data = data.get("Global Quote", {})
            if not quote_data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="No quote data returned",
                )
            
            quote = {
                "price": self._safe_float(quote_data.get("05. price")),
                "open": self._safe_float(quote_data.get("02. open")),
                "high": self._safe_float(quote_data.get("03. high")),
                "low": self._safe_float(quote_data.get("04. low")),
                "volume": self._safe_float(quote_data.get("06. volume")),
                "previous_close": self._safe_float(quote_data.get("08. previous close")),
                "change": self._safe_float(quote_data.get("09. change")),
                "change_percent": quote_data.get("10. change percent", "").replace("%", ""),
            }
            
            return ProviderResponse(
                success=bool(quote.get("price")),
                data=quote,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
