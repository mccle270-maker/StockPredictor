"""
yfinance provider - Primary data source.
Free, no API key required.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional
from datetime import datetime

from .base import BaseProvider, ProviderResponse


class YFinanceProvider(BaseProvider):
    """
    yfinance provider for price history and fundamentals.
    
    Pros:
    - Free, no API key
    - Good coverage
    - Options data available
    
    Cons:
    - Rate limited (can get 429 errors)
    - Data can be delayed
    """
    
    name = "yfinance"
    requires_key = False
    rate_limit_per_minute = 30  # Conservative estimate
    
    def __init__(self):
        super().__init__()
        try:
            import yfinance as yf
            self._yf = yf
            self._available = True
        except ImportError:
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """Fetch OHLCV data from yfinance."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="yfinance not installed",
            )
        
        try:
            self._log_request()
            tk = self._yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, timeout=15)
            
            if hist is None or hist.empty:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No data returned for {ticker}",
                )
            
            # Standardize column names
            hist = hist.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })
            
            # Ensure we have the required columns
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in hist.columns:
                    return ProviderResponse(
                        success=False,
                        data=None,
                        source=self.name,
                        error=f"Missing column: {col}",
                    )
            
            return ProviderResponse(
                success=True,
                data=hist[required],
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
        """Fetch fundamentals from yfinance."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="yfinance not installed",
            )
        
        try:
            self._log_request()
            tk = self._yf.Ticker(ticker)
            info = tk.info
            
            if not info:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error=f"No info for {ticker}",
                )
            
            fundamentals = {
                "fund_pe_trailing": info.get("trailingPE"),
                "fund_pb": info.get("priceToBook"),
                "fund_marketcap": info.get("marketCap"),
                "fund_forward_pe": info.get("forwardPE"),
                "fund_dividend_yield": info.get("dividendYield"),
                "fund_profit_margin": info.get("profitMargins"),
                "fund_revenue_growth": info.get("revenueGrowth"),
                "fund_earnings_growth": info.get("earningsGrowth"),
                "fund_debt_to_equity": info.get("debtToEquity"),
                "fund_current_ratio": info.get("currentRatio"),
                "fund_roe": info.get("returnOnEquity"),
                "fund_roa": info.get("returnOnAssets"),
            }
            
            # Check if we got any data
            has_data = any(v is not None for v in fundamentals.values())
            
            return ProviderResponse(
                success=has_data,
                data=fundamentals if has_data else None,
                source=self.name,
                error=None if has_data else "No fundamental data available",
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
    
    def get_options_chain(self, ticker: str) -> ProviderResponse:
        """Fetch options chain from yfinance."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="yfinance not installed",
            )
        
        try:
            self._log_request()
            tk = self._yf.Ticker(ticker)
            
            # Get expiration dates
            expirations = tk.options
            if not expirations:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="No options available",
                )
            
            # Get first expiration chain
            chain = tk.option_chain(expirations[0])
            
            return ProviderResponse(
                success=True,
                data={
                    "expirations": expirations,
                    "calls": chain.calls,
                    "puts": chain.puts,
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
