"""
Finnhub provider - News sentiment and earnings.
Requires API key: FINNHUB_API_KEY
Free tier: 60 requests/minute
"""
from __future__ import annotations

import pandas as pd
import requests
from typing import Optional
from datetime import datetime, timedelta

from .base import BaseProvider, ProviderResponse


class FinnhubProvider(BaseProvider):
    """
    Finnhub provider for news sentiment and earnings.
    
    Pros:
    - Good news/sentiment data
    - Earnings calendar
    - Free tier 60 req/min
    
    Cons:
    - Limited price history in free tier
    """
    
    name = "finnhub"
    requires_key = True
    rate_limit_per_minute = 60
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            from ...config import FINNHUB_API_KEY
            api_key = FINNHUB_API_KEY
        
        self._api_key = api_key
    
    def is_available(self) -> bool:
        return bool(self._api_key)
    
    def _request(self, endpoint: str, params: dict = None) -> requests.Response:
        """Make authenticated request to Finnhub."""
        if params is None:
            params = {}
        params["token"] = self._api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        return requests.get(url, params=params, timeout=10)
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """Fetch OHLCV data from Finnhub (limited in free tier)."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Finnhub API key not set",
            )
        
        try:
            self._log_request()
            
            # Calculate date range
            end = int(datetime.now().timestamp())
            days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = days_map.get(period, 730)
            start = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # Resolution: 1, 5, 15, 30, 60, D, W, M
            resolution = "D" if interval == "1d" else interval.upper()
            
            resp = self._request("stock/candle", {
                "symbol": ticker,
                "resolution": resolution,
                "from": start,
                "to": end,
            })
            
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
            
            if data.get("s") == "no_data" or "c" not in data:
                return ProviderResponse(
                    success=False,
                    data=None,
                    source=self.name,
                    error="No data available",
                )
            
            # Build DataFrame
            df = pd.DataFrame({
                "Open": data["o"],
                "High": data["h"],
                "Low": data["l"],
                "Close": data["c"],
                "Volume": data["v"],
            }, index=pd.to_datetime(data["t"], unit="s"))
            
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
    
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """Fetch fundamentals from Finnhub."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Finnhub API key not set",
            )
        
        try:
            self._log_request()
            
            # Get company profile
            resp = self._request("stock/profile2", {"symbol": ticker})
            
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
            
            profile = resp.json()
            
            # Get basic financials
            self._log_request()
            resp2 = self._request("stock/metric", {"symbol": ticker, "metric": "all"})
            metrics = resp2.json() if resp2.status_code == 200 else {}
            
            metric_data = metrics.get("metric", {})
            
            fundamentals = {
                "fund_pe_trailing": metric_data.get("peBasicExclExtraTTM"),
                "fund_pb": metric_data.get("pbQuarterly"),
                "fund_marketcap": profile.get("marketCapitalization"),
                "fund_52w_high": metric_data.get("52WeekHigh"),
                "fund_52w_low": metric_data.get("52WeekLow"),
                "fund_beta": metric_data.get("beta"),
                "fund_dividend_yield": metric_data.get("dividendYieldIndicatedAnnual"),
                "fund_eps_ttm": metric_data.get("epsBasicExclExtraItemsTTM"),
                "fund_revenue_per_share": metric_data.get("revenuePerShareTTM"),
                "company_name": profile.get("name"),
                "industry": profile.get("finnhubIndustry"),
                "country": profile.get("country"),
                "ipo_date": profile.get("ipo"),
            }
            
            has_data = fundamentals.get("fund_marketcap") is not None
            
            return ProviderResponse(
                success=has_data,
                data=fundamentals if has_data else None,
                source=self.name,
                error=None if has_data else "No fundamental data",
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_news(self, ticker: str, days: int = 7) -> ProviderResponse:
        """Fetch company news from Finnhub."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Finnhub API key not set",
            )
        
        try:
            self._log_request()
            
            end = datetime.now()
            start = end - timedelta(days=days)
            
            resp = self._request("company-news", {
                "symbol": ticker,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
            })
            
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
            
            # Process news with sentiment
            processed = []
            for item in news:
                processed.append({
                    "datetime": datetime.fromtimestamp(item.get("datetime", 0)),
                    "headline": item.get("headline"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "category": item.get("category"),
                })
            
            return ProviderResponse(
                success=True,
                data=processed,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_earnings(self, ticker: str) -> ProviderResponse:
        """Fetch earnings data from Finnhub."""
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Finnhub API key not set",
            )
        
        try:
            self._log_request()
            
            resp = self._request("stock/earnings", {"symbol": ticker})
            
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
            
            earnings = resp.json()
            
            return ProviderResponse(
                success=True,
                data=earnings,
                source=self.name,
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
    
    def get_sentiment(self, ticker: str) -> ProviderResponse:
        """
        Fetch sentiment data from Finnhub using free endpoints.
        
        Uses:
        - company-news (free) - analyze news volume and recency
        - insider-sentiment (free) - insider trading sentiment
        """
        if not self.is_available():
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error="Finnhub API key not set",
            )
        
        try:
            sentiment_data = {}
            
            # Get company news (free endpoint)
            self._log_request()
            news_resp = self.get_news(ticker, days=7)
            
            if news_resp.success and news_resp.data:
                news = news_resp.data
                sentiment_data["buzz"] = {
                    "articlesInLastWeek": len(news),
                    "buzz": min(len(news) / 10.0, 1.0),  # Normalize 0-1
                }
            else:
                sentiment_data["buzz"] = {"articlesInLastWeek": 0, "buzz": 0}
            
            # Get insider sentiment (free endpoint)
            self._log_request()
            resp = self._request("stock/insider-sentiment", {
                "symbol": ticker,
                "from": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d"),
            })
            
            if resp.status_code == 200:
                insider_data = resp.json()
                data_points = insider_data.get("data", [])
                
                if data_points:
                    # Calculate average MSPR (monthly share purchase ratio)
                    msprs = [d.get("mspr", 0) for d in data_points if d.get("mspr") is not None]
                    changes = [d.get("change", 0) for d in data_points if d.get("change") is not None]
                    
                    avg_mspr = sum(msprs) / len(msprs) if msprs else 0
                    total_change = sum(changes) if changes else 0
                    
                    # MSPR > 0 means more buying than selling
                    bullish_pct = max(0, min(1, (avg_mspr + 100) / 200)) * 100
                    
                    sentiment_data["sentiment"] = {
                        "bullishPercent": bullish_pct,
                        "bearishPercent": 100 - bullish_pct,
                    }
                    sentiment_data["insiderSentiment"] = {
                        "mspr": avg_mspr,
                        "change": total_change,
                        "dataPoints": len(data_points),
                    }
                else:
                    sentiment_data["sentiment"] = {
                        "bullishPercent": 50,
                        "bearishPercent": 50,
                    }
            
            # Get recommendation trends (free endpoint)
            self._log_request()
            rec_resp = self._request("stock/recommendation", {"symbol": ticker})
            
            if rec_resp.status_code == 200:
                recommendations = rec_resp.json()
                if recommendations:
                    latest = recommendations[0]
                    total = (latest.get("buy", 0) + latest.get("hold", 0) + 
                            latest.get("sell", 0) + latest.get("strongBuy", 0) + 
                            latest.get("strongSell", 0))
                    
                    if total > 0:
                        bullish = (latest.get("buy", 0) + latest.get("strongBuy", 0)) / total
                        sentiment_data["analystRecommendations"] = {
                            "period": latest.get("period"),
                            "strongBuy": latest.get("strongBuy", 0),
                            "buy": latest.get("buy", 0),
                            "hold": latest.get("hold", 0),
                            "sell": latest.get("sell", 0),
                            "strongSell": latest.get("strongSell", 0),
                            "bullishPercent": bullish * 100,
                        }
            
            has_data = bool(sentiment_data.get("buzz") or sentiment_data.get("sentiment"))
            
            return ProviderResponse(
                success=has_data,
                data=sentiment_data if has_data else None,
                source=self.name,
                error=None if has_data else "No sentiment data available",
            )
            
        except Exception as e:
            return ProviderResponse(
                success=False,
                data=None,
                source=self.name,
                error=str(e)[:200],
            )
