"""
Data Aggregator - Combines multiple data sources with fallback chains.
"""
from __future__ import annotations

import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .cache_manager import CacheManager, get_cache, CacheConfig
from .providers.base import ProviderResponse
from .providers.yfinance_provider import YFinanceProvider
from .providers.tiingo_provider import TiingoProvider
from .providers.finnhub_provider import FinnhubProvider
from .providers.sec_edgar_provider import SECEdgarProvider
from .providers.alphavantage_provider import AlphaVantageProvider


logger = logging.getLogger(__name__)


@dataclass
class AggregatorConfig:
    """Configuration for data aggregator."""
    # Provider priority for price data
    price_providers: List[str] = field(default_factory=lambda: [
        "yfinance",     # Primary - free, fast, reliable
        "tiingo",       # Backup - good data quality
        "alphavantage", # Last resort - heavily rate limited
    ])
    
    # Provider priority for fundamentals
    fundamentals_providers: List[str] = field(default_factory=lambda: [
        "yfinance",     # Primary - quick fundamentals
        "tiingo",       # Backup - detailed fundamentals
        "sec_edgar",    # Official SEC data
        "alphavantage", # Last resort
    ])
    
    # Provider priority for sentiment
    sentiment_providers: List[str] = field(default_factory=lambda: [
        "finnhub",      # News sentiment
    ])
    
    # Enable caching
    use_cache: bool = True
    
    # Fail over to next provider on any error
    failover_on_error: bool = True


class DataAggregator:
    """
    Multi-source data aggregator with fallback chains.
    
    Features:
    - Automatic failover between providers
    - Aggressive caching
    - Unified output format
    - Health monitoring
    """
    
    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
        cache: Optional[CacheManager] = None,
    ):
        self.config = config or AggregatorConfig()
        self.cache = cache or get_cache()
        
        # Initialize all providers
        self._providers = {
            "yfinance": YFinanceProvider(),
            "tiingo": TiingoProvider(),
            "finnhub": FinnhubProvider(),
            "sec_edgar": SECEdgarProvider(),
            "alphavantage": AlphaVantageProvider(),
        }
        
        # Track provider health
        self._provider_stats: Dict[str, Dict[str, int]] = {
            name: {"success": 0, "failure": 0}
            for name in self._providers
        }
    
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get price history with automatic fallback.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data or None if all sources fail
        """
        # Check cache first
        if use_cache and self.config.use_cache:
            cached = self.cache.get("price", ticker, period=period, interval=interval)
            if cached is not None:
                logger.debug(f"Cache hit for {ticker} price history")
                return cached
        
        # Try providers in order
        for provider_name in self.config.price_providers:
            provider = self._providers.get(provider_name)
            if not provider or not provider.is_available():
                continue
            
            try:
                response = provider.get_price_history(ticker, period=period, interval=interval)
                
                if response.success and response.data is not None:
                    df = response.data
                    
                    # Normalize column names
                    df = self._normalize_price_columns(df)
                    
                    # Cache successful result
                    if use_cache and self.config.use_cache:
                        self.cache.set("price", ticker, df, period=period, interval=interval)
                    
                    self._provider_stats[provider_name]["success"] += 1
                    logger.info(f"Got {ticker} prices from {provider_name}")
                    return df
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {ticker}: {e}")
                self._provider_stats[provider_name]["failure"] += 1
                
                if not self.config.failover_on_error:
                    break
        
        logger.error(f"All providers failed for {ticker} price history")
        return None
    
    def get_fundamentals(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get fundamentals with automatic fallback and merging.
        
        Multiple sources are combined to get the most complete data.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with fundamental data
        """
        # Check cache first
        if use_cache and self.config.use_cache:
            cached = self.cache.get("fundamentals", ticker)
            if cached is not None:
                logger.debug(f"Cache hit for {ticker} fundamentals")
                return cached
        
        combined = {}
        sources_used = []
        
        # Try providers in order, merging results
        for provider_name in self.config.fundamentals_providers:
            provider = self._providers.get(provider_name)
            if not provider or not provider.is_available():
                continue
            
            try:
                response = provider.get_fundamentals(ticker)
                
                if response.success and response.data:
                    # Merge new data (don't overwrite existing)
                    for key, value in response.data.items():
                        if key not in combined or combined[key] is None:
                            combined[key] = value
                    
                    sources_used.append(provider_name)
                    self._provider_stats[provider_name]["success"] += 1
                    
                    # If we have the key metrics, we can stop
                    if self._has_key_fundamentals(combined):
                        break
                        
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed for {ticker} fundamentals: {e}")
                self._provider_stats[provider_name]["failure"] += 1
        
        if combined:
            combined["_sources"] = sources_used
            combined["_timestamp"] = datetime.now().isoformat()
            
            # Cache successful result
            if use_cache and self.config.use_cache:
                self.cache.set("fundamentals", ticker, combined)
            
            logger.info(f"Got {ticker} fundamentals from {sources_used}")
        else:
            logger.error(f"All providers failed for {ticker} fundamentals")
        
        return combined
    
    def get_sentiment(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get sentiment data (news, social, etc.)."""
        # Check cache first
        if use_cache and self.config.use_cache:
            cached = self.cache.get("sentiment", ticker)
            if cached is not None:
                return cached
        
        for provider_name in self.config.sentiment_providers:
            provider = self._providers.get(provider_name)
            if not provider or not provider.is_available():
                continue
            
            try:
                # Finnhub has get_sentiment method
                if hasattr(provider, "get_sentiment"):
                    response = provider.get_sentiment(ticker)
                    
                    if response.success and response.data:
                        result = response.data
                        result["_source"] = provider_name
                        result["_timestamp"] = datetime.now().isoformat()
                        
                        if use_cache and self.config.use_cache:
                            self.cache.set("sentiment", ticker, result)
                        
                        return result
                        
            except Exception as e:
                logger.warning(f"Sentiment provider {provider_name} failed: {e}")
        
        return {}
    
    def _normalize_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize price DataFrame columns to standard names."""
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj Close",
            "adjclose": "Adj Close",
            "adjusted_close": "Adj Close",
        }
        
        # Lowercase all columns first
        df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
        
        # Rename to standard names
        new_columns = {}
        for col in df.columns:
            col_lower = col.lower() if isinstance(col, str) else col
            if col_lower in column_mapping:
                new_columns[col] = column_mapping[col_lower]
        
        if new_columns:
            df = df.rename(columns=new_columns)
        
        return df
    
    def _has_key_fundamentals(self, data: Dict) -> bool:
        """Check if we have the key fundamental metrics."""
        key_metrics = ["fund_pe_trailing", "fund_pb", "fund_marketcap"]
        return sum(1 for m in key_metrics if data.get(m) is not None) >= 2
    
    def get_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers."""
        health = {}
        
        for name, provider in self._providers.items():
            stats = self._provider_stats[name]
            total = stats["success"] + stats["failure"]
            
            health[name] = {
                "available": provider.is_available(),
                "success_count": stats["success"],
                "failure_count": stats["failure"],
                "success_rate": stats["success"] / total if total > 0 else None,
                "requires_key": provider.requires_key,
            }
        
        return health
    
    def check_all_providers(self) -> Dict[str, bool]:
        """Quick health check on all providers."""
        results = {}
        
        for name, provider in self._providers.items():
            try:
                response = provider.check_health()
                results[name] = response.success
            except Exception:
                results[name] = False
        
        return results


# Global aggregator instance
_global_aggregator: Optional[DataAggregator] = None


def get_aggregator() -> DataAggregator:
    """Get or create the global aggregator instance."""
    global _global_aggregator
    if _global_aggregator is None:
        _global_aggregator = DataAggregator()
    return _global_aggregator


# Convenience functions
def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch price history using the global aggregator."""
    return get_aggregator().get_price_history(ticker, period=period, interval=interval)


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """Fetch fundamentals using the global aggregator."""
    return get_aggregator().get_fundamentals(ticker)


def fetch_sentiment(ticker: str) -> Dict[str, Any]:
    """Fetch sentiment using the global aggregator."""
    return get_aggregator().get_sentiment(ticker)
