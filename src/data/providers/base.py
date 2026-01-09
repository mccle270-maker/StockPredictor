"""
Base provider interface for all data sources.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class ProviderResponse:
    """Standardized response from any data provider."""
    success: bool
    data: Any  # DataFrame, dict, or None
    source: str  # Provider name
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    rate_limited: bool = False
    
    def __bool__(self) -> bool:
        return self.success and self.data is not None


class BaseProvider(ABC):
    """
    Abstract base class for data providers.
    
    All providers must implement:
    - get_price_history()
    - get_fundamentals()
    - is_available() - check if provider can be used (API key set, etc.)
    """
    
    name: str = "base"
    requires_key: bool = False
    rate_limit_per_minute: int = 60
    
    def __init__(self):
        self._request_count = 0
        self._last_request_time = None
        self._rate_limit_reset = None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider can be used (API key valid, etc.)."""
        pass
    
    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> ProviderResponse:
        """
        Fetch price history for a ticker.
        
        Returns:
            ProviderResponse with DataFrame containing OHLCV data
        """
        pass
    
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> ProviderResponse:
        """
        Fetch fundamental data for a ticker.
        
        Returns:
            ProviderResponse with dict containing:
            - fund_pe_trailing
            - fund_pb
            - fund_marketcap
            - (optional) other fundamentals
        """
        pass
    
    def get_news(self, ticker: str, days: int = 7) -> ProviderResponse:
        """
        Fetch news for a ticker (optional, not all providers support).
        
        Returns:
            ProviderResponse with list of news items
        """
        return ProviderResponse(
            success=False,
            data=None,
            source=self.name,
            error="News not supported by this provider",
        )
    
    def get_earnings(self, ticker: str) -> ProviderResponse:
        """
        Fetch earnings data (optional).
        
        Returns:
            ProviderResponse with earnings data
        """
        return ProviderResponse(
            success=False,
            data=None,
            source=self.name,
            error="Earnings not supported by this provider",
        )
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        # Simple implementation - can be enhanced
        return True
    
    def _log_request(self):
        """Log a request for rate limiting."""
        self._request_count += 1
        self._last_request_time = datetime.now()
