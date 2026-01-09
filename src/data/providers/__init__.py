"""
Data providers for multi-source data pipeline.

Each provider implements a consistent interface for fetching:
- Price history
- Fundamentals
- News/sentiment
"""

from .base import BaseProvider, ProviderResponse
from .yfinance_provider import YFinanceProvider
from .tiingo_provider import TiingoProvider
from .finnhub_provider import FinnhubProvider
from .sec_edgar_provider import SECEdgarProvider
from .alphavantage_provider import AlphaVantageProvider

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "YFinanceProvider",
    "TiingoProvider",
    "FinnhubProvider",
    "SECEdgarProvider",
    "AlphaVantageProvider",
]
