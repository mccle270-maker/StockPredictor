# Data access layer - external API calls and caching

from .market import get_price_history, get_intraday_history
from .macro import get_macro_df, get_fred_series
from .fundamentals import get_fundamental_features, get_fundamentals
from .news import get_news_for_ticker, get_news_sentiment, detect_big_news
from .options import (
    get_option_chain,
    get_option_snapshot_features,
    get_atm_greeks,
    get_expirations,
)

# New multi-source data pipeline
from .cache_manager import CacheManager, get_cache, cache_get, cache_set
from .aggregator import (
    DataAggregator,
    get_aggregator,
    fetch_prices,
    fetch_fundamentals,
    fetch_sentiment,
    fetch_earnings_warning,
    fetch_earnings_warnings_batch,
)

# Database functions for accuracy tracking
from .database import (
    validate_pending_predictions,
    get_ticker_accuracy,
    get_all_ticker_accuracy,
)

# Provider classes
from .providers import (
    BaseProvider,
    ProviderResponse,
    YFinanceProvider,
    TiingoProvider,
    FinnhubProvider,
    SECEdgarProvider,
    AlphaVantageProvider,
)

__all__ = [
    # Legacy functions
    "get_price_history",
    "get_intraday_history",
    "get_macro_df",
    "get_fred_series",
    "get_fundamental_features",
    "get_fundamentals",
    "get_news_for_ticker",
    "get_news_sentiment",
    "detect_big_news",
    "get_option_chain",
    "get_option_snapshot_features",
    "get_atm_greeks",
    "get_expirations",
    # New multi-source pipeline
    "CacheManager",
    "get_cache",
    "cache_get",
    "cache_set",
    "DataAggregator",
    "get_aggregator",
    "fetch_prices",
    "fetch_fundamentals",
    "fetch_sentiment",
    "fetch_earnings_warning",
    "fetch_earnings_warnings_batch",
    # Database/accuracy tracking
    "validate_pending_predictions",
    "get_ticker_accuracy",
    "get_all_ticker_accuracy",
    # Provider classes
    "BaseProvider",
    "ProviderResponse",
    "YFinanceProvider",
    "TiingoProvider",
    "FinnhubProvider",
    "SECEdgarProvider",
    "AlphaVantageProvider",
]
