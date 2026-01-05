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

__all__ = [
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
]
