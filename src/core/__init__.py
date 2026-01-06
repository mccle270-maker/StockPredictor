# Core module - pure business logic (no I/O, no UI dependencies)

from .features import (
    build_all_features,
    add_returns,
    add_volatility,
    add_gbm_features,
    add_regime_features,
    build_target,
)
from .models import (
    make_model,
    train_model,
    select_features_elasticnet,
    select_features_ols_pvalue,
    get_feature_importance,
)
from .metrics import (
    compute_sharpe,
    compute_sortino,
    compute_drawdown,
    compute_max_drawdown,
    summarize_risk,
    deflated_sharpe_ratio,
)
from .pricing import (
    black_scholes_price,
    black_scholes_greeks,
    suggest_options_strategy,
    normalize_strategy,
    OptionSpec,
    Greeks,
)
from .option_strategy import (
    generate_option_strategy,
    classify_delta_strike,
    determine_strategy,
    get_strategy_summary,
)
from .zscore_filter import (
    ZScoreFilter,
    ZScoreResult,
    get_zscore_filter,
    reset_zscore_filter,
    compute_prediction_zscore,
    evaluate_ticker_zscore,
)
from .trade_limiter import (
    TradeLimiter,
    SkippedSignal,
    TradeCount,
    get_trade_limiter,
    reset_trade_limiter,
    apply_trade_limits,
    rank_signals_by_conviction,
)
from .regime_filter import (
    RegimeFilter,
    MarketRegime,
    RegimeState,
    BlockedTrade,
    get_regime_filter,
    reset_regime_filter,
    apply_regime_filter,
    get_current_regime,
)

__all__ = [
    # Features
    "build_all_features",
    "add_returns",
    "add_volatility",
    "add_gbm_features",
    "add_regime_features",
    "build_target",
    # Models
    "make_model",
    "train_model",
    "select_features_elasticnet",
    "select_features_ols_pvalue",
    "get_feature_importance",
    # Metrics
    "compute_sharpe",
    "compute_sortino",
    "compute_drawdown",
    "compute_max_drawdown",
    "summarize_risk",
    "deflated_sharpe_ratio",
    # Pricing
    "black_scholes_price",
    "black_scholes_greeks",
    "suggest_options_strategy",
    "normalize_strategy",
    "OptionSpec",
    "Greeks",
    # Option Strategy Generator
    "generate_option_strategy",
    "classify_delta_strike",
    "determine_strategy",
    "get_strategy_summary",
    # Z-Score Filtering
    "ZScoreFilter",
    "ZScoreResult",
    "get_zscore_filter",
    "reset_zscore_filter",
    "compute_prediction_zscore",
    "evaluate_ticker_zscore",
    # Trade Limiting
    "TradeLimiter",
    "SkippedSignal",
    "TradeCount",
    "get_trade_limiter",
    "reset_trade_limiter",
    "apply_trade_limits",
    "rank_signals_by_conviction",
    # Regime Filtering
    "RegimeFilter",
    "MarketRegime",
    "RegimeState",
    "BlockedTrade",
    "get_regime_filter",
    "reset_regime_filter",
    "apply_regime_filter",
    "get_current_regime",
]
