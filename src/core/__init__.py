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
]
