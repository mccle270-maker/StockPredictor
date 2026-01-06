"""
Services Layer - Orchestration
==============================

This layer coordinates core business logic with data access.

Modules:
- prediction.py: predict_next_for_ticker, predict_long_horizon_for_ticker
- backtest.py: track_predictions, walk_forward_backtest, backtest_one_ticker
- signals.py: build_signals_from_pred_df, suggest trading actions
"""

from .prediction import (
    predict_next_for_ticker,
    predict_long_horizon_for_ticker,
    build_features_and_target,
)
from .backtest import (
    track_predictions,
    backtest_one_ticker,
    walk_forward_backtest,
)
from .signals import (
    build_signals_from_pred_df,
    is_us_tradeable_symbol,
    compute_prediction_zscore,
)

# Position sizing exports (from core)
from ..core.position_sizing import (
    scale_signals_by_volatility,
    scale_position_size,
    compute_rolling_volatility,
    get_position_sizing_summary,
    PositionSize,
)

__all__ = [
    "predict_next_for_ticker",
    "predict_long_horizon_for_ticker",
    "build_features_and_target",
    "track_predictions",
    "backtest_one_ticker",
    "walk_forward_backtest",
    "build_signals_from_pred_df",
    "is_us_tradeable_symbol",
    "compute_prediction_zscore",
    # Position sizing
    "scale_signals_by_volatility",
    "scale_position_size",
    "compute_rolling_volatility",
    "get_position_sizing_summary",
    "PositionSize",
]
