"""
QuantDesk Dashboard Components

Reusable UI components for the Dash application.
"""

from .sidebar import create_sidebar, TICKER_PRESETS, STRATEGY_OPTIONS, HORIZON_OPTIONS
from .summary_cards import (
    create_ticker_action_card,
    create_portfolio_summary,
    create_summary_tab_content,
    create_empty_state,
    get_signal_color,
    analyze_prediction,
)
from .charts import (
    create_prediction_distribution_chart,
    create_confidence_scatter_chart,
    create_signal_breakdown_chart,
    create_ranked_predictions_chart,
    create_volatility_chart,
    create_price_chart,
    create_backtest_equity_chart,
    create_drawdown_chart,
)
from .tables import (
    create_signals_table,
    create_pnl_table,
    create_options_table,
    create_portfolio_positions_table,
    create_trade_history_table,
)

__all__ = [
    # Sidebar
    "create_sidebar",
    "TICKER_PRESETS",
    "STRATEGY_OPTIONS", 
    "HORIZON_OPTIONS",
    
    # Summary Cards
    "create_ticker_action_card",
    "create_portfolio_summary",
    "create_summary_tab_content",
    "create_empty_state",
    "get_signal_color",
    "analyze_prediction",
    
    # Charts
    "create_prediction_distribution_chart",
    "create_confidence_scatter_chart",
    "create_signal_breakdown_chart",
    "create_ranked_predictions_chart",
    "create_volatility_chart",
    "create_price_chart",
    "create_backtest_equity_chart",
    "create_drawdown_chart",
    
    # Tables
    "create_signals_table",
    "create_pnl_table",
    "create_options_table",
    "create_portfolio_positions_table",
    "create_trade_history_table",
]
