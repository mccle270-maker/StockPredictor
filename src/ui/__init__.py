"""
UI Layer - Streamlit Components
===============================

Clean UI components that delegate to services layer.

Modules:
- components.py: Reusable UI widgets (ticker input, charts, tables)
- pages/: Page-specific layouts
"""

from .components import (
    ticker_input_widget,
    multi_ticker_input_widget,
    model_selector_widget,
    horizon_selector_widget,
    period_selector_widget,
    prediction_display,
    backtest_display,
    risk_metrics_display,
    price_chart,
    options_chain_display,
    signal_display,
    status_indicator,
)

__all__ = [
    "ticker_input_widget",
    "multi_ticker_input_widget",
    "model_selector_widget",
    "horizon_selector_widget",
    "period_selector_widget",
    "prediction_display",
    "backtest_display",
    "risk_metrics_display",
    "price_chart",
    "options_chain_display",
    "signal_display",
    "status_indicator",
]
