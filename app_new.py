"""
Stock Predictor - Clean Streamlit App
=====================================

A full-featured UI that matches the original app.py structure,
using the refactored service/core layers for cleaner architecture.

Run with: streamlit run app_new.py
"""

import os
import sys
import json
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# Add src to path for imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# Import from refactored modules
from src.config import (
    UNIVERSE_PRESETS,
    FRICTION_PRESETS,
    OPTIONS_PRESETS,
    WALKFORWARD_PRESETS,
    ExecutionModel,
    OptionsConfig,
    SIGNALS_PATH,
)
from src.core.pricing import PricingModel
from src.services.prediction import (
    predict_next_for_ticker,
    predict_long_horizon_for_ticker,
)
from src.services.backtest import (
    track_predictions,
    backtest_one_ticker,
    walk_forward_backtest,
)
from src.services.signals import (
    build_signals_from_pred_df,
    is_us_tradeable_symbol,
    suggest_options_strategy,
)
from src.data.market import get_price_history
from src.data.options import get_option_chain, get_option_snapshot_features, get_atm_greeks
from src.data.news import get_news_for_ticker, detect_big_news
from src.core.metrics import (
    compute_sharpe,
    compute_drawdown,
    summarize_risk,
    prepare_risk_timeseries,
)
from src.core.option_strategy import generate_option_strategy, get_strategy_summary
from src.ui.components import (
    ticker_input_widget,
    multi_ticker_input_widget,
    model_selector_widget,
    horizon_selector_widget,
    period_selector_widget,
    prediction_display,
    backtest_display,
    risk_metrics_display,
    price_chart,
    signal_display,
)

# Optional: Database and Live Updates (graceful degradation if not available)
try:
    from src.data.database import (
        save_prediction,
        save_predictions_batch,
        get_prediction_history,
        save_backtest_result,
        get_model_performance_history,
        save_query,
        get_recent_queries,
        toggle_favorite_query,
        delete_query,
        get_accuracy_stats,
        get_database_stats,
        get_best_model_for_ticker,
    )
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False

try:
    from src.ui.live_updates import (
        init_live_state,
        is_market_hours,
        get_live_price,
        get_live_prices_batch,
        render_live_status_badge,
        render_live_controls,
        should_refresh,
        mark_refreshed,
        get_time_until_next_refresh,
        create_live_price_chart,
        auto_refresh_check,
        get_live_config,
        set_live_config,
    )
    HAS_LIVE_UPDATES = True
except ImportError:
    HAS_LIVE_UPDATES = False

# Paths
TRADER_PATH = BASE_DIR / "auto_paper_trade.py"
SIGNALS_OUT_PATH = SIGNALS_PATH

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _fmt_pct(val, decimals=2):
    """Format a value as percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val) * 100:.{decimals}f}%"

def _fmt_num(val, decimals=2):
    """Format a number."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.{decimals}f}"

def _parse_int(s, default):
    try:
        return int(s)
    except:
        return default

def _parse_float(s, default):
    try:
        return float(s)
    except:
        return default

def apply_latency_delay(df: pd.DataFrame, delay_days: int, pred_col: str = "predicted_return") -> pd.DataFrame:
    """Shift predictions by delay_days to simulate execution latency."""
    out = df.copy()
    if delay_days and delay_days > 0 and pred_col in out.columns:
        out[pred_col] = out[pred_col].shift(delay_days)
    return out

def apply_costs_on_trades(
    strat_df: pd.DataFrame,
    exec_model: ExecutionModel,
    actual_ret_col: str = "actual_return",
    pos_col: str = "position",
) -> pd.Series:
    """Apply trading costs based on position changes."""
    pos = strat_df.get(pos_col, pd.Series(index=strat_df.index, data=0.0)).fillna(0.0)
    pos_change = pos.diff().abs().fillna(0.0)
    base = strat_df.get(actual_ret_col, pd.Series(index=strat_df.index, data=0.0)).fillna(0.0) * pos
    per_trade_cost = (exec_model.half_spread_bps + exec_model.slippage_bps + exec_model.fee_bps) / 10000.0
    costs = per_trade_cost * pos_change
    return base - costs

def get_card_theme():
    """Get theme colors for inline HTML cards."""
    is_dark = st.session_state.get("theme", "dark") == "dark"
    if is_dark:
        return {
            "bg": "#161b22",
            "border": "#30363d",
            "label": "#8b949e",
            "text": "#f0f6fc",
            "muted": "#6e7681",
            "bar_bg": "#21262d",
            "green": "#3fb950",
            "red": "#f85149",
            "yellow": "#d29922",
            "blue": "#388bfd",
            "purple": "#a371f7",
        }
    else:
        return {
            "bg": "#ffffff",
            "border": "#d0d7de",
            "label": "#57606a",
            "text": "#1f2328",
            "muted": "#6e7781",
            "bar_bg": "#e5e7eb",
            "green": "#1a7f37",
            "red": "#cf222e",
            "yellow": "#9a6700",
            "blue": "#0969da",
            "purple": "#8250df",
        }

# ============================================================================
# INPUT VALIDATION & ERROR HANDLING
# ============================================================================

import re
from functools import wraps
from typing import List, Tuple, Optional, Any, Callable

# Valid ticker pattern: 1-5 uppercase letters, optionally followed by . and 1-2 letters (e.g., BRK.B)
TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$')

# Valid period values
VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"]

# Valid model types
VALID_MODELS = ["rf", "xgb", "gbrt", "linreg"]


class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


class APIError(Exception):
    """Custom exception for API/service errors."""
    pass


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate a single stock ticker symbol.
    
    Args:
        ticker: The ticker symbol to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not ticker:
        return False, "Ticker symbol cannot be empty"
    
    # Clean and uppercase
    ticker_clean = ticker.strip().upper()
    
    if len(ticker_clean) == 0:
        return False, "Ticker symbol cannot be empty or whitespace only"
    
    if len(ticker_clean) > 6:
        return False, f"Ticker '{ticker_clean}' is too long (max 6 characters)"
    
    if not TICKER_PATTERN.match(ticker_clean):
        return False, f"Invalid ticker format '{ticker_clean}'. Use 1-5 letters (e.g., AAPL, MSFT, BRK.B)"
    
    # Check for obviously invalid tickers
    invalid_tickers = {"TEST", "FAKE", "NULL", "NONE", "NA", "N/A"}
    if ticker_clean in invalid_tickers:
        return False, f"'{ticker_clean}' is not a valid stock ticker"
    
    return True, ""


def validate_tickers(tickers_input: str) -> Tuple[List[str], List[str]]:
    """
    Validate a comma-separated list of ticker symbols.
    
    Args:
        tickers_input: Comma-separated string of tickers
        
    Returns:
        Tuple of (valid_tickers, error_messages)
    """
    if not tickers_input or not tickers_input.strip():
        return [], ["No tickers provided. Please enter at least one ticker symbol."]
    
    # Split by comma, semicolon, space, or newline
    raw_tickers = re.split(r'[,;\s\n]+', tickers_input.strip())
    raw_tickers = [t.strip().upper() for t in raw_tickers if t.strip()]
    
    if not raw_tickers:
        return [], ["No valid tickers found in input."]
    
    valid_tickers = []
    errors = []
    seen = set()
    
    for ticker in raw_tickers:
        # Skip duplicates
        if ticker in seen:
            continue
        seen.add(ticker)
        
        is_valid, error = validate_ticker(ticker)
        if is_valid:
            valid_tickers.append(ticker)
        else:
            errors.append(error)
    
    # Limit number of tickers
    MAX_TICKERS = 50
    if len(valid_tickers) > MAX_TICKERS:
        errors.append(f"Too many tickers ({len(valid_tickers)}). Maximum allowed is {MAX_TICKERS}.")
        valid_tickers = valid_tickers[:MAX_TICKERS]
    
    return valid_tickers, errors


def validate_period(period: str) -> Tuple[bool, str]:
    """
    Validate a time period string.
    
    Args:
        period: Period string (e.g., "1y", "5y", "max")
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not period:
        return False, "Period cannot be empty"
    
    period_clean = period.strip().lower()
    
    if period_clean not in VALID_PERIODS:
        return False, f"Invalid period '{period}'. Valid options: {', '.join(VALID_PERIODS)}"
    
    return True, ""


def validate_horizon(horizon: int) -> Tuple[bool, str]:
    """
    Validate prediction horizon (days).
    
    Args:
        horizon: Number of days for prediction horizon
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(horizon, int):
        try:
            horizon = int(horizon)
        except (ValueError, TypeError):
            return False, f"Horizon must be a number, got '{horizon}'"
    
    if horizon < 1:
        return False, "Horizon must be at least 1 day"
    
    if horizon > 252:
        return False, "Horizon cannot exceed 252 trading days (1 year)"
    
    return True, ""


def validate_model_type(model_type: str) -> Tuple[bool, str]:
    """
    Validate model type selection.
    
    Args:
        model_type: Model type string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_type:
        return False, "Model type cannot be empty"
    
    model_clean = model_type.strip().lower()
    
    if model_clean not in VALID_MODELS:
        return False, f"Invalid model type '{model_type}'. Valid options: {', '.join(VALID_MODELS)}"
    
    return True, ""


def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> Tuple[bool, str]:
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: The value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter (for error messages)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        val = float(value)
    except (ValueError, TypeError):
        return False, f"{name} must be a number, got '{value}'"
    
    if val < min_val:
        return False, f"{name} must be at least {min_val}, got {val}"
    
    if val > max_val:
        return False, f"{name} must be at most {max_val}, got {val}"
    
    return True, ""


def safe_api_call(func: Callable, *args, error_prefix: str = "Error", **kwargs) -> Tuple[Any, Optional[str]]:
    """
    Safely execute an API or service call with error handling.
    
    Args:
        func: The function to call
        *args: Positional arguments for the function
        error_prefix: Prefix for error messages
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (result, error_message) where error_message is None on success
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except ConnectionError as e:
        return None, f"{error_prefix}: Network connection failed. Please check your internet connection."
    except TimeoutError as e:
        return None, f"{error_prefix}: Request timed out. Please try again."
    except ValueError as e:
        return None, f"{error_prefix}: Invalid data received - {str(e)}"
    except KeyError as e:
        return None, f"{error_prefix}: Missing required data - {str(e)}"
    except FileNotFoundError as e:
        return None, f"{error_prefix}: Required file not found - {str(e)}"
    except PermissionError as e:
        return None, f"{error_prefix}: Permission denied - {str(e)}"
    except Exception as e:
        # Log the full error for debugging
        error_type = type(e).__name__
        error_msg = str(e)
        # Truncate very long error messages
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return None, f"{error_prefix}: {error_type} - {error_msg}"


def display_validation_errors(errors: List[str], container=None):
    """
    Display validation errors in a user-friendly format.
    
    Args:
        errors: List of error messages
        container: Optional streamlit container to display in
    """
    if not errors:
        return
    
    target = container if container else st
    
    if len(errors) == 1:
        target.error(f"⚠️ {errors[0]}")
    else:
        error_html = "<div style='color: #f85149;'><strong>⚠️ Please fix the following issues:</strong><ul>"
        for err in errors[:5]:  # Limit to 5 errors
            error_html += f"<li>{err}</li>"
        if len(errors) > 5:
            error_html += f"<li>...and {len(errors) - 5} more errors</li>"
        error_html += "</ul></div>"
        target.markdown(error_html, unsafe_allow_html=True)


def display_api_error(error_msg: str, ticker: str = None, container=None):
    """
    Display an API error with helpful context.
    
    Args:
        error_msg: The error message
        ticker: Optional ticker that caused the error
        container: Optional streamlit container to display in
    """
    target = container if container else st
    
    tc = get_card_theme()
    ticker_info = f" for <strong>{ticker}</strong>" if ticker else ""
    
    target.markdown(f"""
    <div style='
        background: {tc["red"]}15;
        border: 1px solid {tc["red"]};
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
    '>
        <div style='color: {tc["red"]}; font-weight: 600; margin-bottom: 0.5rem;'>
            ❌ Error{ticker_info}
        </div>
        <div style='color: {tc["text"]}; font-size: 0.9rem;'>
            {error_msg}
        </div>
        <div style='color: {tc["muted"]}; font-size: 0.8rem; margin-top: 0.5rem;'>
            💡 Try: Check ticker spelling, verify internet connection, or try a different time period.
        </div>
    </div>
    """, unsafe_allow_html=True)


def validate_prediction_inputs(tickers: List[str], period: str, horizon: int, model_type: str) -> Tuple[bool, List[str]]:
    """
    Validate all inputs for running predictions.
    
    Args:
        tickers: List of ticker symbols
        period: Time period string
        horizon: Prediction horizon in days
        model_type: Model type string
        
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    errors = []
    
    # Validate tickers
    if not tickers:
        errors.append("No tickers provided. Please enter at least one ticker symbol.")
    
    # Validate period
    is_valid, err = validate_period(period)
    if not is_valid:
        errors.append(err)
    
    # Validate horizon
    is_valid, err = validate_horizon(horizon)
    if not is_valid:
        errors.append(err)
    
    # Validate model type
    is_valid, err = validate_model_type(model_type)
    if not is_valid:
        errors.append(err)
    
    return len(errors) == 0, errors


def validate_backtest_inputs(ticker: str, period: str, horizon: int, model_type: str) -> Tuple[bool, List[str]]:
    """
    Validate all inputs for running a backtest.
    
    Args:
        ticker: Single ticker symbol
        period: Time period string
        horizon: Prediction horizon in days
        model_type: Model type string
        
    Returns:
        Tuple of (all_valid, list_of_errors)
    """
    errors = []
    
    # Validate ticker
    is_valid, err = validate_ticker(ticker)
    if not is_valid:
        errors.append(err)
    
    # Validate period
    is_valid, err = validate_period(period)
    if not is_valid:
        errors.append(err)
    
    # Validate horizon
    is_valid, err = validate_horizon(horizon)
    if not is_valid:
        errors.append(err)
    
    # Validate model type
    is_valid, err = validate_model_type(model_type)
    if not is_valid:
        errors.append(err)
    
    return len(errors) == 0, errors


# =============================================================================
# CACHING LAYER - Performance Optimization
# =============================================================================
# TTL Guidelines:
#   - Intraday data: 30 seconds
#   - Price history: 10 minutes (600s)
#   - Predictions: 15 minutes (900s)
#   - Backtests/Walk-forward: 30 minutes (1800s)
#   - Options data: 5 minutes (300s)
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def _cached_price_history(ticker: str, period: str):
    """
    Cached version of get_price_history.
    TTL: 10 minutes - price data doesn't change frequently.
    """
    return get_price_history(ticker, period=period)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_prediction(ticker: str, period: str, model_type: str, horizon: int, 
                       run_gaf: bool = False, auto_optimize: bool = True,
                       use_elasticnet: bool = False):
    """
    Cached version of predict_next_for_ticker.
    TTL: 15 minutes - predictions are expensive and don't need constant refresh.
    
    Note: auto_optimize and use_elasticnet are included in cache key so different
    settings produce different cached results.
    """
    import os
    # Set environment variable for Elastic Net feature selection
    if use_elasticnet:
        os.environ["USE_ELASTICNET_SELECT"] = "1"
    else:
        os.environ.pop("USE_ELASTICNET_SELECT", None)
    
    return predict_next_for_ticker(
        ticker=ticker,
        period=period,
        model_type=model_type,  # Use selected model type
        horizon=horizon,
        run_gaf=run_gaf,
        auto_optimize=auto_optimize  # Pass auto_optimize to the function
    )


@st.cache_data(ttl=300, show_spinner=False)
def _cached_option_chain(ticker: str):
    """
    Cached version of get_option_chain.
    TTL: 5 minutes - options data is more volatile.
    """
    return get_option_chain(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_option_snapshot(ticker: str):
    """
    Cached version of get_option_snapshot_features.
    TTL: 5 minutes - options snapshot data.
    """
    return get_option_snapshot_features(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_atm_greeks(ticker: str):
    """
    Cached version of get_atm_greeks.
    TTL: 5 minutes - ATM greeks calculation.
    """
    return get_atm_greeks(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_track_predictions(ticker, period, model_type, horizon):
    """
    Cached version of track_predictions.
    TTL: 5 minutes - tracking data.
    """
    return track_predictions(ticker, period=period, model_type=model_type, horizon=horizon)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_long_horizon_prediction(ticker, period):
    """
    Cached version of long horizon prediction.
    TTL: 15 minutes - same as regular predictions.
    """
    return predict_long_horizon_for_ticker(ticker, period=period)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_walk_forward_backtest(ticker: str, period: str, horizon: int, model_type: str, 
                                   train_years: float, test_years: float, step_days: int = 21):
    """
    Cached version of walk_forward_backtest.
    TTL: 30 minutes - backtests are very expensive and results don't change.
    """
    return walk_forward_backtest(
        ticker=ticker,
        period=period,
        horizon=horizon,
        model_type=model_type,
        train_years=train_years,
        test_years=test_years,
        step_days=step_days
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_backtest_one_ticker(ticker: str, period: str, model_type: str, horizon: int = 1):
    """
    Cached version of backtest_one_ticker.
    TTL: 30 minutes - backtests are expensive.
    """
    from src.services.backtest import backtest_one_ticker
    return backtest_one_ticker(ticker, period=period, model_type=model_type, horizon=horizon)


def _cached_build_signals(pred_df_json: str, prediction_horizon: int, trade_mode: str, 
                          prefer_spreads: bool, dte_min: int, dte_max: int,
                          max_strike: float, max_premium: float, width_pct: float):
    """
    Build signals from prediction DataFrame.
    Note: Not cached directly because pred_df changes frequently.
    Instead, this provides a consistent interface for signal building.
    """
    pred_df = pd.read_json(pred_df_json)
    # Create execution model with defaults
    from dataclasses import dataclass
    
    @dataclass
    class ExecutionModel:
        delay: int = 1
        spread: float = 0.0002
        slippage: float = 0.0003
        fees: float = 0.0
    
    exec_model = ExecutionModel()
    
    return build_signals_from_pred_df(
        pred_df,
        prediction_horizon=prediction_horizon,
        trade_mode=trade_mode,
        prefer_spreads=prefer_spreads,
        dte_min=dte_min,
        dte_max=dte_max,
        max_strike=max_strike,
        max_premium=max_premium,
        width_pct=width_pct,
        exec_model=exec_model,
    )


def clear_prediction_cache():
    """Clear prediction-related caches for fresh data."""
    _cached_prediction.clear()
    _cached_long_horizon_prediction.clear()
    _cached_track_predictions.clear()


def clear_price_cache():
    """Clear price history cache."""
    _cached_price_history.clear()


def clear_options_cache():
    """Clear options-related caches."""
    _cached_option_chain.clear()
    _cached_option_snapshot.clear()
    _cached_atm_greeks.clear()


def clear_backtest_cache():
    """Clear backtest caches."""
    _cached_walk_forward_backtest.clear()
    _cached_backtest_one_ticker.clear()


def clear_all_caches():
    """Clear all cached data for completely fresh state."""
    clear_prediction_cache()
    clear_price_cache()
    clear_options_cache()
    clear_backtest_cache()


def render_styled_table(df, table_id="styled-table", highlight_cols=None, theme=None):
    """
    Render a DataFrame as a professionally styled HTML table.
    
    Args:
        df: DataFrame to render
        table_id: Unique ID for the table (for CSS scoping)
        highlight_cols: Dict mapping column names to highlight rules
                       e.g., {"PRED %": {"positive": "> 0", "negative": "< 0"}}
        theme: Theme dict (from get_theme_colors()) or None for auto-detect
    """
    if highlight_cols is None:
        highlight_cols = {}
    
    # Get theme colors
    if theme is None:
        is_dark = st.session_state.get("theme", "dark") == "dark"
        if is_dark:
            t = {
                "border": "#30363d", "header_bg": "#161b22", "accent": "#388bfd",
                "text": "#f0f6fc", "row_even": "#0d1117", "row_odd": "#0f141a",
                "hover": "#21262d", "row_border": "#21262d",
                "green": "#3fb950", "red": "#f85149"
            }
        else:
            t = {
                "border": "#d0d7de", "header_bg": "#f6f8fa", "accent": "#0969da",
                "text": "#1f2328", "row_even": "#ffffff", "row_odd": "#f9fafb",
                "hover": "#e5e7eb", "row_border": "#d8dee4",
                "green": "#1a7f37", "red": "#cf222e"
            }
    else:
        t = theme
    
    html = f"""
    <div style="overflow-x: auto; border: 1px solid {t['border']}; border-radius: 8px; margin: 1rem 0;">
    <table id="{table_id}" style="
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-size: 0.85rem;
    ">
        <thead>
            <tr style="background: {t['header_bg']}; border-bottom: 2px solid {t['accent']};">
    """
    
    # Headers
    for col in df.columns:
        html += f"""<th style="
            color: {t['accent']};
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 12px 16px;
            text-align: left;
            text-transform: uppercase;
            font-size: 0.7rem;
            white-space: nowrap;
        ">{col}</th>"""
    
    html += "</tr></thead><tbody>"
    
    # Rows
    for idx, row in df.iterrows():
        row_bg = t['row_even'] if idx % 2 == 0 else t['row_odd']
        html += f'<tr style="background: {row_bg};" onmouseover="this.style.background=\'{t["hover"]}\'" onmouseout="this.style.background=\'{row_bg}\'">'
        
        for col in df.columns:
            val = row[col]
            cell_style = f"color: {t['text']}; padding: 10px 16px; border-bottom: 1px solid {t['row_border']}; white-space: nowrap;"
            
            # Apply highlighting rules
            if col in highlight_cols:
                rules = highlight_cols[col]
                try:
                    num_val = float(val) if not pd.isna(val) else None
                    if num_val is not None:
                        if rules.get("type") == "ticker":
                            cell_style += f"color: {t['accent']}; font-weight: 700;"
                        elif rules.get("type") == "pct_direction":
                            if num_val > 0:
                                cell_style += f"color: {t['green']}; font-weight: 600;"
                            elif num_val < 0:
                                cell_style += f"color: {t['red']}; font-weight: 600;"
                        elif rules.get("type") == "prob":
                            if num_val > 0.55:
                                cell_style += f"color: {t['green']}; font-weight: 600;"
                            elif num_val < 0.45:
                                cell_style += f"color: {t['red']}; font-weight: 600;"
                        elif rules.get("type") == "sharpe":
                            if num_val > 0.5:
                                cell_style += f"color: {t['green']};"
                            elif num_val < 0:
                                cell_style += f"color: {t['red']};"
                        elif rules.get("type") == "accuracy":
                            if num_val > 55:
                                cell_style += f"color: {t['green']}; font-weight: 600;"
                            elif num_val < 50:
                                cell_style += f"color: {t['red']}; font-weight: 600;"
                except:
                    pass
            
            # Format value
            if pd.isna(val):
                display_val = "—"
            elif isinstance(val, float):
                display_val = f"{val:.2f}"
            else:
                display_val = str(val)
            
            html += f'<td style="{cell_style}">{display_val}</td>'
        
        html += "</tr>"
    
    html += "</tbody></table></div>"
    return html


def show_loading_indicator(message="Loading..."):
    """Display a styled loading indicator."""
    is_dark = st.session_state.get("theme", "dark") == "dark"
    bg = "#161b22" if is_dark else "#f6f8fa"
    border = "#30363d" if is_dark else "#d0d7de"
    text = "#8b949e" if is_dark else "#57606a"
    accent = "#388bfd" if is_dark else "#0969da"
    
    return st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1rem;
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <div style="
            width: 20px;
            height: 20px;
            border: 2px solid {border};
            border-top: 2px solid {accent};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <span style="color: {text}; font-size: 0.9rem;">{message}</span>
    </div>
    <style>
        @keyframes spin {{{{
            0% {{{{ transform: rotate(0deg); }}}}
            100% {{{{ transform: rotate(360deg); }}}}
        }}}}
    </style>
    """, unsafe_allow_html=True)

def _enrich_prediction_with_options(pred: dict, ticker: str) -> dict:
    """Add options data to prediction result."""
    try:
        opt = get_option_snapshot_features(ticker) or {}
        if isinstance(opt, dict):
            pred.update(opt)
        
        # IV minus realized
        atm_iv = pred.get("atm_iv")
        if atm_iv is not None and pred.get("vol_20d") is not None:
            try:
                pred["iv_minus_realized"] = float(atm_iv) - float(pred["vol_20d"])
            except:
                pred["iv_minus_realized"] = None
        else:
            pred["iv_minus_realized"] = None
        
        # Theo ATM call (Black-Scholes)
        pred["theo_atm_call_price"] = None
        last_close = pred.get("last_close")
        opt_exp = pred.get("opt_exp")
        if last_close is not None and atm_iv is not None and opt_exp:
            try:
                from src.core.pricing import price_option, OptionSpec
                opt_exp_date = pd.to_datetime(opt_exp).date()
                val_date = pd.Timestamp.today().date()
                opt_spec = OptionSpec(
                    spot=float(last_close),
                    strike=float(last_close),
                    maturity_date=opt_exp_date,
                    valuation_date=val_date,
                    rate=0.05,
                    div_yield=0.0,
                    vol=float(atm_iv),
                    is_call=True,
                )
                theo_price = price_option(opt_spec, model=PricingModel.BLACK_SCHOLES)
                pred["theo_atm_call_price"] = float(theo_price)
            except:
                pass
    except Exception as e:
        print(f"Options enrichment failed for {ticker}: {e}")
    
    return pred


def _enrich_prediction_with_options_cached(pred: dict, ticker: str, pricing_model=None) -> dict:
    """
    Add options data to prediction result using cached option data.
    This version uses the cached option snapshot to avoid redundant API calls.
    
    Args:
        pred: Prediction dictionary to enrich
        ticker: Stock ticker symbol
        pricing_model: PricingModel.BLACK_SCHOLES or PricingModel.HESTON (default: BLACK_SCHOLES)
    """
    if pricing_model is None:
        pricing_model = PricingModel.BLACK_SCHOLES
    
    try:
        # Use cached option snapshot
        opt = _cached_option_snapshot(ticker) or {}
        if isinstance(opt, dict):
            pred.update(opt)
        
        # IV minus realized
        atm_iv = pred.get("atm_iv")
        if atm_iv is not None and pred.get("vol_20d") is not None:
            try:
                pred["iv_minus_realized"] = float(atm_iv) - float(pred["vol_20d"])
            except:
                pred["iv_minus_realized"] = None
        else:
            pred["iv_minus_realized"] = None
        
        # Theo ATM call using selected pricing model
        pred["theo_atm_call_price"] = None
        pred["pricing_model_used"] = pricing_model.name if hasattr(pricing_model, 'name') else str(pricing_model)
        last_close = pred.get("last_close")
        opt_exp = pred.get("opt_exp")
        if last_close is not None and atm_iv is not None and opt_exp:
            try:
                from src.core.pricing import price_option, OptionSpec
                opt_exp_date = pd.to_datetime(opt_exp).date()
                val_date = pd.Timestamp.today().date()
                opt_spec = OptionSpec(
                    spot=float(last_close),
                    strike=float(last_close),
                    maturity_date=opt_exp_date,
                    valuation_date=val_date,
                    rate=0.05,
                    div_yield=0.0,
                    vol=float(atm_iv),
                    is_call=True,
                )
                theo_price = price_option(opt_spec, model=pricing_model)
                pred["theo_atm_call_price"] = float(theo_price)
            except Exception as price_err:
                # If Heston fails, fallback to Black-Scholes
                if pricing_model != PricingModel.BLACK_SCHOLES:
                    try:
                        theo_price = price_option(opt_spec, model=PricingModel.BLACK_SCHOLES)
                        pred["theo_atm_call_price"] = float(theo_price)
                        pred["pricing_model_used"] = "BLACK_SCHOLES (fallback)"
                    except:
                        pass
    except Exception as e:
        print(f"Options enrichment failed for {ticker}: {e}")
    
    return pred

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="QuantDesk | Stock Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# THEME-AWARE CSS GENERATOR
# ============================================================================
def generate_theme_css(is_dark: bool = True) -> str:
    """Generate CSS based on current theme."""
    if is_dark:
        theme_vars = """
        --bg-dark: #0a0e14;
        --bg-panel: #0d1117;
        --bg-card: #161b22;
        --bg-elevated: #1c2128;
        --bg-hover: #21262d;
        
        --border-default: #30363d;
        --border-muted: #21262d;
        --border-accent: #388bfd;
        
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --text-link: #58a6ff;
        
        --accent-green: #3fb950;
        --accent-red: #f85149;
        --accent-blue: #388bfd;
        --accent-yellow: #d29922;
        --accent-purple: #a371f7;
        --accent-cyan: #39c5cf;
        
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
        
        --df-bg: #0d1117;
        --df-header-bg: #161b22;
        --df-row-alt: #0f141a;
        """
    else:
        theme_vars = """
        --bg-dark: #ffffff;
        --bg-panel: #f6f8fa;
        --bg-card: #ffffff;
        --bg-elevated: #f3f4f6;
        --bg-hover: #e5e7eb;
        
        --border-default: #d0d7de;
        --border-muted: #d8dee4;
        --border-accent: #0969da;
        
        --text-primary: #1f2328;
        --text-secondary: #57606a;
        --text-muted: #6e7781;
        --text-link: #0969da;
        
        --accent-green: #1a7f37;
        --accent-red: #cf222e;
        --accent-blue: #0969da;
        --accent-yellow: #9a6700;
        --accent-purple: #8250df;
        --accent-cyan: #0969da;
        
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.15);
        
        --df-bg: #ffffff;
        --df-header-bg: #f6f8fa;
        --df-row-alt: #f9fafb;
        """
    
    return f"""
<style>
    /* ============================================================
       QUANTDESK PROFESSIONAL TRADING DASHBOARD THEME
       Dynamic Light/Dark Mode Support
       ============================================================ */
    
    /* === FONTS === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* === ROOT VARIABLES === */
    :root {{
        {theme_vars}
        
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
        
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
    }}
    
    /* === GLOBAL RESET === */
    .stApp {{
        background: var(--bg-dark);
        font-family: var(--font-sans);
    }}
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, header[data-testid="stHeader"] {{
        visibility: hidden;
        height: 0;
    }}
    .block-container {{
        padding: 1rem 1.5rem 2rem 1.5rem;
        max-width: 100%;
    }}
    
    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {{
        background: var(--bg-panel);
        border-right: 1px solid var(--border-default);
        width: 280px !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding: 0.75rem 1rem;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: var(--text-primary) !important;
        font-size: 0.8rem;
    }}
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stTextInput input {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono);
        font-size: 0.8rem;
        border-radius: var(--radius-sm);
    }}
    section[data-testid="stSidebar"] .stSlider p,
    section[data-testid="stSidebar"] .stSlider span {{
        color: var(--text-secondary) !important;
        font-size: 0.75rem;
    }}
    section[data-testid="stSidebar"] .stCheckbox label {{
        color: var(--text-secondary) !important;
    }}
    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-sm);
        color: var(--text-primary) !important;
        font-size: 0.75rem;
        font-weight: 500;
    }}
    
    /* === TYPOGRAPHY === */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li {{
        color: var(--text-primary) !important;
    }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--text-muted) !important;
        font-size: 0.7rem;
    }}
    
    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {{
        background: transparent;
        border-bottom: 2px solid var(--border-muted);
        gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        color: var(--text-secondary);
        font-family: var(--font-sans);
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        padding: 0.75rem 1.5rem;
        margin-bottom: -2px;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text-primary);
    }}
    .stTabs [aria-selected="true"] {{
        border-bottom-color: var(--accent-blue) !important;
        color: var(--text-primary) !important;
    }}
    
    /* === METRIC CARDS === */
    [data-testid="stMetric"] {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-sm);
    }}
    [data-testid="stMetric"]:hover {{
        border-color: var(--border-default);
        box-shadow: var(--shadow-md);
    }}
    [data-testid="stMetric"] label {{
        color: var(--text-muted) !important;
        font-size: 0.65rem !important;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] div {{
        color: var(--text-primary) !important;
        font-family: var(--font-mono);
        font-size: 1.5rem !important;
        font-weight: 600;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: var(--font-mono);
        font-size: 0.75rem;
    }}
    [data-testid="stMetricDelta"][data-testid-direction="up"] {{
        color: var(--accent-green) !important;
    }}
    [data-testid="stMetricDelta"][data-testid-direction="down"] {{
        color: var(--accent-red) !important;
    }}
    
    /* === BUTTONS === */
    .stButton > button {{
        background: var(--accent-blue);
        border: none;
        border-radius: var(--radius-sm);
        color: white !important;
        font-family: var(--font-sans);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 0.6rem 1.25rem;
        text-transform: uppercase;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        background: #1f6feb;
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}
    .stButton > button[kind="secondary"] {{
        background: transparent;
        border: 1px solid var(--border-default);
        color: var(--text-primary) !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: var(--bg-hover);
        border-color: var(--text-secondary);
    }}
    
    /* === DATAFRAMES/TABLES - THEME AWARE === */
    .stDataFrame,
    [data-testid="stDataFrame"],
    .stDataFrame > div {{
        border: 1px solid var(--border-default) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: var(--df-bg) !important;
    }}
    
    [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrameGlideDataEditor"],
    .dvn-scroller,
    .dvn-underlay,
    .dvn-scroll-inner,
    div[data-testid="glide-data-grid-canvas"],
    .gdg-style,
    [class*="glideDataEditor"] {{
        background: var(--df-bg) !important;
        background-color: var(--df-bg) !important;
    }}
    
    canvas,
    .dvn-scroller canvas {{
        background: var(--df-bg) !important;
    }}
    
    .dvn-header,
    [data-testid="stDataFrameGlideDataEditor"] .dvn-header,
    .gdg-header-row,
    .dvn-scroller .dvn-header {{
        background: var(--df-header-bg) !important;
        background-color: var(--df-header-bg) !important;
    }}
    
    .stDataFrame *,
    [data-testid="stDataFrame"] *,
    .dvn-scroller *,
    .gdg-cell *,
    [data-testid="stDataFrameGlideDataEditor"] * {{
        color: var(--text-primary) !important;
    }}
    
    .stDataFrame > div > div,
    .stDataFrame iframe,
    [data-testid="stDataFrame"] > div {{
        background: var(--df-bg) !important;
    }}
    
    .dvn-resize-handle {{
        background: var(--border-default) !important;
    }}
    
    .dvn-row-number,
    .gdg-row-number {{
        background: var(--df-header-bg) !important;
        color: var(--text-muted) !important;
    }}
    
    .stDataFrame ::-webkit-scrollbar,
    [data-testid="stDataFrame"] ::-webkit-scrollbar {{
        background: var(--df-bg) !important;
        width: 8px;
        height: 8px;
    }}
    .stDataFrame ::-webkit-scrollbar-thumb,
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {{
        background: var(--border-default) !important;
        border-radius: 4px;
    }}
    .stDataFrame ::-webkit-scrollbar-track,
    [data-testid="stDataFrame"] ::-webkit-scrollbar-track {{
        background: var(--df-bg) !important;
    }}
    
    .dvn-cell-selected,
    .gdg-cell-selected {{
        background: rgba(56, 139, 253, 0.2) !important;
    }}
    
    .dvn-cell:hover,
    .gdg-cell:hover {{
        background: var(--bg-hover) !important;
    }}
    
    /* === SELECTBOX DROPDOWN === */
    [data-baseweb="select"] > div,
    [data-baseweb="popover"] > div,
    div[data-baseweb="popover"] {{
        background: var(--bg-card) !important;
    }}
    [data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="menu"] li,
    ul[role="listbox"] li,
    div[role="listbox"] > div,
    [role="option"] {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    [data-baseweb="menu"] li:hover,
    ul[role="listbox"] li:hover,
    [role="option"]:hover {{
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }}
    [data-baseweb="menu"] li[aria-selected="true"],
    [role="option"][aria-selected="true"] {{
        background: var(--accent-blue) !important;
        color: white !important;
    }}
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] > div > div {{
        color: var(--text-primary) !important;
    }}
    .stMultiSelect [data-baseweb="tag"] {{
        background: var(--accent-blue) !important;
        color: white !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span {{
        color: white !important;
    }}
    
    [data-baseweb="select"] {{
        background: var(--bg-card) !important;
    }}
    [data-baseweb="select"] > div {{
        background: var(--bg-card) !important;
        background-color: var(--bg-card) !important;
    }}
    [data-baseweb="select"] [data-baseweb="input"] {{
        background: var(--bg-card) !important;
        background-color: var(--bg-card) !important;
    }}
    [data-baseweb="popover"] {{
        background: var(--bg-card) !important;
    }}
    [data-baseweb="popover"] > div {{
        background: var(--bg-card) !important;
    }}
    .stSelectbox div,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [class*="css"] {{
        background-color: var(--bg-card) !important;
    }}
    [data-baseweb="base-input"],
    [data-baseweb="base-input"] > div {{
        background: var(--bg-card) !important;
        background-color: var(--bg-card) !important;
    }}
    .stSelectbox [data-baseweb="select"]:focus-within,
    .stSelectbox [data-baseweb="select"]:focus,
    .stSelectbox [data-baseweb="select"]:active {{
        background: var(--bg-card) !important;
        background-color: var(--bg-card) !important;
    }}
    
    /* === PROGRESS BAR === */
    .stProgress > div > div {{
        background: var(--bg-elevated);
        border-radius: var(--radius-sm);
    }}
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        border-radius: var(--radius-sm);
    }}
    
    /* === ALERTS === */
    .stAlert {{
        border-radius: var(--radius-md);
        border-left-width: 4px;
    }}
    .stSuccess {{
        background: rgba(63, 185, 80, 0.1);
        border-left-color: var(--accent-green);
    }}
    .stWarning {{
        background: rgba(210, 153, 34, 0.1);
        border-left-color: var(--accent-yellow);
    }}
    .stError {{
        background: rgba(248, 81, 73, 0.1);
        border-left-color: var(--accent-red);
    }}
    .stInfo {{
        background: rgba(56, 139, 253, 0.1);
        border-left-color: var(--accent-blue);
    }}
    
    /* === EXPANDERS === */
    .streamlit-expanderHeader {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-md);
        color: var(--text-primary) !important;
        font-size: 0.85rem;
        font-weight: 500;
    }}
    .streamlit-expanderHeader:hover {{
        background: var(--bg-elevated);
        border-color: var(--border-default);
    }}
    .streamlit-expanderContent {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-top: none;
        border-radius: 0 0 var(--radius-md) var(--radius-md);
    }}
    
    /* === INPUTS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: var(--radius-sm);
        color: var(--text-primary) !important;
        font-family: var(--font-mono);
        font-size: 0.85rem;
    }}
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within {{
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(56, 139, 253, 0.2);
    }}
    
    /* === CUSTOM CLASSES === */
    .section-title {{
        border-bottom: 2px solid var(--border-muted);
        color: var(--text-secondary) !important;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        text-transform: uppercase;
    }}
    
    .metric-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-md);
        padding: 1rem;
    }}
    
    .metric-label {{
        color: var(--text-muted);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}
    
    .metric-value {{
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }}
    
    .metric-value.positive {{ color: var(--accent-green); }}
    .metric-value.negative {{ color: var(--accent-red); }}
    .metric-value.neutral {{ color: var(--text-secondary); }}
    
    .status-badge {{
        border-radius: 12px;
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
    }}
    .status-badge.bullish {{
        background: rgba(63, 185, 80, 0.15);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
    }}
    .status-badge.bearish {{
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid var(--accent-red);
        color: var(--accent-red);
    }}
    .status-badge.neutral {{
        background: rgba(139, 148, 158, 0.15);
        border: 1px solid var(--text-secondary);
        color: var(--text-secondary);
    }}
    
    .ticker-symbol {{
        background: var(--accent-blue);
        border-radius: var(--radius-sm);
        color: white !important;
        font-family: var(--font-mono);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.3rem 0.6rem;
    }}
    
    .data-row {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-md);
        margin: 0.5rem 0;
        padding: 0.75rem 1rem;
    }}
    .data-row:hover {{
        border-color: var(--border-default);
    }}
    
    .chart-container {{
        background: var(--bg-card);
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-md);
        padding: 1rem;
    }}
    
    .empty-state {{
        background: var(--bg-card);
        border: 2px dashed var(--border-muted);
        border-radius: var(--radius-lg);
        color: var(--text-muted) !important;
        padding: 3rem 2rem;
        text-align: center;
    }}
    
    .divider {{
        border: none;
        border-top: 1px solid var(--border-muted);
        margin: 1.5rem 0;
    }}
    
    /* === RESPONSIVE TWEAKS === */
    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            width: 260px !important;
        }}
        [data-testid="stMetric"] {{
            padding: 0.75rem;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] div {{
            font-size: 1.2rem !important;
        }}
    }}
</style>
"""

# NOTE: The static CSS block that was here has been removed.
# Theme CSS is now handled dynamically by generate_theme_css() called below.
# This allows proper light/dark mode switching.

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
st.session_state.setdefault("pred_df", None)
st.session_state.setdefault("model_type", "rf")
st.session_state.setdefault("screener_df", None)
st.session_state.setdefault("prediction_horizon", 5)
st.session_state.setdefault("period", "5y")
st.session_state.setdefault("auto_optimize", True)
st.session_state.setdefault("last_signals", None)
st.session_state.setdefault("theme", "dark")  # Theme toggle state

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
def get_theme_colors():
    """Get color palette based on current theme."""
    is_dark = st.session_state.get("theme", "dark") == "dark"
    
    if is_dark:
        return {
            "bg_dark": "#0a0e14",
            "bg_panel": "#0d1117",
            "bg_card": "#161b22",
            "bg_elevated": "#1c2128",
            "bg_hover": "#21262d",
            "border_default": "#30363d",
            "border_muted": "#21262d",
            "border_accent": "#388bfd",
            "text_primary": "#f0f6fc",
            "text_secondary": "#8b949e",
            "text_muted": "#6e7681",
            "text_link": "#58a6ff",
            "accent_green": "#3fb950",
            "accent_red": "#f85149",
            "accent_blue": "#388bfd",
            "accent_yellow": "#d29922",
            "accent_purple": "#a371f7",
            "accent_cyan": "#39c5cf",
            "chart_bg": "rgba(0,0,0,0)",
            "chart_grid": "rgba(48,54,61,0.5)",
        }
    else:
        return {
            "bg_dark": "#ffffff",
            "bg_panel": "#f6f8fa",
            "bg_card": "#ffffff",
            "bg_elevated": "#f3f4f6",
            "bg_hover": "#e5e7eb",
            "border_default": "#d0d7de",
            "border_muted": "#d8dee4",
            "border_accent": "#0969da",
            "text_primary": "#1f2328",
            "text_secondary": "#57606a",
            "text_muted": "#6e7781",
            "text_link": "#0969da",
            "accent_green": "#1a7f37",
            "accent_red": "#cf222e",
            "accent_blue": "#0969da",
            "accent_yellow": "#9a6700",
            "accent_purple": "#8250df",
            "accent_cyan": "#0969da",
            "chart_bg": "rgba(255,255,255,0)",
            "chart_grid": "rgba(208,215,222,0.5)",
        }

def get_plotly_template():
    """Get Plotly layout settings based on theme."""
    colors = get_theme_colors()
    return {
        "plot_bgcolor": colors["chart_bg"],
        "paper_bgcolor": colors["chart_bg"],
        "font": {"color": colors["text_secondary"], "size": 10},
        "xaxis": {"gridcolor": colors["chart_grid"], "zeroline": False},
        "yaxis": {"gridcolor": colors["chart_grid"]},
    }

# Get current theme colors
THEME = get_theme_colors()

# Inject theme-aware CSS
is_dark_mode = st.session_state.get("theme", "dark") == "dark"
st.markdown(generate_theme_css(is_dark_mode), unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Professional Control Panel
# ============================================================================

# Brand header with theme toggle
theme_col1, theme_col2 = st.sidebar.columns([3, 1])
with theme_col1:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #388bfd 0%, #a371f7 100%);
        border-radius: 6px;
        padding: 0.75rem 1rem;
    '>
        <div style='font-size: 1.1rem; font-weight: 700; color: white; font-family: Inter, sans-serif; letter-spacing: -0.5px;'>
            QuantDesk
        </div>
        <div style='font-size: 0.6rem; color: rgba(255,255,255,0.8); margin-top: 0.15rem; letter-spacing: 1px; text-transform: uppercase;'>
            ML Trading Signals
        </div>
    </div>
    """, unsafe_allow_html=True)

with theme_col2:
    # Theme toggle button
    current_theme = st.session_state.get("theme", "dark")
    theme_icon = "🌙" if current_theme == "dark" else "☀️"
    theme_label = "Light" if current_theme == "dark" else "Dark"
    
    if st.button(theme_icon, key="theme_toggle", help=f"Switch to {theme_label} mode", use_container_width=True):
        st.session_state["theme"] = "light" if current_theme == "dark" else "dark"
        st.rerun()

st.sidebar.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

# === UNIVERSE SECTION ===
st.sidebar.markdown('<p class="section-title">Universe</p>', unsafe_allow_html=True)
watchlist_text = st.sidebar.text_input(
    "Tickers", 
    value="AAPL, NVDA, MSFT",
    label_visibility="collapsed",
    placeholder="Enter tickers (comma-separated)..."
)

# Validate and parse tickers
tickers, ticker_validation_errors = validate_tickers(watchlist_text)

# Show validation feedback in sidebar
if ticker_validation_errors and watchlist_text.strip():
    # Only show first error to avoid cluttering sidebar
    st.sidebar.caption(f"⚠️ {ticker_validation_errors[0]}")
elif tickers:
    st.sidebar.caption(f"✓ {len(tickers)} valid ticker(s)")

# Quick presets
preset_cols = st.sidebar.columns(3)
with preset_cols[0]:
    if st.button("MAG7", use_container_width=True, key="preset_mag7"):
        st.session_state["_preset_tickers"] = "AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA"
with preset_cols[1]:
    if st.button("TECH", use_container_width=True, key="preset_tech"):
        st.session_state["_preset_tickers"] = "AAPL, MSFT, NVDA, AMD, CRM, ADBE, INTC"
with preset_cols[2]:
    if st.button("FANG", use_container_width=True, key="preset_fang"):
        st.session_state["_preset_tickers"] = "META, AMZN, NFLX, GOOGL"

# Apply preset if clicked
if "_preset_tickers" in st.session_state and st.session_state["_preset_tickers"]:
    watchlist_text = st.session_state["_preset_tickers"]
    tickers, _ = validate_tickers(watchlist_text)  # Presets are pre-validated, ignore errors
    st.session_state["_preset_tickers"] = None

# === MODEL CONFIGURATION ===
st.sidebar.markdown('<p class="section-title">Model Configuration</p>', unsafe_allow_html=True)

param_cols = st.sidebar.columns(2)
with param_cols[0]:
    period = st.selectbox(
        "Training Period",
        ["2y", "5y", "10y"],
        index=1,
        key="param_period"
    )
with param_cols[1]:
    prediction_horizon = st.selectbox(
        "Forecast Horizon",
        [1, 2, 3, 4, 5],
        index=4,
        format_func=lambda x: f"{x}D",
        key="param_horizon"
    )

horizon_label = f"{prediction_horizon}D"

model_type = st.sidebar.selectbox(
    "Algorithm",
    ["rf", "gbrt", "xgb"],
    index=0,
    format_func=lambda x: {"rf": "Random Forest", "gbrt": "Gradient Boosting", "xgb": "XGBoost"}[x],
    key="param_model"
)

auto_optimize = st.sidebar.checkbox("Auto-optimize hyperparameters", value=True)

# === SIGNAL FILTERS ===
st.sidebar.markdown('<p class="section-title">Signal Filters</p>', unsafe_allow_html=True)

signal_threshold_pct = st.sidebar.slider(
    "Min signal threshold",
    min_value=0.0,
    max_value=2.0,
    value=0.25,
    step=0.05,
    format="%.2f%%",
    key="signal_thresh"
) / 100.0

with st.sidebar.expander("Advanced Filters", expanded=False):
    max_tickers = st.slider("Max tickers", 1, 50, 10)
    ret_thresh = st.slider("Min return %", 0.0, 10.0, 3.0, 0.5)
    vol_spike_thresh = st.slider("Vol spike (x)", 0.5, 5.0, 1.5, 0.1)
    min_move = st.slider("Min pred %", 0.0, 5.0, 1.0, 0.1)
    min_iv, max_iv = st.slider("IV range", 0.0, 1.0, (0.2, 0.8), 0.05)
    exclude_disagree = st.checkbox("Hide disagree", value=True)

# === EXECUTION SETTINGS ===
st.sidebar.markdown('<p class="section-title">Execution</p>', unsafe_allow_html=True)

with st.sidebar.expander("Pricing & Features", expanded=False):
    pricing_model_label = st.selectbox("Pricing Model", ["Black-Scholes", "Heston"], index=0)
    pricing_model = PricingModel.BLACK_SCHOLES if pricing_model_label == "Black-Scholes" else PricingModel.HESTON

    st.markdown("**Feature Selection**")
    use_elasticnet_select = st.checkbox("Elastic Net", value=False)
    en_l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.05)
    en_cv_folds = st.slider("CV folds", 3, 8, 5, 1)
    en_min_features = st.slider("Min features", 6, 40, 12, 1)

    if use_elasticnet_select:
        os.environ["USE_ELASTICNET_SELECT"] = "1"
        os.environ["ELASTICNET_L1_RATIO"] = str(en_l1_ratio)
        os.environ["ELASTICNET_CV_FOLDS"] = str(en_cv_folds)
        os.environ["ELASTICNET_MINFEATURES"] = str(en_min_features)
    else:
        os.environ["USE_ELASTICNET_SELECT"] = "0"

    run_gaf = st.checkbox("GAF-CNN", value=False)
    fetch_live_price = st.checkbox("Live price", value=False)
    run_mc = st.checkbox("Monte Carlo", value=False)

with st.sidebar.expander("Trading Costs", expanded=False):
    friction_preset = st.selectbox("Friction Preset", list(FRICTION_PRESETS.keys()), index=0)
    fp = FRICTION_PRESETS[friction_preset]
    exec_model = ExecutionModel(
        delay_days=int(fp.delay_days),
        half_spread_bps=float(fp.half_spread_bps),
        slippage_bps=float(fp.slippage_bps),
        fee_bps=float(fp.fee_bps),
    )

    with st.expander("Override Costs", expanded=False):
        exec_model = ExecutionModel(
            delay_days=_parse_int(st.text_input("Delay days", value=str(exec_model.delay_days)), exec_model.delay_days),
            half_spread_bps=_parse_float(st.text_input("Spread (bps)", value=str(exec_model.half_spread_bps)), exec_model.half_spread_bps),
            slippage_bps=_parse_float(st.text_input("Slippage (bps)", value=str(exec_model.slippage_bps)), exec_model.slippage_bps),
            fee_bps=_parse_float(st.text_input("Fee (bps)", value=str(exec_model.fee_bps)), exec_model.fee_bps),
        )

with st.sidebar.expander("Options Trading", expanded=False):
    trade_mode = st.selectbox("Mode", ["Stocks", "Options if suggested", "Options"], index=1)
    options_preset = st.selectbox("Options Preset", list(OPTIONS_PRESETS.keys()), index=0)
    op = OPTIONS_PRESETS[options_preset]

    budget_per_contract = _parse_float(st.text_input("Max premium ($)", value="500"), 500.0)
    max_premium = float(budget_per_contract)
    max_strike = _parse_float(st.text_input("Max strike", value="500"), 500.0)
    dte_min = _parse_int(st.text_input("DTE min", value=str(op.dte_min)), int(op.dte_min))
    dte_max = _parse_int(st.text_input("DTE max", value=str(op.dte_max)), int(op.dte_max))
    width_pct_in = _parse_float(st.text_input("Width %", value=str(op.width_pct * 100.0)), op.width_pct * 100.0)
    width_pct = float(width_pct_in) / 100.0
    prefer_spreads = st.checkbox("Prefer spreads", value=bool(op.prefer_spreads))
    auto_run_trader = st.checkbox("Auto-execute trades", value=False)

# === CACHE MANAGEMENT ===
st.sidebar.markdown('<p class="section-title">Cache Management</p>', unsafe_allow_html=True)
with st.sidebar.expander("🗑️ Clear Caches", expanded=False):
    st.caption("Clear cached data for fresh results:")
    cache_cols = st.columns(2)
    with cache_cols[0]:
        if st.button("Clear Predictions", use_container_width=True):
            clear_prediction_cache()
            st.success("✓ Predictions cleared")
    with cache_cols[1]:
        if st.button("Clear Prices", use_container_width=True):
            clear_price_cache()
            st.success("✓ Prices cleared")
    cache_cols2 = st.columns(2)
    with cache_cols2[0]:
        if st.button("Clear Options", use_container_width=True):
            clear_options_cache()
            st.success("✓ Options cleared")
    with cache_cols2[1]:
        if st.button("Clear Backtests", use_container_width=True):
            clear_backtest_cache()
            st.success("✓ Backtests cleared")
    if st.button("🔄 Clear All Caches", use_container_width=True, type="primary"):
        clear_all_caches()
        st.success("✓ All caches cleared!")

# === OPTIONAL FEATURES ===
if HAS_LIVE_UPDATES:
    st.sidebar.markdown('<p class="section-title">Live Updates</p>', unsafe_allow_html=True)
    init_live_state()
    live_enabled = st.sidebar.toggle(
        "🔴 Enable Live Mode",
        value=st.session_state.get("live_enabled", False),
        key="sidebar_live_toggle",
        help="Auto-refresh prices and charts during market hours"
    )
    st.session_state["live_enabled"] = live_enabled
    
    if live_enabled:
        interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[10, 30, 60, 120, 300],
            index=1,
            format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}m",
            key="sidebar_live_interval"
        )
        set_live_config(refresh_interval=interval)
        
        # Show market status
        if is_market_hours():
            st.sidebar.success("🟢 Market Open")
        else:
            st.sidebar.warning("🟠 After Hours")

if HAS_DATABASE:
    st.sidebar.markdown('<p class="section-title">History</p>', unsafe_allow_html=True)
    with st.sidebar.expander("📚 Query History", expanded=False):
        recent_queries = get_recent_queries(limit=5, query_type="prediction")
        if recent_queries:
            for q in recent_queries:
                tickers_str = ", ".join(q["tickers"][:3])
                if len(q["tickers"]) > 3:
                    tickers_str += f" +{len(q['tickers'])-3}"
                
                label = q.get("label") or f"{tickers_str} • {q['model_type'].upper()}"
                fav_icon = "⭐" if q.get("is_favorite") else "☆"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"{fav_icon} {label}", key=f"query_{q['id']}", use_container_width=True):
                        # Load query into current session
                        st.session_state["_preset_tickers"] = ", ".join(q["tickers"])
                        st.session_state["param_period"] = q["period"]
                        st.session_state["param_model"] = q["model_type"]
                        st.session_state["param_horizon"] = q["horizon"]
                        st.rerun()
                with col2:
                    if st.button("⭐" if not q.get("is_favorite") else "★", key=f"fav_{q['id']}"):
                        toggle_favorite_query(q["id"])
                        st.rerun()
        else:
            st.caption("No recent queries")
        
        # Database stats
        try:
            stats = get_database_stats()
            st.caption(f"📊 {stats.get('prediction_history_count', 0)} predictions stored")
        except Exception:
            pass

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.caption(f"📊 {len(tickers)} ticker(s) • {period} • {horizon_label}")

# ============================================================================
# MAIN PANEL - Tabbed Layout
# ============================================================================

tab_dash, tab_backtests, tab_port = st.tabs(["📈 DASHBOARD", "🔬 BACKTEST", "📊 PORTFOLIO"])

# ============================================================================
# TAB: Dashboard - Professional Trading View
# ============================================================================
with tab_dash:
    
    # === HEADER ROW: Run Button + Status ===
    header_cols = st.columns([2, 6, 2])
    with header_cols[0]:
        run_btn = st.button("🚀 RUN PREDICTIONS", type="primary", key="run_predictions", use_container_width=True)
    with header_cols[1]:
        tc = get_card_theme()
        st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0;'>
            <span class='ticker-symbol'>{len(tickers)} TICKERS</span>
            <span style='color: {tc["label"]}; font-family: JetBrains Mono, monospace; font-size: 0.8rem;'>
                {model_type.upper()} • {period} • {horizon_label} Horizon
            </span>
        </div>
        """, unsafe_allow_html=True)
    with header_cols[2]:
        # Show live status badge if available
        if HAS_LIVE_UPDATES and st.session_state.get("live_enabled", False):
            render_live_status_badge()
        elif st.session_state.get("pred_df") is not None:
            st.markdown("<span class='status-badge bullish'>READY</span>", unsafe_allow_html=True)
    
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
    # === RUN PREDICTIONS ===
    if run_btn:
        # Validate inputs before running
        is_valid, validation_errors = validate_prediction_inputs(tickers, period, prediction_horizon, model_type)
        
        if not is_valid:
            display_validation_errors(validation_errors)
        elif not tickers:
            st.warning("⚠️ Enter at least one ticker symbol in the sidebar")
        else:
            results = []
            errors = []
            progress = st.progress(0)
            status = st.empty()
            
            # Get current settings for toggles
            current_auto_optimize = auto_optimize if 'auto_optimize' in dir() else True
            current_use_elasticnet = use_elasticnet_select if 'use_elasticnet_select' in dir() else False
            current_pricing_model = pricing_model if 'pricing_model' in dir() else PricingModel.BLACK_SCHOLES
            
            for i, tk in enumerate(tickers):
                status.text(f"Processing {tk} ({i+1}/{len(tickers)})...")
                
                # Use cached prediction with safe API call wrapper
                run_gaf_val = run_gaf if 'run_gaf' in dir() else False
                pred, error = safe_api_call(
                    _cached_prediction,
                    ticker=tk,
                    period=period,
                    model_type=model_type,
                    horizon=prediction_horizon,
                    run_gaf=run_gaf_val,
                    auto_optimize=current_auto_optimize,
                    use_elasticnet=current_use_elasticnet,
                    error_prefix=f"Prediction failed for {tk}"
                )
                
                if error:
                    errors.append((tk, error))
                elif pred:
                    # Enrich with options data using cached functions (with error handling)
                    enriched_pred, enrich_error = safe_api_call(
                        _enrich_prediction_with_options_cached, 
                        pred, 
                        tk,
                        current_pricing_model,
                        error_prefix=f"Options data failed for {tk}"
                    )
                    if enriched_pred:
                        results.append(enriched_pred)
                    else:
                        # Use unenriched prediction if options fail
                        results.append(pred)
                        if enrich_error:
                            errors.append((tk, f"Options data unavailable: {enrich_error}"))
                
                progress.progress((i + 1) / len(tickers))
            
            status.empty()
            progress.empty()
            
            # Display any errors that occurred
            if errors:
                with st.expander(f"⚠️ {len(errors)} ticker(s) had issues (click to expand)", expanded=False):
                    for tk, err in errors:
                        st.warning(f"**{tk}**: {err}")
            
            if results:
                pred_df = pd.DataFrame(results)
                
                # Add convenience columns
                if "pred_next_ret" in pred_df.columns:
                    pred_df["pred_next_ret_pct"] = pred_df["pred_next_ret"] * 100.0
                if "vol_20d" in pred_df.columns and "pred_next_ret" in pred_df.columns:
                    pred_df["vol_adjusted_edge"] = pred_df["pred_next_ret"] / pred_df["vol_20d"].replace(0, np.nan)
                
                st.session_state["pred_df"] = pred_df
                st.session_state["model_type"] = model_type
                st.session_state["period"] = period
                st.session_state["prediction_horizon"] = prediction_horizon
                
                # === OPTIONAL: Save to database ===
                if HAS_DATABASE:
                    try:
                        # Save predictions batch
                        save_predictions_batch(results, period, model_type, prediction_horizon)
                        
                        # Save query for history
                        save_query(
                            query_type="prediction",
                            tickers=tickers,
                            period=period,
                            model_type=model_type,
                            horizon=prediction_horizon,
                            extra_params={"trade_mode": trade_mode, "auto_optimize": auto_optimize}
                        )
                    except Exception as db_err:
                        # Silently fail - database is optional
                        pass
                
                # Build and save signals
                signals = build_signals_from_pred_df(
                    pred_df,
                    prediction_horizon=prediction_horizon,
                    trade_mode=trade_mode,
                    prefer_spreads=prefer_spreads,
                    dte_min=dte_min,
                    dte_max=dte_max,
                    max_strike=max_strike,
                    max_premium=max_premium,
                    width_pct=width_pct,
                    exec_model=exec_model,
                )
                
                # Write signals to file
                try:
                    with open(SIGNALS_OUT_PATH, "w") as f:
                        # Convert signals to JSON-serializable format
                        json_signals = {}
                        for k, v in signals.items():
                            if isinstance(v, dict):
                                json_signals[k] = {
                                    sk: (float(sv) if isinstance(sv, (np.floating, np.integer)) else sv)
                                    for sk, sv in v.items()
                                }
                            else:
                                json_signals[k] = v
                        json.dump(json_signals, f, indent=2, default=str)
                    st.session_state["last_signals"] = signals
                    st.success(f"✅ Wrote signals.json to: {SIGNALS_OUT_PATH}")
                except Exception as e:
                    st.warning(f"Failed to write signals: {e}")
                
                # Auto-run trader if enabled
                st.session_state["last_trader_stdout"] = ""
                st.session_state["last_trader_stderr"] = ""
                st.session_state["last_trader_rc"] = None
                
                if auto_run_trader:
                    if not TRADER_PATH.exists():
                        st.error(f"Trader not found: {TRADER_PATH}")
                    else:
                        with st.spinner("Executing trades..."):
                            res = subprocess.run(
                                [sys.executable, str(TRADER_PATH)],
                                cwd=str(BASE_DIR),
                                capture_output=True,
                                text=True,
                            )
                            st.session_state["last_trader_stdout"] = res.stdout or ""
                            st.session_state["last_trader_stderr"] = res.stderr or ""
                            st.session_state["last_trader_rc"] = res.returncode
                            
                            if res.returncode == 0:
                                st.success("Trade execution complete")
                            else:
                                st.error(f"Trade failed: code {res.returncode}")
                
                st.rerun()
    
    # Display results if available
    pred_df = st.session_state.get("pred_df")
    
    if pred_df is not None and not pred_df.empty:
        display_horizon = st.session_state.get("prediction_horizon", prediction_horizon)
        display_horizon_label = f"{display_horizon}D"
        
        # === HERO METRICS PANEL ===
        tc = get_card_theme()
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {tc["bg"]} 0%, {"#0d1117" if st.session_state.get("theme", "dark") == "dark" else "#f3f4f6"} 100%);
            border: 1px solid {tc["border"]};
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        '>
            <p style='color: {tc["label"]}; font-size: 0.7rem; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 1rem;'>
                📊 PORTFOLIO OVERVIEW
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate comprehensive stats
        avg_pred = pred_df["pred_next_ret"].mean() * 100 if "pred_next_ret" in pred_df.columns else 0
        max_pred = pred_df["pred_next_ret"].max() * 100 if "pred_next_ret" in pred_df.columns else 0
        min_pred = pred_df["pred_next_ret"].min() * 100 if "pred_next_ret" in pred_df.columns else 0
        bullish_count = (pred_df["pred_next_ret"] > 0).sum() if "pred_next_ret" in pred_df.columns else 0
        bearish_count = (pred_df["pred_next_ret"] < 0).sum() if "pred_next_ret" in pred_df.columns else 0
        avg_prob = pred_df["prob_up"].mean() * 100 if "prob_up" in pred_df.columns else 50
        avg_vol = pred_df["vol_20d"].mean() * 100 if "vol_20d" in pred_df.columns else 0
        avg_iv = pred_df["atm_iv"].mean() * 100 if "atm_iv" in pred_df.columns else 0
        avg_iv_premium = (pred_df["atm_iv"] - pred_df["vol_20d"]).mean() * 100 if "atm_iv" in pred_df.columns and "vol_20d" in pred_df.columns else 0
        
        # Top performer
        if "pred_next_ret" in pred_df.columns and "ticker" in pred_df.columns:
            top_idx = pred_df["pred_next_ret"].idxmax()
            top_ticker = pred_df.loc[top_idx, "ticker"]
            top_return = pred_df.loc[top_idx, "pred_next_ret"] * 100
        else:
            top_ticker = "—"
            top_return = 0
        
        # Row 1: Primary Metrics
        st.markdown('<p class="section-title">Key Metrics</p>', unsafe_allow_html=True)
        
        row1 = st.columns(6)
        with row1[0]:
            st.metric(
                label="TOTAL TICKERS",
                value=len(pred_df),
                delta=f"{bullish_count}↑ {bearish_count}↓"
            )
        with row1[1]:
            delta_val = "bullish" if avg_pred > 0 else "bearish"
            st.metric(
                label="AVG PREDICTION",
                value=f"{avg_pred:+.2f}%",
                delta=f"{delta_val}"
            )
        with row1[2]:
            st.metric(
                label="P(UP) AVG",
                value=f"{avg_prob:.0f}%",
                delta=f"{'positive bias' if avg_prob > 50 else 'negative bias'}"
            )
        with row1[3]:
            st.metric(
                label="AVG VOLATILITY",
                value=f"{avg_vol:.1f}%",
                delta="20D annualized"
            )
        with row1[4]:
            st.metric(
                label="MODEL",
                value=st.session_state.get("model_type", model_type).upper(),
                delta=display_horizon_label
            )
        with row1[5]:
            st.metric(
                label="TOP PICK",
                value=top_ticker,
                delta=f"{top_return:+.1f}%"
            )
        
        # Row 2: Detailed Analytics
        st.markdown('<p class="section-title">Market Analytics</p>', unsafe_allow_html=True)
        
        row2 = st.columns(5)
        with row2[0]:
            st.metric(
                label="MAX UPSIDE",
                value=f"{max_pred:+.2f}%",
                delta="best case"
            )
        with row2[1]:
            st.metric(
                label="MAX DOWNSIDE",
                value=f"{min_pred:+.2f}%",
                delta="worst case"
            )
        with row2[2]:
            if "atm_iv" in pred_df.columns:
                st.metric(
                    label="AVG IV",
                    value=f"{avg_iv:.1f}%",
                    delta=f"{avg_iv_premium:+.1f}% vs realized"
                )
            else:
                st.metric(label="AVG IV", value="—", delta="no data")
        with row2[3]:
            # Calculate spread (max - min prediction)
            pred_spread = max_pred - min_pred
            st.metric(
                label="SIGNAL SPREAD",
                value=f"{pred_spread:.1f}%",
                delta="high=dispersion"
            )
        with row2[4]:
            # Confidence indicator
            if "prob_up" in pred_df.columns:
                high_conf = ((pred_df["prob_up"] > 0.6) | (pred_df["prob_up"] < 0.4)).sum()
                conf_pct = high_conf / len(pred_df) * 100
                st.metric(
                    label="HIGH CONFIDENCE",
                    value=f"{high_conf}",
                    delta=f"{conf_pct:.0f}% of signals"
                )
            else:
                st.metric(label="HIGH CONFIDENCE", value="—", delta="no data")
        
        # Row 3: Quick Stats Cards using st.columns for reliable rendering
        st.markdown('<p class="section-title">Market Summary</p>', unsafe_allow_html=True)
        
        bull_pct = (bullish_count / max(len(pred_df), 1)) * 100
        bear_pct = (bearish_count / max(len(pred_df), 1)) * 100
        
        # Get theme colors for cards
        tc = get_card_theme()
        
        card_cols = st.columns(4)
        
        # Card 1: Bull/Bear Ratio
        with card_cols[0]:
            ratio_html = f"""
            <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                    BULL / BEAR RATIO
                </div>
                <div style='margin-top:0.5rem;'>
                    <span style='color:{tc["green"]};font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;'>{bullish_count}</span>
                    <span style='color:{tc["muted"]};font-size:1rem;'> / </span>
                    <span style='color:{tc["red"]};font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;'>{bearish_count}</span>
                </div>
                <div style='background:{tc["bar_bg"]};border-radius:4px;height:6px;margin-top:0.75rem;overflow:hidden;display:flex;'>
                    <div style='background:{tc["green"]};width:{bull_pct:.0f}%;height:100%;'></div>
                    <div style='background:{tc["red"]};width:{bear_pct:.0f}%;height:100%;'></div>
                </div>
            </div>
            """
            st.markdown(ratio_html, unsafe_allow_html=True)
        
        # Card 2: Market Bias
        with card_cols[1]:
            if avg_pred > 0.5:
                bias_color = tc["green"]
                bias_text = "▲ BULLISH"
            elif avg_pred < -0.5:
                bias_color = tc["red"]
                bias_text = "▼ BEARISH"
            else:
                bias_color = tc["yellow"]
                bias_text = "◆ NEUTRAL"
            
            bias_html = f"""
            <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                    MARKET BIAS
                </div>
                <div style='color:{bias_color};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;margin-top:0.5rem;'>
                    {bias_text}
                </div>
                <div style='color:{tc["muted"]};font-size:0.75rem;margin-top:0.25rem;'>
                    Avg: {avg_pred:+.2f}%
                </div>
            </div>
            """
            st.markdown(bias_html, unsafe_allow_html=True)
        
        # Card 3: Risk Level
        with card_cols[2]:
            if avg_vol < 20:
                risk_color = tc["green"]
                risk_text = "LOW"
            elif avg_vol > 40:
                risk_color = tc["red"]
                risk_text = "HIGH"
            else:
                risk_color = tc["yellow"]
                risk_text = "MODERATE"
            
            risk_html = f"""
            <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                    RISK LEVEL
                </div>
                <div style='color:{risk_color};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;margin-top:0.5rem;'>
                    {risk_text}
                </div>
                <div style='color:{tc["muted"]};font-size:0.75rem;margin-top:0.25rem;'>
                    Vol: {avg_vol:.1f}%
                </div>
            </div>
            """
            st.markdown(risk_html, unsafe_allow_html=True)
        
        # Card 4: Period
        with card_cols[3]:
            period_val = st.session_state.get("period", period).upper()
            period_html = f"""
            <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                    PERIOD
                </div>
                <div style='color:{tc["blue"]};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;margin-top:0.5rem;'>
                    {period_val}
                </div>
                <div style='color:{tc["muted"]};font-size:0.75rem;margin-top:0.25rem;'>
                    {display_horizon_label} forecast
                </div>
            </div>
            """
            st.markdown(period_html, unsafe_allow_html=True)
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # === SIGNALS TABLE ===
        st.markdown('<p class="section-title">Trading Signals</p>', unsafe_allow_html=True)
        
        cand_df = pred_df.copy()

        # === Add new columns for z-score, volatility scaling, ensemble ===
        # Z-score: rolling mean/std of pred_next_ret per ticker (window=60, min 20)
        if "pred_next_ret" in cand_df.columns:
            window = 60
            min_periods = 20
            # If multi-ticker, group by ticker; else, just rolling
            if "ticker" in cand_df.columns:
                cand_df["prediction_zscore"] = cand_df.groupby("ticker")["pred_next_ret"].transform(
                    lambda x: (x - x.rolling(window, min_periods=min_periods).mean()) / x.rolling(window, min_periods=min_periods).std()
                )
            else:
                cand_df["prediction_zscore"] = (cand_df["pred_next_ret"] - cand_df["pred_next_ret"].rolling(window, min_periods=min_periods).mean()) / cand_df["pred_next_ret"].rolling(window, min_periods=min_periods).std()
            cand_df["pred_next_ret_pct"] = cand_df["pred_next_ret"] * 100
            cand_df["abs_pred_pct"] = cand_df["pred_next_ret_pct"].abs()
        else:
            cand_df["abs_pred_pct"] = np.nan

        # Volatility-scaled position sizing (if available)
        if "vol_scaled_quantity" not in cand_df.columns:
            # If not present, try to compute from pred_next_ret and vol_20d
            if "pred_next_ret" in cand_df.columns and "vol_20d" in cand_df.columns:
                base_qty = 100  # Example base quantity
                cand_df["base_quantity"] = base_qty
                # scale = edge/vol, capped at 3x
                cand_df["vol_scale_factor"] = (cand_df["pred_next_ret"].abs() / cand_df["vol_20d"]).clip(upper=3.0)
                cand_df["vol_scaled_quantity"] = (cand_df["base_quantity"] * cand_df["vol_scale_factor"]).round(0)
            else:
                cand_df["base_quantity"] = np.nan
                cand_df["vol_scale_factor"] = np.nan
                cand_df["vol_scaled_quantity"] = np.nan

        # Ensemble output (if present)
        if "ensemble_prediction" not in cand_df.columns:
            cand_df["ensemble_prediction"] = cand_df.get("ensemble_prediction", np.nan)
        if "ensemble_confidence" not in cand_df.columns and "ensemble_confidence" in pred_df.columns:
            cand_df["ensemble_confidence"] = pred_df["ensemble_confidence"]

        # Apply filters
        mask = pd.Series(True, index=cand_df.index)
        if "atm_iv" in cand_df.columns:
            mask &= cand_df["atm_iv"].between(min_iv, max_iv)
        if "abs_pred_pct" in cand_df.columns:
            mask &= (cand_df["abs_pred_pct"] >= min_move)

        cand_df = cand_df[mask].copy()

        sort_col = "abs_pred_pct" if "abs_pred_pct" in cand_df.columns else "pred_next_ret"
        if sort_col in cand_df.columns:
            cand_df = cand_df.sort_values(sort_col, ascending=False)

        # Always use ALL tickers for analysis, not just filtered ones
        detail_universe = pred_df["ticker"].tolist()

        # === Build columns for display (ensure Z-SCORE always present) ===
        cols = [
            "ticker", "pred_next_ret_pct", "prediction_zscore", "prob_up", "vol_20d",
            "base_quantity", "vol_scale_factor", "vol_scaled_quantity",
            "ensemble_prediction", "ensemble_confidence",
            "atm_iv", "pred_next_price", "num_features", "prob_up_gaf",
        ]
        # Always include Z-SCORE if possible
        if "prediction_zscore" not in cand_df.columns:
            cand_df["prediction_zscore"] = np.nan
        cols = [c for c in cols if c in cand_df.columns]

        # Style the dataframe with better column names
        styled_df = cand_df[cols].copy()
        styled_df = styled_df.rename(columns={
            "ticker": "SYMBOL",
            "pred_next_ret_pct": "PRED %",
            "prediction_zscore": "Z-SCORE",
            "prob_up": "P(UP)",
            "vol_20d": "VOL",
            "base_quantity": "BASE QTY",
            "vol_scale_factor": "VOL SCALE",
            "vol_scaled_quantity": "VOL QTY",
            "ensemble_prediction": "ENSEMBLE",
            "ensemble_confidence": "ENS CONF",
            "atm_iv": "IV",
            "pred_next_price": "TARGET",
            "num_features": "FEAT",
            "prob_up_gaf": "GAF"
        })

        # Custom styled HTML table for signals (with Z-score and weak signal highlighting)
        def render_signals_table(df):
            is_dark = st.session_state.get("theme", "dark") == "dark"
            if is_dark:
                colors = {
                    "header_bg": "#161b22", "header_border": "#388bfd", "header_text": "#58a6ff",
                    "cell_bg": "#0d1117", "cell_bg_alt": "#0f141a", "cell_border": "#21262d",
                    "cell_text": "#f0f6fc", "hover_bg": "#21262d", "border": "#30363d",
                    "ticker": "#388bfd", "bullish": "#3fb950", "bearish": "#f85149", "weak": "#d29922"
                }
            else:
                colors = {
                    "header_bg": "#f6f8fa", "header_border": "#0969da", "header_text": "#0969da",
                    "cell_bg": "#ffffff", "cell_bg_alt": "#f9fafb", "cell_border": "#d8dee4",
                    "cell_text": "#1f2328", "hover_bg": "#e5e7eb", "border": "#d0d7de",
                    "ticker": "#0969da", "bullish": "#1a7f37", "bearish": "#cf222e", "weak": "#9a6700"
                }
            html = f"""
            <style>
                .signals-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: 'JetBrains Mono', 'SF Mono', monospace;
                    font-size: 0.85rem;
                    margin: 1rem 0;
                }}
                .signals-table th {{
                    background: {colors['header_bg']};
                    border-bottom: 2px solid {colors['header_border']};
                    color: {colors['header_text']};
                    font-weight: 600;
                    letter-spacing: 0.5px;
                    padding: 12px 16px;
                    text-align: left;
                    text-transform: uppercase;
                    font-size: 0.7rem;
                }}
                .signals-table td {{
                    background: {colors['cell_bg']};
                    border-bottom: 1px solid {colors['cell_border']};
                    color: {colors['cell_text']};
                    padding: 10px 16px;
                }}
                .signals-table tr:nth-child(even) td {{
                    background: {colors['cell_bg_alt']};
                }}
                .signals-table tr:hover td {{
                    background: {colors['hover_bg']};
                }}
                .signals-table .ticker-cell {{
                    font-weight: 700;
                    color: {colors['ticker']};
                }}
                .signals-table .bullish {{
                    color: {colors['bullish']};
                    font-weight: 600;
                }}
                .signals-table .bearish {{
                    color: {colors['bearish']};
                    font-weight: 600;
                }}
                .signals-table .weak {{
                    color: {colors['weak']};
                    font-weight: 600;
                }}
            </style>
            <div style="overflow-x: auto; border: 1px solid {colors['border']}; border-radius: 8px; max-height: 350px; overflow-y: auto;">
            <table class="signals-table">
                <thead><tr>
            """
            for col in df.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"
            for _, row in df.iterrows():
                html += "<tr>"
                for col in df.columns:
                    val = row[col]
                    cell_class = ""
                    if col == "SYMBOL":
                        cell_class = "ticker-cell"
                        html += f'<td class="{cell_class}">{val}</td>'
                    elif col == "PRED %":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            cell_class = "bullish" if val > 0 else "bearish"
                            html += f'<td class="{cell_class}">{val:+.2f}%</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "Z-SCORE":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            # Highlight weak signals (|z| < 0.5)
                            if abs(val) < 0.5:
                                cell_class = "weak"
                            elif val > 1:
                                cell_class = "bullish"
                            elif val < -1:
                                cell_class = "bearish"
                            html += f'<td class="{cell_class}">{val:+.2f}</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "P(UP)":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            cell_class = "bullish" if val > 0.55 else "bearish" if val < 0.45 else ""
                            html += f'<td class="{cell_class}">{val:.0%}</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "VOL":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            html += f'<td>{val*100:.1f}%</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "BASE QTY":
                        html += f'<td>{int(val) if not pd.isna(val) else "—"}</td>'
                    elif col == "VOL SCALE":
                        html += f'<td>{val:.2f}' + ('' if pd.isna(val) else 'x') + '</td>' if not pd.isna(val) else '<td>—</td>'
                    elif col == "VOL QTY":
                        html += f'<td>{int(val) if not pd.isna(val) else "—"}</td>'
                    elif col == "ENSEMBLE":
                        html += f'<td>{val:+.2f}' + ('%' if not pd.isna(val) else '') + '</td>' if not pd.isna(val) else '<td>—</td>'
                    elif col == "ENS CONF":
                        html += f'<td>{val:.0%}</td>' if not pd.isna(val) else '<td>—</td>'
                    elif col == "IV":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            html += f'<td>{val*100:.1f}%</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "TARGET":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            html += f'<td>${val:.2f}</td>'
                        else:
                            html += f'<td>—</td>'
                    elif col == "FEAT":
                        html += f'<td>{int(val) if not pd.isna(val) else "—"}</td>'
                    elif col == "GAF":
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            cell_class = "bullish" if val > 0.55 else "bearish" if val < 0.45 else ""
                            html += f'<td class="{cell_class}">{val:.0%}</td>'
                        else:
                            html += f'<td>—</td>'
                    else:
                        html += f'<td>{val if not pd.isna(val) else "—"}</td>'
                html += "</tr>"
            html += "</tbody></table></div>"
            return html

        st.markdown(render_signals_table(styled_df), unsafe_allow_html=True)
        # detail_universe already set above to include ALL tickers

        # === P&L and Risk Summary Table (theme-aware, styled) ===
        st.markdown('<p class="section-title">P&L and Risk Summary</p>', unsafe_allow_html=True)
        pnl_cols = [
            "ticker", "realized_pnl", "simulated_pnl", "max_drawdown", "sharpe_ratio", "win_rate", "position_size"
        ]
        pnl_cols = [c for c in pnl_cols if c in pred_df.columns]
        if pnl_cols:
            pnl_df = pred_df[pnl_cols].copy()
            pnl_df = pnl_df.rename(columns={
                "ticker": "SYMBOL",
                "realized_pnl": "REALIZED P&L",
                "simulated_pnl": "SIM P&L",
                "max_drawdown": "MAX DD",
                "sharpe_ratio": "SHARPE",
                "win_rate": "WIN %",
                "position_size": "POS SIZE"
            })
            def render_pnl_table(df):
                tc = get_card_theme()
                is_dark = st.session_state.get("theme", "dark") == "dark"
                html = f"""
                <style>
                    .pnl-table {{
                        width: 100%;
                        border-collapse: collapse;
                        font-family: 'JetBrains Mono', 'SF Mono', monospace;
                        font-size: 0.85rem;
                        margin: 1rem 0;
                    }}
                    .pnl-table th {{
                        background: {tc['bg']};
                        border-bottom: 2px solid {tc['blue']};
                        color: {tc['blue']};
                        font-weight: 600;
                        letter-spacing: 0.5px;
                        padding: 12px 16px;
                        text-align: left;
                        text-transform: uppercase;
                        font-size: 0.7rem;
                    }}
                    .pnl-table td {{
                        background: {tc['bg']};
                        border-bottom: 1px solid {tc['border']};
                        color: {tc['text']};
                        padding: 10px 16px;
                    }}
                    .pnl-table tr:nth-child(even) td {{
                        background: {tc['bar_bg']};
                    }}
                    .pnl-table tr:hover td {{
                        background: {tc['muted']}22;
                    }}
                    .pnl-table .positive {{ color: {tc['green']}; }}
                    .pnl-table .negative {{ color: {tc['red']}; }}
                </style>
                <div style="overflow-x: auto; border: 1px solid {tc['border']}; border-radius: 8px;">
                <table class="pnl-table">
                    <thead><tr>
                """
                for col in df.columns:
                    html += f"<th>{col}</th>"
                html += "</tr></thead><tbody>"
                for _, row in df.iterrows():
                    html += "<tr>"
                    for col in df.columns:
                        val = row[col]
                        cell_class = ""
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            if col in ["REALIZED P&L", "SIM P&L", "SHARPE"]:
                                cell_class = "positive" if val > 0 else "negative" if val < 0 else ""
                                val = f"{val:.2f}"
                            elif col == "MAX DD":
                                cell_class = "negative" if val < 0 else ""
                                val = f"{val:.2f}"
                            elif col == "WIN %":
                                val = f"{val:.1f}%"
                            elif col == "POS SIZE":
                                val = f"{val:.0f}"
                        elif pd.isna(val):
                            val = "—"
                        html += f'<td class="{cell_class}">{val}</td>'
                    html += "</tr>"
                html += "</tbody></table></div>"
                return html
            st.markdown(render_pnl_table(pnl_df), unsafe_allow_html=True)
        else:
            st.info("No P&L or risk summary data available in current predictions.")

        # === Ensemble as selectable model type ===
        # Add 'Ensemble' to model_type selectbox if not present
        if 'Ensemble' not in st.session_state.get('model_type_options', []):
            # Patch model_type selectbox to include Ensemble
            # (Assumes model_type selectbox is defined above as model_type = st.sidebar.selectbox(...))
            model_type_options = ["rf", "gbrt", "xgb", "ensemble"]
            st.session_state['model_type_options'] = model_type_options
        # If user selects 'ensemble', core logic should already handle weighted/voting output
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # === INTERACTIVE CHARTS SECTION ===
        st.markdown('<p class="section-title">Signal Analytics</p>', unsafe_allow_html=True)
        
        # Only show charts if we have prediction data
        if pred_df is not None and not pred_df.empty and "pred_next_ret" in pred_df.columns:
            
            chart_cols = st.columns(2)
            
            # Chart 1: Prediction Distribution (Histogram)
            with chart_cols[0]:
                st.markdown("##### 📊 Prediction Distribution")
                
                pred_values = pred_df["pred_next_ret"].dropna() * 100
                
                fig_hist = go.Figure()
                
                # Separate positive and negative predictions
                pos_preds = pred_values[pred_values >= 0]
                neg_preds = pred_values[pred_values < 0]
                
                if len(neg_preds) > 0:
                    fig_hist.add_trace(go.Histogram(
                        x=neg_preds,
                        name="Bearish",
                        marker_color="#f85149",
                        opacity=0.8,
                        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra>Bearish</extra>"
                    ))
                
                if len(pos_preds) > 0:
                    fig_hist.add_trace(go.Histogram(
                        x=pos_preds,
                        name="Bullish",
                        marker_color="#3fb950",
                        opacity=0.8,
                        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra>Bullish</extra>"
                    ))
                
                fig_hist.add_vline(x=0, line_dash="dash", line_color=THEME["text_secondary"], line_width=1)
                fig_hist.add_vline(x=pred_values.mean(), line_dash="dot", line_color=THEME["accent_blue"], line_width=2,
                                   annotation_text=f"Avg: {pred_values.mean():.2f}%", annotation_position="top")
                
                fig_hist.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME["text_secondary"], size=10),
                    xaxis=dict(
                        title="Predicted Return (%)",
                        gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        title="Count",
                        gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10, color=THEME["text_primary"])
                    ),
                    barmode='overlay',
                    bargap=0.1,
                )
                st.plotly_chart(fig_hist, width="stretch")
            
            # Chart 2: Prediction vs Probability Scatter
            with chart_cols[1]:
                st.markdown("##### 🎯 Prediction vs Confidence")
                
                if "prob_up" in pred_df.columns:
                    scatter_df = pred_df[["ticker", "pred_next_ret", "prob_up"]].dropna()
                    
                    fig_scatter = go.Figure()
                    
                    # Color based on prediction direction
                    colors = ["#3fb950" if p > 0 else "#f85149" for p in scatter_df["pred_next_ret"]]
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=scatter_df["prob_up"] * 100,
                        y=scatter_df["pred_next_ret"] * 100,
                        mode="markers+text",
                        marker=dict(
                            size=12,
                            color=colors,
                            line=dict(width=1, color='#21262d')
                        ),
                        text=scatter_df["ticker"],
                        textposition="top center",
                        textfont=dict(size=9, color=THEME["text_primary"]),
                        hovertemplate="<b>%{text}</b><br>P(Up): %{x:.0f}%<br>Pred: %{y:+.2f}%<extra></extra>"
                    ))
                    
                    # Add quadrant lines
                    fig_scatter.add_hline(y=0, line_dash="dash", line_color=THEME["text_muted"], line_width=1)
                    fig_scatter.add_vline(x=50, line_dash="dash", line_color=THEME["text_muted"], line_width=1)
                    
                    fig_scatter.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=40),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=THEME["text_secondary"], size=10),
                        xaxis=dict(
                            title="P(Up) %",
                            gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                            range=[0, 100],
                        ),
                        yaxis=dict(
                            title="Predicted Return (%)",
                            gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                        ),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_scatter, width="stretch")
                else:
                    st.info("P(Up) data not available for scatter plot")
            
            # Second row of charts
            chart_cols2 = st.columns(2)
            
            # Chart 3: Ranked Bar Chart of Predictions
            with chart_cols2[0]:
                st.markdown("##### 📈 Prediction Ranking")
                
                ranked_df = pred_df[["ticker", "pred_next_ret"]].dropna().copy()
                ranked_df["pred_pct"] = ranked_df["pred_next_ret"] * 100
                ranked_df = ranked_df.sort_values("pred_pct", ascending=True)
                
                fig_bar = go.Figure()
                
                colors = ["#3fb950" if p > 0 else "#f85149" for p in ranked_df["pred_pct"]]
                
                fig_bar.add_trace(go.Bar(
                    x=ranked_df["pred_pct"],
                    y=ranked_df["ticker"],
                    orientation='h',
                    marker_color=colors,
                    hovertemplate="<b>%{y}</b><br>Prediction: %{x:+.2f}%<extra></extra>"
                ))
                
                fig_bar.add_vline(x=0, line_color=THEME["text_muted"], line_width=1)
                
                fig_bar.update_layout(
                    height=max(200, len(ranked_df) * 28),
                    margin=dict(l=0, r=0, t=10, b=40),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=THEME["text_secondary"], size=10),
                    xaxis=dict(
                        title="Predicted Return (%)",
                        gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                        zeroline=False,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10, color=THEME["text_primary"]),
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig_bar, width="stretch")
            
            # Chart 4: Volatility vs IV Scatter (if available)
            with chart_cols2[1]:
                if "vol_20d" in pred_df.columns and "atm_iv" in pred_df.columns:
                    st.markdown("##### ⚡ Volatility Analysis")
                    
                    vol_df = pred_df[["ticker", "vol_20d", "atm_iv", "pred_next_ret"]].dropna()
                    
                    if not vol_df.empty:
                        fig_vol_scatter = go.Figure()
                        
                        # Size based on absolute prediction
                        sizes = (vol_df["pred_next_ret"].abs() * 100 + 5).clip(upper=25)
                        colors = ["#3fb950" if p > 0 else "#f85149" for p in vol_df["pred_next_ret"]]
                        
                        fig_vol_scatter.add_trace(go.Scatter(
                            x=vol_df["vol_20d"] * 100,
                            y=vol_df["atm_iv"] * 100,
                            mode="markers+text",
                            marker=dict(
                                size=sizes,
                                color=colors,
                                line=dict(width=1, color='#21262d'),
                                opacity=0.8
                            ),
                            text=vol_df["ticker"],
                            textposition="top center",
                            textfont=dict(size=9, color=THEME["text_primary"]),
                            hovertemplate="<b>%{text}</b><br>RV: %{x:.1f}%<br>IV: %{y:.1f}%<br>Pred: " + 
                                          vol_df["pred_next_ret"].apply(lambda x: f"{x*100:+.2f}%").tolist()[0] + "<extra></extra>"
                        ))
                        
                        # Add diagonal line (IV = RV)
                        max_val = max(vol_df["vol_20d"].max(), vol_df["atm_iv"].max()) * 100 * 1.1
                        fig_vol_scatter.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode="lines",
                            line=dict(color=THEME["text_muted"], dash="dash", width=1),
                            name="IV = RV",
                            hoverinfo="skip"
                        ))
                        
                        fig_vol_scatter.update_layout(
                            height=280,
                            margin=dict(l=0, r=0, t=10, b=40),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=THEME["text_secondary"], size=10),
                            xaxis=dict(
                                title="Realized Vol (20D) %",
                                gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                            ),
                            yaxis=dict(
                                title="Implied Vol %",
                                gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                            ),
                            showlegend=False,
                        )
                        
                        # Add annotation for IV premium zone
                        fig_vol_scatter.add_annotation(
                            x=max_val * 0.3,
                            y=max_val * 0.7,
                            text="IV Rich ↑",
                            showarrow=False,
                            font=dict(size=10, color="#d29922"),
                            opacity=0.7
                        )
                        fig_vol_scatter.add_annotation(
                            x=max_val * 0.7,
                            y=max_val * 0.3,
                            text="IV Cheap ↓",
                            showarrow=False,
                            font=dict(size=10, color="#a371f7"),
                            opacity=0.7
                        )
                        
                        st.plotly_chart(fig_vol_scatter, width="stretch")
                    else:
                        st.info("Insufficient volatility data")
                else:
                    # Fallback: P(Up) Distribution
                    st.markdown("##### 🎲 Confidence Distribution")
                    
                    if "prob_up" in pred_df.columns:
                        prob_values = pred_df["prob_up"].dropna() * 100
                        
                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Histogram(
                            x=prob_values,
                            nbinsx=10,
                            marker_color="#388bfd",
                            opacity=0.8,
                            hovertemplate="P(Up): %{x:.0f}%<br>Count: %{y}<extra></extra>"
                        ))
                        
                        fig_prob.add_vline(x=50, line_dash="dash", line_color=THEME["text_muted"], line_width=1)
                        fig_prob.add_vline(x=prob_values.mean(), line_dash="dot", line_color=THEME["accent_green"], line_width=2,
                                           annotation_text=f"Avg: {prob_values.mean():.0f}%", annotation_position="top")
                        
                        fig_prob.update_layout(
                            height=280,
                            margin=dict(l=0, r=0, t=10, b=40),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color=THEME["text_secondary"], size=10),
                            xaxis=dict(
                                title="P(Up) %",
                                gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                                range=[0, 100],
                            ),
                            yaxis=dict(
                                title="Count",
                                gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)',
                            ),
                        )
                        st.plotly_chart(fig_prob, width="stretch")
                    else:
                        st.info("No probability data available")
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # === OPTION STRATEGIES SECTION ===
        st.markdown('<p class="section-title">Option Strategies</p>', unsafe_allow_html=True)
        
        # Generate option strategies from predictions
        try:
            strategy_df = generate_option_strategy(pred_df)
            
            if strategy_df is not None and not strategy_df.empty:
                # Display strategy summary cards
                summary = get_strategy_summary(strategy_df)
                
                strat_cols = st.columns(4)
                # Get theme colors for option strategy cards
                tc = get_card_theme()
                
                with strat_cols[0]:
                    st.markdown(f"""
                    <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                        <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                            TOTAL STRATEGIES
                        </div>
                        <div style='color:{tc["blue"]};font-family:JetBrains Mono,monospace;font-size:1.5rem;font-weight:700;margin-top:0.5rem;'>
                            {summary['total_strategies']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with strat_cols[1]:
                    st.markdown(f"""
                    <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                        <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                            BULLISH / BEARISH
                        </div>
                        <div style='margin-top:0.5rem;'>
                            <span style='color:{tc["green"]};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;'>{summary['bullish_count']}</span>
                            <span style='color:{tc["muted"]};font-size:0.85rem;'> / </span>
                            <span style='color:{tc["red"]};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;'>{summary['bearish_count']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with strat_cols[2]:
                    avg_conf = summary['avg_confidence'] * 100
                    conf_color = tc["green"] if avg_conf >= 60 else tc["yellow"] if avg_conf >= 40 else tc["red"]
                    st.markdown(f"""
                    <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                        <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                            AVG CONFIDENCE
                        </div>
                        <div style='color:{conf_color};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;margin-top:0.5rem;'>
                            {avg_conf:.0f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with strat_cols[3]:
                    total_cost = summary['total_estimated_cost']
                    st.markdown(f"""
                    <div style='background:{tc["bg"]};border:1px solid {tc["border"]};border-radius:8px;padding:1rem;text-align:center;'>
                        <div style='color:{tc["label"]};font-size:0.65rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>
                            EST. TOTAL COST
                        </div>
                        <div style='color:{tc["purple"]};font-family:JetBrains Mono,monospace;font-size:1.25rem;font-weight:700;margin-top:0.5rem;'>
                            ${total_cost:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                
                # Strategy breakdown by type
                if summary['strategy_breakdown']:
                    st.markdown("##### Strategy Breakdown")
                    breakdown_cols = st.columns(len(summary['strategy_breakdown']))
                    for i, (strat_name, count) in enumerate(summary['strategy_breakdown'].items()):
                        with breakdown_cols[i]:
                            # Color based on strategy type
                            if "Call" in strat_name or "Bull" in strat_name:
                                badge_color = tc["green"]
                            elif "Put" in strat_name or "Bear" in strat_name:
                                badge_color = tc["red"]
                            else:
                                badge_color = tc["yellow"]
                            
                            st.markdown(f"""
                            <div style='background:{tc["bar_bg"]};border-radius:6px;padding:0.5rem;text-align:center;'>
                                <span style='color:{badge_color};font-weight:600;font-size:0.8rem;'>{strat_name}</span>
                                <span style='color:{tc["text"]};font-family:JetBrains Mono,monospace;font-size:1rem;margin-left:0.5rem;'>{count}</span>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                
                # Display strategies table
                def render_strategy_table(df):
                    """Render option strategy dataframe as a theme-aware HTML table."""
                    # Get theme colors
                    is_dark = st.session_state.get("theme", "dark") == "dark"
                    if is_dark:
                        colors = {
                            "header_bg": "#161b22", "header_border": "#a371f7", "header_text": "#a371f7",
                            "cell_bg": "#0d1117", "cell_bg_alt": "#0f141a", "cell_border": "#21262d",
                            "cell_text": "#f0f6fc", "hover_bg": "#21262d", "border": "#30363d",
                            "ticker": "#388bfd", "bullish": "#3fb950", "bearish": "#f85149",
                            "neutral": "#d29922", "iv_rich": "#f85149", "iv_cheap": "#3fb950"
                        }
                    else:
                        colors = {
                            "header_bg": "#f6f8fa", "header_border": "#8250df", "header_text": "#8250df",
                            "cell_bg": "#ffffff", "cell_bg_alt": "#f9fafb", "cell_border": "#d8dee4",
                            "cell_text": "#1f2328", "hover_bg": "#e5e7eb", "border": "#d0d7de",
                            "ticker": "#0969da", "bullish": "#1a7f37", "bearish": "#cf222e",
                            "neutral": "#9a6700", "iv_rich": "#cf222e", "iv_cheap": "#1a7f37"
                        }
                    
                    html = f"""
                    <style>
                        .strategy-table {{
                            width: 100%;
                            border-collapse: collapse;
                            font-family: 'JetBrains Mono', 'SF Mono', monospace;
                            font-size: 0.85rem;
                            margin: 0.5rem 0;
                        }}
                        .strategy-table th {{
                            background: {colors['header_bg']};
                            border-bottom: 2px solid {colors['header_border']};
                            color: {colors['header_text']};
                            font-weight: 600;
                            letter-spacing: 0.5px;
                            padding: 12px 14px;
                            text-align: left;
                            text-transform: uppercase;
                            font-size: 0.7rem;
                        }}
                        .strategy-table td {{
                            background: {colors['cell_bg']};
                            border-bottom: 1px solid {colors['cell_border']};
                            color: {colors['cell_text']};
                            padding: 10px 14px;
                        }}
                        .strategy-table tr:nth-child(even) td {{
                            background: {colors['cell_bg_alt']};
                        }}
                        .strategy-table tr:hover td {{
                            background: {colors['hover_bg']};
                        }}
                        .strategy-table .ticker-cell {{
                            font-weight: 700;
                            color: {colors['ticker']};
                        }}
                        .strategy-table .bullish {{
                            color: {colors['bullish']};
                            font-weight: 600;
                        }}
                        .strategy-table .bearish {{
                            color: {colors['bearish']};
                            font-weight: 600;
                        }}
                        .strategy-table .neutral {{
                            color: {colors['neutral']};
                            font-weight: 600;
                        }}
                        .strategy-table .iv-rich {{
                            color: {colors['iv_rich']};
                        }}
                        .strategy-table .iv-cheap {{
                            color: {colors['iv_cheap']};
                        }}
                    </style>
                    <div style="overflow-x: auto; border: 1px solid {colors['border']}; border-radius: 8px; max-height: 350px; overflow-y: auto;">
                    <table class="strategy-table">
                        <thead><tr>
                            <th>SYMBOL</th>
                            <th>STRATEGY</th>
                            <th>STRIKE</th>
                            <th>BIAS</th>
                            <th>EST. COST</th>
                            <th>CONF</th>
                            <th>IV RICH</th>
                        </tr></thead>
                        <tbody>
                    """
                    
                    for _, row in df.iterrows():
                        # Ticker
                        html += f"<tr><td class='ticker-cell'>{row['symbol']}</td>"
                        
                        # Strategy - color based on type
                        action = row.get('recommended_action', 'Hold')
                        if 'Call' in action or 'Bull' in action:
                            html += f"<td class='bullish'>{action}</td>"
                        elif 'Put' in action or 'Bear' in action:
                            html += f"<td class='bearish'>{action}</td>"
                        else:
                            html += f"<td class='neutral'>{action}</td>"
                        
                        # Strike
                        strike = row.get('suggested_strike', 'ATM')
                        html += f"<td>{strike}</td>"
                        
                        # Directional Bias
                        bias = row.get('directional_bias', 'Neutral')
                        if bias == 'Bullish':
                            html += "<td class='bullish'>▲ BULL</td>"
                        elif bias == 'Bearish':
                            html += "<td class='bearish'>▼ BEAR</td>"
                        else:
                            html += "<td class='neutral'>◆ NEUT</td>"
                        
                        # Estimated Cost
                        cost = row.get('estimated_cost', 0)
                        html += f"<td>${cost:.2f}</td>"
                        
                        # Confidence
                        conf = row.get('confidence', 0)
                        conf_pct = conf * 100 if conf <= 1 else conf
                        if conf_pct >= 60:
                            html += f"<td class='bullish'>{conf_pct:.0f}%</td>"
                        elif conf_pct >= 40:
                            html += f"<td class='neutral'>{conf_pct:.0f}%</td>"
                        else:
                            html += f"<td class='bearish'>{conf_pct:.0f}%</td>"
                        
                        # IV Richness
                        iv_rich = row.get('IV_richness', 0)
                        iv_pct = iv_rich * 100 if abs(iv_rich) <= 1 else iv_rich
                        if iv_pct > 10:
                            html += f"<td class='iv-rich'>+{iv_pct:.0f}%</td>"
                        elif iv_pct < -10:
                            html += f"<td class='iv-cheap'>{iv_pct:.0f}%</td>"
                        else:
                            html += f"<td>{iv_pct:+.0f}%</td>"
                        
                        html += "</tr>"
                    
                    html += "</tbody></table></div>"
                    return html
                
                st.markdown(render_strategy_table(strategy_df), unsafe_allow_html=True)
            else:
                st.info("No option strategies could be generated. Ensure predictions include required fields (pred_next_ret, atm_iv, vol_20d).")
        except Exception as e:
            st.warning(f"Could not generate option strategies: {e}")
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # === VALIDATION SECTION ===
        st.markdown('<p class="section-title">Model Validation</p>', unsafe_allow_html=True)
        
        val_cols = st.columns([2, 8])
        with val_cols[0]:
            run_all_acc = st.button("🔬 VALIDATE ALL", key="dash_run_all_acc", type="secondary", use_container_width=True)
        with val_cols[1]:
            st.caption("Run historical accuracy test on all predicted tickers")
        
        valdf = st.session_state.get("dash_validation_table", None)
        
        if run_all_acc:
            if pred_df is None or pred_df.empty:
                st.warning("Run predictions first")
            else:
                val_tickers = pred_df["ticker"].astype(str).str.upper().tolist()
                rows = []
                prog = st.progress(0.0)
                status = st.empty()
                
                for i, tk in enumerate(val_tickers, start=1):
                    status.text(f"Testing {tk} ({i}/{len(val_tickers)})")
                    row = {
                        "SYM": tk,
                        "ACC %": np.nan,
                        "SR B&H": np.nan,
                        "SR SIG": np.nan,
                        "SR NET": np.nan,
                        "DAYS": 0,
                        "ERR": "",
                    }
                    try:
                        results_test, acc = _cached_track_predictions(
                            tk, period=period, model_type=model_type, horizon=display_horizon
                        )
                        if results_test is None or results_test.empty:
                            row["ERR"] = "No data"
                        else:
                            row["ACC %"] = float(acc) * 100.0 if acc else np.nan
                            row["DAYS"] = int(results_test.shape[0])
                            baseline_returns = results_test["actual_return"].dropna()
                            row["SR B&H"] = compute_sharpe(baseline_returns)
                            
                            # Apply latency delay
                            results_exec = apply_latency_delay(
                                results_test,
                                delay_days=exec_model.delay_days,
                                pred_col="predicted_return"
                            )
                            
                            # Apply strategy
                            strat = results_exec.copy()
                            strat["position"] = np.where(
                                strat["predicted_return"] > signal_threshold_pct, 1.0, 0.0
                            )
                            stratret_nocost = strat["actual_return"] * strat["position"]
                            stratret_withcost = apply_costs_on_trades(strat, exec_model)
                            
                            row["SR SIG"] = compute_sharpe(stratret_nocost)
                            row["SR NET"] = compute_sharpe(stratret_withcost)
                    except Exception as e:
                        row["ERR"] = str(e)[:20]
                    
                    rows.append(row)
                    prog.progress(i / max(1, len(val_tickers)))
                
                status.empty()
                prog.empty()
                valdf = pd.DataFrame(rows)
                st.session_state["dash_validation_table"] = valdf
        
        # Show validation table with custom HTML styling
        if valdf is not None and not valdf.empty:
            showdf = valdf.copy()
            if "ACC %" in showdf.columns:
                showdf = showdf.sort_values(by="ACC %", ascending=False, na_position="last")
            
            # Create custom styled HTML table
            def render_dark_table(df):
                """Render a dataframe as a theme-aware HTML table."""
                tc = get_card_theme()
                is_dark = st.session_state.get("theme", "dark") == "dark"
                html = f"""
                <style>
                    .dark-table {{
                        width: 100%;
                        border-collapse: collapse;
                        font-family: 'JetBrains Mono', 'SF Mono', monospace;
                        font-size: 0.85rem;
                        margin: 1rem 0;
                    }}
                    .dark-table th {{
                        background: {tc["bg"]};
                        border-bottom: 2px solid {tc["blue"]};
                        color: {tc["blue"]};
                        font-weight: 600;
                        letter-spacing: 0.5px;
                        padding: 12px 16px;
                        text-align: left;
                        text-transform: uppercase;
                        font-size: 0.7rem;
                    }}
                    .dark-table td {{
                        background: {"#0d1117" if is_dark else "#ffffff"};
                        border-bottom: 1px solid {tc["border"]};
                        color: {tc["text"]};
                        padding: 10px 16px;
                    }}
                    .dark-table tr:nth-child(even) td {{
                        background: {"#0f141a" if is_dark else "#f9fafb"};
                    }}
                    .dark-table tr:hover td {{
                        background: {"#21262d" if is_dark else "#e5e7eb"};
                    }}
                    .dark-table .positive {{
                        color: {tc["green"]};
                    }}
                    .dark-table .negative {{
                        color: {tc["red"]};
                    }}
                </style>
                <div style="overflow-x: auto; border: 1px solid {tc["border"]}; border-radius: 8px;">
                <table class="dark-table">
                    <thead><tr>
                """
                for col in df.columns:
                    html += f"<th>{col}</th>"
                html += "</tr></thead><tbody>"
                
                for _, row in df.iterrows():
                    html += "<tr>"
                    for col in df.columns:
                        val = row[col]
                        cell_class = ""
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            if col in ["SR B&H", "SR SIG", "SR NET"]:
                                cell_class = "positive" if val > 0.5 else "negative" if val < 0 else ""
                                val = f"{val:.4f}"
                            elif col == "ACC %":
                                cell_class = "positive" if val > 55 else "negative" if val < 50 else ""
                                val = f"{val:.1f}%"
                            else:
                                val = f"{val:.2f}" if isinstance(val, float) else str(val)
                        elif pd.isna(val):
                            val = "—"
                        html += f'<td class="{cell_class}">{val}</td>'
                    html += "</tr>"
                
                html += "</tbody></table></div>"
                return html
            
            st.markdown(render_dark_table(showdf), unsafe_allow_html=True)
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        
        # === TICKER ANALYSIS SECTION ===
        st.markdown('<p class="section-title">Ticker Analysis</p>', unsafe_allow_html=True)
        
        if not detail_universe:
            st.markdown("""
            <div class='empty-state'>
                <p style='font-size: 1rem; margin-bottom: 0.5rem;'>No tickers available</p>
                <p style='font-size: 0.8rem;'>Run predictions to analyze individual stocks</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            sorted_universe = sorted(set(detail_universe))
            
            # Ticker selector row
            ticker_cols = st.columns([3, 7])
            with ticker_cols[0]:
                selected = st.selectbox(
                    "Select Ticker", 
                    sorted_universe, 
                    key="dash_selected_ticker"
                )
            
            matching_rows = pred_df[pred_df["ticker"] == selected]
            if matching_rows.empty:
                st.error(f"No data for {selected}")
            else:
                row = matching_rows.iloc[0]
                pred_ret = row.get("pred_next_ret")
                prob_up = row.get("prob_up")
                last_close = row.get("last_close")
                pred_price = row.get("pred_next_price")
                vol_20d = row.get("vol_20d")
                atm_iv = row.get("atm_iv")
                prob_up_gaf = row.get("prob_up_gaf")
                
                # === TICKER HEADER WITH SIGNAL ===
                signal_color = "#3fb950" if pred_ret and pred_ret > 0 else "#f85149" if pred_ret and pred_ret < 0 else "#d29922"
                signal_text = "BULLISH" if pred_ret and pred_ret > 0 else "BEARISH" if pred_ret and pred_ret < 0 else "NEUTRAL"
                signal_icon = "▲" if pred_ret and pred_ret > 0 else "▼" if pred_ret and pred_ret < 0 else "◆"
                
                tc = get_card_theme()
                is_dark = st.session_state.get("theme", "dark") == "dark"
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, {tc["bg"]} 0%, {"#0d1117" if is_dark else "#f3f4f6"} 100%);
                    border: 1px solid {tc["border"]};
                    border-left: 4px solid {signal_color};
                    border-radius: 8px;
                    padding: 1.25rem;
                    margin-bottom: 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                '>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <span style='
                            background: {tc["blue"]};
                            border-radius: 6px;
                            color: white;
                            font-family: JetBrains Mono, monospace;
                            font-size: 1.5rem;
                            font-weight: 700;
                            padding: 0.5rem 1rem;
                        '>{selected}</span>
                        <div>
                            <div style='color: {tc["text"]}; font-size: 1.1rem; font-weight: 600;'>
                                ${_fmt_num(last_close, 2) if last_close else "—"}
                            </div>
                            <div style='color: {tc["label"]}; font-size: 0.75rem;'>Last Close</div>
                        </div>
                    </div>
                    <div style='
                        background: {signal_color}20;
                        border: 1px solid {signal_color};
                        border-radius: 6px;
                        padding: 0.5rem 1rem;
                        text-align: center;
                    '>
                        <div style='color: {signal_color}; font-size: 1.25rem; font-weight: 700;'>
                            {signal_icon} {signal_text}
                        </div>
                        <div style='color: {tc["label"]}; font-size: 0.7rem;'>{display_horizon_label} Outlook</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # === PRIMARY METRICS ROW ===
                st.markdown('<p class="section-title">Prediction Metrics</p>', unsafe_allow_html=True)
                
                m1 = st.columns(6)
                with m1[0]:
                    st.metric(
                        label="PREDICTED RETURN",
                        value=_fmt_pct(pred_ret, 2),
                        delta=f"{display_horizon_label} forecast"
                    )
                with m1[1]:
                    prob_delta = "high confidence" if prob_up and (prob_up > 0.65 or prob_up < 0.35) else "moderate"
                    st.metric(
                        label="P(UP)",
                        value="—" if prob_up is None else f"{float(prob_up):.0%}",
                        delta=prob_delta
                    )
                with m1[2]:
                    price_change = ((pred_price / last_close) - 1) * 100 if pred_price and last_close else 0
                    st.metric(
                        label="TARGET PRICE",
                        value=f"${_fmt_num(pred_price, 2)}",
                        delta=f"{price_change:+.1f}%" if price_change else None
                    )
                with m1[3]:
                    st.metric(
                        label="20D VOLATILITY",
                        value=_fmt_pct(vol_20d, 1),
                        delta="annualized"
                    )
                with m1[4]:
                    if atm_iv:
                        iv_premium = (atm_iv - vol_20d) * 100 if vol_20d else 0
                        st.metric(
                            label="IMPLIED VOL",
                            value=f"{float(atm_iv)*100:.1f}%",
                            delta=f"{iv_premium:+.1f}% premium" if iv_premium else None
                        )
                    else:
                        st.metric(label="IMPLIED VOL", value="—", delta="no data")
                with m1[5]:
                    if prob_up_gaf is not None:
                        st.metric(
                            label="GAF-CNN P(UP)",
                            value=f"{float(prob_up_gaf):.0%}",
                            delta="image model"
                        )
                    else:
                        st.metric(label="GAF-CNN P(UP)", value="—", delta="not run")
                
                # === SECONDARY METRICS ROW ===
                st.markdown('<p class="section-title">Additional Data</p>', unsafe_allow_html=True)
                
                m2 = st.columns(5)
                with m2[0]:
                    num_features = row.get("num_features")
                    st.metric(
                        label="FEATURES USED",
                        value=int(num_features) if num_features else "—",
                        delta="after selection"
                    )
                with m2[1]:
                    confidence = row.get("confidence_score")
                    if confidence:
                        conf_label = "high" if confidence > 0.7 else "low" if confidence < 0.4 else "moderate"
                        st.metric(
                            label="CONFIDENCE",
                            value=f"{float(confidence):.0%}",
                            delta=conf_label
                        )
                    else:
                        st.metric(label="CONFIDENCE", value="—", delta="no data")
                with m2[2]:
                    put_call = row.get("put_call_oi_ratio")
                    if put_call:
                        pc_label = "bearish bias" if put_call > 1.2 else "bullish bias" if put_call < 0.8 else "neutral"
                        st.metric(
                            label="PUT/CALL OI",
                            value=f"{float(put_call):.2f}",
                            delta=pc_label
                        )
                    else:
                        st.metric(label="PUT/CALL OI", value="—", delta="no data")
                with m2[3]:
                    iv_minus_rv = row.get("iv_minus_realized")
                    if iv_minus_rv is not None:
                        iv_label = "IV rich" if iv_minus_rv > 0.05 else "IV cheap" if iv_minus_rv < -0.05 else "fair"
                        st.metric(
                            label="IV - REALIZED",
                            value=f"{float(iv_minus_rv)*100:+.1f}%",
                            delta=iv_label
                        )
                    else:
                        st.metric(label="IV - REALIZED", value="—", delta="no data")
                with m2[4]:
                    theo_call = row.get("theo_atm_call_price")
                    if theo_call:
                        st.metric(
                            label="THEO ATM CALL",
                            value=f"${float(theo_call):.2f}",
                            delta="Black-Scholes"
                        )
                    else:
                        st.metric(label="THEO ATM CALL", value="—", delta="no data")
                
                # === PRICE CHART ===
                st.markdown('<p class="section-title" style="margin-top: 1.5rem;">Price Chart</p>', unsafe_allow_html=True)
                
                hist = _cached_price_history(selected, period="3mo")
                if hist is None or hist.empty or "Close" not in hist.columns:
                    st.info("No price data available")
                else:
                    close = hist["Close"].dropna()
                    last_dt = close.index[-1]
                    pred_price = row.get("pred_next_price")
                    
                    fig = go.Figure()
                    
                    # Price line
                    fig.add_trace(go.Scatter(
                        x=close.index, 
                        y=close.values, 
                        name="Price", 
                        mode="lines",
                        line=dict(color="#388bfd", width=2),
                        hovertemplate="$%{y:.2f}<extra></extra>"
                    ))
                    
                    # Prediction marker
                    if pred_price is not None:
                        future_dt = last_dt + pd.Timedelta(days=int(display_horizon))
                        pred_color = "#3fb950" if pred_price > float(close.iloc[-1]) else "#f85149"
                        fig.add_trace(go.Scatter(
                            x=[last_dt, future_dt],
                            y=[float(close.iloc[-1]), pred_price],
                            name="Forecast", 
                            mode="lines+markers",
                            line=dict(color=pred_color, dash="dash", width=2),
                            marker=dict(size=10, symbol="diamond"),
                            hovertemplate="$%{y:.2f}<extra>Forecast</extra>"
                        ))
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(
                            showgrid=False, 
                            zeroline=False,
                            color=THEME["text_secondary"]
                        ),
                        yaxis=dict(
                            showgrid=True, 
                            gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', 
                            zeroline=False,
                            color=THEME["text_secondary"],
                            tickprefix="$"
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", color=THEME["text_secondary"], size=11),
                        showlegend=True,
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="left",
                            x=0,
                            font=dict(size=10)
                        ),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, width="stretch")
                
                # === OPTIONS & RISK DETAILS ===
                detail_cols = st.columns(2)
                
                with detail_cols[0]:
                    with st.expander("📊 Options Data", expanded=False):
                        strat_suggestion, _bias = suggest_options_strategy(
                            pred_ret=float(row.get("pred_next_ret") or 0.0),
                            put_call_ratio=row.get("put_call_oi_ratio"),
                            atm_iv=row.get("atm_iv"),
                            horizon=int(display_horizon),
                        )
                        
                        o1, o2 = st.columns(2)
                        with o1:
                            st.metric("ATM IV", _fmt_num(row.get("atm_iv"), 3))
                            st.metric("Put/Call OI", _fmt_num(row.get("put_call_oi_ratio"), 3))
                        with o2:
                            st.metric("IV - Realized", _fmt_num(row.get("iv_minus_realized"), 3))
                            st.metric("Theo ATM Call", f"${_fmt_num(row.get('theo_atm_call_price'), 2)}")
                        
                        st.markdown(f"**Suggested Strategy:** `{strat_suggestion}`")
                
                with detail_cols[1]:
                    with st.expander("📈 30D Outlook", expanded=False):
                        lh_result = _cached_long_horizon_prediction(selected, period=st.session_state.get("period", "5y"))
                        
                        if "error" in lh_result:
                            st.info(f"Unavailable: {lh_result.get('error', '')[:50]}")
                        else:
                            lh1, lh2 = st.columns(2)
                            with lh1:
                                st.metric("P(Up) 30D", f"{float(lh_result.get('p_up_30d', 0.5)) * 100:.0f}%")
                                st.metric("P10 Return", _fmt_pct(lh_result.get("ret_p10_30d"), 1))
                            with lh2:
                                st.metric("P50 Return", _fmt_pct(lh_result.get("ret_p50_30d"), 1))
                                st.metric("P90 Return", _fmt_pct(lh_result.get("ret_p90_30d"), 1))
                
                # === RISK DASHBOARD ===
                with st.expander("📉 Risk Dashboard", expanded=False):
                    hist_full = _cached_price_history(selected, period=period)
                    if hist_full is None or hist_full.empty or "Close" not in hist_full.columns:
                        st.info("No history available to compute risk metrics.")
                    else:
                        returns = hist_full["Close"].pct_change().dropna()
                        risk_summary = summarize_risk(returns)
                        risk = prepare_risk_timeseries(returns)
                        
                        dd_df = risk.get("drawdown", pd.DataFrame())
                        sharpe_60 = risk.get("sharpe_60", pd.Series())
                        sortino_60 = risk.get("sortino_60", pd.Series())
                        vol_20 = risk.get("vol_20", pd.Series())
                        var95 = risk.get("var95", pd.Series())
                        es95 = risk.get("es95", pd.Series())
                        
                        # Risk badge
                        tc = get_card_theme()
                        risk_label = risk_summary.get("label", "unknown")
                        badge_color = {"low": tc["green"], "medium": tc["yellow"], "high": tc["red"]}.get(risk_label, tc["label"])
                        st.markdown(
                            f"<div style='padding:12px;border-radius:6px;background:{badge_color}15;border:1px solid {badge_color};margin-bottom:1rem;'>"
                            f"<strong style='color:{badge_color};'>Risk: {risk_label.upper()}</strong> — {risk_summary.get('summary','')}<br>"
                            f"<span style='color:{tc['label']};font-size:0.8rem;'>{risk_summary.get('check','')}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        
                        # 1. Drawdown chart
                        st.markdown("##### 📉 Drawdown History")
                        if not dd_df.empty and "drawdown" in dd_df.columns:
                            max_dd = dd_df["drawdown"].min()
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=dd_df.index, y=dd_df["drawdown"],
                                name="Drawdown", fill="tozeroy", mode="lines",
                                line=dict(color="#f85149", width=1.5),
                                fillcolor="rgba(248,81,73,0.2)"
                            ))
                            fig_dd.update_layout(
                                height=220,
                                title=dict(text=f"Max Drawdown: {max_dd:.1%}", font=dict(size=11, color=THEME["text_primary"])),
                                margin=dict(l=0, r=0, t=30, b=0),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=THEME["text_secondary"], size=10),
                                yaxis=dict(gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', tickformat='.0%', title=""),
                                xaxis=dict(showgrid=False),
                            )
                            st.plotly_chart(fig_dd, width="stretch")
                        else:
                            st.info("Drawdown data unavailable")
                        
                        # 2. Rolling Sharpe & Sortino (two columns)
                        risk_cols = st.columns(2)
                        
                        with risk_cols[0]:
                            st.markdown("##### 📈 Rolling Sharpe (60D)")
                            if not sharpe_60.empty:
                                fig_sharpe = go.Figure()
                                fig_sharpe.add_trace(go.Scatter(
                                    x=sharpe_60.index, y=sharpe_60.values,
                                    name="Sharpe 60D", mode="lines",
                                    line=dict(color="#388bfd", width=1.5)
                                ))
                                fig_sharpe.add_hline(y=0, line_dash="dash", line_color=THEME["text_muted"], line_width=1)
                                fig_sharpe.add_hline(y=1, line_dash="dot", line_color=THEME["accent_green"], line_width=1)
                                fig_sharpe.add_hline(y=-1, line_dash="dot", line_color=THEME["accent_red"], line_width=1)
                                fig_sharpe.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=THEME["text_secondary"], size=10),
                                    yaxis=dict(gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', title=""),
                                    xaxis=dict(showgrid=False),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_sharpe, width="stretch")
                            else:
                                st.info("Sharpe data unavailable")
                        
                        with risk_cols[1]:
                            st.markdown("##### 📈 Rolling Sortino (60D)")
                            if not sortino_60.empty:
                                fig_sortino = go.Figure()
                                fig_sortino.add_trace(go.Scatter(
                                    x=sortino_60.index, y=sortino_60.values,
                                    name="Sortino 60D", mode="lines",
                                    line=dict(color="#a371f7", width=1.5)
                                ))
                                fig_sortino.add_hline(y=0, line_dash="dash", line_color=THEME["text_muted"], line_width=1)
                                fig_sortino.add_hline(y=1, line_dash="dot", line_color=THEME["accent_green"], line_width=1)
                                fig_sortino.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=THEME["text_secondary"], size=10),
                                    yaxis=dict(gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', title=""),
                                    xaxis=dict(showgrid=False),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_sortino, width="stretch")
                            else:
                                st.info("Sortino data unavailable")
                        
                        # 3. Rolling Volatility
                        st.markdown("##### 📊 Rolling Volatility (20D Annualized)")
                        if not vol_20.empty:
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Scatter(
                                x=vol_20.index, y=vol_20.values,
                                name="Volatility 20D", mode="lines", fill="tozeroy",
                                line=dict(color="#d29922", width=1.5),
                                fillcolor="rgba(210,153,34,0.15)"
                            ))
                            # Add average line
                            avg_vol = vol_20.mean()
                            fig_vol.add_hline(y=avg_vol, line_dash="dash", line_color=THEME["text_primary"], line_width=1,
                                             annotation_text=f"Avg: {avg_vol:.1%}", annotation_position="right")
                            fig_vol.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=10, b=0),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=THEME["text_secondary"], size=10),
                                yaxis=dict(gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', tickformat='.0%', title=""),
                                xaxis=dict(showgrid=False),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_vol, width="stretch")
                        else:
                            st.info("Volatility data unavailable")
                        
                        # 4. VaR & Expected Shortfall (combined chart)
                        st.markdown("##### ⚠️ Value-at-Risk & Expected Shortfall (95%, 250D Rolling)")
                        if not var95.empty and not es95.empty:
                            fig_var = go.Figure()
                            fig_var.add_trace(go.Scatter(
                                x=var95.index, y=var95.values,
                                name="VaR 95%", mode="lines",
                                line=dict(color="#f85149", width=1.5)
                            ))
                            fig_var.add_trace(go.Scatter(
                                x=es95.index, y=es95.values,
                                name="ES 95%", mode="lines",
                                line=dict(color="#ff7b72", width=1.5, dash="dot")
                            ))
                            fig_var.update_layout(
                                height=200,
                                margin=dict(l=0, r=0, t=10, b=30),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=THEME["text_secondary"], size=10),
                                yaxis=dict(gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', tickformat='.1%', title=""),
                                xaxis=dict(showgrid=False),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.15,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(size=10, color=THEME["text_primary"])
                                ),
                            )
                            st.plotly_chart(fig_var, width="stretch")
                        else:
                            st.info("VaR/ES data unavailable")
                
                # === NEWS & HEADLINES ===
                with st.expander("📰 News & Headlines", expanded=False):
                    try:
                        news = get_news_for_ticker(selected, limit=5)
                        if news:
                            # Check for significant news
                            if detect_big_news(news):
                                st.warning("⚠️ **Recent significant news detected!** Consider increased volatility.")
                            
                            tc = get_card_theme()
                            st.markdown(f'<p class="section-title">Recent Headlines</p>', unsafe_allow_html=True)
                            
                            for i, art in enumerate(news, 1):
                                title = art.get("title", "No title")
                                source = art.get("source", "")
                                published = art.get("published_utc", art.get("published", ""))
                                url = art.get("url", art.get("article_url", ""))
                                sentiment = art.get("sentiment_score", art.get("sentiment", 0))
                                
                                # Format published date
                                if published:
                                    try:
                                        pub_dt = pd.to_datetime(published)
                                        pub_str = pub_dt.strftime("%b %d, %Y")
                                    except:
                                        pub_str = str(published)[:10]
                                else:
                                    pub_str = ""
                                
                                # Sentiment indicator
                                if sentiment and isinstance(sentiment, (int, float)):
                                    if sentiment > 0.3:
                                        sent_icon = "🟢"
                                        sent_label = "Positive"
                                    elif sentiment < -0.3:
                                        sent_icon = "🔴"
                                        sent_label = "Negative"
                                    else:
                                        sent_icon = "⚪"
                                        sent_label = "Neutral"
                                else:
                                    sent_icon = "⚪"
                                    sent_label = ""
                                
                                # Display article
                                st.markdown(f"""
                                <div style='padding: 0.75rem; margin-bottom: 0.5rem; border-radius: 6px; 
                                            background: {tc["bg"]}; border: 1px solid {tc["border"]};'>
                                    <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                                        <div style='flex: 1;'>
                                            <a href='{url}' target='_blank' style='color: {tc["accent"]}; text-decoration: none; font-weight: 500;'>
                                                {title}
                                            </a>
                                            <div style='font-size: 0.75rem; color: {tc["label"]}; margin-top: 0.25rem;'>
                                                {source} • {pub_str}
                                            </div>
                                        </div>
                                        <div style='font-size: 0.8rem; margin-left: 0.5rem;' title='{sent_label}'>
                                            {sent_icon}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("📰 No recent news available for this ticker. News API may not be configured.")
                    except Exception as e:
                        st.info(f"📰 News unavailable: API not configured or rate limited.")
                
                # === SIGNALS OUTPUT ===
                with st.expander("📋 Signals & Output", expanded=False):
                    st.markdown(f"**Signals File:** `{SIGNALS_OUT_PATH}`")
                    
                    if st.session_state.get("last_signals"):
                        sig_rows = []
                        strong_count = 0
                        weak_count = 0
                        for tk, s in st.session_state.last_signals.items():
                            if isinstance(s, dict):
                                z_score = float(s.get("z_score", 0.0))
                                z_passes = s.get("z_score_passes", True)
                                z_strength = s.get("z_score_strength", "unknown")
                                
                                if z_passes:
                                    strong_count += 1
                                    status = "✅ Strong"
                                else:
                                    weak_count += 1
                                    status = "⚠️ Weak"
                                
                                # Check trade limit status
                                trade_allowed = s.get("trade_allowed", True)
                                trade_rank = s.get("trade_rank", 0)
                                skip_reason = s.get("skip_reason", "")
                                
                                # Check regime blocked status
                                regime_blocked = s.get("regime_blocked", False)
                                regime_block_reason = s.get("regime_block_reason", "")
                                
                                if regime_blocked:
                                    status = "🚫 Regime"
                                elif not trade_allowed:
                                    status = "⏭️ Limit"
                                
                                sig_rows.append({
                                    "Symbol": tk,
                                    "Type": s.get("asset", "stock").upper(),
                                    "Action": s.get("action", s.get("strategy", "—")),
                                    "Pred %": round(float(s.get("pred_next_ret", 0.0)) * 100, 2),
                                    "Z-Score": round(z_score, 2),
                                    "Rank": trade_rank if trade_rank > 0 else "—",
                                    "Status": status,
                                })
                        
                        # Count skipped signals
                        skipped_count = sum(1 for r in sig_rows if r.get("Status") == "⏭️ Limit")
                        regime_blocked_count = sum(1 for r in sig_rows if r.get("Status") == "🚫 Regime")
                        allowed_count = len(sig_rows) - skipped_count - regime_blocked_count
                        
                        # Get regime status from first signal (same for all)
                        regime_info = None
                        for tk, s in st.session_state.last_signals.items():
                            if isinstance(s, dict) and "regime" in s:
                                regime_info = {
                                    "regime": s.get("regime", "neutral"),
                                    "spy_vs_200dma_pct": s.get("spy_vs_200dma_pct", 0.0),
                                    "vix_level": s.get("vix_level", 20.0),
                                    "longs_allowed": s.get("regime_longs_allowed", True),
                                    "shorts_allowed": s.get("regime_shorts_allowed", True),
                                }
                                break
                        
                        # Show regime status banner if available
                        if regime_info:
                            regime_name = regime_info["regime"].upper().replace("_", " ")
                            spy_pct = regime_info["spy_vs_200dma_pct"]
                            vix = regime_info["vix_level"]
                            longs_ok = "✅" if regime_info["longs_allowed"] else "❌"
                            shorts_ok = "✅" if regime_info["shorts_allowed"] else "❌"
                            
                            # Color based on regime
                            if "bull" in regime_info["regime"].lower():
                                regime_color = "🟢"
                            elif "bear" in regime_info["regime"].lower() or "crash" in regime_info["regime"].lower():
                                regime_color = "🔴"
                            else:
                                regime_color = "🟡"
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(90deg, rgba(100,100,100,0.1), transparent); 
                                        padding: 10px 15px; border-radius: 8px; margin-bottom: 15px;">
                                <strong>{regime_color} Market Regime: {regime_name}</strong> &nbsp;|&nbsp;
                                SPY vs 200DMA: <strong>{spy_pct:+.1f}%</strong> &nbsp;|&nbsp;
                                VIX: <strong>{vix:.1f}</strong> &nbsp;|&nbsp;
                                Longs: {longs_ok} &nbsp; Shorts: {shorts_ok}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Signals", len(sig_rows))
                        with col2:
                            st.metric("Allowed", allowed_count)
                        with col3:
                            st.metric("Strong (|z|≥1)", strong_count)
                        with col4:
                            if skipped_count > 0:
                                st.metric("Skipped (Limit)", skipped_count, delta=f"-{skipped_count}")
                            else:
                                st.metric("Skipped (Limit)", 0)
                        with col5:
                            if regime_blocked_count > 0:
                                st.metric("Blocked (Regime)", regime_blocked_count, delta=f"-{regime_blocked_count}")
                            else:
                                st.metric("Blocked (Regime)", 0)
                        
                        if sig_rows:
                            sig_df = pd.DataFrame(sig_rows)
                            sig_table = render_styled_table(
                                sig_df,
                                highlight_cols={
                                    "Symbol": {"type": "ticker"},
                                    "Pred %": {"type": "pct_direction"},
                                    "Z-Score": {"type": "pct_direction"},
                                }
                            )
                            st.markdown(sig_table, unsafe_allow_html=True)
                            
                            # Show warnings
                            if weak_count > 0:
                                st.warning(f"⚠️ {weak_count} weak signal(s) included. These have |z-score| < 1.0 and may be less reliable.")
                            if skipped_count > 0:
                                st.info(f"⏭️ {skipped_count} signal(s) skipped due to trade limits. Only top-ranked signals per ticker are allowed.")
                            if regime_blocked_count > 0:
                                st.error(f"🚫 {regime_blocked_count} signal(s) blocked by market regime filter. Trade direction not allowed in current regime.")
                    else:
                        st.info("No signals generated yet. Run predictions first.")
                    
                    if st.session_state.get("last_trader_rc") is not None:
                        rc = st.session_state.last_trader_rc
                        if rc == 0:
                            st.success("✅ Trade execution completed successfully")
                        else:
                            st.error(f"❌ Trade execution failed (code {rc})")
                        if st.session_state.get("last_trader_stdout"):
                            with st.expander("Trader Output", expanded=False):
                                st.code(st.session_state.last_trader_stdout[:1000], language="text")
                
                # === FULL DATA ===
                with st.expander("📋 Full Predictions Data", expanded=False):
                    pred_table = render_styled_table(pred_df)
                    st.markdown(pred_table, unsafe_allow_html=True)
    
    else:
        # Empty state - no predictions yet
        tc = get_card_theme()
        st.markdown(f"""
        <div class='empty-state'>
            <p style='font-size: 1.2rem; margin-bottom: 0.5rem; color: {tc["text"]};'>No Predictions Yet</p>
            <p style='font-size: 0.85rem; color: {tc["label"]};'>Enter tickers in the sidebar and click <strong>RUN PREDICTIONS</strong> to get started</p>
        </div>
        """, unsafe_allow_html=True)
    
    # === Z-SCORE WEAK SIGNALS LOG ===
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    with st.expander("📉 Z-Score Weak Signals Log", expanded=False):
        st.markdown("""
        **What is this?**  
        Signals with |z-score| < threshold are considered "weak" - the prediction is not statistically 
        unusual relative to recent history. These signals are logged here for analysis.
        """)
        
        try:
            from src.config import get_zscore_log_path, ZSCORE_GATING_CONFIG
            log_path = get_zscore_log_path()
            
            threshold = ZSCORE_GATING_CONFIG.get("min_zscore", 1.0)
            st.info(f"Current threshold: |z-score| ≥ {threshold}")
            
            if log_path.exists():
                import json
                weak_logs = []
                with open(log_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                weak_logs.append(json.loads(line))
                            except Exception:
                                pass
                
                if weak_logs:
                    # Take last 100
                    recent = weak_logs[-100:]
                    weak_df = pd.DataFrame(recent)
                    
                    # Format for display
                    if "timestamp" in weak_df.columns:
                        weak_df["timestamp"] = pd.to_datetime(weak_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
                    
                    display_cols = [c for c in ["timestamp", "ticker", "pred", "z_score", "threshold", "strength"] if c in weak_df.columns]
                    
                    st.dataframe(weak_df[display_cols].tail(50), use_container_width=True)
                    st.caption(f"Showing last 50 of {len(weak_logs)} weak signals logged")
                    
                    # Clear log button
                    if st.button("🗑️ Clear Weak Signals Log", key="clear_weak_log"):
                        log_path.unlink()
                        st.success("Log cleared!")
                        st.rerun()
                else:
                    st.success("✅ No weak signals logged yet!")
            else:
                st.info("No weak signals log file found. Signals will be logged as predictions are made.")
        except Exception as e:
            st.warning(f"Could not load weak signals log: {e}")
    
    # === OPTIONAL: Prediction History Explorer ===
    if HAS_DATABASE:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        with st.expander("📚 Prediction History Explorer", expanded=False):
            hist_cols = st.columns([2, 2, 2, 2])
            with hist_cols[0]:
                hist_ticker = st.text_input("Filter by Ticker", value="", placeholder="e.g., AAPL", key="hist_ticker")
            with hist_cols[1]:
                hist_model = st.selectbox("Filter by Model", ["All", "rf", "xgb", "gbrt"], index=0, key="hist_model")
            with hist_cols[2]:
                hist_days = st.selectbox("Time Range", [7, 14, 30, 90], index=2, format_func=lambda x: f"Last {x} days", key="hist_days")
            with hist_cols[3]:
                hist_limit = st.selectbox("Max Results", [25, 50, 100, 250], index=1, key="hist_limit")
            
            try:
                history_df = get_prediction_history(
                    ticker=hist_ticker if hist_ticker else None,
                    model_type=hist_model if hist_model != "All" else None,
                    days_back=hist_days,
                    limit=hist_limit
                )
                
                if not history_df.empty:
                    # Format for display
                    display_cols = ["timestamp", "ticker", "model_type", "horizon", "pred_next_ret", "prob_up", "confidence_score", "was_correct"]
                    display_df = history_df[[c for c in display_cols if c in history_df.columns]].copy()
                    
                    if "pred_next_ret" in display_df.columns:
                        display_df["pred_next_ret"] = display_df["pred_next_ret"].apply(lambda x: f"{x*100:.2f}%" if x else "—")
                    if "prob_up" in display_df.columns:
                        display_df["prob_up"] = display_df["prob_up"].apply(lambda x: f"{x*100:.0f}%" if x else "—")
                    if "confidence_score" in display_df.columns:
                        display_df["confidence_score"] = display_df["confidence_score"].apply(lambda x: f"{x*100:.0f}%" if x else "—")
                    if "was_correct" in display_df.columns:
                        display_df["was_correct"] = display_df["was_correct"].apply(lambda x: "✅" if x == 1 else "❌" if x == 0 else "⏳")
                    
                    display_df.columns = ["Time", "Ticker", "Model", "Horizon", "Pred %", "P(Up)", "Conf", "Correct"]
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Accuracy summary
                    accuracy_stats = get_accuracy_stats(
                        ticker=hist_ticker if hist_ticker else None,
                        model_type=hist_model if hist_model != "All" else None,
                        days_back=hist_days
                    )
                    
                    if accuracy_stats.get("total", 0) > 0:
                        acc_cols = st.columns(3)
                        with acc_cols[0]:
                            st.metric("Total Predictions", accuracy_stats["total"])
                        with acc_cols[1]:
                            acc = accuracy_stats.get("accuracy")
                            st.metric("Accuracy", f"{acc*100:.1f}%" if acc else "—")
                        with acc_cols[2]:
                            st.metric("Data Points", len(history_df))
                else:
                    st.info("No prediction history found for the selected filters.")
                    
            except Exception as e:
                st.warning(f"Could not load history: {e}")
    
    # === OPTIONAL: Live Updates Auto-Refresh ===
    if HAS_LIVE_UPDATES and st.session_state.get("live_enabled", False):
        # Check if we should auto-refresh
        auto_refresh_check()

# ============================================================================
# TAB: Backtest - Single Stock Historical Performance
# ============================================================================
with tab_backtests:
    st.markdown('<p class="section-title">Single Stock Backtest</p>', unsafe_allow_html=True)
    
    # Input row
    input_cols = st.columns([2, 2, 2, 2, 2])
    with input_cols[0]:
        bt_ticker = st.text_input("Symbol", "NVDA", key="backtest_ticker")
    with input_cols[1]:
        bt_model = st.selectbox("Model", ["rf", "xgb", "gbrt"], index=0, key="bt_model",
                                format_func=lambda x: {"rf": "Random Forest", "xgb": "XGBoost", "gbrt": "Gradient Boost"}[x])
    with input_cols[2]:
        bt_horizon = st.selectbox("Horizon", [1, 2, 3, 4, 5], index=4, key="bt_horizon",
                                  format_func=lambda x: f"{x} Day")
    with input_cols[3]:
        bt_period = st.selectbox("Period", ["2y", "5y", "10y"], index=1, key="bt_period")
    with input_cols[4]:
        run_backtest_btn = st.button("🔬 RUN BACKTEST", key="run_backtest", type="primary", use_container_width=True)
    
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
    if run_backtest_btn:
        # Validate backtest inputs
        is_valid, validation_errors = validate_backtest_inputs(bt_ticker, bt_period, bt_horizon, bt_model)
        
        if not is_valid:
            display_validation_errors(validation_errors)
        else:
            # Clean the ticker
            bt_ticker_clean = bt_ticker.strip().upper()
            
            # Use safe API call wrapper
            result_data, api_error = safe_api_call(
                _cached_track_predictions,
                bt_ticker_clean, 
                period=bt_period, 
                model_type=bt_model, 
                horizon=bt_horizon,
                error_prefix=f"Backtest failed for {bt_ticker_clean}"
            )
            
            if api_error:
                display_api_error(api_error, bt_ticker_clean)
            elif result_data is None:
                st.warning(f"⚠️ No data returned for {bt_ticker_clean}. The ticker may be invalid or have insufficient history.")
            else:
                results_test, accuracy = result_data
                
                if results_test is None or results_test.empty:
                    st.warning(f"⚠️ Insufficient historical data for backtest on {bt_ticker_clean}. Try a shorter time period or different ticker.")
                else:
                    try:
                        baseline_returns = results_test["actual_return"].dropna()
                        
                        if len(baseline_returns) < 20:
                            st.warning(f"⚠️ Only {len(baseline_returns)} data points available. Results may not be statistically significant.")
                        
                        # Apply strategy
                        strat = results_test.copy()
                        strat["position"] = np.where(strat["predicted_return"] > signal_threshold_pct, 1.0, 0.0)
                        strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]
                        
                        sharpe_baseline = compute_sharpe(baseline_returns)
                        sharpe_strategy = compute_sharpe(strat["strategy_ret_no_cost"])
                        
                        total_return_baseline = (1 + baseline_returns).prod() - 1
                        total_return_strategy = (1 + strat["strategy_ret_no_cost"].dropna()).prod() - 1
                        num_trades = strat["position"].diff().abs().sum() / 2
                        
                        # Results header
                        tc = get_card_theme()
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
                            <span class='ticker-symbol'>{bt_ticker_clean}</span>
                            <span style='color: {tc["label"]}; font-family: JetBrains Mono; font-size: 0.85rem;'>
                                {bt_period} • {bt_horizon}D Horizon • {bt_model.upper()}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics row
                        metric_cols = st.columns(5)
                        with metric_cols[0]:
                            delta = sharpe_strategy - sharpe_baseline if sharpe_strategy and sharpe_baseline else None
                            st.metric("Strategy Sharpe", f"{sharpe_strategy:.2f}" if sharpe_strategy else "—",
                                      delta=f"{delta:+.2f} vs B&H" if delta else None)
                        with metric_cols[1]:
                            st.metric("Hit Rate", f"{accuracy * 100:.0f}%" if accuracy else "—")
                        with metric_cols[2]:
                            st.metric("Total Return", f"{total_return_strategy * 100:+.1f}%")
                        with metric_cols[3]:
                            st.metric("B&H Return", f"{total_return_baseline * 100:+.1f}%")
                        with metric_cols[4]:
                            st.metric("Trades", int(num_trades))
                        
                        # === OPTIONAL: Save backtest results to database ===
                        if HAS_DATABASE:
                            try:
                                save_backtest_result(
                                    ticker=bt_ticker_clean,
                                    model_type=bt_model,
                                    horizon=bt_horizon,
                                    period=bt_period,
                                    metrics={
                                        "accuracy": accuracy,
                                        "sharpe_ratio": sharpe_strategy,
                                        "total_return": total_return_strategy,
                                        "max_drawdown": compute_drawdown(strat["strategy_ret_no_cost"]).get("max_dd") if len(strat["strategy_ret_no_cost"]) > 0 else None,
                                        "win_rate": accuracy,
                                        "n_trades": int(num_trades)
                                    }
                                )
                            except Exception:
                                pass  # Database is optional
                        
                        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                        
                        # Cumulative returns chart
                        st.markdown('<p class="section-title">Cumulative Performance</p>', unsafe_allow_html=True)
                        
                        cum_baseline = (1 + baseline_returns).cumprod()
                        cum_strategy = (1 + strat["strategy_ret_no_cost"].fillna(0)).cumprod()
                        
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=cum_baseline.index, y=cum_baseline.values,
                            name="Buy & Hold", mode="lines", 
                            line=dict(color=THEME["text_secondary"], width=2),
                            hovertemplate="%{y:.2f}x<extra>Buy & Hold</extra>"
                        ))
                        fig_cum.add_trace(go.Scatter(
                            x=cum_strategy.index, y=cum_strategy.values,
                            name="ML Strategy", mode="lines", 
                            line=dict(color=THEME["accent_green"], width=2),
                            hovertemplate="%{y:.2f}x<extra>ML Strategy</extra>"
                        ))
                        fig_cum.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(showgrid=False, color=THEME["text_secondary"]),
                            yaxis=dict(
                                showgrid=True, 
                                gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)', 
                                color=THEME["text_secondary"],
                                title="Growth of $1"
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", color=THEME["text_secondary"], size=11),
                            legend=dict(
                                orientation="h", 
                                yanchor="bottom", 
                                y=1.02, 
                                xanchor="left", 
                                x=0,
                                font=dict(size=11)
                            ),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_cum, width="stretch")
                        
                        # Store results for download
                        st.session_state["backtest_results"] = results_test
                        
                        with st.expander("📋 Detailed Results", expanded=False):
                            backtest_table = render_styled_table(results_test.tail(50).reset_index())
                            st.markdown(backtest_table, unsafe_allow_html=True)
                
                    except Exception as e:
                        # Catch any calculation errors within the backtest results processing
                        display_api_error(f"Error processing backtest results: {str(e)}", bt_ticker_clean)
    else:
        tc = get_card_theme()
        st.markdown(f"""
        <div class='empty-state'>
            <p style='font-size: 1rem; color: {tc["text"]};'>Configure Backtest</p>
            <p style='font-size: 0.85rem; color: {tc["label"]};'>Select a ticker, model, and time period, then click RUN BACKTEST</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB: Portfolio Walk-Forward Analysis
# ============================================================================
with tab_port:
    st.markdown('<p class="section-title">Portfolio Walk-Forward Analysis</p>', unsafe_allow_html=True)
    
    # Configuration panel
    config_cols = st.columns([3, 3, 2, 2])
    
    with config_cols[0]:
        universe_text = st.text_input(
            "Universe", 
            "AAPL, MSFT, NVDA, GOOGL, AMZN", 
            key="port_universe",
            help="Enter tickers separated by commas"
        )
        
        # Quick preset buttons
        preset_cols = st.columns(4)
        with preset_cols[0]:
            if st.button("MAG7", use_container_width=True, key="port_mag7"):
                st.session_state["_port_universe"] = "AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA"
        with preset_cols[1]:
            if st.button("TECH", use_container_width=True, key="port_tech"):
                st.session_state["_port_universe"] = "AAPL, MSFT, NVDA, AMD, CRM, ADBE, INTC"
        with preset_cols[2]:
            if st.button("FANG", use_container_width=True, key="port_fang"):
                st.session_state["_port_universe"] = "META, AMZN, NFLX, GOOGL"
        with preset_cols[3]:
            if st.button("BANKS", use_container_width=True, key="port_banks"):
                st.session_state["_port_universe"] = "JPM, BAC, WFC, GS, MS"
        
        if "_port_universe" in st.session_state and st.session_state["_port_universe"]:
            universe_text = st.session_state["_port_universe"]
            st.session_state["_port_universe"] = None
    
    with config_cols[1]:
        port_model = st.selectbox(
            "Model", 
            ["rf", "xgb", "gbrt"], 
            key="port_model",
            format_func=lambda x: {"rf": "Random Forest", "xgb": "XGBoost", "gbrt": "Gradient Boost"}[x]
        )
        
        preset_option = st.selectbox(
            "Window Preset",
            ["Conservative", "Balanced", "Aggressive"],
            key="port_preset"
        )
        
        presets = {
            "Conservative": {"train": 2.0, "test": 0.15},
            "Balanced": {"train": 1.5, "test": 0.2},
            "Aggressive": {"train": 1.0, "test": 0.1},
        }
        default_train = presets[preset_option]["train"]
        default_test = presets[preset_option]["test"]
    
    with config_cols[2]:
        port_horizon = st.selectbox(
            "Horizon", 
            [1, 3, 5], 
            index=2,
            key="port_horizon",
            format_func=lambda x: f"{x} Day"
        )
        
        with st.expander("Advanced", expanded=False):
            train_years = st.slider("Train years", 0.5, 4.0, float(default_train), 0.25)
            test_years = st.slider("Test years", 0.05, 1.0, float(default_test), 0.05)
    
    with config_cols[3]:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)  # Spacer
        run_wf = st.button("🚀 RUN ANALYSIS", type="primary", key="run_port_wf", use_container_width=True)
    
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    
    if run_wf:
        # Validate and parse tickers
        valid_tickers, ticker_errors = validate_tickers(universe_text)
        
        if ticker_errors and not valid_tickers:
            display_validation_errors(ticker_errors)
        elif not valid_tickers:
            st.error("⚠️ No valid tickers provided. Please enter ticker symbols separated by commas.")
        else:
            # Show warnings for invalid tickers but continue with valid ones
            if ticker_errors:
                with st.expander(f"⚠️ {len(ticker_errors)} ticker(s) skipped (click to see details)", expanded=False):
                    for err in ticker_errors:
                        st.warning(err)
            
            port_tickers = valid_tickers
            
            # Validate other inputs
            is_valid, err = validate_model_type(port_model)
            if not is_valid:
                st.error(err)
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                errors_occurred = []
                
                try:
                    status_text.text(f"Processing {len(port_tickers)} assets...")
                    
                    # Run walk-forward for each ticker using cached version
                    all_folds = []
                    for i, tk in enumerate(port_tickers):
                        status_text.text(f"Processing {tk} ({i+1}/{len(port_tickers)})...")
                        
                        # Use cached walk-forward backtest with safe API call
                        train_yrs = train_years if 'train_years' in dir() else default_train
                        test_yrs = test_years if 'test_years' in dir() else default_test
                        
                        folds, wf_error = safe_api_call(
                            _cached_walk_forward_backtest,
                            ticker=tk,
                            period="5y",
                            horizon=port_horizon,
                            model_type=port_model,
                            train_years=train_yrs,
                            test_years=test_yrs,
                            step_days=21,
                            error_prefix=f"Walk-forward failed for {tk}"
                        )
                        
                        if wf_error:
                            errors_occurred.append((tk, wf_error))
                        elif folds:
                            for f in folds:
                                f["ticker"] = tk
                            all_folds.extend(folds)
                        
                        progress_bar.progress((i + 1) / len(port_tickers))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Show any errors that occurred
                    if errors_occurred:
                        with st.expander(f"⚠️ {len(errors_occurred)} ticker(s) had issues", expanded=False):
                            for tk, err in errors_occurred:
                                st.warning(f"**{tk}**: {err}")
                    
                    if all_folds:
                        results_df = pd.DataFrame(all_folds)
                        st.session_state["port_results"] = results_df
                        success_count = len(port_tickers) - len(errors_occurred)
                        st.success(f"✅ Completed: {len(results_df)} folds across {success_count} assets")
                    else:
                        st.error("❌ No results generated. All tickers failed or had insufficient data.")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    display_api_error(f"Analysis failed: {str(e)}", container=st)
    
    # Display results
    if "port_results" in st.session_state and st.session_state.port_results is not None:
        results_df = st.session_state.port_results
        
        if not results_df.empty:
            # Summary metrics
            median_sharpe = results_df["sharpe"].median() if "sharpe" in results_df.columns else 0
            avg_accuracy = results_df["accuracy"].mean() * 100 if "accuracy" in results_df.columns else 0
            total_folds = len(results_df)
            num_tickers = results_df["ticker"].nunique() if "ticker" in results_df.columns else 0
            
            st.markdown('<p class="section-title">Performance Summary</p>', unsafe_allow_html=True)
            
            # Metrics row
            m_cols = st.columns(5)
            with m_cols[0]:
                st.metric("Median Sharpe", f"{median_sharpe:.2f}")
            with m_cols[1]:
                st.metric("Avg Accuracy", f"{avg_accuracy:.0f}%")
            with m_cols[2]:
                st.metric("Total Folds", total_folds)
            with m_cols[3]:
                st.metric("Assets", num_tickers)
            with m_cols[4]:
                # Model quality badge
                if median_sharpe > 1.2:
                    st.markdown("<span class='status-badge bullish'>EXCELLENT</span>", unsafe_allow_html=True)
                elif median_sharpe > 0.5:
                    st.markdown("<span class='status-badge neutral'>GOOD</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-badge bearish'>WEAK</span>", unsafe_allow_html=True)
            
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            
            # Results in two columns
            result_cols = st.columns([3, 2])
            
            with result_cols[0]:
                st.markdown('<p class="section-title">Fold Results</p>', unsafe_allow_html=True)
                fold_table = render_styled_table(results_df.round(3).reset_index(drop=True))
                st.markdown(f"<div style='max-height: 350px; overflow-y: auto;'>{fold_table}</div>", unsafe_allow_html=True)
            
            with result_cols[1]:
                st.markdown('<p class="section-title">Sharpe Distribution</p>', unsafe_allow_html=True)
                
                if "sharpe" in results_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=results_df["sharpe"],
                        nbinsx=20,
                        marker_color="#388bfd",
                        opacity=0.8
                    ))
                    fig.add_vline(x=median_sharpe, line_dash="dash", line_color="#3fb950",
                                  annotation_text=f"Median: {median_sharpe:.2f}")
                    fig.add_vline(x=0, line_dash="solid", line_color="#f85149",
                                  annotation_text="Breakeven")
                    fig.update_layout(
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(title="Sharpe Ratio", color=THEME["text_secondary"]),
                        yaxis=dict(title="Frequency", color=THEME["text_secondary"], gridcolor='rgba(48,54,61,0.5)' if st.session_state.get("theme", "dark") == "dark" else 'rgba(208,215,222,0.5)'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", color=THEME["text_secondary"], size=11),
                    )
                    st.plotly_chart(fig, width="stretch")
            
            # Export row
            export_cols = st.columns([1, 4])
            with export_cols[0]:
                csv = results_df.round(3).to_csv(index=False)
                st.download_button(
                    "📥 Export CSV",
                    csv,
                    "walkforward_results.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        tc = get_card_theme()
        st.markdown(f"""
        <div class='empty-state'>
            <p style='font-size: 1rem; color: {tc["text"]};'>Walk-Forward Analysis</p>
            <p style='font-size: 0.85rem; color: {tc["label"]};'>Configure your portfolio universe and click RUN ANALYSIS to evaluate model robustness</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style='
    border-top: 1px solid #30363d;
    margin-top: 2rem;
    padding: 1rem 0;
    text-align: center;
'>
    <span style='color: #6e7681; font-size: 0.75rem;'>
        QuantDesk v2.0 | ML-Powered Trading Signals | Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)
