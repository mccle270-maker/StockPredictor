"""
UI Components
=============

Reusable Streamlit widgets for the Stock Predictor dashboard.
All components are pure UI - they receive data and render it.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Any, Optional


def ticker_input_widget(
    key: str = "ticker_input",
    default: str = "AAPL",
    label: str = "Enter Ticker Symbol",
) -> str:
    """
    Ticker symbol input with uppercase normalization.
    """
    return st.text_input(label, value=default, key=key).upper().strip()


def multi_ticker_input_widget(
    key: str = "multi_ticker",
    default: str = "AAPL, MSFT, NVDA",
    label: str = "Enter Tickers (comma-separated)",
) -> list[str]:
    """
    Multi-ticker input with parsing.
    """
    text = st.text_input(label, value=default, key=key)
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def model_selector_widget(
    key: str = "model_type",
    default: str = "rf",
) -> str:
    """
    Model type dropdown.
    
    NOTE: GBRT removed from options (2026-01-07) due to severe overfitting.
    See GBRT_INVESTIGATION_REPORT.md for details.
    """
    options = {"rf": "Random Forest", "xgb": "XGBoost", "xgb_binary": "XGBoost Binary"}
    return st.selectbox(
        "Model Type",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        index=list(options.keys()).index(default) if default in options else 0,
        key=key,
    )


def horizon_selector_widget(
    key: str = "horizon",
    default: int = 1,
    max_horizon: int = 5,
) -> int:
    """
    Prediction horizon slider.
    """
    return st.slider(
        "Prediction Horizon (days)",
        min_value=1,
        max_value=max_horizon,
        value=default,
        key=key,
    )


def period_selector_widget(
    key: str = "period",
    default: str = "5y",
) -> str:
    """
    Historical data period selector.
    """
    options = ["1y", "2y", "3y", "5y", "10y", "max"]
    return st.selectbox(
        "History Period",
        options=options,
        index=options.index(default) if default in options else 3,
        key=key,
    )


def prediction_display(
    prediction: dict[str, Any],
    show_details: bool = True,
) -> None:
    """
    Display prediction results in a formatted layout.
    """
    if not prediction:
        st.warning("No prediction available")
        return
    
    ticker = prediction.get("ticker", "???")
    pred_ret = prediction.get("pred_next_ret", 0)
    pred_price = prediction.get("pred_next_price", 0)
    prob_up = prediction.get("prob_up")
    last_close = prediction.get("last_close", 0)
    horizon = prediction.get("horizon", 1)
    
    # Header
    direction = "🟢" if pred_ret > 0 else "🔴" if pred_ret < 0 else "⚪"
    st.subheader(f"{direction} {ticker} - {horizon}-Day Prediction")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Close", f"${last_close:,.2f}")
    
    with col2:
        st.metric(
            "Predicted Price",
            f"${pred_price:,.2f}",
            delta=f"{pred_ret * 100:+.2f}%",
        )
    
    with col3:
        if prob_up is not None:
            st.metric("Prob Up", f"{prob_up * 100:.1f}%")
        else:
            st.metric("Prob Up", "N/A")
    
    with col4:
        confidence = prediction.get("confidence_score", 0)
        st.metric("Confidence", f"{confidence:.4f}")
    
    if show_details:
        with st.expander("Prediction Details"):
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write(f"**Model:** {prediction.get('model_type', 'N/A')}")
                st.write(f"**Features:** {prediction.get('num_features', 'N/A')}")
                st.write(f"**Vol 20D:** {prediction.get('vol_20d', 0):.4f}")
            
            with details_col2:
                st.write(f"**P/E Ratio:** {prediction.get('pe_ratio', 'N/A')}")
                st.write(f"**ElasticNet:** {'Enabled' if prediction.get('elasticnet_enabled') else 'Disabled'}")
                st.write(f"**Top Features:** {prediction.get('top_features', 'N/A')}")


def backtest_display(
    results: dict[str, Any] | list[dict[str, Any]],
) -> None:
    """
    Display backtest results.
    """
    if isinstance(results, dict):
        results = [results]
    
    if not results:
        st.warning("No backtest results")
        return
    
    # Summary table
    df = pd.DataFrame(results)
    
    if "sharpe" in df.columns:
        df["sharpe"] = df["sharpe"].round(3)
    if "accuracy" in df.columns:
        df["accuracy"] = (df["accuracy"] * 100).round(1).astype(str) + "%"
    if "max_drawdown" in df.columns:
        df["max_drawdown"] = (df["max_drawdown"] * 100).round(1).astype(str) + "%"
    
    st.dataframe(df, use_container_width=True)


def risk_metrics_display(
    returns: pd.Series,
    title: str = "Risk Metrics",
) -> None:
    """
    Display risk metrics for a return series.
    """
    from ..core.metrics import compute_sharpe, compute_sortino, compute_drawdown
    
    if returns is None or returns.empty:
        st.warning("No return data available")
        return
    
    st.subheader(title)
    
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    dd = compute_drawdown(returns)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "N/A")
    
    with col2:
        st.metric("Sortino Ratio", f"{sortino:.3f}" if sortino else "N/A")
    
    with col3:
        max_dd = dd.get("max_drawdown", 0) * 100
        st.metric("Max Drawdown", f"{max_dd:.1f}%")
    
    with col4:
        total_ret = (1 + returns).prod() - 1
        st.metric("Total Return", f"{total_ret * 100:.1f}%")


def price_chart(
    hist: pd.DataFrame,
    title: str = "Price History",
    show_volume: bool = True,
) -> None:
    """
    Display interactive price chart with Plotly.
    """
    if hist is None or hist.empty:
        st.warning("No price data available")
        return
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Price",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def options_chain_display(
    chain: pd.DataFrame,
    title: str = "Options Chain",
) -> None:
    """
    Display options chain table.
    """
    if chain is None or chain.empty:
        st.info("No options data available")
        return
    
    st.subheader(title)
    
    # Format numeric columns
    display_cols = ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]
    available_cols = [c for c in display_cols if c in chain.columns]
    
    if available_cols:
        display_df = chain[available_cols].copy()
        if "impliedVolatility" in display_df.columns:
            display_df["impliedVolatility"] = (display_df["impliedVolatility"] * 100).round(1)
            display_df = display_df.rename(columns={"impliedVolatility": "IV %"})
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(chain, use_container_width=True)


def signal_display(
    signals: dict[str, dict[str, Any]],
    title: str = "Trading Signals",
) -> None:
    """
    Display trading signals in a formatted table.
    """
    if not signals:
        st.info("No signals generated")
        return
    
    st.subheader(title)
    
    rows = []
    for ticker, sig in signals.items():
        row = {
            "Ticker": ticker,
            "Asset": sig.get("asset", "stock"),
            "Action": sig.get("action") or sig.get("strategy", "N/A"),
            "Qty": sig.get("qty", 1),
            "Pred Return": f"{sig.get('pred_next_ret', 0) * 100:.2f}%",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Color coding
    def color_action(val):
        if val in ["BUY", "BUY_CALL", "BULL_CALL_SPREAD"]:
            return "color: green"
        elif val in ["SHORT", "SELL", "BUY_PUT", "BEAR_PUT_SPREAD"]:
            return "color: red"
        return ""
    
    styled = df.style.applymap(color_action, subset=["Action"])
    st.dataframe(styled, use_container_width=True)


def status_indicator(
    success: bool,
    success_msg: str = "Success",
    error_msg: str = "Error",
) -> None:
    """
    Display status indicator.
    """
    if success:
        st.success(success_msg)
    else:
        st.error(error_msg)
