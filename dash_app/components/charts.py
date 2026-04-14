"""
Chart Components for QuantDesk Dashboard

Plotly chart factories for the Dashboard tab.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Common dark theme layout
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#f0f6fc"),
    margin=dict(l=40, r=20, t=30, b=40),
)


def create_prediction_distribution_chart(pred_df: pd.DataFrame, height: int = 300) -> go.Figure:
    """
    Create a histogram of predicted returns.
    
    Args:
        pred_df: DataFrame with prediction data
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if "pred_next_ret" in pred_df.columns:
        returns = pred_df["pred_next_ret"] * 100
        
        # Color bins based on positive/negative
        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in returns]
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=20,
            marker=dict(
                color="#388bfd",
                opacity=0.7,
                line=dict(color="#58a6ff", width=1),
            ),
            name="Predicted Returns",
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
        
        # Add zero reference line
        fig.add_vline(
            x=0, 
            line_dash="dash", 
            line_color="#95a5a6",
            line_width=2,
            annotation_text="0%",
            annotation_position="top",
        )
        
        # Add mean line
        mean_ret = returns.mean()
        fig.add_vline(
            x=mean_ret,
            line_dash="dot",
            line_color="#d29922",
            line_width=2,
            annotation_text=f"Avg: {mean_ret:.2f}%",
            annotation_position="bottom",
        )
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Predicted Return (%)",
        yaxis_title="Count",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
    )
    
    return fig


def create_confidence_scatter_chart(pred_df: pd.DataFrame, height: int = 300) -> go.Figure:
    """
    Create a scatter plot of confidence vs predicted return.
    
    Args:
        pred_df: DataFrame with prediction data
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if "pred_next_ret" in pred_df.columns:
        # Get confidence (may be named differently)
        if "confidence" in pred_df.columns:
            confidence = pred_df["confidence"] * 100
        elif "prob_up" in pred_df.columns:
            # Use prob_up as a proxy for confidence
            confidence = pred_df["prob_up"].apply(lambda x: abs(x - 0.5) * 200)  # 0-100 scale
        else:
            confidence = pd.Series([50] * len(pred_df))
        
        returns = pred_df["pred_next_ret"] * 100
        tickers = pred_df.get("ticker", pd.Series(range(len(pred_df))))
        signals = pred_df.get("signal", pd.Series(["HOLD"] * len(pred_df)))
        
        # Color by signal
        colors = []
        for signal in signals:
            if signal and signal.upper() in ["BUY", "STRONG BUY"]:
                colors.append("#2ecc71")
            elif signal and signal.upper() in ["SELL", "STRONG SELL"]:
                colors.append("#e74c3c")
            else:
                colors.append("#95a5a6")
        
        fig.add_trace(go.Scatter(
            x=confidence,
            y=returns,
            mode="markers+text",
            text=tickers,
            textposition="top center",
            textfont=dict(size=10, color="#c9d1d9"),
            marker=dict(
                size=14,
                color=colors,
                opacity=0.8,
                line=dict(color="white", width=1),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Confidence: %{x:.1f}%<br>"
                "Predicted Return: %{y:.2f}%"
                "<extra></extra>"
            ),
            name="Predictions",
        ))
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="#95a5a6", line_width=1)
        fig.add_vline(x=50, line_dash="dash", line_color="#95a5a6", line_width=1)
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Confidence (%)",
        yaxis_title="Predicted Return (%)",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False, range=[0, 100]),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
    )
    
    return fig


def create_signal_breakdown_chart(predictions: list, height: int = 250) -> go.Figure:
    """
    Create a donut chart showing signal breakdown.
    
    Args:
        predictions: List of prediction dictionaries
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    if not predictions:
        return go.Figure()
    
    # Count signals
    n_buy = sum(1 for p in predictions if p.get("signal", "").upper() in ["BUY", "STRONG BUY"])
    n_sell = sum(1 for p in predictions if p.get("signal", "").upper() in ["SELL", "STRONG SELL"])
    n_hold = len(predictions) - n_buy - n_sell
    
    fig = go.Figure(data=[go.Pie(
        labels=["BUY", "SELL", "HOLD"],
        values=[n_buy, n_sell, n_hold],
        hole=0.5,
        marker=dict(colors=["#2ecc71", "#e74c3c", "#95a5a6"]),
        textinfo="label+value",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        showlegend=False,
        height=height,
        annotations=[dict(
            text=f"{len(predictions)}<br>Total",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font_color="#f0f6fc",
        )],
    )
    
    return fig


def create_ranked_predictions_chart(pred_df: pd.DataFrame, height: int = 400) -> go.Figure:
    """
    Create a horizontal bar chart of ranked predictions.
    
    Args:
        pred_df: DataFrame with prediction data
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if "pred_next_ret" in pred_df.columns and "ticker" in pred_df.columns:
        # Sort by predicted return
        df_sorted = pred_df.sort_values("pred_next_ret", ascending=True)
        
        returns = df_sorted["pred_next_ret"] * 100
        tickers = df_sorted["ticker"]
        
        # Color by positive/negative
        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in returns]
        
        fig.add_trace(go.Bar(
            y=tickers,
            x=returns,
            orientation="h",
            marker=dict(
                color=colors,
                opacity=0.8,
                line=dict(color="white", width=1),
            ),
            text=[f"{r:+.2f}%" for r in returns],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Predicted: %{x:.2f}%<extra></extra>",
        ))
        
        # Add zero line
        fig.add_vline(x=0, line_color="#95a5a6", line_width=2)
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Predicted Return (%)",
        yaxis_title="",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    
    return fig


def create_volatility_chart(pred_df: pd.DataFrame, height: int = 300) -> go.Figure:
    """
    Create a chart comparing predicted return vs volatility.
    
    Args:
        pred_df: DataFrame with prediction data
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if "pred_next_ret" in pred_df.columns:
        vol_col = None
        for col in ["vol_20d", "volatility", "vol", "atm_iv"]:
            if col in pred_df.columns:
                vol_col = col
                break
        
        if vol_col:
            volatility = pred_df[vol_col] * 100
            returns = pred_df["pred_next_ret"] * 100
            tickers = pred_df.get("ticker", pd.Series(range(len(pred_df))))
            
            # Size by absolute return
            sizes = (returns.abs() + 5) * 2
            
            fig.add_trace(go.Scatter(
                x=volatility,
                y=returns,
                mode="markers+text",
                text=tickers,
                textposition="top center",
                textfont=dict(size=9, color="#c9d1d9"),
                marker=dict(
                    size=sizes,
                    color=returns,
                    colorscale="RdYlGn",
                    opacity=0.7,
                    line=dict(color="white", width=1),
                    colorbar=dict(title="Return %", tickformat="+.1f"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Volatility: %{x:.1f}%<br>"
                    "Return: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title="Volatility (%)",
        yaxis_title="Predicted Return (%)",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zeroline=False),
    )
    
    return fig


def create_price_chart(price_df: pd.DataFrame, ticker: str, 
                       prediction: dict = None, height: int = 400) -> go.Figure:
    """
    Create a candlestick price chart with optional prediction overlay.
    
    Args:
        price_df: DataFrame with OHLC price data
        ticker: Ticker symbol
        prediction: Optional prediction dict with target price
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if price_df is not None and not price_df.empty:
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=price_df.index,
            open=price_df["Open"],
            high=price_df["High"],
            low=price_df["Low"],
            close=price_df["Close"],
            name=ticker,
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
        ))
        
        # Add volume as bar chart on secondary y-axis
        if "Volume" in price_df.columns:
            colors = ["#2ecc71" if c >= o else "#e74c3c" 
                     for c, o in zip(price_df["Close"], price_df["Open"])]
            
            fig.add_trace(go.Bar(
                x=price_df.index,
                y=price_df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.3,
                yaxis="y2",
            ))
        
        # Add prediction target line
        if prediction:
            target = prediction.get("pred_next_price", 0)
            if target and target > 0:
                fig.add_hline(
                    y=target,
                    line_dash="dash",
                    line_color="#388bfd",
                    line_width=2,
                    annotation_text=f"Target: ${target:.2f}",
                    annotation_position="right",
                )
    
    fig.update_layout(
        **DARK_LAYOUT,
        title=f"{ticker} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=False,
        height=height,
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,
            range=[0, price_df["Volume"].max() * 4] if "Volume" in price_df.columns else None,
        ),
    )
    
    return fig


def create_backtest_equity_chart(equity_curve: list, dates: list = None, 
                                  height: int = 350) -> go.Figure:
    """
    Create an equity curve chart from backtest results.
    
    Args:
        equity_curve: List of equity values
        dates: Optional list of dates
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if equity_curve:
        x_vals = dates if dates else list(range(len(equity_curve)))
        
        # Calculate returns for coloring
        returns = pd.Series(equity_curve).pct_change().fillna(0)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=equity_curve,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#388bfd", width=2),
            fillcolor="rgba(56, 139, 253, 0.2)",
            name="Equity",
            hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        ))
        
        # Add starting line
        fig.add_hline(
            y=equity_curve[0],
            line_dash="dot",
            line_color="#95a5a6",
            annotation_text=f"Start: ${equity_curve[0]:,.0f}",
        )
    
    fig.update_layout(
        **DARK_LAYOUT,
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickformat="$,.0f"),
    )
    
    return fig


def create_drawdown_chart(equity_curve: list, dates: list = None,
                          height: int = 200) -> go.Figure:
    """
    Create a drawdown chart from equity curve.
    
    Args:
        equity_curve: List of equity values
        dates: Optional list of dates
        height: Chart height in pixels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if equity_curve:
        equity = pd.Series(equity_curve)
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        x_vals = dates if dates else list(range(len(equity_curve)))
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#e74c3c", width=1),
            fillcolor="rgba(231, 76, 60, 0.3)",
            name="Drawdown",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))
        
        # Highlight max drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        fig.add_annotation(
            x=x_vals[max_dd_idx] if dates else max_dd_idx,
            y=max_dd,
            text=f"Max DD: {max_dd:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e74c3c",
            font=dict(color="#e74c3c"),
        )
    
    fig.update_layout(
        **DARK_LAYOUT,
        title="Drawdown",
        xaxis_title="",
        yaxis_title="Drawdown (%)",
        showlegend=False,
        height=height,
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickformat=".1f"),
    )
    
    return fig
