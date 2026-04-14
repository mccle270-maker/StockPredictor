"""
QuantDesk Dashboard - Full Featured Research App
Summary Tab + Dashboard Tab with app_new.py style controls
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from flask_caching import Cache
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Import Project Modules
# =============================================================================
try:
    from src.core.production_predictor import ProductionPredictor, quick_predict, TradingMode
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

try:
    from src.data import fetch_prices
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False

try:
    from prediction_model import build_features_and_target
    LEGACY_MODEL_AVAILABLE = True
except ImportError:
    LEGACY_MODEL_AVAILABLE = False

# =============================================================================
# Initialize Dash App
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    title="QuantDesk"
)

cache = Cache(app.server, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 300})

# =============================================================================
# Presets & Configuration
# =============================================================================
TICKER_PRESETS = {
    "MAG7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "TECH": ["AAPL", "MSFT", "NVDA", "AMD", "CRM", "ADBE", "INTC"],
    "FANG": ["META", "AMZN", "NFLX", "GOOGL"],
    "FINANCIALS": ["JPM", "BAC", "WFC", "GS", "MS"],
    "ENERGY": ["XOM", "CVX", "COP", "SLB"],
}

STRATEGY_PRESETS = {
    "Default": {"zscore": 2.0, "desc": "Balanced approach"},
    "Strict": {"zscore": 2.5, "desc": "Higher quality signals"},
    "Loose": {"zscore": 1.5, "desc": "More signals"},
    "Aggressive": {"zscore": 1.0, "desc": "Maximum signals"},
}

MODEL_OPTIONS = [
    {"label": "Random Forest", "value": "rf"},
    {"label": "XGBoost", "value": "xgb"},
    {"label": "Gradient Boosting", "value": "gbrt"},
    {"label": "Ensemble (RF+XGB)", "value": "ensemble"},
]

TRADING_MODES = {
    "conservative": {"long": 0.45, "short": 0.70, "desc": "Capital preservation", "sharpe": "0.68"},
    "balanced": {"long": 0.42, "short": 0.55, "desc": "Best risk/reward", "sharpe": "1.10"},
    "aggressive": {"long": 0.38, "short": 0.45, "desc": "Maximum returns", "sharpe": "1.17"},
}

# =============================================================================
# Data Functions
# =============================================================================
@cache.memoize(timeout=3600)
def get_price_data(ticker, period="1y"):
    if DATA_AVAILABLE:
        try:
            return fetch_prices(ticker, period=period)
        except:
            pass
    return None

@cache.memoize(timeout=300)
def get_prediction(ticker, mode="balanced", use_adaptive=True):
    """Get prediction using adaptive or legacy model."""
    result = {
        "ticker": ticker,
        "signal": "HOLD",
        "confidence": 0,
        "predicted_return": 0,
        "predicted_price": None,
        "position_size": 0,
        "current_price": None,
        "change_1d": None,
        "volume": None,
        "market_cap": None,
        "pe_ratio": None,
        "warnings": [],
    }
    
    # Get current price info
    df = get_price_data(ticker, "5d")
    if df is not None and not df.empty and "Close" in df.columns:
        result["current_price"] = df["Close"].iloc[-1]
        if len(df) > 1:
            result["change_1d"] = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1)
        if "Volume" in df.columns:
            result["volume"] = df["Volume"].iloc[-1]
    
    # Get prediction
    if use_adaptive and PREDICTOR_AVAILABLE:
        try:
            pred = quick_predict(ticker, mode=mode)
            result["signal"] = pred.signal
            result["confidence"] = pred.confidence
            result["predicted_return"] = pred.predicted_return
            result["predicted_price"] = pred.predicted_price
            result["position_size"] = pred.position_size
            result["warnings"] = getattr(pred, 'warnings', [])
        except Exception as e:
            result["warnings"] = [str(e)]
    
    return result

def get_all_predictions(tickers, mode="balanced", use_adaptive=True):
    return [get_prediction(t, mode, use_adaptive) for t in tickers]

# =============================================================================
# Sidebar Component
# =============================================================================
def create_sidebar():
    return html.Div([
        # Logo
        html.Div([
            html.Div([
                html.Div("QuantDesk", style={"fontSize": "1.1rem", "fontWeight": "700", "color": "#fff"}),
                html.Div("ML Trading Signals", style={"fontSize": "0.6rem", "color": "rgba(255,255,255,0.7)", "letterSpacing": "1px", "textTransform": "uppercase"}),
            ], style={
                "background": "linear-gradient(135deg, #388bfd 0%, #a371f7 100%)",
                "borderRadius": "6px",
                "padding": "12px 16px",
            }),
        ], className="mb-3"),
        
        # Universe Section
        html.Div([
            html.Label("UNIVERSE", className="sidebar-label"),
            dcc.Input(
                id="ticker-input",
                type="text",
                placeholder="AAPL, MSFT, GOOGL...",
                value="AAPL, NVDA, MSFT",
                debounce=True,
                className="sidebar-input",
            ),
            html.Div([
                dbc.Button("MAG7", id="btn-mag7", size="sm", outline=True, color="info", className="preset-btn"),
                dbc.Button("TECH", id="btn-tech", size="sm", outline=True, color="info", className="preset-btn"),
                dbc.Button("FANG", id="btn-fang", size="sm", outline=True, color="info", className="preset-btn"),
            ], className="preset-row"),
        ], className="sidebar-section"),
        
        # Strategy Preset
        html.Div([
            html.Label("TRADING STRATEGY", className="sidebar-label"),
            dbc.Select(
                id="strategy-preset",
                options=[{"label": f"{k} - {v['desc']}", "value": k} for k, v in STRATEGY_PRESETS.items()],
                value="Default",
                className="sidebar-select",
            ),
        ], className="sidebar-section"),
        
        # Quick Settings
        html.Div([
            html.Label("QUICK SETTINGS", className="sidebar-label"),
            dbc.Row([
                dbc.Col([
                    html.Div("Horizon", className="small text-muted"),
                    dbc.Select(
                        id="horizon-select",
                        options=[{"label": f"{h}D", "value": h} for h in [1, 2, 3, 5]],
                        value=1,
                        size="sm",
                        className="sidebar-select-sm",
                    ),
                ], width=6),
                dbc.Col([
                    html.Div("Period", className="small text-muted"),
                    dbc.Select(
                        id="period-select",
                        options=[{"label": p, "value": p} for p in ["1y", "2y", "5y"]],
                        value="2y",
                        size="sm",
                        className="sidebar-select-sm",
                    ),
                ], width=6),
            ]),
        ], className="sidebar-section"),
        
        # Trading Mode (Adaptive Model)
        html.Div([
            html.Label("TRADING MODE", className="sidebar-label"),
            dbc.Checklist(
                id="use-adaptive",
                options=[{"label": " Use Adaptive Model", "value": "adaptive"}],
                value=["adaptive"],
                switch=True,
                className="mb-2",
            ),
            dbc.RadioItems(
                id="trading-mode",
                options=[
                    {"label": f"🛡️ Conservative (Sharpe {TRADING_MODES['conservative']['sharpe']})", "value": "conservative"},
                    {"label": f"⚖️ Balanced (Sharpe {TRADING_MODES['balanced']['sharpe']})", "value": "balanced"},
                    {"label": f"🚀 Aggressive (Sharpe {TRADING_MODES['aggressive']['sharpe']})", "value": "aggressive"},
                ],
                value="balanced",
                className="trading-mode-radio",
            ),
            html.Div(id="mode-info", className="mode-info"),
        ], className="sidebar-section"),
        
        # Advanced Options (Collapsible)
        html.Div([
            dbc.Button("⚙️ Advanced Options", id="btn-advanced", color="link", className="collapse-btn"),
            dbc.Collapse([
                html.Div([
                    html.Label("Model Algorithm", className="small text-muted"),
                    dbc.Select(
                        id="model-type",
                        options=MODEL_OPTIONS,
                        value="rf",
                        size="sm",
                        className="sidebar-select-sm mb-2",
                    ),
                    html.Label("Z-Score Threshold", className="small text-muted"),
                    dcc.Slider(
                        id="zscore-slider",
                        min=0.5, max=3.0, step=0.1, value=2.0,
                        marks={0.5: "0.5", 1.5: "1.5", 2.5: "2.5"},
                        className="mb-3",
                    ),
                    dbc.Checklist(
                        id="advanced-toggles",
                        options=[
                            {"label": " Auto-optimize", "value": "auto_opt"},
                            {"label": " Elastic Net Selection", "value": "elastic"},
                        ],
                        value=["auto_opt"],
                        className="small",
                    ),
                ]),
            ], id="collapse-advanced", is_open=False),
        ], className="sidebar-section"),
        
        # Run Button
        dbc.Button([html.I(className="fas fa-play me-2"), "RUN PREDICTIONS"], 
                  id="btn-run", color="primary", className="w-100 mt-3 mb-3"),
        
        # Cache Management
        html.Div([
            dbc.Button([html.I(className="fas fa-sync-alt me-1"), "Refresh"], id="btn-refresh", 
                      size="sm", color="outline-secondary", className="me-2"),
            dbc.Button([html.I(className="fas fa-trash me-1"), "Clear"], id="btn-clear", 
                      size="sm", color="outline-danger"),
        ], className="d-flex"),
        
        # Status
        html.Div([
            html.Div([
                html.Span("●", style={"color": "#00d4aa" if PREDICTOR_AVAILABLE else "#dc3545"}),
                html.Span(" Adaptive Model", className="ms-1"),
            ], className="status-item"),
            html.Div([
                html.Span("●", style={"color": "#00d4aa" if DATA_AVAILABLE else "#dc3545"}),
                html.Span(" Data Feed", className="ms-1"),
            ], className="status-item"),
        ], className="sidebar-status"),
        
    ], className="sidebar")


# =============================================================================
# Summary Tab - Clean Ticker Cards
# =============================================================================
def create_summary_card(pred):
    """Create a clean summary card for one ticker."""
    ticker = pred.get("ticker", "???")
    signal = pred.get("signal", "HOLD")
    confidence = pred.get("confidence", 0) or 0
    pred_ret = pred.get("predicted_return", 0) or 0
    pred_price = pred.get("predicted_price")
    current_price = pred.get("current_price")
    change_1d = pred.get("change_1d")
    
    # Signal styling
    if signal == "BUY":
        signal_color = "#00d4aa"
        signal_bg = "rgba(0, 212, 170, 0.1)"
        signal_icon = "▲"
    elif signal == "SELL":
        signal_color = "#dc3545"
        signal_bg = "rgba(220, 53, 69, 0.1)"
        signal_icon = "▼"
    else:
        signal_color = "#6c757d"
        signal_bg = "rgba(108, 117, 125, 0.1)"
        signal_icon = "—"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(ticker, className="ticker-symbol"),
                html.Span([signal_icon, " ", signal], style={"color": signal_color, "fontWeight": "600"}),
            ], className="d-flex justify-content-between align-items-center"),
        ], style={"background": signal_bg, "borderBottom": f"2px solid {signal_color}"}),
        dbc.CardBody([
            # Current Price Row
            dbc.Row([
                dbc.Col([
                    html.Div("Current", className="metric-label"),
                    html.Div(f"${current_price:.2f}" if current_price else "—", className="metric-value"),
                ], width=6),
                dbc.Col([
                    html.Div("Change", className="metric-label"),
                    html.Div(
                        f"{change_1d:+.2%}" if change_1d else "—", 
                        className="metric-value",
                        style={"color": "#00d4aa" if change_1d and change_1d > 0 else "#dc3545" if change_1d and change_1d < 0 else "#fff"}
                    ),
                ], width=6),
            ], className="mb-3"),
            
            html.Hr(style={"borderColor": "#333", "margin": "8px 0"}),
            
            # Prediction Row
            dbc.Row([
                dbc.Col([
                    html.Div("Predicted Return", className="metric-label"),
                    html.Div(
                        f"{pred_ret:+.2%}", 
                        className="metric-value-lg",
                        style={"color": "#00d4aa" if pred_ret > 0 else "#dc3545" if pred_ret < 0 else "#fff"}
                    ),
                ], width=6),
                dbc.Col([
                    html.Div("Confidence", className="metric-label"),
                    html.Div(f"{confidence:.0%}", className="metric-value-lg", style={"color": "#58a6ff"}),
                ], width=6),
            ], className="mb-2"),
            
            # Target Price
            html.Div([
                html.Span("Target: ", className="metric-label"),
                html.Span(f"${pred_price:.2f}" if pred_price else "—", style={"color": "#a371f7", "fontFamily": "monospace"}),
            ]),
        ]),
    ], className="summary-card")


def create_summary_tab(predictions):
    """Create the Summary tab content."""
    if not predictions:
        return html.Div("Run predictions to see summary", className="empty-state")
    
    # Portfolio stats
    total = len(predictions)
    buys = len([p for p in predictions if p.get("signal") == "BUY"])
    sells = len([p for p in predictions if p.get("signal") == "SELL"])
    avg_ret = np.mean([p.get("predicted_return", 0) or 0 for p in predictions])
    avg_conf = np.mean([p.get("confidence", 0) or 0 for p in predictions])
    
    # Market bias
    if avg_ret > 0.003:
        bias, bias_color = "BULLISH", "#00d4aa"
    elif avg_ret < -0.003:
        bias, bias_color = "BEARISH", "#dc3545"
    else:
        bias, bias_color = "NEUTRAL", "#ffc107"
    
    return html.Div([
        # Portfolio Overview
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("SIGNALS", className="stat-label"),
                    html.Div([
                        html.Span(f"{buys}", style={"color": "#00d4aa", "fontSize": "1.5rem", "fontWeight": "700"}),
                        html.Span(" BUY ", style={"color": "#6c757d", "fontSize": "0.8rem"}),
                        html.Span(f"{sells}", style={"color": "#dc3545", "fontSize": "1.5rem", "fontWeight": "700"}),
                        html.Span(" SELL", style={"color": "#6c757d", "fontSize": "0.8rem"}),
                    ]),
                ], className="stat-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("MARKET BIAS", className="stat-label"),
                    html.Div(bias, style={"color": bias_color, "fontSize": "1.25rem", "fontWeight": "700"}),
                ], className="stat-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("AVG RETURN", className="stat-label"),
                    html.Div(f"{avg_ret:+.2%}", style={
                        "color": "#00d4aa" if avg_ret > 0 else "#dc3545",
                        "fontSize": "1.25rem", "fontWeight": "700", "fontFamily": "monospace"
                    }),
                ], className="stat-card"),
            ], md=3),
            dbc.Col([
                html.Div([
                    html.Div("AVG CONFIDENCE", className="stat-label"),
                    html.Div(f"{avg_conf:.0%}", style={"color": "#58a6ff", "fontSize": "1.25rem", "fontWeight": "700"}),
                ], className="stat-card"),
            ], md=3),
        ], className="mb-4"),
        
        # Ticker Cards Grid
        html.Div([
            create_summary_card(p) for p in sorted(predictions, key=lambda x: -(x.get("predicted_return") or 0))
        ], className="summary-cards-grid"),
    ])


# =============================================================================
# Dashboard Tab - Full Analysis
# =============================================================================
def create_dashboard_tab(predictions, selected_ticker=None):
    """Create the Dashboard tab content matching app_new.py."""
    if not predictions:
        return html.Div("Run predictions to see dashboard", className="empty-state")
    
    # Default to first ticker
    if not selected_ticker and predictions:
        selected_ticker = predictions[0]["ticker"]
    
    selected_pred = next((p for p in predictions if p["ticker"] == selected_ticker), predictions[0] if predictions else None)
    
    return html.Div([
        dbc.Row([
            # Left Column: Charts
            dbc.Col([
                # Prediction Distribution
                html.Div([
                    html.H6("📊 Prediction Distribution", className="chart-title"),
                    dcc.Graph(id="dist-chart", config={"displayModeBar": False}),
                ], className="chart-container mb-3"),
                
                # Prediction Ranking
                html.Div([
                    html.H6("📈 Prediction Ranking", className="chart-title"),
                    dcc.Graph(id="ranking-chart", config={"displayModeBar": False}),
                ], className="chart-container"),
            ], md=5),
            
            # Right Column: Ticker Analysis
            dbc.Col([
                # Ticker Selector
                html.Div([
                    html.H6("🎯 Ticker Analysis", className="chart-title"),
                    dbc.Select(
                        id="ticker-select",
                        options=[{"label": p["ticker"], "value": p["ticker"]} for p in predictions],
                        value=selected_ticker,
                        className="mb-3",
                    ),
                ]),
                
                # Ticker Detail
                html.Div(id="ticker-detail-content"),
                
                # Price Chart
                html.Div([
                    dcc.Graph(id="price-chart", config={"displayModeBar": False}),
                ], className="chart-container"),
            ], md=7),
        ]),
        
        # Signals Table
        html.Div([
            html.H6("📋 All Signals", className="chart-title mt-4"),
            html.Div(id="signals-table-container"),
        ]),
    ])


def create_ticker_detail(pred, period="1y"):
    """Create detailed ticker analysis panel."""
    if not pred:
        return html.Div()
    
    ticker = pred.get("ticker")
    signal = pred.get("signal", "HOLD")
    confidence = pred.get("confidence", 0) or 0
    pred_ret = pred.get("predicted_return", 0) or 0
    pred_price = pred.get("predicted_price")
    current_price = pred.get("current_price")
    change_1d = pred.get("change_1d")
    warnings = pred.get("warnings", [])
    
    signal_color = "#00d4aa" if signal == "BUY" else "#dc3545" if signal == "SELL" else "#6c757d"
    
    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span(ticker, style={"fontSize": "1.5rem", "fontWeight": "700", "color": "#fff"}),
                html.Span(f"${current_price:.2f}" if current_price else "", 
                         style={"fontSize": "1rem", "color": "#8b949e", "marginLeft": "12px"}),
                html.Span(f"{change_1d:+.2%}" if change_1d else "", 
                         style={"fontSize": "0.85rem", "marginLeft": "8px",
                                "color": "#00d4aa" if change_1d and change_1d > 0 else "#dc3545"}),
            ]),
            html.Div([
                html.Span(f"{'▲' if signal == 'BUY' else '▼' if signal == 'SELL' else '—'} {signal}",
                         style={"color": signal_color, "fontSize": "1rem", "fontWeight": "600",
                                "background": f"{signal_color}20", "padding": "4px 12px", "borderRadius": "4px"}),
            ]),
        ], className="d-flex justify-content-between align-items-center mb-3"),
        
        # Metrics Grid
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("PREDICTED RETURN", className="metric-label"),
                    html.Div(f"{pred_ret:+.2%}", className="metric-value-xl",
                            style={"color": "#00d4aa" if pred_ret > 0 else "#dc3545"}),
                ], className="detail-metric"),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("CONFIDENCE", className="metric-label"),
                    html.Div(f"{confidence:.0%}", className="metric-value-xl", style={"color": "#58a6ff"}),
                ], className="detail-metric"),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("TARGET PRICE", className="metric-label"),
                    html.Div(f"${pred_price:.2f}" if pred_price else "—", 
                            className="metric-value-xl", style={"color": "#a371f7"}),
                ], className="detail-metric"),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("POSITION SIZE", className="metric-label"),
                    html.Div(f"{pred.get('position_size', 0):.0%}", className="metric-value-xl"),
                ], className="detail-metric"),
            ], width=3),
        ], className="mb-3"),
        
        # Warnings
        html.Div([
            html.Span([html.I(className="fas fa-exclamation-triangle me-1"), f"{len(warnings)} warning(s)"],
                     className="text-warning small") if warnings else 
            html.Span([html.I(className="fas fa-check me-1"), "No warnings"], className="text-success small"),
        ]),
    ], className="ticker-detail-panel")


def create_distribution_chart(predictions):
    """Create prediction distribution histogram."""
    if not predictions:
        return go.Figure()
    
    returns = [p.get("predicted_return", 0) or 0 for p in predictions]
    returns_pct = [r * 100 for r in returns]
    
    pos = [r for r in returns_pct if r >= 0]
    neg = [r for r in returns_pct if r < 0]
    
    fig = go.Figure()
    if neg:
        fig.add_trace(go.Histogram(x=neg, name="Bearish", marker_color="#dc3545", opacity=0.8))
    if pos:
        fig.add_trace(go.Histogram(x=pos, name="Bullish", marker_color="#00d4aa", opacity=0.8))
    
    fig.add_vline(x=0, line_dash="dash", line_color="#6c757d")
    if returns_pct:
        fig.add_vline(x=np.mean(returns_pct), line_dash="dot", line_color="#58a6ff", 
                      annotation_text=f"Avg: {np.mean(returns_pct):.2f}%")
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#8b949e", size=10),
        xaxis=dict(title="Predicted Return (%)", gridcolor='rgba(48,54,61,0.5)'),
        yaxis=dict(title="Count", gridcolor='rgba(48,54,61,0.5)'),
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def create_ranking_chart(predictions):
    """Create prediction ranking bar chart."""
    if not predictions:
        return go.Figure()
    
    df = pd.DataFrame(predictions)
    df["pred_pct"] = df["predicted_return"].fillna(0) * 100
    df = df.sort_values("pred_pct", ascending=True)
    
    colors = ["#00d4aa" if p > 0 else "#dc3545" for p in df["pred_pct"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["pred_pct"],
        y=df["ticker"],
        orientation='h',
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>%{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#6c757d")
    
    fig.update_layout(
        height=max(180, len(df) * 30),
        margin=dict(l=0, r=20, t=10, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#8b949e", size=10),
        xaxis=dict(title="Predicted Return (%)", gridcolor='rgba(48,54,61,0.5)'),
        yaxis=dict(tickfont=dict(color="#fff")),
        showlegend=False,
    )
    return fig


def create_price_chart(ticker, period="1y"):
    """Create price chart with moving averages."""
    df = get_price_data(ticker, period)
    
    if df is None or df.empty or "Close" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig
    
    fig = go.Figure()
    
    # Candlestick
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name=ticker, increasing_line_color="#00d4aa", decreasing_line_color="#dc3545",
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=ticker, line=dict(color="#00d4aa")))
    
    # Moving averages
    if len(df) > 20:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), mode="lines",
                                 name="MA20", line=dict(color="#ffc107", width=1)))
    if len(df) > 50:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), mode="lines",
                                 name="MA50", line=dict(color="#58a6ff", width=1)))
    
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        font=dict(color="#8b949e"),
        xaxis=dict(gridcolor='rgba(48,54,61,0.3)'),
        yaxis=dict(gridcolor='rgba(48,54,61,0.3)', tickprefix="$"),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def create_signals_table(predictions):
    """Create signals data table."""
    if not predictions:
        return html.Div()
    
    data = []
    for p in predictions:
        data.append({
            "Ticker": p.get("ticker", ""),
            "Signal": p.get("signal", ""),
            "Confidence": f"{(p.get('confidence') or 0):.0%}",
            "Pred Return": f"{(p.get('predicted_return') or 0):+.2%}",
            "Target": f"${p.get('predicted_price'):.2f}" if p.get('predicted_price') else "—",
            "Current": f"${p.get('current_price'):.2f}" if p.get('current_price') else "—",
            "1D Change": f"{p.get('change_1d'):+.2%}" if p.get('change_1d') else "—",
        })
    
    return dash_table.DataTable(
        data=data,
        columns=[{"name": c, "id": c} for c in ["Ticker", "Signal", "Confidence", "Pred Return", "Target", "Current", "1D Change"]],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#161b22",
            "color": "#58a6ff",
            "fontWeight": "600",
            "textTransform": "uppercase",
            "fontSize": "0.7rem",
            "letterSpacing": "0.5px",
            "padding": "12px",
        },
        style_cell={
            "backgroundColor": "#0d1117",
            "color": "#f0f6fc",
            "padding": "10px 12px",
            "border": "1px solid #21262d",
            "fontFamily": "monospace",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{Signal} = "BUY"'}, "backgroundColor": "rgba(0, 212, 170, 0.1)"},
            {"if": {"filter_query": '{Signal} = "SELL"'}, "backgroundColor": "rgba(220, 53, 69, 0.1)"},
        ],
        sort_action="native",
        page_size=10,
    )


# =============================================================================
# Main Layout
# =============================================================================
app.layout = html.Div([
    # Stores
    dcc.Store(id="predictions-store", data=[]),
    
    # Sidebar
    create_sidebar(),
    
    # Main Content
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span(id="ticker-count", className="header-stat"),
                html.Span(id="model-info", className="header-info"),
            ]),
            html.Span(f"Updated: {datetime.now().strftime('%H:%M')}", className="text-muted small"),
        ], className="main-header"),
        
        # Tabs
        dbc.Tabs([
            dbc.Tab(
                html.Div(id="summary-content", className="tab-content"),
                label="📊 Summary",
                tab_id="tab-summary",
            ),
            dbc.Tab(
                html.Div(id="dashboard-content", className="tab-content"),
                label="🎯 Dashboard", 
                tab_id="tab-dashboard",
            ),
        ], id="main-tabs", active_tab="tab-summary"),
        
    ], className="main-content"),
])


# =============================================================================
# Callbacks
# =============================================================================

# Preset buttons
@callback(Output("ticker-input", "value"), 
          [Input(f"btn-{p.lower()}", "n_clicks") for p in ["mag7", "tech", "fang"]],
          prevent_initial_call=True)
def set_preset(*args):
    triggered = ctx.triggered_id
    presets = {"btn-mag7": "MAG7", "btn-tech": "TECH", "btn-fang": "FANG"}
    if triggered in presets:
        return ", ".join(TICKER_PRESETS[presets[triggered]])
    return dash.no_update

# Toggle advanced options
@callback(Output("collapse-advanced", "is_open"), Input("btn-advanced", "n_clicks"), 
          State("collapse-advanced", "is_open"), prevent_initial_call=True)
def toggle_advanced(n, is_open):
    return not is_open

# Update mode info
@callback(Output("mode-info", "children"), Input("trading-mode", "value"))
def update_mode_info(mode):
    if mode in TRADING_MODES:
        m = TRADING_MODES[mode]
        return f"📊 Long @ {m['long']:.0%} conf | Short @ {m['short']:.0%} conf"
    return ""

# Run predictions
@callback(
    [
        Output("predictions-store", "data"),
        Output("summary-content", "children"),
        Output("dashboard-content", "children"),
        Output("ticker-count", "children"),
        Output("model-info", "children"),
    ],
    Input("btn-run", "n_clicks"),
    [
        State("ticker-input", "value"),
        State("trading-mode", "value"),
        State("use-adaptive", "value"),
        State("model-type", "value"),
        State("period-select", "value"),
        State("horizon-select", "value"),
    ],
    prevent_initial_call=True,
)
def run_predictions(n_clicks, ticker_input, trading_mode, use_adaptive, model_type, period, horizon):
    if not n_clicks:
        return dash.no_update
    
    # Parse tickers
    tickers = [t.strip().upper() for t in ticker_input.replace(",", " ").split() if t.strip()]
    
    if not tickers:
        return [], html.Div("Enter tickers", className="empty-state"), html.Div("Enter tickers", className="empty-state"), "", ""
    
    # Get predictions
    use_adaptive_bool = "adaptive" in use_adaptive if use_adaptive else False
    predictions = get_all_predictions(tickers, trading_mode, use_adaptive_bool)
    
    # Create tab content
    summary_content = create_summary_tab(predictions)
    dashboard_content = create_dashboard_tab(predictions)
    
    # Header info
    ticker_count = f"{len(predictions)} TICKERS"
    mode_label = trading_mode.upper() if use_adaptive_bool else model_type.upper()
    model_info = f"{mode_label} • {period} • {horizon}D"
    
    return predictions, summary_content, dashboard_content, ticker_count, model_info


# Update dashboard charts
@callback(
    [
        Output("dist-chart", "figure"),
        Output("ranking-chart", "figure"),
    ],
    Input("predictions-store", "data"),
)
def update_charts(predictions):
    if not predictions:
        empty = go.Figure()
        empty.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return empty, empty
    return create_distribution_chart(predictions), create_ranking_chart(predictions)


# Ticker selection
@callback(
    [
        Output("ticker-detail-content", "children"),
        Output("price-chart", "figure"),
    ],
    [
        Input("ticker-select", "value"),
        Input("period-select", "value"),
    ],
    State("predictions-store", "data"),
)
def update_ticker_detail(ticker, period, predictions):
    if not ticker or not predictions:
        return html.Div(), go.Figure()
    
    pred = next((p for p in predictions if p["ticker"] == ticker), None)
    detail = create_ticker_detail(pred, period)
    chart = create_price_chart(ticker, period)
    
    return detail, chart


# Signals table
@callback(
    Output("signals-table-container", "children"),
    Input("predictions-store", "data"),
)
def update_signals_table(predictions):
    return create_signals_table(predictions)


# Clear cache
@callback(Output("btn-clear", "children"), Input("btn-clear", "n_clicks"), prevent_initial_call=True)
def clear_cache_btn(n):
    cache.clear()
    return [html.I(className="fas fa-check me-1"), "Cleared!"]


# =============================================================================
# Run Server
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 QuantDesk Dashboard")
    print("=" * 50)
    print(f"Adaptive Model: {'✅' if PREDICTOR_AVAILABLE else '❌'}")
    print(f"Data Feed: {'✅' if DATA_AVAILABLE else '❌'}")
    print("=" * 50)
    print("→ http://localhost:8050")
    print("=" * 50)
    
    app.run(debug=True, port=8050)
