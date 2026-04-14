"""
Sidebar Component for QuantDesk Dashboard
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# Quick ticker presets
TICKER_PRESETS = {
    "MAG7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "TECH": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ORCL", "IBM"],
    "FANG": ["META", "AMZN", "NFLX", "GOOGL"],
    "BANKS": ["JPM", "BAC", "GS", "MS", "C", "WFC"],
    "HEALTH": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY"],
    "ENERGY": ["XOM", "CVX", "COP", "SLB", "EOG"],
}

# Strategy options
STRATEGY_OPTIONS = [
    {"label": "🎯 Conservative - Capital preservation", "value": "conservative"},
    {"label": "⚖️ Balanced - Best risk/reward", "value": "balanced"},
    {"label": "🚀 Aggressive - Maximum returns", "value": "aggressive"},
]

# Horizon options
HORIZON_OPTIONS = [
    {"label": "1 Day", "value": 1},
    {"label": "5 Days", "value": 5},
    {"label": "10 Days", "value": 10},
    {"label": "20 Days", "value": 20},
]

# Model options
MODEL_OPTIONS = [
    {"label": "XGBoost (Recommended)", "value": "xgb"},
    {"label": "Random Forest", "value": "rf"},
    {"label": "Gradient Boosting", "value": "gbrt"},
    {"label": "Ensemble (RF + XGB)", "value": "ensemble"},
]

# Period options
PERIOD_OPTIONS = [
    {"label": "1 Year", "value": "1y"},
    {"label": "2 Years", "value": "2y"},
    {"label": "3 Years", "value": "3y"},
    {"label": "5 Years", "value": "5y"},
]


def create_sidebar():
    """Create the sidebar with all controls."""
    return dbc.Col([
        # Logo/Title
        html.Div([
            html.H2([
                html.I(className="fas fa-chart-line me-2"),
                "QuantDesk"
            ], className="sidebar-title"),
            html.P("Stock Prediction Dashboard", className="sidebar-subtitle"),
        ], className="sidebar-header"),
        
        html.Hr(),
        
        # Ticker Input Section
        html.Div([
            html.Label("Tickers", className="control-label"),
            dbc.Input(
                id="ticker-input",
                type="text",
                placeholder="AAPL, NVDA, MSFT...",
                value="AAPL, NVDA, MSFT",
                className="ticker-input",
            ),
            # Quick presets
            html.Div([
                dbc.Button(
                    name, 
                    id={"type": "preset-btn", "name": name}, 
                    size="sm", 
                    color="secondary", 
                    className="preset-btn me-1 mb-1"
                )
                for name in TICKER_PRESETS.keys()
            ], className="preset-buttons mt-2"),
        ], className="control-group"),
        
        html.Hr(className="my-3"),
        
        # Strategy Dropdown
        html.Div([
            html.Label("Strategy", className="control-label"),
            dcc.Dropdown(
                id="strategy-dropdown",
                options=STRATEGY_OPTIONS,
                value="balanced",
                clearable=False,
                className="strategy-dropdown",
            ),
        ], className="control-group"),
        
        # Horizon Dropdown
        html.Div([
            html.Label("Horizon", className="control-label"),
            dcc.Dropdown(
                id="horizon-dropdown",
                options=HORIZON_OPTIONS,
                value=5,
                clearable=False,
            ),
        ], className="control-group"),
        
        html.Hr(className="my-3"),
        
        # Advanced Options (Collapsed by default)
        dbc.Button(
            [html.I(className="fas fa-cog me-2"), "Advanced Options"],
            id="advanced-toggle",
            color="secondary",
            outline=True,
            className="w-100 mb-2",
        ),
        dbc.Collapse([
            html.Div([
                html.Label("Model Type", className="control-label-sm"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=MODEL_OPTIONS,
                    value="xgb",
                    clearable=False,
                    className="dropdown-sm",
                ),
            ], className="control-group-sm"),
            
            html.Div([
                html.Label("Data Period", className="control-label-sm"),
                dcc.Dropdown(
                    id="period-dropdown",
                    options=PERIOD_OPTIONS,
                    value="2y",
                    clearable=False,
                    className="dropdown-sm",
                ),
            ], className="control-group-sm"),
            
            html.Div([
                html.Label("Z-Score Filter", className="control-label-sm"),
                dcc.Slider(
                    id="zscore-slider",
                    min=0,
                    max=3,
                    step=0.1,
                    value=1.0,
                    marks={0: "0", 1: "1", 2: "2", 3: "3"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], className="control-group-sm mt-3"),
            
            html.Hr(className="my-2"),
            
            dbc.Checklist(
                id="advanced-options",
                options=[
                    {"label": " Use Adaptive Model", "value": "adaptive"},
                    {"label": " Run GAF-CNN", "value": "gaf"},
                    {"label": " Auto-Optimize Features", "value": "auto_opt"},
                    {"label": " Include Options Data", "value": "options"},
                ],
                value=["adaptive", "auto_opt"],
                className="mt-2",
            ),
        ], id="advanced-collapse", is_open=False),
        
        html.Hr(className="my-3"),
        
        # Run Button
        dbc.Button(
            [html.I(className="fas fa-rocket me-2"), "RUN PREDICTIONS"],
            id="run-predictions-btn",
            color="success",
            size="lg",
            className="w-100 run-btn",
        ),
        
        # Loading indicator
        dbc.Spinner(
            html.Div(id="loading-indicator"),
            color="success",
            type="grow",
            size="sm",
        ),
        
        # Spacer
        html.Div(style={"flex": "1"}),
        
        # Bottom controls
        html.Div([
            html.Hr(),
            dbc.Button(
                [html.I(className="fas fa-sync-alt me-1"), "Clear Cache"],
                id="clear-cache-btn",
                color="link",
                size="sm",
                className="text-muted w-100",
            ),
            html.P([
                html.I(className="fas fa-clock me-1"),
                "Last run: Never"
            ], id="last-run-time", className="text-muted small text-center mt-2"),
        ], className="sidebar-footer mt-auto"),
        
    ], width=3, className="sidebar d-flex flex-column")
