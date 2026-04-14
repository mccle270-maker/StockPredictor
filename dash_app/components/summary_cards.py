"""
Summary Cards Components for QuantDesk Dashboard

These are the key new feature - clean action cards for each ticker
with bullish signals, warnings, and model quality metrics.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np


def get_signal_color(signal: str) -> dict:
    """Get color configuration for a signal type."""
    signal_upper = signal.upper() if signal else "HOLD"
    
    if signal_upper in ["BUY", "STRONG BUY"]:
        return {
            "bg": "rgba(46, 204, 113, 0.15)",
            "border": "#2ecc71",
            "text": "#2ecc71",
            "emoji": "🟢",
            "class": "buy"
        }
    elif signal_upper in ["SELL", "STRONG SELL"]:
        return {
            "bg": "rgba(231, 76, 60, 0.15)",
            "border": "#e74c3c",
            "text": "#e74c3c",
            "emoji": "🔴",
            "class": "sell"
        }
    else:
        return {
            "bg": "rgba(149, 165, 166, 0.15)",
            "border": "#95a5a6",
            "text": "#95a5a6",
            "emoji": "⚪",
            "class": "hold"
        }


def analyze_prediction(pred: dict) -> tuple:
    """
    Analyze a prediction and return bullish signals and warnings.
    
    Args:
        pred: Prediction dictionary
        
    Returns:
        (bullish_points, warning_points) - lists of string descriptions
    """
    bullish = []
    warnings = []
    
    # P(Up) analysis
    prob_up = pred.get("prob_up", 0.5)
    if prob_up is not None:
        if prob_up > 0.60:
            bullish.append(f"P(Up): {prob_up:.0%} - Very strong upward probability")
        elif prob_up > 0.55:
            bullish.append(f"P(Up): {prob_up:.0%} - Strong upward probability")
        elif prob_up < 0.40:
            bullish.append(f"P(Down): {1-prob_up:.0%} - Strong downward probability")
        elif prob_up < 0.45:
            bullish.append(f"P(Down): {1-prob_up:.0%} - Moderate downward probability")
        elif 0.48 < prob_up < 0.52:
            warnings.append(f"P(Up) near 50% ({prob_up:.0%}) - Low directional confidence")
    
    # Z-Score analysis
    z_score = pred.get("pred_zscore", pred.get("prediction_zscore", 0))
    if z_score is not None and z_score != 0:
        abs_z = abs(z_score)
        if abs_z > 2.0:
            bullish.append(f"Z-Score: {z_score:+.2f} - Very high statistical significance")
        elif abs_z > 1.5:
            bullish.append(f"Z-Score: {z_score:+.2f} - Highly significant signal")
        elif abs_z > 1.0:
            bullish.append(f"Z-Score: {z_score:+.2f} - Statistically significant")
        elif abs_z < 0.5:
            warnings.append(f"Weak signal (Z-Score: {z_score:+.2f})")
    
    # Predicted return analysis
    pred_ret = pred.get("pred_next_ret", 0)
    if pred_ret is not None:
        pred_pct = pred_ret * 100
        if abs(pred_pct) > 3:
            bullish.append(f"Predicted return: {pred_pct:+.2f}% - Large move expected")
        elif abs(pred_pct) > 2:
            bullish.append(f"Predicted return: {pred_pct:+.2f}% - Strong move expected")
        elif abs(pred_pct) > 1:
            pass  # Normal range, no special note
        elif abs(pred_pct) < 0.5:
            warnings.append("Low predicted move (<0.5%)")
    
    # Confidence analysis (for adaptive model)
    confidence = pred.get("confidence", 0)
    if confidence and confidence > 0:
        if confidence > 0.65:
            bullish.append(f"Confidence: {confidence:.0%} - Very high conviction signal")
        elif confidence > 0.55:
            bullish.append(f"Confidence: {confidence:.0%} - High conviction signal")
        elif confidence < 0.40:
            warnings.append(f"Low model confidence ({confidence:.0%})")
    
    # IV analysis
    iv = pred.get("atm_iv", pred.get("iv", 0))
    if iv and iv > 0:
        if iv > 0.50:
            warnings.append(f"Very high IV ({iv:.0%}) - Options very expensive")
        elif iv > 0.40:
            warnings.append(f"High IV ({iv:.0%}) - Options expensive")
        elif iv < 0.15:
            bullish.append(f"Low IV ({iv:.0%}) - Options cheap")
    
    # RSI analysis (if available)
    rsi = pred.get("rsi14", pred.get("rsi", None))
    if rsi is not None:
        if rsi < 25:
            bullish.append(f"RSI: {rsi:.0f} - Very oversold territory")
        elif rsi < 30:
            bullish.append(f"RSI: {rsi:.0f} - Oversold territory")
        elif rsi > 75:
            warnings.append(f"RSI: {rsi:.0f} - Very overbought")
        elif rsi > 70:
            warnings.append(f"RSI: {rsi:.0f} - Overbought")
    
    # Volume analysis (if available)
    vol_ratio = pred.get("volume_ratio", pred.get("vol_ratio", None))
    if vol_ratio is not None:
        if vol_ratio > 2.0:
            bullish.append(f"Volume {vol_ratio:.1f}x average - High interest")
        elif vol_ratio > 1.5:
            bullish.append(f"Volume {vol_ratio:.1f}x average - Above normal")
        elif vol_ratio < 0.5:
            warnings.append(f"Low volume ({vol_ratio:.1f}x avg)")
    
    return bullish, warnings


def create_ticker_action_card(pred: dict) -> dbc.Card:
    """
    Create an action card for a single ticker prediction.
    
    This is the primary display component showing:
    - Current price and target
    - Signal (BUY/SELL/HOLD) with color coding
    - Key bullish indicators
    - Warning signs
    - Model quality metrics
    - Action buttons
    """
    ticker = pred.get("ticker", "N/A")
    signal = pred.get("signal", pred.get("signal_label", "HOLD"))
    colors = get_signal_color(signal)
    
    last_close = pred.get("last_close", 0) or 0
    target_price = pred.get("pred_next_price", last_close) or last_close
    pred_ret = (pred.get("pred_next_ret", 0) or 0) * 100
    confidence = pred.get("confidence", 0) or 0
    
    # Analyze the prediction
    bullish_points, warning_points = analyze_prediction(pred)
    
    # Get backtest/quality metrics if available
    accuracy = pred.get("accuracy", 0)
    sharpe = pred.get("sharpe", 0)
    win_rate = pred.get("win_rate", 0)
    
    # Build the card
    return dbc.Card([
        # Header with ticker and signal badge
        dbc.CardHeader([
            html.Div([
                html.Span(ticker, className="ticker-symbol"),
                html.Span([
                    colors["emoji"], " ", signal.upper()
                ], className="signal-badge", style={
                    "background": colors["bg"],
                    "border": f"1px solid {colors['border']}",
                    "color": colors["text"],
                }),
            ], className="card-header-content"),
        ], className="ticker-card-header"),
        
        # Card body
        dbc.CardBody([
            # Price Target Row
            html.Div([
                html.Span(f"Current: ${last_close:.2f}", className="current-price"),
                html.Span(" → ", className="arrow"),
                html.Span(f"Target: ${target_price:.2f}", className="target-price"),
                html.Span(f" ({pred_ret:+.1f}%)", className="return-pct", style={
                    "color": "#2ecc71" if pred_ret > 0 else "#e74c3c" if pred_ret < 0 else "#95a5a6"
                }),
            ], className="price-row"),
            
            # Confidence bar (for adaptive model)
            html.Div([
                html.Div([
                    html.Span(f"Confidence: {confidence:.0%}", className="confidence-label"),
                    dbc.Progress(
                        value=confidence * 100, 
                        className="confidence-bar", 
                        color="success" if confidence > 0.5 else "warning" if confidence > 0.35 else "danger",
                        style={"height": "8px"}
                    ),
                ], className="confidence-row"),
            ], className="mb-3") if confidence > 0 else None,
            
            # Bullish Signals Section
            html.Div([
                html.Strong([
                    html.I(className="fas fa-check-circle me-1", style={"color": "#2ecc71"}),
                    "Key Signals:"
                ], className="section-label", style={"color": "#2ecc71"}),
                html.Ul([
                    html.Li(p, className="signal-point") 
                    for p in bullish_points[:3]  # Limit to 3 points
                ], style={"paddingLeft": "1.25rem", "marginBottom": "0.5rem"}),
            ], className="signals-section") if bullish_points else None,
            
            # Warning Signals Section
            html.Div([
                html.Strong([
                    html.I(className="fas fa-exclamation-triangle me-1", style={"color": "#d29922"}),
                    "Watch Out:"
                ], className="section-label", style={"color": "#d29922"}),
                html.Ul([
                    html.Li(w, className="warning-point") 
                    for w in warning_points[:3]  # Limit to 3 warnings
                ], style={"paddingLeft": "1.25rem", "marginBottom": "0.5rem"}),
            ], className="warnings-section") if warning_points else None,
            
            # Model Quality Section
            html.Div([
                html.Strong([
                    html.I(className="fas fa-chart-bar me-1", style={"color": "#388bfd"}),
                    "Model Quality:"
                ], className="section-label"),
                html.Div([
                    html.Span(
                        f"Accuracy: {accuracy:.0%}", 
                        className="quality-metric"
                    ) if accuracy else None,
                    html.Span(
                        f"Sharpe: {sharpe:.2f}", 
                        className="quality-metric"
                    ) if sharpe else None,
                    html.Span(
                        f"Win Rate: {win_rate:.0%}", 
                        className="quality-metric"
                    ) if win_rate else None,
                ], className="quality-row"),
            ], className="quality-section mt-3") if any([accuracy, sharpe, win_rate]) else None,
            
            # Action Buttons
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-chart-line me-1"), 
                        "Analysis"
                    ], id={"type": "view-analysis", "ticker": ticker},
                       color="primary", size="sm", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-chart-area me-1"), 
                        "Charts"
                    ], id={"type": "view-charts", "ticker": ticker},
                       color="info", size="sm", outline=True),
                    dbc.Button([
                        html.I(className="fas fa-cog me-1"), 
                        "Options"
                    ], id={"type": "view-options", "ticker": ticker},
                       color="secondary", size="sm", outline=True),
                ], className="action-buttons w-100"),
            ], className="mt-3"),
        ]),
    ], className="ticker-card")


def create_portfolio_summary(predictions: list) -> dbc.Card:
    """
    Create the portfolio summary panel that appears at the top of the Summary tab.
    
    Shows:
    - BUY/SELL/HOLD signal counts
    - Top pick with highest confidence
    - Average predicted return
    - High confidence signal count
    - Weak signal warnings
    """
    if not predictions:
        return html.Div("No predictions available", className="empty-state")
    
    # Count signals
    n_buy = sum(1 for p in predictions if p.get("signal", "").upper() in ["BUY", "STRONG BUY"])
    n_sell = sum(1 for p in predictions if p.get("signal", "").upper() in ["SELL", "STRONG SELL"])
    n_hold = len(predictions) - n_buy - n_sell
    
    # Find top pick (highest confidence among actionable signals)
    actionable = [p for p in predictions if p.get("signal", "").upper() in ["BUY", "SELL", "STRONG BUY", "STRONG SELL"]]
    if actionable:
        top_pick = max(actionable, key=lambda p: p.get("confidence", 0))
    else:
        top_pick = max(predictions, key=lambda p: abs(p.get("pred_next_ret", 0)))
    
    top_ticker = top_pick.get("ticker", "N/A")
    top_ret = (top_pick.get("pred_next_ret", 0) or 0) * 100
    top_conf = top_pick.get("confidence", 0) or 0
    top_signal = top_pick.get("signal", "HOLD")
    
    # Calculate averages
    avg_ret = np.mean([(p.get("pred_next_ret", 0) or 0) * 100 for p in predictions])
    
    # Count high confidence signals
    high_conf = sum(1 for p in predictions if (p.get("confidence", 0) or 0) > 0.5)
    
    # Count weak signals (low z-score)
    weak_signals = sum(1 for p in predictions 
                       if abs(p.get("pred_zscore", p.get("prediction_zscore", 0)) or 0) < 0.5)
    
    # Determine overall sentiment for gradient
    if n_buy > n_sell and n_buy > n_hold:
        gradient = "linear-gradient(135deg, #1a472a 0%, #134e5e 50%, #1a237e 100%)"
        sentiment = "BULLISH"
    elif n_sell > n_buy and n_sell > n_hold:
        gradient = "linear-gradient(135deg, #4a1a1a 0%, #6b2d2d 50%, #3d0a0a 100%)"
        sentiment = "BEARISH"
    else:
        gradient = "linear-gradient(135deg, #2d3748 0%, #374151 50%, #1f2937 100%)"
        sentiment = "NEUTRAL"
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span([
                    html.I(className="fas fa-briefcase me-2"),
                    "PORTFOLIO SUMMARY"
                ]),
                html.Span([
                    html.Span(f"{len(predictions)} tickers", className="badge bg-secondary ms-2"),
                    html.Span(sentiment, className="badge bg-dark ms-2"),
                ]),
            ], className="d-flex justify-content-between align-items-center"),
        ], className="summary-header"),
        
        dbc.CardBody([
            # Signal counts row
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("🟢", className="signal-emoji", style={"fontSize": "1.5rem"}),
                        html.Div([
                            html.Div(f"BUY", className="text-muted small"),
                            html.Div(f"{n_buy}", className="signal-count", 
                                    style={"color": "#2ecc71", "fontSize": "1.5rem", "fontWeight": "700"}),
                        ]),
                    ], className="d-flex align-items-center gap-2"),
                ], width=4, className="text-center"),
                dbc.Col([
                    html.Div([
                        html.Span("🔴", className="signal-emoji", style={"fontSize": "1.5rem"}),
                        html.Div([
                            html.Div(f"SELL", className="text-muted small"),
                            html.Div(f"{n_sell}", className="signal-count",
                                    style={"color": "#e74c3c", "fontSize": "1.5rem", "fontWeight": "700"}),
                        ]),
                    ], className="d-flex align-items-center gap-2"),
                ], width=4, className="text-center"),
                dbc.Col([
                    html.Div([
                        html.Span("⚪", className="signal-emoji", style={"fontSize": "1.5rem"}),
                        html.Div([
                            html.Div(f"HOLD", className="text-muted small"),
                            html.Div(f"{n_hold}", className="signal-count",
                                    style={"color": "#95a5a6", "fontSize": "1.5rem", "fontWeight": "700"}),
                        ]),
                    ], className="d-flex align-items-center gap-2"),
                ], width=4, className="text-center"),
            ], className="signal-row mb-3"),
            
            html.Hr(),
            
            # Top pick
            html.Div([
                html.Strong("🏆 Top Pick: ", className="me-2"),
                html.Span(top_ticker, style={
                    "color": "#388bfd", 
                    "fontWeight": "700", 
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "1.1rem"
                }),
                html.Span(f" ({top_ret:+.1f}%", style={
                    "color": "#2ecc71" if top_ret > 0 else "#e74c3c"
                }),
                html.Span(f", {top_conf:.0%} confidence)" if top_conf else ")", className="text-muted"),
                html.Span(f" - {top_signal}", className="ms-1", style={
                    "color": "#2ecc71" if "BUY" in top_signal.upper() else "#e74c3c" if "SELL" in top_signal.upper() else "#95a5a6"
                }),
            ], className="top-pick-row"),
            
            # Average return
            html.Div([
                html.Strong("📊 Avg Predicted Return: ", className="me-2"),
                html.Span(f"{avg_ret:+.2f}%", style={
                    "color": "#2ecc71" if avg_ret > 0 else "#e74c3c",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontWeight": "600"
                }),
            ], className="avg-row mt-2"),
            
            # High confidence count
            html.Div([
                html.Strong("✅ High Confidence Signals: ", className="me-2"),
                html.Span(f"{high_conf}/{len(predictions)}", style={
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
            ], className="conf-row mt-2"),
            
            # Weak signals warning
            html.Div([
                html.I(className="fas fa-exclamation-triangle me-1", style={"color": "#d29922"}),
                html.Span(
                    f"{weak_signals} weak signal{'s' if weak_signals != 1 else ''} (|z| < 0.5) - consider skipping",
                    style={"color": "#d29922", "fontSize": "0.9rem"}
                ),
            ], className="warning-row mt-2") if weak_signals > 0 else None,
        ]),
    ], style={"background": gradient}, className="portfolio-summary-card mb-4")


def create_empty_state(message: str = "Run predictions to see results") -> html.Div:
    """Create an empty state placeholder."""
    return html.Div([
        html.I(className="fas fa-chart-line fa-3x mb-3", style={"color": "#6c757d"}),
        html.H5(message, className="text-muted"),
        html.P("Click 'RUN PREDICTIONS' to get started", className="text-muted small"),
    ], className="empty-state text-center py-5")


def create_summary_tab_content(predictions: list = None):
    """
    Create the complete Summary tab content.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dash layout for Summary tab
    """
    if not predictions:
        return create_empty_state()
    
    # Portfolio summary at top
    summary = create_portfolio_summary(predictions)
    
    # Create ticker cards grid
    cards = [create_ticker_action_card(p) for p in predictions]
    
    return html.Div([
        summary,
        html.Div(cards, className="ticker-cards-grid"),
    ])
