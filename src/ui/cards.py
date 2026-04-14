"""
Card Components - Metric cards, info cards
==========================================

Professional card components for displaying metrics, predictions,
and information in a Bloomberg/TradingView style.

All functions return HTML strings for flexibility.
Use st.markdown(html, unsafe_allow_html=True) to render.

Usage:
    from src.ui.cards import metric_card, info_card, stat_card_row
    st.markdown(metric_card("Revenue", "$1.2M", delta="+12%", trend="up"), unsafe_allow_html=True)
"""

from __future__ import annotations

import streamlit as st
from typing import Optional, Literal, List, Dict

from .theme import get_colors


# =============================================================================
# METRIC CARD
# =============================================================================

def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    trend: Literal["up", "down", "neutral"] = "neutral",
    size: Literal["small", "default", "large"] = "default",
) -> str:
    """
    Generate HTML for a metric card with hover effects.
    
    Args:
        label: Card label/title
        value: Main value to display (formatted string)
        delta: Optional change value (e.g., "+12%", "-5.2%")
        trend: Color coding - "up" (green), "down" (red), "neutral" (gray)
        size: Card size - "small", "default", or "large"
        
    Returns:
        HTML string for the metric card
        
    Example:
        >>> html = metric_card("Revenue", "$1.2M", delta="+12%", trend="up")
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    c = get_colors()
    
    # Size configurations
    sizes = {
        "small": {
            "padding": "0.75rem",
            "label_size": "0.6rem",
            "value_size": "1.25rem",
            "delta_size": "0.7rem",
            "min_width": "120px",
        },
        "default": {
            "padding": "1rem",
            "label_size": "0.7rem",
            "value_size": "1.5rem",
            "delta_size": "0.8rem",
            "min_width": "150px",
        },
        "large": {
            "padding": "1.25rem",
            "label_size": "0.75rem",
            "value_size": "2rem",
            "delta_size": "0.9rem",
            "min_width": "180px",
        },
    }
    s = sizes.get(size, sizes["default"])
    
    # Trend colors
    trend_colors = {
        "up": c["accent_green"],
        "down": c["accent_red"],
        "neutral": c["text_muted"],
    }
    trend_color = trend_colors.get(trend, c["text_muted"])
    
    # Trend icon
    trend_icons = {
        "up": "↑",
        "down": "↓",
        "neutral": "",
    }
    trend_icon = trend_icons.get(trend, "")
    
    # Delta HTML
    delta_html = ""
    if delta:
        delta_html = f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            color: {trend_color};
            font-size: {s['delta_size']};
            font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            margin-left: 0.5rem;
        ">
            <span>{trend_icon}</span>
            <span>{delta}</span>
        </div>
        """
    
    # Generate unique ID for hover effect
    import hashlib
    card_id = f"mc_{hashlib.md5(f'{label}{value}'.encode()).hexdigest()[:8]}"
    
    return f"""
    <style>
    #{card_id} {{
        background: {c["bg_card"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
        padding: {s["padding"]};
        min-width: {s["min_width"]};
        transition: all 0.2s ease;
        cursor: default;
    }}
    #{card_id}:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 16px {c["shadow"]};
        border-color: {c["accent_blue"]};
    }}
    </style>
    <div id="{card_id}">
        <div style="
            color: {c["text_secondary"]};
            font-size: {s["label_size"]};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
            font-family: 'Inter', sans-serif;
        ">
            {label}
        </div>
        <div style="
            display: flex;
            align-items: baseline;
            flex-wrap: wrap;
        ">
            <span style="
                color: {c["text_primary"]};
                font-size: {s["value_size"]};
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                line-height: 1.2;
            ">{value}</span>
            {delta_html}
        </div>
    </div>
    """


# =============================================================================
# INFO CARD
# =============================================================================

def info_card(
    title: str,
    content: str,
    icon: str = "ℹ️",
    variant: Literal["default", "success", "warning", "error"] = "default",
) -> str:
    """
    Generate HTML for an informational card with left accent border.
    
    Args:
        title: Card title
        content: Card content/message
        icon: Emoji or icon character
        variant: Style variant - "default", "success", "warning", "error"
        
    Returns:
        HTML string for the info card
        
    Example:
        >>> html = info_card("Notice", "Market closes at 4pm EST", icon="🔔", variant="warning")
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    c = get_colors()
    
    # Variant colors
    variant_styles = {
        "default": {
            "accent": c["accent_blue"],
            "bg": f"{c['accent_blue']}10",
        },
        "success": {
            "accent": c["accent_green"],
            "bg": f"{c['accent_green']}10",
        },
        "warning": {
            "accent": c["accent_yellow"],
            "bg": f"{c['accent_yellow']}10",
        },
        "error": {
            "accent": c["accent_red"],
            "bg": f"{c['accent_red']}10",
        },
    }
    style = variant_styles.get(variant, variant_styles["default"])
    
    return f"""
    <div style="
        background: {style['bg']};
        border: 1px solid {style['accent']}30;
        border-left: 4px solid {style['accent']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    ">
        <div style="
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
        ">
            <span style="
                font-size: 1.1rem;
                line-height: 1.4;
            ">{icon}</span>
            <div style="flex: 1;">
                <div style="
                    color: {c["text_primary"]};
                    font-weight: 600;
                    font-size: 0.9rem;
                    margin-bottom: 0.35rem;
                    font-family: 'Inter', sans-serif;
                ">{title}</div>
                <div style="
                    color: {c["text_secondary"]};
                    font-size: 0.85rem;
                    line-height: 1.5;
                    font-family: 'Inter', sans-serif;
                ">{content}</div>
            </div>
        </div>
    </div>
    """


# =============================================================================
# STAT CARD ROW
# =============================================================================

def stat_card_row(stats: List[Dict]) -> str:
    """
    Generate HTML for a responsive flex row of metric cards.
    
    Args:
        stats: List of dicts with keys:
            - label (str): Card label
            - value (str): Main value
            - delta (str, optional): Change value
            - trend (str, optional): "up", "down", or "neutral"
            
    Returns:
        HTML string for the stat card row
        
    Example:
        >>> stats = [
        ...     {"label": "Revenue", "value": "$1.2M", "delta": "+12%", "trend": "up"},
        ...     {"label": "Users", "value": "45.2K", "delta": "-3%", "trend": "down"},
        ...     {"label": "Conversion", "value": "3.2%", "trend": "neutral"},
        ... ]
        >>> html = stat_card_row(stats)
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    c = get_colors()
    
    if not stats:
        return ""
    
    # Generate cards
    cards_html = ""
    for stat in stats:
        card = metric_card(
            label=stat.get("label", ""),
            value=stat.get("value", ""),
            delta=stat.get("delta"),
            trend=stat.get("trend", "neutral"),
            size="default",
        )
        cards_html += f"""
        <div style="flex: 1; min-width: 150px;">
            {card}
        </div>
        """
    
    return f"""
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 0.5rem 0;
    ">
        {cards_html}
    </div>
    """


# =============================================================================
# ADDITIONAL CARD COMPONENTS
# =============================================================================

def prediction_card(
    ticker: str,
    predicted_return: float,
    probability: float,
    current_price: float,
    predicted_price: float,
    signal: Literal["BUY", "SELL", "HOLD"] = "HOLD",
    confidence: float = 0.5,
    horizon: int = 1,
) -> str:
    """
    Generate HTML for a prediction card for a single ticker.
    
    Args:
        ticker: Stock symbol
        predicted_return: Predicted return (e.g., 0.025 for 2.5%)
        probability: Probability of up move (0-1)
        current_price: Current stock price
        predicted_price: Predicted future price
        signal: Trading signal
        confidence: Model confidence (0-1)
        horizon: Prediction horizon in days
        
    Returns:
        HTML string for the prediction card
    """
    c = get_colors()
    
    # Signal colors and styling
    signal_styles = {
        "BUY": {"color": c["accent_green"], "bg": f"{c['accent_green']}20", "icon": "▲"},
        "SELL": {"color": c["accent_red"], "bg": f"{c['accent_red']}20", "icon": "▼"},
        "HOLD": {"color": c["text_secondary"], "bg": f"{c['text_secondary']}20", "icon": "◆"},
    }
    style = signal_styles.get(signal, signal_styles["HOLD"])
    
    # Format values
    ret_str = f"{predicted_return * 100:+.2f}%"
    prob_str = f"{probability * 100:.0f}%"
    conf_str = f"{confidence * 100:.0f}%"
    
    return f"""
    <div style="
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-left: 4px solid {style['color']};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: {c['text_primary']};
                    margin-bottom: 0.25rem;
                    font-family: 'JetBrains Mono', monospace;
                ">{ticker}</div>
                <div style="
                    color: {c['text_secondary']};
                    font-size: 0.75rem;
                    font-family: 'Inter', sans-serif;
                ">{horizon}-Day Forecast</div>
            </div>
            <div style="
                background: {style['bg']};
                color: {style['color']};
                padding: 0.35rem 0.75rem;
                border-radius: 6px;
                font-weight: 600;
                font-size: 0.8rem;
                font-family: 'Inter', sans-serif;
            ">
                {style['icon']} {signal}
            </div>
        </div>
        
        <div style="
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid {c['border_muted']};
        ">
            <div>
                <div style="color: {c['text_muted']}; font-size: 0.65rem; text-transform: uppercase; font-family: 'Inter', sans-serif;">Current</div>
                <div style="color: {c['text_primary']}; font-weight: 600; font-family: 'JetBrains Mono', monospace;">${current_price:,.2f}</div>
            </div>
            <div>
                <div style="color: {c['text_muted']}; font-size: 0.65rem; text-transform: uppercase; font-family: 'Inter', sans-serif;">Target</div>
                <div style="color: {c['text_primary']}; font-weight: 600; font-family: 'JetBrains Mono', monospace;">${predicted_price:,.2f}</div>
            </div>
            <div>
                <div style="color: {c['text_muted']}; font-size: 0.65rem; text-transform: uppercase; font-family: 'Inter', sans-serif;">Return</div>
                <div style="color: {style['color']}; font-weight: 600; font-family: 'JetBrains Mono', monospace;">{ret_str}</div>
            </div>
            <div>
                <div style="color: {c['text_muted']}; font-size: 0.65rem; text-transform: uppercase; font-family: 'Inter', sans-serif;">P(Up)</div>
                <div style="color: {c['text_primary']}; font-weight: 600; font-family: 'JetBrains Mono', monospace;">{prob_str}</div>
            </div>
        </div>
    </div>
    """


def stats_row(stats: List[Dict]) -> str:
    """
    Generate HTML for a compact stats row (no cards, just values).
    
    Args:
        stats: List of {"label": "Sharpe", "value": "1.25", "delta": "+0.3"}
        
    Returns:
        HTML string for the stats row
    """
    c = get_colors()
    
    stats_html = ""
    for stat in stats:
        label = stat.get("label", "")
        value = stat.get("value", "")
        delta = stat.get("delta", "")
        
        delta_html = ""
        if delta:
            is_positive = delta.startswith("+") or (delta[0].isdigit() and float(delta) > 0)
            delta_color = c["accent_green"] if is_positive else c["accent_red"]
            delta_html = f"<span style='color: {delta_color}; font-size: 0.75rem; margin-left: 0.25rem; font-family: JetBrains Mono, monospace;'>{delta}</span>"
        
        stats_html += f"""
        <div style="text-align: center; padding: 0 1rem;">
            <div style="color: {c['text_muted']}; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.5px; font-family: Inter, sans-serif;">{label}</div>
            <div style="color: {c['text_primary']}; font-size: 1.1rem; font-weight: 600; font-family: JetBrains Mono, monospace;">{value}{delta_html}</div>
        </div>
        """
    
    return f"""
    <div style="
        display: flex;
        justify-content: space-around;
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 10px;
        padding: 1rem;
    ">
        {stats_html}
    </div>
    """


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    trend: Literal["up", "down", "neutral"] = "neutral",
    size: Literal["small", "default", "large"] = "default",
) -> None:
    """Convenience function to render metric_card directly."""
    st.markdown(metric_card(label, value, delta, trend, size), unsafe_allow_html=True)


def render_info_card(
    title: str,
    content: str,
    icon: str = "ℹ️",
    variant: Literal["default", "success", "warning", "error"] = "default",
) -> None:
    """Convenience function to render info_card directly."""
    st.markdown(info_card(title, content, icon, variant), unsafe_allow_html=True)


def render_stat_card_row(stats: List[Dict]) -> None:
    """Convenience function to render stat_card_row directly."""
    st.markdown(stat_card_row(stats), unsafe_allow_html=True)
