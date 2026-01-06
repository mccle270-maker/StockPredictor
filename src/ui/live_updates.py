"""
Live Updates Module for Stock Predictor
========================================

Provides real-time price updates and chart refreshing functionality
without blocking the main dashboard UI.

Features:
- Configurable refresh intervals
- Background price fetching
- Live chart data updates
- Session-based tracking

This is an optional enhancement - the dashboard works without it.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
import threading
from queue import Queue

# Try to import database for snapshot storage
try:
    from src.data.database import save_price_snapshot, get_price_snapshots
    HAS_DATABASE = True
except ImportError:
    HAS_DATABASE = False


@dataclass
class LiveConfig:
    """Configuration for live updates."""
    enabled: bool = False
    refresh_interval: int = 30  # seconds
    price_refresh_interval: int = 10  # seconds for intraday prices
    chart_refresh_interval: int = 60  # seconds for chart updates
    max_history_points: int = 100  # max data points to keep in memory


# Global configuration
_live_config = LiveConfig()


def get_live_config() -> LiveConfig:
    """Get the current live update configuration."""
    return _live_config


def set_live_config(
    enabled: Optional[bool] = None,
    refresh_interval: Optional[int] = None,
    price_refresh_interval: Optional[int] = None,
    chart_refresh_interval: Optional[int] = None
):
    """Update live configuration settings."""
    global _live_config
    
    if enabled is not None:
        _live_config.enabled = enabled
    if refresh_interval is not None:
        _live_config.refresh_interval = refresh_interval
    if price_refresh_interval is not None:
        _live_config.price_refresh_interval = price_refresh_interval
    if chart_refresh_interval is not None:
        _live_config.chart_refresh_interval = chart_refresh_interval


def init_live_state():
    """Initialize session state for live updates."""
    if "live_enabled" not in st.session_state:
        st.session_state["live_enabled"] = False
    if "live_last_refresh" not in st.session_state:
        st.session_state["live_last_refresh"] = None
    if "live_price_cache" not in st.session_state:
        st.session_state["live_price_cache"] = {}
    if "live_refresh_count" not in st.session_state:
        st.session_state["live_refresh_count"] = 0


def is_market_hours() -> bool:
    """
    Check if US stock market is currently open.
    Returns True if within trading hours (9:30 AM - 4:00 PM ET).
    """
    from datetime import datetime
    import pytz
    
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Check if weekend
        if now.weekday() >= 5:
            return False
        
        # Check if within trading hours
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception:
        # If pytz not available, assume market hours
        return True


def get_live_price(ticker: str, fallback_price: Optional[float] = None) -> Optional[float]:
    """
    Get the latest price for a ticker with caching.
    Uses a fast source for intraday data.
    """
    try:
        import yfinance as yf
        
        # Check cache first
        cache = st.session_state.get("live_price_cache", {})
        cache_entry = cache.get(ticker)
        
        if cache_entry:
            cache_time, cached_price = cache_entry
            # Use cache if less than 10 seconds old
            if (datetime.now() - cache_time).total_seconds() < 10:
                return cached_price
        
        # Fetch new price
        tk = yf.Ticker(ticker)
        data = tk.fast_info
        
        price = data.get("lastPrice") or data.get("regularMarketPrice")
        
        if price:
            # Update cache
            if "live_price_cache" not in st.session_state:
                st.session_state["live_price_cache"] = {}
            st.session_state["live_price_cache"][ticker] = (datetime.now(), price)
            
            # Save to database if available
            if HAS_DATABASE:
                try:
                    change_pct = data.get("regularMarketChangePercent")
                    volume = data.get("lastVolume")
                    save_price_snapshot(ticker, price, volume, change_pct)
                except Exception:
                    pass
            
            return price
        
        return fallback_price
    except Exception as e:
        return fallback_price


def get_live_prices_batch(tickers: List[str]) -> Dict[str, Optional[float]]:
    """
    Get live prices for multiple tickers efficiently.
    """
    results = {}
    
    try:
        import yfinance as yf
        
        # Download all at once
        data = yf.download(tickers, period="1d", interval="1m", progress=False)
        
        if not data.empty:
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        price = data["Close"].iloc[-1]
                    else:
                        price = data["Close"][ticker].iloc[-1]
                    results[ticker] = float(price) if not pd.isna(price) else None
                except Exception:
                    results[ticker] = None
        
    except Exception as e:
        for ticker in tickers:
            results[ticker] = None
    
    return results


def create_live_metric_container(
    label: str,
    ticker: str,
    initial_value: Optional[float] = None,
    format_fn: Callable[[float], str] = lambda x: f"${x:.2f}",
    show_change: bool = True
):
    """
    Create a live-updating metric container.
    
    Returns an st.empty() container that can be updated.
    """
    container = st.empty()
    
    price = get_live_price(ticker, initial_value)
    delta = None
    
    if show_change and initial_value and price:
        change = price - initial_value
        delta = f"{change:+.2f} ({(change/initial_value)*100:+.1f}%)"
    
    if price:
        container.metric(
            label=label,
            value=format_fn(price),
            delta=delta
        )
    else:
        container.metric(label=label, value="—", delta="no data")
    
    return container


def should_refresh() -> bool:
    """
    Check if it's time for a refresh based on the interval.
    """
    if not st.session_state.get("live_enabled", False):
        return False
    
    last_refresh = st.session_state.get("live_last_refresh")
    if last_refresh is None:
        return True
    
    config = get_live_config()
    elapsed = (datetime.now() - last_refresh).total_seconds()
    
    return elapsed >= config.refresh_interval


def mark_refreshed():
    """Mark that a refresh just occurred."""
    st.session_state["live_last_refresh"] = datetime.now()
    st.session_state["live_refresh_count"] = st.session_state.get("live_refresh_count", 0) + 1


def get_time_until_next_refresh() -> int:
    """Get seconds until next scheduled refresh."""
    if not st.session_state.get("live_enabled", False):
        return -1
    
    last_refresh = st.session_state.get("live_last_refresh")
    if last_refresh is None:
        return 0
    
    config = get_live_config()
    elapsed = (datetime.now() - last_refresh).total_seconds()
    remaining = max(0, config.refresh_interval - int(elapsed))
    
    return remaining


def render_live_status_badge():
    """Render a status badge showing live update status."""
    init_live_state()
    
    if st.session_state.get("live_enabled", False):
        remaining = get_time_until_next_refresh()
        refresh_count = st.session_state.get("live_refresh_count", 0)
        
        if is_market_hours():
            status_color = "#3fb950"  # Green
            status_text = f"🟢 LIVE • Next: {remaining}s"
        else:
            status_color = "#f0883e"  # Orange
            status_text = f"🟠 LIVE (After Hours) • {remaining}s"
        
        st.markdown(
            f"""<div style='display: inline-flex; align-items: center; 
            padding: 4px 12px; border-radius: 20px; 
            background: {status_color}20; border: 1px solid {status_color};
            font-size: 0.75rem; font-weight: 500;'>
            {status_text} • {refresh_count} updates
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div style='display: inline-flex; align-items: center; 
            padding: 4px 12px; border-radius: 20px; 
            background: #30363d; border: 1px solid #484f58;
            font-size: 0.75rem; font-weight: 500; color: #8b949e;'>
            ⚪ Live Updates Off
            </div>""",
            unsafe_allow_html=True
        )


def render_live_controls():
    """Render the live update control panel."""
    init_live_state()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        live_enabled = st.toggle(
            "Enable Live Updates",
            value=st.session_state.get("live_enabled", False),
            key="live_toggle"
        )
        st.session_state["live_enabled"] = live_enabled
    
    with col2:
        if live_enabled:
            interval = st.selectbox(
                "Refresh Interval",
                options=[10, 30, 60, 120],
                index=1,
                format_func=lambda x: f"{x}s",
                key="live_interval"
            )
            set_live_config(refresh_interval=interval)
    
    with col3:
        if live_enabled:
            if st.button("🔄 Refresh Now", key="manual_refresh"):
                st.session_state["live_last_refresh"] = None
                st.rerun()
    
    return live_enabled


def auto_refresh_check():
    """
    Check if auto-refresh should trigger and rerun if needed.
    Call this at the end of your main content.
    """
    if should_refresh():
        mark_refreshed()
        time.sleep(0.1)  # Small delay to prevent too-rapid refreshes
        st.rerun()


def create_live_price_chart(
    ticker: str,
    base_history: Optional[pd.DataFrame] = None,
    height: int = 300
):
    """
    Create a live-updating price chart.
    
    Args:
        ticker: Stock ticker symbol
        base_history: Optional historical data to include
        height: Chart height in pixels
    """
    import plotly.graph_objects as go
    
    # Get live snapshots if database available
    live_data = None
    if HAS_DATABASE:
        try:
            live_data = get_price_snapshots(ticker, minutes_back=120)
        except Exception:
            pass
    
    fig = go.Figure()
    
    # Add historical data if provided
    if base_history is not None and not base_history.empty:
        if "Close" in base_history.columns:
            fig.add_trace(go.Scatter(
                x=base_history.index,
                y=base_history["Close"],
                name="Historical",
                mode="lines",
                line=dict(color="#388bfd", width=2)
            ))
    
    # Add live data if available
    if live_data is not None and not live_data.empty:
        live_data["timestamp"] = pd.to_datetime(live_data["timestamp"])
        fig.add_trace(go.Scatter(
            x=live_data["timestamp"],
            y=live_data["price"],
            name="Live",
            mode="lines+markers",
            line=dict(color="#3fb950", width=2),
            marker=dict(size=4)
        ))
    
    # Get current live price
    current_price = get_live_price(ticker)
    if current_price:
        fig.add_trace(go.Scatter(
            x=[datetime.now()],
            y=[current_price],
            name="Current",
            mode="markers",
            marker=dict(color="#f85149", size=12, symbol="star")
        ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="",
        yaxis_title="Price ($)",
        template="plotly_dark"
    )
    
    return fig


# Streamlit component for auto-refresh
def auto_refresh_component(interval_seconds: int = 30):
    """
    JavaScript-based auto-refresh component.
    More reliable than Python-based refresh for browser tabs.
    """
    if st.session_state.get("live_enabled", False):
        st.markdown(
            f"""
            <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {interval_seconds * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )
