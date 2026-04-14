"""
Notification System - Toasts and alerts
=======================================

Non-intrusive notification components for status updates,
confirmations, and error messages.

Usage:
    from src.ui.notifications import toast, inline_alert
    toast("Changes saved successfully!", type="success")
    inline_alert("Please review before submitting", type="warning", dismissible=True)
"""

from __future__ import annotations

import streamlit as st
from typing import Optional, Literal
import hashlib

from .theme import get_colors


# =============================================================================
# TOAST NOTIFICATION
# =============================================================================

def toast(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    duration: int = 3000,
) -> str:
    """
    Display a toast notification in the bottom-right corner.
    
    Auto-dismisses after the specified duration using CSS animations.
    
    Args:
        message: Toast message to display
        type: Notification type - "success", "error", "warning", "info"
        duration: Auto-dismiss duration in milliseconds (default 3000ms)
    
    Returns:
        HTML string for the toast notification
        
    Example:
        >>> st.markdown(toast("Changes saved!", type="success"), unsafe_allow_html=True)
    """
    c = get_colors()
    
    # Type-specific styling
    type_styles = {
        "success": {
            "icon": "✓",
            "color": c["accent_green"],
            "bg": c["bg_card"],
            "border": c["accent_green"],
        },
        "error": {
            "icon": "✕",
            "color": c["accent_red"],
            "bg": c["bg_card"],
            "border": c["accent_red"],
        },
        "warning": {
            "icon": "⚠",
            "color": c["accent_yellow"],
            "bg": c["bg_card"],
            "border": c["accent_yellow"],
        },
        "info": {
            "icon": "ℹ",
            "color": c["accent_blue"],
            "bg": c["bg_card"],
            "border": c["accent_blue"],
        },
    }
    style = type_styles.get(type, type_styles["info"])
    
    # Generate unique ID for this toast
    toast_id = f"toast_{hashlib.md5(f'{message}{type}'.encode()).hexdigest()[:8]}"
    
    # Calculate fade out start time (duration - 300ms for fade animation)
    fade_start = max(duration - 300, 0)
    
    return f"""
    <style>
    @keyframes toastSlideIn {{
        from {{
            transform: translateX(120%);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    @keyframes toastFadeOut {{
        from {{
            transform: translateX(0);
            opacity: 1;
        }}
        to {{
            transform: translateX(50%);
            opacity: 0;
        }}
    }}
    
    #{toast_id} {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        display: flex;
        align-items: center;
        gap: 12px;
        background: {style["bg"]};
        border: 1px solid {style["border"]}40;
        border-left: 4px solid {style["border"]};
        border-radius: 10px;
        padding: 14px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3), 0 4px 8px rgba(0,0,0,0.2);
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 0.9rem;
        max-width: 400px;
        animation: 
            toastSlideIn 0.3s ease forwards,
            toastFadeOut 0.3s ease {fade_start}ms forwards;
    }}
    
    #{toast_id} .toast-icon {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: {style["color"]}20;
        color: {style["color"]};
        border-radius: 50%;
        font-size: 0.8rem;
        font-weight: 600;
        flex-shrink: 0;
    }}
    
    #{toast_id} .toast-message {{
        color: {c["text_primary"]};
        line-height: 1.4;
    }}
    
    #{toast_id} .toast-close {{
        color: {c["text_muted"]};
        cursor: pointer;
        padding: 4px;
        margin-left: 8px;
        font-size: 1rem;
        opacity: 0.6;
        transition: opacity 0.15s ease;
    }}
    
    #{toast_id} .toast-close:hover {{
        opacity: 1;
    }}
    </style>
    
    <div id="{toast_id}">
        <span class="toast-icon">{style["icon"]}</span>
        <span class="toast-message">{message}</span>
        <span class="toast-close">×</span>
    </div>
    """


# =============================================================================
# INLINE ALERT
# =============================================================================

def inline_alert(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    dismissible: bool = False,
    title: Optional[str] = None,
) -> str:
    """
    Display an inline alert that stays in place.
    
    Args:
        message: Alert message
        type: Alert type - "success", "error", "warning", "info"
        dismissible: Show dismiss X button (visual only)
        title: Optional bold title before message
    
    Returns:
        HTML string for the inline alert
        
    Example:
        >>> st.markdown(inline_alert("Please save your work", type="warning"), unsafe_allow_html=True)
    """
    c = get_colors()
    
    # Type-specific styling
    type_styles = {
        "success": {
            "icon": "✓",
            "color": c["accent_green"],
            "bg": f"{c['accent_green']}10",
            "border": c["accent_green"],
        },
        "error": {
            "icon": "✕",
            "color": c["accent_red"],
            "bg": f"{c['accent_red']}10",
            "border": c["accent_red"],
        },
        "warning": {
            "icon": "⚠",
            "color": c["accent_yellow"],
            "bg": f"{c['accent_yellow']}10",
            "border": c["accent_yellow"],
        },
        "info": {
            "icon": "ℹ",
            "color": c["accent_blue"],
            "bg": f"{c['accent_blue']}10",
            "border": c["accent_blue"],
        },
    }
    style = type_styles.get(type, type_styles["info"])
    
    # Generate unique ID
    alert_id = f"alert_{hashlib.md5(f'{message}{type}'.encode()).hexdigest()[:8]}"
    
    # Title HTML
    title_html = ""
    if title:
        title_html = f"""
        <span style="
            font-weight: 600;
            color: {c["text_primary"]};
            margin-right: 6px;
        ">{title}</span>
        """
    
    # Dismiss button HTML
    dismiss_html = ""
    if dismissible:
        dismiss_html = f"""
        <span style="
            color: {c["text_muted"]};
            cursor: pointer;
            padding: 4px 8px;
            margin-left: auto;
            font-size: 1.1rem;
            opacity: 0.6;
            transition: opacity 0.15s ease;
            flex-shrink: 0;
        " onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.6'">×</span>
        """
    
    return f"""
    <style>
    #{alert_id} {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: {style["bg"]};
        border: 1px solid {style["border"]}30;
        border-left: 4px solid {style["border"]};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-family: 'Inter', -apple-system, sans-serif;
        animation: alertFadeIn 0.25s ease;
    }}
    
    @keyframes alertFadeIn {{
        from {{
            opacity: 0;
            transform: translateY(-8px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    #{alert_id} .alert-icon {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        background: {style["color"]}20;
        color: {style["color"]};
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 600;
        flex-shrink: 0;
        margin-top: 1px;
    }}
    
    #{alert_id} .alert-content {{
        flex: 1;
        color: {c["text_secondary"]};
        font-size: 0.875rem;
        line-height: 1.5;
    }}
    </style>
    
    <div id="{alert_id}">
        <span class="alert-icon">{style["icon"]}</span>
        <div class="alert-content">
            {title_html}{message}
        </div>
        {dismiss_html}
    </div>
    """


# =============================================================================
# ADDITIONAL NOTIFICATION COMPONENTS
# =============================================================================

def status_badge(
    label: str,
    status: Literal["active", "inactive", "pending", "error"] = "active",
    size: Literal["sm", "md"] = "md",
) -> str:
    """
    Generate HTML for a status badge.
    
    Args:
        label: Badge label
        status: Status variant
        size: Badge size
        
    Returns:
        HTML string for the badge
    """
    c = get_colors()
    
    status_styles = {
        "active": {"color": c["accent_green"], "bg": f"{c['accent_green']}20"},
        "inactive": {"color": c["text_muted"], "bg": f"{c['text_muted']}20"},
        "pending": {"color": c["accent_yellow"], "bg": f"{c['accent_yellow']}20"},
        "error": {"color": c["accent_red"], "bg": f"{c['accent_red']}20"},
    }
    style = status_styles.get(status, status_styles["active"])
    
    sizes = {
        "sm": {"padding": "2px 8px", "font": "0.65rem"},
        "md": {"padding": "4px 12px", "font": "0.75rem"},
    }
    s = sizes.get(size, sizes["md"])
    
    # Pulse dot for active status
    dot_html = ""
    if status == "active":
        dot_html = f"""
        <span style="
            display: inline-block;
            width: 6px;
            height: 6px;
            background: {style['color']};
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse 2s infinite;
        "></span>
        """
    
    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        background: {style['bg']};
        color: {style['color']};
        padding: {s['padding']};
        border-radius: 20px;
        font-size: {s['font']};
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    ">
        {dot_html}{label}
    </span>
    """


def render_status_badge(
    label: str,
    status: Literal["active", "inactive", "pending", "error"] = "active",
    size: Literal["sm", "md"] = "md",
) -> None:
    """Render a status badge inline."""
    st.markdown(status_badge(label, status, size), unsafe_allow_html=True)


def progress_indicator(
    current: int,
    total: int,
    label: Optional[str] = None,
    show_percentage: bool = True,
) -> None:
    """
    Display a custom progress indicator.
    
    Args:
        current: Current progress value
        total: Total/max value
        label: Optional label
        show_percentage: Show percentage text
    """
    c = get_colors()
    
    pct = (current / total * 100) if total > 0 else 0
    pct_text = f"{pct:.0f}%" if show_percentage else ""
    label_text = f"{label}: " if label else ""
    
    st.markdown(f"""
    <div style="margin: 8px 0; font-family: 'Inter', sans-serif;">
        <div style="
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
        ">
            <span style="color: {c['text_secondary']}; font-size: 0.8rem;">{label_text}{current}/{total}</span>
            <span style="color: {c['text_primary']}; font-weight: 600; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">{pct_text}</span>
        </div>
        <div style="
            background: {c['bg_elevated']};
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, {c['accent_blue']}, {c['accent_purple']});
                width: {pct}%;
                height: 100%;
                border-radius: 6px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def loading_toast(message: str = "Loading...") -> None:
    """
    Display a loading toast with spinner.
    
    Args:
        message: Loading message
    """
    c = get_colors()
    
    toast_id = f"loading_toast_{hashlib.md5(message.encode()).hexdigest()[:8]}"
    
    st.markdown(f"""
    <style>
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    #{toast_id} {{
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 99999;
        display: flex;
        align-items: center;
        gap: 12px;
        background: {c["bg_card"]};
        border: 1px solid {c["border"]};
        border-radius: 10px;
        padding: 14px 20px;
        box-shadow: 0 8px 24px {c["shadow_strong"]};
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }}
    
    #{toast_id} .spinner {{
        width: 18px;
        height: 18px;
        border: 2px solid {c["bg_elevated"]};
        border-top: 2px solid {c["accent_blue"]};
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }}
    </style>
    
    <div id="{toast_id}">
        <div class="spinner"></div>
        <span style="color: {c['text_primary']};">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def banner(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    icon: Optional[str] = None,
) -> None:
    """
    Display a full-width banner at the top.
    
    Args:
        message: Banner message
        type: Banner type
        icon: Optional custom icon
    """
    c = get_colors()
    
    type_styles = {
        "success": {"icon": "✓", "bg": c["accent_green"], "text": "#ffffff"},
        "error": {"icon": "✕", "bg": c["accent_red"], "text": "#ffffff"},
        "warning": {"icon": "⚠", "bg": c["accent_yellow"], "text": "#000000"},
        "info": {"icon": "ℹ", "bg": c["accent_blue"], "text": "#ffffff"},
    }
    style = type_styles.get(type, type_styles["info"])
    display_icon = icon or style["icon"]
    
    st.markdown(f"""
    <div style="
        background: {style['bg']};
        color: {style['text']};
        padding: 12px 20px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
    ">
        <span>{display_icon}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)
