"""
Navigation Components - Nav bar and headers
============================================

Professional navigation components including sticky nav bars,
section headers, and page headers.

All components return HTML strings for flexible rendering.

Usage:
    from src.ui.navigation import nav_bar, section_header, page_header
    
    # Render navigation bar
    html = nav_bar("dashboard", [
        {"key": "dashboard", "icon": "📈", "label": "Dashboard"},
        {"key": "backtest", "icon": "📊", "label": "Backtest"},
    ])
    st.markdown(html, unsafe_allow_html=True)
"""

from __future__ import annotations

from typing import Optional, List, Dict
from datetime import datetime

from .theme import get_colors


# =============================================================================
# NAV BAR
# =============================================================================

def nav_bar(
    active_tab: str,
    tabs: List[Dict[str, str]],
    show_status: bool = True,
) -> str:
    """
    Sticky top navigation bar.
    
    Features:
    - Active tab with blue background tint and bottom border
    - Inactive tabs with hover effect
    - Optional live status indicator with pulsing green dot
    - Current time display
    
    Args:
        active_tab: Key of the currently active tab
        tabs: List of tab dictionaries with keys: "key", "icon", "label"
            Example: [{"key": "dashboard", "icon": "📈", "label": "Dashboard"}]
        show_status: Show live status indicator and time on right side
    
    Returns:
        HTML string for the navigation bar
    
    Example:
        html = nav_bar("dashboard", [
            {"key": "dashboard", "icon": "📈", "label": "Dashboard"},
            {"key": "backtest", "icon": "📊", "label": "Backtest"},
            {"key": "settings", "icon": "⚙️", "label": "Settings"},
        ])
        st.markdown(html, unsafe_allow_html=True)
    """
    colors = get_colors()
    
    # Build nav items
    nav_items_html = ""
    for tab in tabs:
        is_active = tab["key"] == active_tab
        icon = tab.get("icon", "")
        label = tab.get("label", tab["key"])
        
        if is_active:
            item_style = f"""
                background: {colors['accent_blue']}15;
                color: {colors['accent_blue']};
                border-bottom: 2px solid {colors['accent_blue']};
            """
        else:
            item_style = f"""
                background: transparent;
                color: {colors['text_secondary']};
                border-bottom: 2px solid transparent;
            """
        
        nav_items_html += f"""
        <div class="nav-tab" data-key="{tab['key']}" style="
            padding: 0.75rem 1.25rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-radius: 8px 8px 0 0;
            {item_style}
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <span>{icon}</span>
            <span>{label}</span>
        </div>
        """
    
    # Status indicator (right side)
    status_html = ""
    if show_status:
        current_time = datetime.now().strftime("%H:%M:%S")
        status_html = f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <!-- Pulsing status dot -->
            <div style="
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.4rem 0.75rem;
                background: {colors['accent_green']}15;
                border-radius: 20px;
                border: 1px solid {colors['accent_green']}30;
            ">
                <div class="status-dot" style="
                    width: 8px;
                    height: 8px;
                    background: {colors['accent_green']};
                    border-radius: 50%;
                    animation: pulse 2s ease-in-out infinite;
                "></div>
                <span style="
                    color: {colors['accent_green']};
                    font-size: 0.75rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                ">Live</span>
            </div>
            
            <!-- Current time -->
            <div style="
                color: {colors['text_muted']};
                font-size: 0.8rem;
                font-family: 'JetBrains Mono', monospace;
            ">{current_time}</div>
        </div>
        """
    
    return f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(1.1); }}
    }}
    .nav-tab:hover {{
        background: {colors['bg_hover']} !important;
        color: {colors['text_primary']} !important;
    }}
    </style>
    <div style="
        position: sticky;
        top: 0;
        z-index: 999;
        background: {colors['bg_secondary']};
        border-bottom: 1px solid {colors['border']};
        padding: 0 1.5rem;
        margin: -1rem -1rem 1.5rem -1rem;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 56px;
        ">
            <!-- Navigation tabs (left side) -->
            <div style="
                display: flex;
                gap: 0.25rem;
                height: 100%;
                align-items: flex-end;
            ">
                {nav_items_html}
            </div>
            
            <!-- Status indicator (right side) -->
            {status_html}
        </div>
    </div>
    """


# =============================================================================
# SECTION HEADER
# =============================================================================

def section_header(
    title: str,
    subtitle: Optional[str] = None,
    action_label: Optional[str] = None,
) -> str:
    """
    Section divider with title.
    
    Features:
    - Uppercase title with letter-spacing
    - Optional subtitle in muted color
    - Optional action button on the right (visual only)
    
    Args:
        title: Section title (will be uppercased)
        subtitle: Optional description text
        action_label: Optional action button label (visual only)
    
    Returns:
        HTML string for the section header
    
    Example:
        html = section_header(
            "Market Overview",
            subtitle="Real-time market data and signals",
            action_label="Refresh"
        )
        st.markdown(html, unsafe_allow_html=True)
    """
    colors = get_colors()
    
    subtitle_html = ""
    if subtitle:
        subtitle_html = f"""
        <div style="
            color: {colors['text_muted']};
            font-size: 0.8rem;
            font-weight: 400;
            margin-top: 0.25rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">{subtitle}</div>
        """
    
    action_html = ""
    if action_label:
        action_html = f"""
        <div style="
            color: {colors['accent_blue']};
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            padding: 0.4rem 0.75rem;
            border: 1px solid {colors['accent_blue']}40;
            border-radius: 6px;
            transition: all 0.2s ease;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        " class="section-action">{action_label}</div>
        """
    
    return f"""
    <style>
    .section-action:hover {{
        background: {colors['accent_blue']}15;
        border-color: {colors['accent_blue']};
    }}
    </style>
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid {colors['border_muted']};
    ">
        <div>
            <div style="
                color: {colors['text_secondary']};
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            ">{title.upper()}</div>
            {subtitle_html}
        </div>
        {action_html}
    </div>
    """


# =============================================================================
# PAGE HEADER
# =============================================================================

def page_header(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> str:
    """
    Large page title with optional badge.
    
    Features:
    - Large, bold title
    - Optional subtitle
    - Optional status badge (e.g., "LIVE", "READY", "BETA")
    
    Args:
        title: Main page title
        subtitle: Optional description text
        badge: Optional status badge text
    
    Returns:
        HTML string for the page header
    
    Example:
        html = page_header(
            "Stock Predictor",
            subtitle="ML-powered trading signals",
            badge="LIVE"
        )
        st.markdown(html, unsafe_allow_html=True)
    """
    colors = get_colors()
    
    subtitle_html = ""
    if subtitle:
        subtitle_html = f"""
        <div style="
            color: {colors['text_secondary']};
            font-size: 1rem;
            font-weight: 400;
            margin-top: 0.5rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">{subtitle}</div>
        """
    
    badge_html = ""
    if badge:
        # Determine badge color based on content
        badge_upper = badge.upper()
        if badge_upper in ("LIVE", "ACTIVE", "ONLINE"):
            badge_color = colors['accent_green']
        elif badge_upper in ("READY", "STANDBY"):
            badge_color = colors['accent_blue']
        elif badge_upper in ("BETA", "ALPHA", "DEV"):
            badge_color = colors['accent_purple']
        elif badge_upper in ("ERROR", "OFFLINE", "DOWN"):
            badge_color = colors['accent_red']
        elif badge_upper in ("WARNING", "CAUTION"):
            badge_color = colors['accent_yellow']
        else:
            badge_color = colors['accent_blue']
        
        badge_html = f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.6rem;
            background: {badge_color}15;
            border: 1px solid {badge_color}40;
            border-radius: 4px;
            margin-left: 1rem;
        ">
            <div style="
                width: 6px;
                height: 6px;
                background: {badge_color};
                border-radius: 50%;
                animation: pulse 2s ease-in-out infinite;
            "></div>
            <span style="
                color: {badge_color};
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            ">{badge_upper}</span>
        </div>
        """
    
    return f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}
    </style>
    <div style="
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid {colors['border']};
    ">
        <div style="
            display: flex;
            align-items: center;
        ">
            <h1 style="
                color: {colors['text_primary']};
                font-size: 2rem;
                font-weight: 700;
                margin: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            ">{title}</h1>
            {badge_html}
        </div>
        {subtitle_html}
    </div>
    """


# =============================================================================
# ADDITIONAL NAVIGATION COMPONENTS
# =============================================================================

def breadcrumbs(items: List[Dict[str, str]]) -> str:
    """
    Breadcrumb navigation.
    
    Args:
        items: List of {"label": "Home"} dicts. Last item is current page.
    
    Returns:
        HTML string for breadcrumbs
    """
    colors = get_colors()
    
    crumbs_html = ""
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        label = item.get("label", "")
        
        if is_last:
            crumbs_html += f"""
            <span style="
                color: {colors['text_primary']};
                font-weight: 500;
            ">{label}</span>
            """
        else:
            crumbs_html += f"""
            <span style="
                color: {colors['text_muted']};
                cursor: pointer;
            " class="breadcrumb-link">{label}</span>
            <span style="
                color: {colors['text_muted']};
                margin: 0 0.5rem;
            ">/</span>
            """
    
    return f"""
    <style>
    .breadcrumb-link:hover {{
        color: {colors['accent_blue']};
        text-decoration: underline;
    }}
    </style>
    <div style="
        font-size: 0.8rem;
        margin-bottom: 1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    ">
        {crumbs_html}
    </div>
    """


def divider(margin: str = "1.5rem") -> str:
    """
    Simple horizontal divider.
    
    Args:
        margin: Vertical margin (CSS value)
    
    Returns:
        HTML string for divider
    """
    colors = get_colors()
    
    return f"""
    <div style="
        height: 1px;
        background: {colors['border_muted']};
        margin: {margin} 0;
    "></div>
    """


# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_nav_bar(
    active_tab: str,
    tabs: List[Dict[str, str]],
    show_status: bool = True,
) -> None:
    """Convenience function that renders nav_bar directly via st.markdown."""
    import streamlit as st
    st.markdown(nav_bar(active_tab, tabs, show_status), unsafe_allow_html=True)


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    action_label: Optional[str] = None,
) -> None:
    """Convenience function that renders section_header directly via st.markdown."""
    import streamlit as st
    st.markdown(section_header(title, subtitle, action_label), unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Convenience function that renders page_header directly via st.markdown."""
    import streamlit as st
    st.markdown(page_header(title, subtitle, badge), unsafe_allow_html=True)
