"""
Theme System - Centralized colors and CSS
==========================================

Provides a unified theming system for the Stock Predictor dashboard.
Includes color palettes, CSS generation, and theme switching utilities.

Usage:
    from src.ui.theme import get_theme_mode, get_colors, inject_theme
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict

import streamlit as st


# =============================================================================
# THEME MODE MANAGEMENT
# =============================================================================

def get_theme_mode() -> Literal["dark", "light"]:
    """
    Get current theme mode from session state.
    
    Returns:
        "dark" or "light"
    """
    return st.session_state.get("theme", "dark")


def set_theme_mode(mode: Literal["dark", "light"]) -> None:
    """Set the theme mode in session state."""
    st.session_state["theme"] = mode


def toggle_theme() -> None:
    """Toggle between dark and light theme."""
    current = get_theme_mode()
    set_theme_mode("light" if current == "dark" else "dark")


# =============================================================================
# COLOR SYSTEM
# =============================================================================

def get_colors() -> Dict[str, str]:
    """
    Get all UI colors based on current theme mode.
    
    Returns:
        Dictionary of color name -> hex value
    """
    mode = get_theme_mode()
    
    if mode == "dark":
        return {
            # Backgrounds
            "bg_primary": "#0d1117",
            "bg_secondary": "#161b22",
            "bg_card": "#21262d",
            "bg_elevated": "#30363d",
            "bg_hover": "#484f58",
            
            # Borders
            "border": "#30363d",
            "border_muted": "#21262d",
            "border_accent": "#388bfd",
            
            # Text
            "text_primary": "#e6edf3",
            "text_secondary": "#8b949e",
            "text_muted": "#6e7681",
            "text_link": "#58a6ff",
            
            # Accents
            "accent_blue": "#58a6ff",
            "accent_green": "#3fb950",
            "accent_red": "#f85149",
            "accent_yellow": "#d29922",
            "accent_purple": "#a371f7",
            "accent_cyan": "#39c5cf",
            "accent_orange": "#db6d28",
            
            # Semantic
            "success": "#238636",
            "success_bg": "#238636",
            "warning": "#9e6a03",
            "warning_bg": "#9e6a03",
            "error": "#da3633",
            "error_bg": "#da3633",
            "info": "#1f6feb",
            "info_bg": "#1f6feb",
            
            # Charts
            "chart_bg": "rgba(13,17,23,0)",
            "chart_grid": "rgba(48,54,61,0.6)",
            "chart_line": "#58a6ff",
            
            # Gradients
            "gradient_start": "#388bfd",
            "gradient_end": "#a371f7",
            
            # Shadows
            "shadow": "rgba(0,0,0,0.3)",
            "shadow_strong": "rgba(0,0,0,0.5)",
        }
    else:
        return {
            # Backgrounds
            "bg_primary": "#ffffff",
            "bg_secondary": "#f6f8fa",
            "bg_card": "#ffffff",
            "bg_elevated": "#f3f4f6",
            "bg_hover": "#e5e7eb",
            
            # Borders
            "border": "#d0d7de",
            "border_muted": "#d8dee4",
            "border_accent": "#0969da",
            
            # Text
            "text_primary": "#1f2328",
            "text_secondary": "#57606a",
            "text_muted": "#6e7781",
            "text_link": "#0969da",
            
            # Accents
            "accent_blue": "#0969da",
            "accent_green": "#1a7f37",
            "accent_red": "#cf222e",
            "accent_yellow": "#9a6700",
            "accent_purple": "#8250df",
            "accent_cyan": "#0891b2",
            "accent_orange": "#bc4c00",
            
            # Semantic
            "success": "#1a7f37",
            "success_bg": "#dafbe1",
            "warning": "#9a6700",
            "warning_bg": "#fff8c5",
            "error": "#cf222e",
            "error_bg": "#ffebe9",
            "info": "#0969da",
            "info_bg": "#ddf4ff",
            
            # Charts
            "chart_bg": "rgba(255,255,255,0)",
            "chart_grid": "rgba(208,215,222,0.5)",
            "chart_line": "#0969da",
            
            # Gradients
            "gradient_start": "#0969da",
            "gradient_end": "#8250df",
            
            # Shadows
            "shadow": "rgba(0,0,0,0.1)",
            "shadow_strong": "rgba(0,0,0,0.15)",
        }


# =============================================================================
# CSS GENERATION
# =============================================================================

def generate_base_css() -> str:
    """
    Generate complete base CSS for the application.
    
    Includes:
    - Streamlit branding hiding
    - Font imports (Inter, JetBrains Mono)
    - CSS variables
    - Keyframe animations
    - Sidebar and layout styling
    
    Returns:
        Complete CSS string
    """
    c = get_colors()
    mode = get_theme_mode()
    
    return f"""
    <style>
    /* =========================================================================
       FONT IMPORTS
       ========================================================================= */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* =========================================================================
       CSS VARIABLES
       ========================================================================= */
    :root {{
        /* Colors */
        --bg-primary: {c["bg_primary"]};
        --bg-secondary: {c["bg_secondary"]};
        --bg-card: {c["bg_card"]};
        --bg-elevated: {c["bg_elevated"]};
        --bg-hover: {c["bg_hover"]};
        --border: {c["border"]};
        --border-muted: {c["border_muted"]};
        --border-accent: {c["border_accent"]};
        --text-primary: {c["text_primary"]};
        --text-secondary: {c["text_secondary"]};
        --text-muted: {c["text_muted"]};
        --accent-blue: {c["accent_blue"]};
        --accent-green: {c["accent_green"]};
        --accent-red: {c["accent_red"]};
        --accent-yellow: {c["accent_yellow"]};
        --accent-purple: {c["accent_purple"]};
        
        /* Typography */
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
        
        /* Transitions */
        --transition-fast: 0.15s ease;
        --transition-normal: 0.25s ease;
        --transition-slow: 0.4s ease;
        
        /* Spacing */
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --radius-xl: 20px;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px {c["shadow"]};
        --shadow-md: 0 4px 6px {c["shadow"]};
        --shadow-lg: 0 10px 15px {c["shadow_strong"]};
    }}
    
    /* =========================================================================
       KEYFRAME ANIMATIONS
       ========================================================================= */
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    @keyframes slideInRight {{
        from {{
            transform: translateX(100%);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    @keyframes slideInUp {{
        from {{
            transform: translateY(20px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes fadeOut {{
        from {{ opacity: 1; }}
        to {{ opacity: 0; }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); }}
    }}
    
    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 5px {c["accent_blue"]}40; }}
        50% {{ box-shadow: 0 0 20px {c["accent_blue"]}60; }}
    }}
    
    /* =========================================================================
       HIDE ALL STREAMLIT BRANDING
       ========================================================================= */
    
    /* Main menu hamburger */
    #MainMenu {{
        visibility: hidden !important;
        display: none !important;
    }}
    
    button[kind="header"] {{
        display: none !important;
    }}
    
    /* Footer "Made with Streamlit" */
    footer {{
        visibility: hidden !important;
        display: none !important;
    }}
    
    footer::after {{
        visibility: hidden !important;
        display: none !important;
    }}
    
    /* Header bar and deploy button */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}
    
    .stDeployButton {{
        display: none !important;
    }}
    
    [data-testid="stDeployButton"] {{
        display: none !important;
    }}
    
    /* Toolbar icons */
    .stToolbar {{
        display: none !important;
    }}
    
    [data-testid="stToolbar"] {{
        display: none !important;
    }}
    
    /* Viewer badge */
    .viewerBadge_container__r5tak,
    .viewerBadge_link__qRIco,
    [data-testid="viewerBadge"] {{
        display: none !important;
    }}
    
    /* Status widget (running indicator) */
    [data-testid="stStatusWidget"] {{
        display: none !important;
    }}
    
    .stStatusWidget {{
        display: none !important;
    }}
    
    /* Decoration (top colored bar) */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}
    
    /* App chrome */
    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 1rem !important;
    }}
    
    /* =========================================================================
       BASE LAYOUT & TYPOGRAPHY
       ========================================================================= */
    
    /* Main app container */
    .stApp {{
        background-color: {c["bg_primary"]} !important;
        font-family: var(--font-sans) !important;
    }}
    
    /* Block container padding */
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }}
    
    .appview-container .main .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }}
    
    /* Remove extra padding */
    .main > div:first-child {{
        padding-top: 0 !important;
    }}
    
    .element-container {{
        margin-bottom: 0.5rem !important;
    }}
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-sans) !important;
        color: {c["text_primary"]} !important;
        font-weight: 600 !important;
    }}
    
    p, span, div, label {{
        font-family: var(--font-sans) !important;
    }}
    
    code, pre, .stCode {{
        font-family: var(--font-mono) !important;
    }}
    
    /* Text colors */
    .stMarkdown, .stText, p {{
        color: {c["text_primary"]} !important;
    }}
    
    /* =========================================================================
       SIDEBAR STYLING
       ========================================================================= */
    
    [data-testid="stSidebar"] {{
        background-color: {c["bg_secondary"]} !important;
        border-right: 1px solid {c["border"]} !important;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {c["bg_secondary"]} !important;
        padding-top: 1rem !important;
    }}
    
    [data-testid="stSidebarNav"] {{
        background-color: {c["bg_secondary"]} !important;
    }}
    
    /* Sidebar section titles */
    .section-title {{
        color: {c["text_secondary"]} !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin: 1rem 0 0.5rem 0 !important;
        padding: 0 !important;
    }}
    
    /* =========================================================================
       FORM ELEMENTS
       ========================================================================= */
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {c["bg_card"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: var(--radius-md) !important;
        color: {c["text_primary"]} !important;
        font-family: var(--font-sans) !important;
        transition: border-color var(--transition-fast) !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {c["accent_blue"]} !important;
        box-shadow: 0 0 0 3px {c["accent_blue"]}20 !important;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background-color: {c["bg_card"]} !important;
        border-color: {c["border"]} !important;
        border-radius: var(--radius-md) !important;
    }}
    
    /* Dropdowns */
    [data-baseweb="select"] {{
        background-color: {c["bg_card"]} !important;
    }}
    
    [data-baseweb="popover"] {{
        background-color: {c["bg_elevated"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: var(--radius-md) !important;
    }}
    
    /* Sliders */
    .stSlider > div > div > div {{
        background-color: {c["bg_elevated"]} !important;
    }}
    
    .stSlider > div > div > div > div {{
        background-color: {c["accent_blue"]} !important;
    }}
    
    /* Checkboxes */
    .stCheckbox > label > div[data-testid="stCheckbox"] {{
        background-color: {c["bg_card"]} !important;
        border-color: {c["border"]} !important;
    }}
    
    /* =========================================================================
       BUTTONS
       ========================================================================= */
    
    .stButton > button {{
        background-color: {c["bg_card"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: var(--radius-md) !important;
        color: {c["text_primary"]} !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        transition: all var(--transition-fast) !important;
        padding: 0.5rem 1rem !important;
    }}
    
    .stButton > button:hover {{
        background-color: {c["bg_hover"]} !important;
        border-color: {c["accent_blue"]} !important;
        transform: translateY(-1px) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    
    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, {c["gradient_start"]} 0%, {c["gradient_end"]} 100%) !important;
        border: none !important;
        color: white !important;
    }}
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {{
        box-shadow: var(--shadow-md) !important;
    }}
    
    /* =========================================================================
       TABS
       ========================================================================= */
    
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {c["bg_secondary"]} !important;
        border-radius: var(--radius-md) !important;
        padding: 4px !important;
        gap: 4px !important;
        border: 1px solid {c["border_muted"]} !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        border-radius: var(--radius-sm) !important;
        color: {c["text_secondary"]} !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all var(--transition-fast) !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {c["bg_hover"]} !important;
        color: {c["text_primary"]} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {c["bg_card"]} !important;
        color: {c["text_primary"]} !important;
        box-shadow: var(--shadow-sm) !important;
    }}
    
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    
    /* =========================================================================
       METRICS
       ========================================================================= */
    
    [data-testid="stMetric"] {{
        background-color: {c["bg_card"]} !important;
        border: 1px solid {c["border_muted"]} !important;
        border-radius: var(--radius-md) !important;
        padding: 1rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {c["text_secondary"]} !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {c["text_primary"]} !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-weight: 500 !important;
    }}
    
    [data-testid="stMetricDelta"] svg {{
        display: none !important;
    }}
    
    /* =========================================================================
       DATAFRAMES & TABLES
       ========================================================================= */
    
    .stDataFrame {{
        border: 1px solid {c["border"]} !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
    }}
    
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background-color: {c["bg_card"]} !important;
    }}
    
    /* =========================================================================
       EXPANDERS
       ========================================================================= */
    
    .streamlit-expanderHeader {{
        background-color: {c["bg_secondary"]} !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        color: {c["text_primary"]} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {c["bg_card"]} !important;
        border: 1px solid {c["border_muted"]} !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }}
    
    /* =========================================================================
       ALERTS & MESSAGES
       ========================================================================= */
    
    .stAlert {{
        border-radius: var(--radius-md) !important;
    }}
    
    .stSuccess {{
        background-color: {c["success_bg"]}20 !important;
        border-left: 4px solid {c["success"]} !important;
    }}
    
    .stWarning {{
        background-color: {c["warning_bg"]}20 !important;
        border-left: 4px solid {c["warning"]} !important;
    }}
    
    .stError {{
        background-color: {c["error_bg"]}20 !important;
        border-left: 4px solid {c["error"]} !important;
    }}
    
    .stInfo {{
        background-color: {c["info_bg"]}20 !important;
        border-left: 4px solid {c["info"]} !important;
    }}
    
    /* =========================================================================
       SCROLLBARS
       ========================================================================= */
    
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {c["bg_secondary"]};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {c["border"]};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {c["text_muted"]};
    }}
    
    /* =========================================================================
       UTILITY CLASSES
       ========================================================================= */
    
    .fade-in {{
        animation: fadeIn var(--transition-normal);
    }}
    
    .slide-in {{
        animation: slideInUp var(--transition-normal);
    }}
    
    .shimmer {{
        background: linear-gradient(
            90deg,
            {c["bg_card"]} 0%,
            {c["bg_elevated"]} 50%,
            {c["bg_card"]} 100%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Card hover effect */
    .card-hover {{
        transition: all var(--transition-fast);
    }}
    
    .card-hover:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: {c["accent_blue"]} !important;
    }}
    
    /* Gradient text */
    .gradient-text {{
        background: linear-gradient(135deg, {c["gradient_start"]} 0%, {c["gradient_end"]} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    /* Stale indicator during rerun */
    div[data-stale="true"] {{
        opacity: 0.4;
        transition: opacity var(--transition-fast);
    }}
    
    </style>
    """


# =============================================================================
# THEME INJECTION
# =============================================================================

def inject_theme() -> None:
    """
    Inject all theme CSS into the Streamlit app.
    
    Call this at the top of your Streamlit app, after st.set_page_config().
    """
    st.markdown(generate_base_css(), unsafe_allow_html=True)


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

@dataclass
class ThemeColors:
    """Color palette for a theme (legacy compatibility)."""
    
    bg_primary: str
    bg_secondary: str
    bg_card: str
    bg_elevated: str
    bg_hover: str
    border_default: str
    border_muted: str
    border_accent: str
    text_primary: str
    text_secondary: str
    text_muted: str
    text_link: str
    accent_green: str
    accent_red: str
    accent_blue: str
    accent_yellow: str
    accent_purple: str
    chart_bg: str
    chart_grid: str
    success: str
    warning: str
    error: str
    info: str


def get_theme() -> ThemeColors:
    """Get current theme as ThemeColors dataclass (legacy compatibility)."""
    c = get_colors()
    return ThemeColors(
        bg_primary=c["bg_primary"],
        bg_secondary=c["bg_secondary"],
        bg_card=c["bg_card"],
        bg_elevated=c["bg_elevated"],
        bg_hover=c["bg_hover"],
        border_default=c["border"],
        border_muted=c["border_muted"],
        border_accent=c.get("border_accent", c["accent_blue"]),
        text_primary=c["text_primary"],
        text_secondary=c["text_secondary"],
        text_muted=c["text_muted"],
        text_link=c.get("text_link", c["accent_blue"]),
        accent_green=c["accent_green"],
        accent_red=c["accent_red"],
        accent_blue=c["accent_blue"],
        accent_yellow=c["accent_yellow"],
        accent_purple=c["accent_purple"],
        chart_bg=c["chart_bg"],
        chart_grid=c["chart_grid"],
        success=c["success"],
        warning=c["warning"],
        error=c["error"],
        info=c["info"],
    )


def get_theme_name() -> Literal["dark", "light"]:
    """Alias for get_theme_mode() for backward compatibility."""
    return get_theme_mode()


def generate_hide_streamlit_css() -> str:
    """Legacy function - now included in generate_base_css()."""
    return generate_base_css()


def inject_theme_css() -> None:
    """Legacy alias for inject_theme()."""
    inject_theme()

