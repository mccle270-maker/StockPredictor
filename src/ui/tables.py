"""
Table Components - Styled data tables
=====================================

Professional data table components with highlighting, formatting,
and custom styling for financial data display.

All components return HTML strings for flexible rendering.

Usage:
    from src.ui.tables import styled_table, mini_table
    
    # Render a DataFrame with highlighting
    html = styled_table(df, highlight_rules={"SYMBOL": {"type": "ticker"}})
    st.markdown(html, unsafe_allow_html=True)
"""

from __future__ import annotations

import pandas as pd
import math
from typing import Optional, Dict, List, Any, Union

from .theme import get_colors


# =============================================================================
# VALUE FORMATTERS
# =============================================================================

def _format_value(value: Any, rule: Optional[Dict] = None, colors: Dict[str, str] = None) -> str:
    """
    Format a cell value based on its highlight rule.
    
    Args:
        value: The cell value
        rule: Highlight rule dict with "type" key
        colors: Color palette from get_colors()
    
    Returns:
        Formatted HTML string for the cell content
    """
    if colors is None:
        colors = get_colors()
    
    # Handle NaN/None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<span style="color: {colors["text_muted"]};">—</span>'
    
    if rule is None:
        # Default formatting
        if isinstance(value, float):
            return f'{value:.2f}'
        return str(value)
    
    rule_type = rule.get("type", "default")
    
    if rule_type == "ticker":
        # Blue, bold ticker symbol
        return f'<span style="color: {colors["accent_blue"]}; font-weight: 600;">{value}</span>'
    
    elif rule_type == "pct_direction":
        # Green if positive, red if negative, with +/- and %
        try:
            num = float(value)
            if num > 0:
                color = colors["accent_green"]
                formatted = f"+{num:.2f}%"
            elif num < 0:
                color = colors["accent_red"]
                formatted = f"{num:.2f}%"
            else:
                color = colors["text_secondary"]
                formatted = "0.00%"
            return f'<span style="color: {color}; font-weight: 500;">{formatted}</span>'
        except (ValueError, TypeError):
            return str(value)
    
    elif rule_type == "prob":
        # Green if >55%, red if <45%, neutral otherwise
        try:
            num = float(value)
            # Handle both 0-1 and 0-100 ranges
            if num <= 1:
                num_pct = num * 100
            else:
                num_pct = num
            
            if num_pct > 55:
                color = colors["accent_green"]
            elif num_pct < 45:
                color = colors["accent_red"]
            else:
                color = colors["text_secondary"]
            
            return f'<span style="color: {color}; font-weight: 500;">{num_pct:.1f}%</span>'
        except (ValueError, TypeError):
            return str(value)
    
    elif rule_type == "currency":
        # Format as $X,XXX.XX
        try:
            num = float(value)
            formatted = f"${num:,.2f}"
            return f'<span style="font-family: \'JetBrains Mono\', monospace;">{formatted}</span>'
        except (ValueError, TypeError):
            return str(value)
    
    elif rule_type == "number":
        # Plain number with optional decimals
        decimals = rule.get("decimals", 2)
        try:
            num = float(value)
            return f'{num:.{decimals}f}'
        except (ValueError, TypeError):
            return str(value)
    
    elif rule_type == "signal":
        # BUY/SELL/HOLD with colors
        val_upper = str(value).upper()
        if val_upper in ("BUY", "LONG"):
            color = colors["accent_green"]
            icon = "▲"
        elif val_upper in ("SELL", "SHORT"):
            color = colors["accent_red"]
            icon = "▼"
        else:
            color = colors["text_secondary"]
            icon = "●"
        return f'<span style="color: {color}; font-weight: 600;">{icon} {val_upper}</span>'
    
    else:
        # Default
        return str(value)


# =============================================================================
# STYLED TABLE
# =============================================================================

def styled_table(
    df: pd.DataFrame,
    highlight_rules: Optional[Dict[str, Dict]] = None,
    max_height: int = 400,
    compact: bool = False,
) -> str:
    """
    Render a pandas DataFrame as a professional HTML table.
    
    Features:
    - Sticky header row
    - Alternating row colors
    - Hover effect on rows
    - Scrollable with max_height
    - Configurable highlighting rules
    
    Args:
        df: DataFrame to display
        highlight_rules: Dict mapping column names to highlight rules.
            Example: {"SYMBOL": {"type": "ticker"}, "PRED %": {"type": "pct_direction"}}
            Supported types:
            - "ticker": blue, bold
            - "pct_direction": green if positive, red if negative, shows +/- and %
            - "prob": green if >55%, red if <45%
            - "currency": formats as $X,XXX.XX
            - "signal": BUY/SELL/HOLD with colors and icons
            - "number": plain number with optional "decimals" key
        max_height: Maximum table height in pixels before scrolling
        compact: Use smaller padding for dense displays
    
    Returns:
        HTML string for the styled table
    
    Example:
        html = styled_table(
            df,
            highlight_rules={
                "SYMBOL": {"type": "ticker"},
                "PRED %": {"type": "pct_direction"},
                "P(UP)": {"type": "prob"},
                "PRICE": {"type": "currency"},
            }
        )
        st.markdown(html, unsafe_allow_html=True)
    """
    colors = get_colors()
    
    if df is None or df.empty:
        return f"""
        <div style="
            background: {colors['bg_card']};
            border: 1px solid {colors['border']};
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            color: {colors['text_muted']};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            No data available
        </div>
        """
    
    if highlight_rules is None:
        highlight_rules = {}
    
    # Sizing
    cell_padding = "0.5rem 0.75rem" if compact else "0.75rem 1rem"
    font_size = "0.8rem" if compact else "0.875rem"
    header_font_size = "0.7rem" if compact else "0.75rem"
    
    # Build header row
    header_cells = ""
    for col in df.columns:
        header_cells += f"""
        <th style="
            padding: {cell_padding};
            text-align: left;
            font-weight: 600;
            font-size: {header_font_size};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: {colors['text_secondary']};
            background: {colors['bg_secondary']};
            border-bottom: 2px solid {colors['border']};
            position: sticky;
            top: 0;
            z-index: 10;
        ">{col}</th>
        """
    
    # Build body rows
    body_rows = ""
    for idx, row in df.iterrows():
        row_idx = df.index.get_loc(idx) if hasattr(df.index, 'get_loc') else idx
        bg_color = colors['bg_card'] if row_idx % 2 == 0 else colors['bg_secondary']
        
        cells = ""
        for col in df.columns:
            value = row[col]
            rule = highlight_rules.get(col)
            formatted = _format_value(value, rule, colors)
            
            cells += f"""
            <td style="
                padding: {cell_padding};
                font-size: {font_size};
                color: {colors['text_primary']};
                border-bottom: 1px solid {colors['border_muted']};
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            ">{formatted}</td>
            """
        
        body_rows += f"""
        <tr style="background: {bg_color};" class="table-row">
            {cells}
        </tr>
        """
    
    # Generate unique ID for scoped styles
    import random
    table_id = f"styled-table-{random.randint(1000, 9999)}"
    
    return f"""
    <style>
    #{table_id} .table-row:hover {{
        background: {colors['bg_hover']} !important;
    }}
    #{table_id} .table-row:hover td {{
        background: transparent !important;
    }}
    </style>
    <div id="{table_id}" style="
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        overflow: hidden;
        max-height: {max_height}px;
        overflow-y: auto;
    ">
        <table style="
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {body_rows}
            </tbody>
        </table>
    </div>
    """


# =============================================================================
# MINI TABLE
# =============================================================================

def mini_table(
    data: List[Dict[str, Any]],
    columns: List[str],
) -> str:
    """
    Simple inline table for small data displays.
    
    Simpler styling, no hover effects. Good for summary info
    or small data snippets.
    
    Args:
        data: List of dictionaries, each dict is a row
        columns: List of column names to display (in order)
    
    Returns:
        HTML string for the mini table
    
    Example:
        data = [
            {"Ticker": "AAPL", "Price": 150.25, "Change": 2.5},
            {"Ticker": "MSFT", "Price": 380.10, "Change": -1.2},
        ]
        html = mini_table(data, columns=["Ticker", "Price", "Change"])
        st.markdown(html, unsafe_allow_html=True)
    """
    colors = get_colors()
    
    if not data or not columns:
        return f"""
        <div style="
            color: {colors['text_muted']};
            font-size: 0.8rem;
            padding: 0.5rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">No data</div>
        """
    
    # Build header
    header_cells = ""
    for col in columns:
        header_cells += f"""
        <th style="
            padding: 0.4rem 0.6rem;
            text-align: left;
            font-weight: 600;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            color: {colors['text_secondary']};
            border-bottom: 1px solid {colors['border']};
        ">{col}</th>
        """
    
    # Build body rows
    body_rows = ""
    for row in data:
        cells = ""
        for col in columns:
            value = row.get(col, "")
            
            # Handle NaN/None
            if value is None or (isinstance(value, float) and math.isnan(value)):
                formatted = f'<span style="color: {colors["text_muted"]};">—</span>'
            elif isinstance(value, float):
                formatted = f'{value:.2f}'
            else:
                formatted = str(value)
            
            cells += f"""
            <td style="
                padding: 0.4rem 0.6rem;
                font-size: 0.8rem;
                color: {colors['text_primary']};
                border-bottom: 1px solid {colors['border_muted']};
            ">{formatted}</td>
            """
        
        body_rows += f"<tr>{cells}</tr>"
    
    return f"""
    <div style="
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        overflow: hidden;
    ">
        <table style="
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {body_rows}
            </tbody>
        </table>
    </div>
    """


# =============================================================================
# HELPER: Render functions for convenience
# =============================================================================

def render_styled_table(
    df: pd.DataFrame,
    highlight_rules: Optional[Dict[str, Dict]] = None,
    max_height: int = 400,
    compact: bool = False,
) -> None:
    """Convenience function that renders styled_table directly via st.markdown."""
    import streamlit as st
    st.markdown(
        styled_table(df, highlight_rules, max_height, compact),
        unsafe_allow_html=True
    )


def render_mini_table(
    data: List[Dict[str, Any]],
    columns: List[str],
) -> None:
    """Convenience function that renders mini_table directly via st.markdown."""
    import streamlit as st
    st.markdown(mini_table(data, columns), unsafe_allow_html=True)
