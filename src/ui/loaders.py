"""
Loading Components - Skeleton loaders and spinners
===================================================

Skeleton placeholders and loading indicators for polished UX
during data fetching and model inference.

All components return HTML strings for flexible rendering.

Usage:
    from src.ui.loaders import skeleton_card, skeleton_table, skeleton_chart, loading_spinner
    
    # Display skeleton while loading
    st.markdown(skeleton_card(), unsafe_allow_html=True)
    
    # Show spinner with message
    st.markdown(loading_spinner(message="Loading predictions..."), unsafe_allow_html=True)
"""

from __future__ import annotations

from typing import Optional

from .theme import get_colors


# =============================================================================
# SHIMMER ANIMATION CSS (shared across components)
# =============================================================================

def _get_shimmer_css() -> str:
    """Get the shimmer animation keyframes."""
    return """
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    """


def _shimmer_gradient(colors: dict) -> str:
    """Get the shimmer gradient background style."""
    return f"""
        background: linear-gradient(
            90deg,
            {colors['bg_elevated']} 0%,
            {colors['bg_hover']} 50%,
            {colors['bg_elevated']} 100%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s ease-in-out infinite;
    """


# =============================================================================
# SKELETON CARD
# =============================================================================

def skeleton_card(
    height: int = 120,
    width: str = "100%",
) -> str:
    """
    Skeleton placeholder for a metric card.
    
    Shows 3 animated bars representing label, value, and delta.
    Uses shimmer animation (gradient moving left to right).
    
    Args:
        height: Card height in pixels (default 120)
        width: Card width as CSS value (default "100%")
    
    Returns:
        HTML string for the skeleton card
    
    Example:
        st.markdown(skeleton_card(), unsafe_allow_html=True)
    """
    colors = get_colors()
    shimmer = _shimmer_gradient(colors)
    
    return f"""
    <style>{_get_shimmer_css()}</style>
    <div style="
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 1.25rem;
        height: {height}px;
        width: {width};
        box-sizing: border-box;
    ">
        <!-- Label skeleton (small, 40% width) -->
        <div style="
            {shimmer}
            height: 12px;
            width: 40%;
            border-radius: 4px;
            margin-bottom: 16px;
        "></div>
        
        <!-- Value skeleton (large, 60% width) -->
        <div style="
            {shimmer}
            height: 28px;
            width: 60%;
            border-radius: 6px;
            margin-bottom: 12px;
        "></div>
        
        <!-- Delta skeleton (small, 30% width) -->
        <div style="
            {shimmer}
            height: 14px;
            width: 30%;
            border-radius: 4px;
        "></div>
    </div>
    """


# =============================================================================
# SKELETON TABLE
# =============================================================================

def skeleton_table(
    rows: int = 5,
    cols: int = 4,
) -> str:
    """
    Skeleton placeholder for a data table.
    
    Shows animated header row and body rows with staggered animation delays.
    
    Args:
        rows: Number of body rows (default 5)
        cols: Number of columns (default 4)
    
    Returns:
        HTML string for the skeleton table
    
    Example:
        st.markdown(skeleton_table(rows=10, cols=6), unsafe_allow_html=True)
    """
    colors = get_colors()
    shimmer = _shimmer_gradient(colors)
    
    # Build header cells
    header_cells = ""
    for c in range(cols):
        delay = c * 0.1  # Stagger columns
        header_cells += f"""
        <div style="
            {shimmer}
            animation-delay: {delay}s;
            height: 14px;
            width: 70%;
            border-radius: 4px;
        "></div>
        """
    
    # Build body rows with staggered animation
    body_rows = ""
    for r in range(rows):
        row_delay = r * 0.05  # Stagger rows
        cells = ""
        for c in range(cols):
            cell_delay = row_delay + (c * 0.03)
            # Vary widths for realism
            widths = ["55%", "70%", "45%", "60%", "50%", "65%"]
            width = widths[(r + c) % len(widths)]
            cells += f"""
            <div style="
                {shimmer}
                animation-delay: {cell_delay}s;
                height: 12px;
                width: {width};
                border-radius: 4px;
            "></div>
            """
        
        body_rows += f"""
        <div style="
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            gap: 1rem;
            padding: 0.875rem 1rem;
            border-bottom: 1px solid {colors['border_muted']};
            align-items: center;
        ">
            {cells}
        </div>
        """
    
    return f"""
    <style>{_get_shimmer_css()}</style>
    <div style="
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        overflow: hidden;
    ">
        <!-- Header row -->
        <div style="
            display: grid;
            grid-template-columns: repeat({cols}, 1fr);
            gap: 1rem;
            padding: 1rem;
            background: {colors['bg_secondary']};
            border-bottom: 1px solid {colors['border']};
            align-items: center;
        ">
            {header_cells}
        </div>
        
        <!-- Body rows -->
        {body_rows}
    </div>
    """


# =============================================================================
# SKELETON CHART
# =============================================================================

def skeleton_chart(height: int = 300) -> str:
    """
    Skeleton placeholder for a chart.
    
    Shows fake bar chart shapes with shimmer animation.
    
    Args:
        height: Chart height in pixels (default 300)
    
    Returns:
        HTML string for the skeleton chart
    
    Example:
        st.markdown(skeleton_chart(height=400), unsafe_allow_html=True)
    """
    colors = get_colors()
    shimmer = _shimmer_gradient(colors)
    
    # Generate varying bar heights for realistic appearance
    bar_heights = [65, 45, 80, 55, 70, 40, 85, 50, 75, 60, 90, 48]
    
    bars_html = ""
    for i, bar_pct in enumerate(bar_heights):
        delay = i * 0.08
        bars_html += f"""
        <div style="
            {shimmer}
            animation-delay: {delay}s;
            width: 100%;
            height: {bar_pct}%;
            border-radius: 4px 4px 0 0;
            align-self: flex-end;
        "></div>
        """
    
    return f"""
    <style>{_get_shimmer_css()}</style>
    <div style="
        background: {colors['bg_card']};
        border: 1px solid {colors['border']};
        border-radius: 12px;
        padding: 1.5rem;
        height: {height}px;
        box-sizing: border-box;
    ">
        <!-- Chart title skeleton -->
        <div style="
            {shimmer}
            height: 16px;
            width: 30%;
            border-radius: 4px;
            margin-bottom: 1.5rem;
        "></div>
        
        <!-- Bar chart area -->
        <div style="
            display: flex;
            gap: 8px;
            height: calc(100% - 50px);
            align-items: flex-end;
            padding-bottom: 1rem;
            border-bottom: 1px solid {colors['border_muted']};
        ">
            {bars_html}
        </div>
        
        <!-- X-axis labels skeleton -->
        <div style="
            display: flex;
            justify-content: space-between;
            padding-top: 0.75rem;
        ">
            <div style="{shimmer} height: 10px; width: 8%; border-radius: 3px;"></div>
            <div style="{shimmer} height: 10px; width: 8%; border-radius: 3px;"></div>
            <div style="{shimmer} height: 10px; width: 8%; border-radius: 3px;"></div>
            <div style="{shimmer} height: 10px; width: 8%; border-radius: 3px;"></div>
        </div>
    </div>
    """


# =============================================================================
# LOADING SPINNER
# =============================================================================

def loading_spinner(
    size: int = 40,
    message: Optional[str] = None,
) -> str:
    """
    Custom loading spinner (not the Streamlit default).
    
    Shows a circular spinner with optional message below.
    Uses spin animation for rotation.
    
    Args:
        size: Spinner diameter in pixels (default 40)
        message: Optional message to display below spinner
    
    Returns:
        HTML string for the loading spinner
    
    Example:
        st.markdown(loading_spinner(message="Analyzing..."), unsafe_allow_html=True)
    """
    colors = get_colors()
    
    border_width = max(3, size // 12)
    
    message_html = ""
    if message:
        message_html = f"""
        <div style="
            color: {colors['text_secondary']};
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 1rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">{message}</div>
        """
    
    return f"""
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    </style>
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    ">
        <div style="
            width: {size}px;
            height: {size}px;
            border: {border_width}px solid {colors['bg_elevated']};
            border-top-color: {colors['accent_blue']};
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        "></div>
        {message_html}
    </div>
    """


# =============================================================================
# ADDITIONAL LOADERS
# =============================================================================

def loading_dots(message: str = "Processing") -> str:
    """
    Animated loading dots indicator.
    
    Args:
        message: Text to display before the dots
    
    Returns:
        HTML string for the loading dots
    """
    colors = get_colors()
    
    return f"""
    <style>
    @keyframes blink {{
        0%, 20% {{ opacity: 0.2; }}
        50% {{ opacity: 1; }}
        80%, 100% {{ opacity: 0.2; }}
    }}
    .loader-dot {{
        display: inline-block;
        animation: blink 1.4s infinite both;
        color: {colors['accent_blue']};
        font-size: 1.2em;
    }}
    .loader-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .loader-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    </style>
    <div style="
        color: {colors['text_secondary']};
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    ">
        {message}
        <span class="loader-dot">•</span>
        <span class="loader-dot">•</span>
        <span class="loader-dot">•</span>
    </div>
    """


def skeleton_text(lines: int = 3) -> str:
    """
    Skeleton placeholder for text content.
    
    Args:
        lines: Number of text lines to show
    
    Returns:
        HTML string for the skeleton text
    """
    colors = get_colors()
    shimmer = _shimmer_gradient(colors)
    
    widths = ["100%", "85%", "70%", "90%", "75%", "95%", "65%"]
    
    lines_html = ""
    for i in range(lines):
        width = widths[i % len(widths)]
        delay = i * 0.1
        lines_html += f"""
        <div style="
            {shimmer}
            animation-delay: {delay}s;
            height: 14px;
            width: {width};
            border-radius: 4px;
            margin-bottom: 10px;
        "></div>
        """
    
    return f"""
    <style>{_get_shimmer_css()}</style>
    <div style="padding: 0.5rem 0;">
        {lines_html}
    </div>
    """


def progress_bar(
    progress: float,
    label: Optional[str] = None,
    show_percentage: bool = True,
) -> str:
    """
    Custom progress bar.
    
    Args:
        progress: Progress value from 0.0 to 1.0
        label: Optional label above the bar
        show_percentage: Show percentage text
    
    Returns:
        HTML string for the progress bar
    """
    colors = get_colors()
    
    pct = max(0, min(100, progress * 100))
    
    label_html = ""
    if label:
        label_html = f"""
        <div style="
            color: {colors['text_secondary']};
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        ">{label}</div>
        """
    
    pct_html = ""
    if show_percentage:
        pct_html = f"""
        <span style="
            color: {colors['text_primary']};
            font-size: 0.85rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        ">{pct:.0f}%</span>
        """
    
    return f"""
    <div style="width: 100%;">
        {label_html}
        <div style="
            display: flex;
            align-items: center;
            gap: 0.75rem;
        ">
            <div style="
                flex: 1;
                height: 8px;
                background: {colors['bg_elevated']};
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct}%;
                    height: 100%;
                    background: linear-gradient(90deg, {colors['accent_blue']}, {colors['accent_cyan']});
                    border-radius: 4px;
                    transition: width 0.3s ease;
                "></div>
            </div>
            {pct_html}
        </div>
    </div>
    """
