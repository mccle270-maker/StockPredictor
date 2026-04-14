#!/usr/bin/env python3
"""
Script to modernize app_modern.py with the new UI components.
This applies all necessary changes to remove Streamlit-looking elements.
"""

import re

def modernize_app():
    # Read the file
    with open('app_modern.py', 'r') as f:
        content = f.read()
    
    # 1. Update docstring
    content = content.replace(
        '"""Stock Predictor - Clean Streamlit App',
        '"""Stock Predictor - Modern Professional Dashboard'
    )
    content = content.replace(
        'Run with: streamlit run app_new.py',
        'Run with: streamlit run app_modern.py'
    )
    
    # 2. Add new UI imports after existing imports
    old_imports = "from src.ui.components import ("
    new_imports = """# =============================================================================
# NEW MODERN UI COMPONENTS - Professional Component Library
# =============================================================================
from src.ui.theme import inject_theme, get_colors, get_theme_mode, toggle_theme
from src.ui.cards import metric_card, stat_card_row, info_card
from src.ui.notifications import toast, inline_alert, status_badge
from src.ui.loaders import skeleton_card, skeleton_table, loading_spinner, skeleton_chart
from src.ui.tables import styled_table, mini_table
from src.ui.navigation import nav_bar, section_header, page_header, divider

from src.ui.components import ("""
    
    content = content.replace(old_imports, new_imports)
    
    # 3. Add helper functions after the existing helper functions section
    helper_section = "# ============================================================================\n# HELPER FUNCTIONS\n# ============================================================================\n"
    new_helper_section = helper_section + """
def show_toast(message: str, msg_type: str = "info"):
    \"\"\"Show a toast notification using the modern component.\"\"\"
    st.markdown(toast(message, type=msg_type), unsafe_allow_html=True)

def show_alert(message: str, msg_type: str = "info"):
    \"\"\"Show an inline alert using the modern component.\"\"\"
    st.markdown(inline_alert(message, type=msg_type), unsafe_allow_html=True)

"""
    content = content.replace(helper_section, new_helper_section)
    
    # 4. Add theme injection after existing CSS injection
    old_theme_inject = "# Inject theme-aware CSS\nis_dark_mode = st.session_state.get(\"theme\", \"dark\") == \"dark\"\nst.markdown(generate_theme_css(is_dark_mode), unsafe_allow_html=True)"
    new_theme_inject = old_theme_inject + "\n\n# Inject new component library theme\ninject_theme()"
    content = content.replace(old_theme_inject, new_theme_inject)
    
    # 5. Replace st.tabs with custom navigation
    old_tabs = '''tab_summary, tab_dash, tab_backtests, tab_port, tab_monitor = st.tabs([
    "⚡ SUMMARY", "📈 DASHBOARD", "🔬 BACKTEST", "📊 PORTFOLIO", "🔔 MONITORING"
])'''
    
    new_tabs = '''# Initialize active tab in session state
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "summary"

# Define navigation tabs
NAV_TABS = [
    {"key": "summary", "icon": "⚡", "label": "Summary"},
    {"key": "dashboard", "icon": "📈", "label": "Dashboard"},
    {"key": "backtest", "icon": "🔬", "label": "Backtest"},
    {"key": "portfolio", "icon": "📊", "label": "Portfolio"},
    {"key": "monitor", "icon": "🔔", "label": "Monitoring"},
]

# Custom navigation styling
colors = get_colors()
st.markdown(f"""
<style>
.nav-container {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid {colors['border']};
    margin-bottom: 1.5rem;
}}
</style>
""", unsafe_allow_html=True)

# Navigation buttons
nav_cols = st.columns(len(NAV_TABS))
for i, tab in enumerate(NAV_TABS):
    with nav_cols[i]:
        is_active = st.session_state["active_tab"] == tab["key"]
        btn_label = f"{tab['icon']} {tab['label']}"
        if is_active:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 0.75rem 0;
                color: {colors['accent_blue']};
                border-bottom: 2px solid {colors['accent_blue']};
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 0.85rem;
                font-weight: 600;
            ">{btn_label}</div>
            """, unsafe_allow_html=True)
        else:
            if st.button(btn_label, key=f"nav_{tab['key']}", use_container_width=True):
                st.session_state["active_tab"] = tab["key"]
                st.rerun()

st.markdown(divider(), unsafe_allow_html=True)

# Get current active tab
active_tab = st.session_state["active_tab"]

# Dummy variables to avoid NameError (will be overwritten in if blocks)
tab_summary = tab_dash = tab_backtests = tab_port = tab_monitor = None'''
    
    content = content.replace(old_tabs, new_tabs)
    
    # 6. Replace "with tab_xxx:" with "if active_tab == 'xxx':"
    content = content.replace('with tab_summary:', 'if active_tab == "summary":')
    content = content.replace('with tab_dash:', 'elif active_tab == "dashboard":')
    content = content.replace('with tab_backtests:', 'elif active_tab == "backtest":')
    content = content.replace('with tab_port:', 'elif active_tab == "portfolio":')
    content = content.replace('with tab_monitor:', 'elif active_tab == "monitor":')
    
    # 7. Add floating run button at the end of file
    floating_button = '''
# ============================================================================
# FLOATING RUN PREDICTIONS BUTTON
# ============================================================================
colors = get_colors()
st.markdown(f"""
<style>
.floating-btn {{
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 9999;
    background: linear-gradient(135deg, {colors['accent_blue']} 0%, {colors['accent_purple']} 100%);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 1rem 1.5rem;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.floating-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
}}
</style>
<div class="floating-btn" onclick="document.querySelector('[data-testid=\\'baseButton-primary\\']').click()">
    🚀 Run Predictions
</div>
""", unsafe_allow_html=True)
'''
    
    content += floating_button
    
    # Write the modified content
    with open('app_modern.py', 'w') as f:
        f.write(content)
    
    print("✓ Modernization complete!")
    print("  - Added new UI component imports")
    print("  - Added helper functions (show_toast, show_alert)")
    print("  - Replaced st.tabs with custom navigation")
    print("  - Added floating Run Predictions button")
    print("  - Injected new theme")

if __name__ == "__main__":
    modernize_app()
