# Portfolio Dashboard & Money Management Guide

## Overview

You want to enhance your Streamlit UI to include:
1. **Portfolio tracking** (holdings, P&L, allocation)
2. **Recommendations** (buy/sell signals based on predictions)
3. **Integration pathway** to real money management

This guide covers **GitHub Apps**, **Streamlit components**, **libraries**, and **architectural patterns**.

---

## 🏗️ ARCHITECTURE OVERVIEW

### Current State
```
Your Streamlit App (app.py)
├── Dashboard Tab: Predictions for selected tickers
├── Backtest Tab: Historical performance
└── Portfolio Tab: (PLACEHOLDER - needs implementation)
```

### Target State
```
Your Streamlit App (app.py) + Backend
├── Dashboard Tab: Predictions + Live prices
├── Backtest Tab: Historical performance
├── Portfolio Tab: Holdings + P&L + Allocation
├── Recommendations Tab: AI-generated buy/sell/hold signals
├── Account Integration Tab: Real/Paper trading connection
└── Settings Tab: Risk limits, rebalancing rules, API keys
```

---

## 📊 PART 1: PORTFOLIO TRACKING LIBRARIES

### Option A: **Portfolio-Lab** (Recommended for Stock Traders)

```bash
pip install portfolio-lab
```

**Pros:**
- Built for stock portfolio tracking
- Automatic P&L calculation
- Dividend handling
- Position sizing analytics
- Correlations & beta tracking
- **Perfect for**: Individual stock traders like you

**Cons:**
- Less mature than yfinance
- Fewer real-time features

**Example Usage:**
```python
from portfolio_lab import Portfolio

portfolio = Portfolio(
    assets=['AAPL', 'NVDA', 'PLTR'],
    quantities=[10, 5, 50],
    purchase_prices=[180, 120, 25],
    purchase_dates=['2023-01-01', '2023-06-15', '2024-01-10']
)

# Auto-calculate current values + P&L
print(portfolio.total_value)
print(portfolio.positions)
print(portfolio.allocation)
```

---

### Option B: **RiskLabs Portfolio** (Professional Grade)

```bash
pip install risk-labs
```

**Pros:**
- Professional-grade risk metrics (VaR, CVaR, Sharpe)
- Correlation matrices
- Rebalancing recommendations
- Sector allocation tracking
- **Perfect for**: More sophisticated analysis

**Cons:**
- Overkill if you just want simple tracking
- Steeper learning curve

---

### Option C: **Pickle + Pandas** (DIY - Most Control, Recommended for Now)

```python
# storage/portfolio.json
{
  "holdings": [
    {
      "ticker": "AAPL",
      "quantity": 10,
      "purchase_price": 180.50,
      "purchase_date": "2024-01-15",
      "current_price": 195.00,
      "allocation_pct": 0.35
    }
  ],
  "cash_available": 5000.00,
  "total_portfolio_value": 50000.00
}
```

**Pros:**
- Full control
- Can easily modify for your needs
- No external dependencies
- **Perfect for**: Starting simple, then evolving

**Cons:**
- Must code calculations yourself
- Manual update needed

---

## 🎯 PART 2: STREAMLIT PORTFOLIO COMPONENTS

### 📈 Add to Your `app.py` - Portfolio Tab

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# In your main tabs section (~line 763):
tab_dash, tab_backtests, tab_port, tab_recommend, tab_account = st.tabs(
    ["🔮 Dashboard", "📊 Backtests", "💼 Portfolio", "🎯 Recommendations", "⚙️ Account"]
)

# ============ PORTFOLIO TAB ============
with tab_port:
    st.header("📊 Portfolio Overview")
    
    # Load portfolio data
    portfolio_df = load_portfolio_data()  # Load from JSON or DB
    current_prices = fetch_current_prices(portfolio_df['ticker'].tolist())
    
    # Update portfolio with current prices
    portfolio_df['current_price'] = portfolio_df['ticker'].map(current_prices)
    portfolio_df['current_value'] = portfolio_df['quantity'] * portfolio_df['current_price']
    portfolio_df['cost_basis'] = portfolio_df['quantity'] * portfolio_df['purchase_price']
    portfolio_df['unrealized_pl'] = portfolio_df['current_value'] - portfolio_df['cost_basis']
    portfolio_df['unrealized_pl_pct'] = (portfolio_df['unrealized_pl'] / portfolio_df['cost_basis'] * 100).round(2)
    
    # ===== Summary Metrics =====
    col1, col2, col3, col4 = st.columns(4)
    
    total_value = portfolio_df['current_value'].sum() + 5000  # +5000 cash
    total_cost = portfolio_df['cost_basis'].sum() + 45000  # -5000 from cost
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    with col1:
        st.metric("Portfolio Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Total P&L", f"${total_pl:,.2f}", f"{total_pl_pct:+.2f}%")
    with col3:
        st.metric("Cash Available", f"${5000:,.2f}")
    with col4:
        st.metric("Buying Power", f"${5000 + (total_pl * 0.8):,.2f}")  # 80% margin
    
    # ===== Holdings Table =====
    st.subheader("Holdings")
    display_cols = ['ticker', 'quantity', 'purchase_price', 'current_price', 'current_value', 'unrealized_pl', 'unrealized_pl_pct']
    st.dataframe(
        portfolio_df[display_cols].sort_values('current_value', ascending=False),
        use_container_width=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "quantity": st.column_config.NumberColumn("Qty", width="small"),
            "purchase_price": st.column_config.NumberColumn("Entry", width="small", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Current", width="small", format="$%.2f"),
            "current_value": st.column_config.NumberColumn("Value", width="small", format="$%.2f"),
            "unrealized_pl": st.column_config.NumberColumn("P&L", width="small", format="$%.2f"),
            "unrealized_pl_pct": st.column_config.NumberColumn("Return %", width="small", format="%.2f%%"),
        }
    )
    
    # ===== Allocation Pie Chart =====
    fig_allocation = go.Figure(data=[go.Pie(
        labels=portfolio_df['ticker'].tolist() + ['Cash'],
        values=portfolio_df['current_value'].tolist() + [5000],
        hole=0.3
    )])
    fig_allocation.update_layout(title="Portfolio Allocation", height=400)
    st.plotly_chart(fig_allocation, use_container_width=True)
    
    # ===== Performance Waterfall =====
    fig_waterfall = go.Figure(data=[go.Waterfall(
        name="P&L",
        x=portfolio_df['ticker'].tolist() + ['Total'],
        y=portfolio_df['unrealized_pl'].tolist() + [total_pl],
        connector={"line": {"color": "rgba(63, 63, 63, 0.5)"}},
        increasing={"marker": {"color": "green"}},
        decreasing={"marker": {"color": "red"}},
    )])
    fig_waterfall.update_layout(title="Unrealized P&L Breakdown", height=400)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # ===== Add/Remove Positions =====
    st.subheader("Manage Positions")
    col_add, col_remove = st.columns(2)
    
    with col_add:
        st.write("**Add Position**")
        add_ticker = st.text_input("Ticker to add")
        add_qty = st.number_input("Quantity", min_value=1, value=1)
        add_price = st.number_input("Purchase price", min_value=0.01, value=100.0, format="%.2f")
        if st.button("Add Position"):
            add_portfolio_position(add_ticker, add_qty, add_price)
            st.success(f"Added {add_qty} shares of {add_ticker}")
    
    with col_remove:
        st.write("**Remove Position**")
        remove_ticker = st.selectbox("Ticker to remove", portfolio_df['ticker'].tolist())
        if st.button("Remove Position"):
            remove_portfolio_position(remove_ticker)
            st.success(f"Removed {remove_ticker}")
```

---

## 🎯 PART 3: RECOMMENDATIONS TAB

### Add to `app.py` - Recommendations Tab

```python
# ============ RECOMMENDATIONS TAB ============
with tab_recommend:
    st.header("🎯 AI Recommendations")
    
    # Get predictions for all portfolio holdings + watchlist
    all_tickers = list(set(portfolio_df['ticker'].tolist() + ['PLTR', 'SMCI', 'TSLA']))
    
    recommendations = []
    for tk in all_tickers:
        try:
            pred = predict_next_for_ticker(tk, period="5y", model_type="rf", horizon=1)
            
            # Calculate recommendation strength
            expected_ret = pred.get('pred_next_ret', 0)
            prob_up = pred.get('prob_up', 0.5)
            sharpe_ratio = pred.get('backtest_sharpe', 0)
            
            # Signal logic
            if expected_ret > 0.02 and prob_up > 0.65 and sharpe_ratio > 1.0:
                signal = "🟢 STRONG BUY"
                strength = 10
            elif expected_ret > 0.01 and prob_up > 0.55:
                signal = "🟡 BUY"
                strength = 7
            elif expected_ret < -0.02 and prob_up < 0.35:
                signal = "🔴 STRONG SELL"
                strength = -10
            elif expected_ret < -0.01 and prob_up < 0.45:
                signal = "🟠 SELL"
                strength = -7
            else:
                signal = "⚪ HOLD"
                strength = 0
            
            recommendations.append({
                'ticker': tk,
                'signal': signal,
                'expected_ret': expected_ret,
                'prob_up': prob_up,
                'sharpe': sharpe_ratio,
                'strength': strength,
                'current_price': current_prices.get(tk, 0),
                'in_portfolio': tk in portfolio_df['ticker'].values
            })
        except:
            pass
    
    rec_df = pd.DataFrame(recommendations)
    
    # ===== Filter by Signal =====
    col1, col2 = st.columns(2)
    with col1:
        show_buys = st.checkbox("Show BUY signals", value=True)
        show_sells = st.checkbox("Show SELL signals", value=True)
    with col2:
        show_holds = st.checkbox("Show HOLD signals", value=False)
        show_portfolio_only = st.checkbox("Portfolio holdings only", value=False)
    
    # Filter recommendations
    filtered_rec = rec_df.copy()
    if show_portfolio_only:
        filtered_rec = filtered_rec[filtered_rec['in_portfolio']]
    
    if show_buys or show_sells or show_holds:
        signals_to_show = []
        if show_buys:
            signals_to_show.extend(['🟢 STRONG BUY', '🟡 BUY'])
        if show_sells:
            signals_to_show.extend(['🔴 STRONG SELL', '🟠 SELL'])
        if show_holds:
            signals_to_show.append('⚪ HOLD')
        
        filtered_rec = filtered_rec[filtered_rec['signal'].isin(signals_to_show)]
    
    # ===== Display Recommendations =====
    st.subheader("Current Recommendations")
    
    # Sort by strength (best opportunities first)
    filtered_rec = filtered_rec.sort_values('strength', ascending=False)
    
    for idx, row in filtered_rec.iterrows():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.write(f"**{row['ticker']}**")
        with col2:
            st.write(row['signal'])
        with col3:
            color = "🟢" if row['expected_ret'] > 0 else "🔴"
            st.write(f"{color} {row['expected_ret']:+.2%}")
        with col4:
            st.write(f"P(up): {row['prob_up']:.1%}")
        with col5:
            if st.button("Details", key=f"details_{row['ticker']}"):
                with st.expander(f"{row['ticker']} Analysis"):
                    pred = predict_next_for_ticker(row['ticker'], period="5y")
                    st.write(f"**Expected Return**: {pred.get('pred_next_ret', 0):+.2%}")
                    st.write(f"**Expected Price**: ${pred.get('pred_next_price', 0):.2f}")
                    st.write(f"**Backtest Sharpe**: {pred.get('backtest_sharpe', 0):.2f}")
                    st.write(f"**Hit Rate**: {pred.get('hit_rate', 0):.1%}")
    
    # ===== Suggested Actions =====
    st.subheader("📋 Suggested Portfolio Actions")
    
    strong_buys = filtered_rec[filtered_rec['strength'] == 10]
    strong_sells = filtered_rec[filtered_rec['strength'] == -10]
    
    if len(strong_buys) > 0:
        st.success(f"**🟢 BUY SIGNALS**: {', '.join(strong_buys['ticker'].tolist())}")
        for _, row in strong_buys.iterrows():
            suggested_qty = int(5000 / row['current_price'])  # Allocate $5k per buy
            st.write(f"  └─ Buy {suggested_qty} shares of {row['ticker']} at current price")
    
    if len(strong_sells) > 0:
        st.error(f"**🔴 SELL SIGNALS**: {', '.join(strong_sells['ticker'].tolist())}")
        for _, row in strong_sells.iterrows():
            in_portfolio_qty = portfolio_df[portfolio_df['ticker'] == row['ticker']]['quantity'].values
            if len(in_portfolio_qty) > 0:
                st.write(f"  └─ Sell {in_portfolio_qty[0]} shares of {row['ticker']}")
```

---

## 🔌 PART 4: ACCOUNT INTEGRATION TAB

### Add to `app.py` - Account Integration Tab

```python
# ============ ACCOUNT INTEGRATION TAB ============
with tab_account:
    st.header("⚙️ Account & Trading Integration")
    
    st.info("""
    Ready to connect your real Alpaca account? This tab lets you:
    - Connect paper trading (SANDBOX)
    - Connect live trading (with real $)
    - Set risk limits
    - Schedule automatic trades
    - Monitor executions
    """)
    
    st.subheader("1️⃣ Choose Trading Mode")
    trading_mode = st.radio("Select mode:", ["Paper Trading (Sandbox)", "Live Trading (Real Money)"])
    
    if trading_mode == "Paper Trading (Sandbox)":
        st.write("""
        **Paper Trading** is perfect for testing:
        - No real money at risk
        - Same API, same execution logic
        - Good for validating your strategy
        - Recommended: Start here!
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            paper_api_key = st.text_input("Alpaca Paper API Key", type="password", key="paper_api")
        with col2:
            paper_secret = st.text_input("Alpaca Paper Secret", type="password", key="paper_secret")
        
        if st.button("Connect Paper Account"):
            # Save to .streamlit/secrets.toml
            save_alpaca_credentials("paper", paper_api_key, paper_secret)
            st.success("Paper account connected!")
    
    else:
        st.warning("⚠️ LIVE TRADING - REAL MONEY")
        st.write("""
        Connecting to live trading means:
        - Real money will be used
        - Losses are real
        - Strategy must be thoroughly tested
        - Recommend: Only after 3+ months of paper trading
        """)
        
        if st.checkbox("I understand the risks and want to proceed"):
            col1, col2 = st.columns(2)
            with col1:
                live_api_key = st.text_input("Alpaca Live API Key", type="password", key="live_api")
            with col2:
                live_secret = st.text_input("Alpaca Live Secret", type="password", key="live_secret")
            
            if st.button("⚠️ Connect LIVE Account"):
                save_alpaca_credentials("live", live_api_key, live_secret)
                st.success("LIVE account connected - be careful!")
    
    st.divider()
    
    st.subheader("2️⃣ Risk Management Rules")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_position_size = st.number_input("Max % per position", 0.0, 100.0, 10.0) / 100
    with col2:
        max_loss_pct = st.number_input("Max daily loss %", 0.0, 10.0, 2.0) / 100
    with col3:
        max_leverage = st.number_input("Max leverage", 1.0, 4.0, 2.0)
    
    if st.button("Save Risk Rules"):
        save_risk_rules({
            "max_position_size": max_position_size,
            "max_loss_pct": max_loss_pct,
            "max_leverage": max_leverage
        })
        st.success("Risk rules saved")
    
    st.divider()
    
    st.subheader("3️⃣ Automated Trading Schedule")
    
    col1, col2 = st.columns(2)
    with col1:
        trading_enabled = st.checkbox("Enable automated trading", value=False)
    with col2:
        trade_frequency = st.selectbox("Trade frequency", ["Daily", "Weekly", "Monthly"])
    
    execution_time = st.time_input("Execute at (ET)", value=datetime.time(10, 30))
    
    if st.button("Enable Schedule"):
        if trading_enabled:
            schedule_trades(trade_frequency, execution_time)
            st.success(f"Trading scheduled for {execution_time} {trade_frequency}")
        else:
            st.warning("Enable automated trading first")
    
    st.divider()
    
    st.subheader("4️⃣ Recent Executions")
    
    # Load execution history from auto_paper_trade.py
    executions = load_recent_executions()
    
    if executions and len(executions) > 0:
        exec_df = pd.DataFrame(executions)
        st.dataframe(
            exec_df,
            use_container_width=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time"),
                "ticker": st.column_config.TextColumn("Ticker"),
                "action": st.column_config.TextColumn("Action"),
                "quantity": st.column_config.NumberColumn("Qty"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "status": st.column_config.TextColumn("Status"),
            }
        )
    else:
        st.info("No executions yet")
```

---

## 💾 PART 5: BACKEND HELPERS (New File: `portfolio_manager.py`)

Create a new file to manage portfolio data:

```python
# portfolio_manager.py

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from yfinance import Ticker
import streamlit as st

PORTFOLIO_FILE = Path("storage/portfolio.json")
PORTFOLIO_FILE.parent.mkdir(exist_ok=True)

def load_portfolio_data():
    """Load portfolio from JSON"""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            data = json.load(f)
        return pd.DataFrame(data.get('holdings', []))
    return pd.DataFrame(columns=['ticker', 'quantity', 'purchase_price', 'purchase_date'])

def save_portfolio_data(portfolio_df):
    """Save portfolio to JSON"""
    data = {
        'holdings': portfolio_df.to_dict('records'),
        'last_updated': datetime.now().isoformat(),
        'cash_available': 5000.0
    }
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_portfolio_position(ticker, quantity, purchase_price):
    """Add a position to portfolio"""
    portfolio = load_portfolio_data()
    new_position = pd.DataFrame([{
        'ticker': ticker.upper(),
        'quantity': quantity,
        'purchase_price': purchase_price,
        'purchase_date': datetime.now().strftime('%Y-%m-%d')
    }])
    portfolio = pd.concat([portfolio, new_position], ignore_index=True)
    save_portfolio_data(portfolio)

def remove_portfolio_position(ticker):
    """Remove a position from portfolio"""
    portfolio = load_portfolio_data()
    portfolio = portfolio[portfolio['ticker'] != ticker.upper()]
    save_portfolio_data(portfolio)

def fetch_current_prices(tickers):
    """Fetch current prices for tickers"""
    prices = {}
    for tk in tickers:
        try:
            t = Ticker(tk)
            prices[tk] = t.info.get('currentPrice', 0)
        except:
            prices[tk] = 0
    return prices

@st.cache_data(ttl=60)
def get_portfolio_with_current_prices():
    """Get portfolio with current market prices"""
    portfolio = load_portfolio_data()
    if portfolio.empty:
        return portfolio
    
    prices = fetch_current_prices(portfolio['ticker'].unique().tolist())
    portfolio['current_price'] = portfolio['ticker'].map(prices)
    portfolio['current_value'] = portfolio['quantity'] * portfolio['current_price']
    portfolio['cost_basis'] = portfolio['quantity'] * portfolio['purchase_price']
    portfolio['unrealized_pl'] = portfolio['current_value'] - portfolio['cost_basis']
    
    return portfolio

def save_alpaca_credentials(mode, api_key, secret):
    """Save Alpaca credentials to .streamlit/secrets.toml"""
    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    
    secrets_file = secrets_dir / "secrets.toml"
    
    # Read existing secrets
    if secrets_file.exists():
        with open(secrets_file) as f:
            content = f.read()
    else:
        content = ""
    
    # Add/update credentials
    if mode == "paper":
        content += f'\nalpaca_paper_api_key = "{api_key}"\n'
        content += f'alpaca_paper_secret = "{secret}"\n'
    else:
        content += f'\nalpaca_live_api_key = "{api_key}"\n'
        content += f'alpaca_live_secret = "{secret}"\n'
    
    with open(secrets_file, 'w') as f:
        f.write(content)

def save_risk_rules(rules):
    """Save risk management rules"""
    with open("storage/risk_rules.json", 'w') as f:
        json.dump(rules, f, indent=2)

def schedule_trades(frequency, execution_time):
    """Schedule automated trades"""
    config = {
        'enabled': True,
        'frequency': frequency,
        'execution_time': execution_time.isoformat(),
        'last_run': None
    }
    with open("storage/trading_schedule.json", 'w') as f:
        json.dump(config, f, indent=2)

def load_recent_executions(limit=10):
    """Load recent trade executions"""
    if not Path("storage/executions.json").exists():
        return []
    
    with open("storage/executions.json") as f:
        executions = json.load(f)
    
    return executions[-limit:]
```

---

## 🚀 PART 6: TOP GITHUB APPS & STREAMLIT COMPONENTS

### Best UI Component Libraries

#### **1. Streamlit-Plotly-Events** ⭐ RECOMMENDED
```bash
pip install streamlit-plotly-events
```
- **Use for**: Interactive charts with click/drag events
- **Example**: Click on stock to see detailed predictions
- **GitHub**: https://github.com/null-jones/streamlit-plotly-events

#### **2. Streamlit-Extras** ⭐ HIGHLY RECOMMENDED
```bash
pip install streamlit-extras
```
- **Use for**: Enhanced UI (tabs, metric cards, progress bars)
- **Features**: `grid`, `metric_cards`, `stateful_button`
- **GitHub**: https://github.com/arnaudmiribel/streamlit-extras

#### **3. Streamlit-Aggrid** ⭐ FOR DATA TABLES
```bash
pip install streamlit-aggrid
```
- **Use for**: Interactive data tables with sorting, filtering, editing
- **Example**: Click portfolio row to edit quantity
- **GitHub**: https://github.com/PablocFonseca/streamlit-aggrid

#### **4. Streamlit-Lottie** (For Animations)
```bash
pip install streamlit-lottie
```
- **Use for**: Nice loading animations, success/error states
- **GitHub**: https://github.com/andfanilo/streamlit-lottie

#### **5. Streamlit-Authenticator** (For Login)
```bash
pip install streamlit-authenticator
```
- **Use for**: User authentication before showing portfolio
- **GitHub**: https://github.com/mkhorasani/streamlit-authenticator

---

### Best Execution/Trading Libraries

#### **1. Alpaca Trade API** ⭐ YOU ALREADY HAVE THIS
```bash
pip install alpaca-trade-api
```
- Direct integration with Alpaca paper trading
- Already used in `auto_paper_trade.py`

#### **2. CCXT** (For Crypto + Multiple Brokers)
```bash
pip install ccxt
```
- Support for 100+ exchanges (Binance, Kraken, Coinbase)
- Unified API across brokers
- **Use if you expand to crypto**

#### **3. VectorBT** (For Advanced Backtesting)
```bash
pip install vectorbt
```
- Ultra-fast vectorized backtesting
- Portfolio optimization
- Much faster than walk-forward
- **Consider for next phase**

---

## 📦 PART 7: GITHUB APPS FOR WORKFLOW

### For Monitoring & Alerts

#### **1. GitHub Actions** (Free, Built-in)
```yaml
# .github/workflows/daily-prediction.yml
name: Daily Predictions

on:
  schedule:
    - cron: '09:30 14 * * MON-FRI'  # 9:30 AM ET, weekdays

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run predictions
        run: |
          python -c "from prediction_model import predict_next_for_ticker; import json; print(json.dumps(predict_next_for_ticker('AAPL'), default=str))"
      - name: Send to Discord
        uses: 8398a7/action-slack@v3
        with:
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          text: 'Daily predictions ready!'
```

#### **2. GitHub Issues as Portfolio Tracker**
Use GitHub Issues template for manual trades:
```markdown
---
name: Execute Trade
about: Log a trade execution
---

**Ticker**: AAPL
**Action**: BUY
**Quantity**: 10
**Price**: $195.00
**Reason**: Strong AI signal + positive sentiment
```

#### **3. Deploy to Heroku/Railway** (Free tier available)
```bash
# Create Procfile
web: streamlit run app.py --logger.level=error

# Deploy
git push heroku main
```

---

## 🔧 PART 8: QUICK START IMPLEMENTATION PLAN

### Phase 1: Portfolio Tracking (2-3 hours)
1. Create `portfolio_manager.py` (copy code from Part 5)
2. Add portfolio JSON storage to `app.py`
3. Add **Portfolio Tab** code (from Part 2)
4. Test by adding 2-3 manual positions

### Phase 2: Recommendations (1-2 hours)
1. Add **Recommendations Tab** code (from Part 3)
2. Integrate with your existing `predict_next_for_ticker()` function
3. Create signal logic (BUY/SELL/HOLD based on predictions)
4. Test recommendations on your current positions

### Phase 3: Account Integration (2-3 hours)
1. Get free Alpaca Paper Trading account (https://app.alpaca.markets)
2. Add **Account Tab** code (from Part 4)
3. Connect paper account and test 1 live trade
4. Monitor execution in your `paper_trading_tracker.py`

### Phase 4: UI Polish (1 hour)
1. Install `streamlit-aggrid` for fancy tables
2. Install `streamlit-extras` for better metrics
3. Add color coding (green buys, red sells)
4. Add emojis + icons for visual appeal

### Phase 5: Automation (2-3 hours)
1. Set up GitHub Actions workflow
2. Schedule daily predictions at 8:30 AM ET
3. Send alerts to Slack/Discord
4. Auto-execute top signals (optional - start with manual approval)

---

## ⚡ OPTIONAL: LIVE DASHBOARD FEATURES

### Real-time P&L Updates
```python
# Auto-refresh every 60 seconds
if st.button("🔄 Refresh Prices"):
    st.cache_data.clear()
    st.rerun()

# Or automatic (requires pro Streamlit)
with st.spinner("Updating prices..."):
    portfolio = get_portfolio_with_current_prices()
    st.rerun()
```

### Mobile-Friendly Design
```python
# Streamlit config (.streamlit/config.toml)
[client]
showSidebarNavigation = false  # Hide on mobile

[theme]
primaryColor = "#00D9FF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

### Dark Mode Toggle
```python
is_dark = st.toggle("🌙 Dark mode", value=True)
if is_dark:
    st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    </style>
    """, unsafe_allow_html=True)
```

---

## 🎯 RECOMMENDED NEXT STEPS

1. **This week**: Implement Portfolio Tab + add 3 real positions
2. **Next week**: Add Recommendations Tab + test signals
3. **Week 3**: Connect Alpaca paper account
4. **Week 4**: Start executing automated trades on paper account
5. **Month 2**: If 50+ Sharpe on paper, consider live account

---

## 📚 RESOURCES

- **Streamlit Docs**: https://docs.streamlit.io
- **Alpaca Docs**: https://alpaca.markets/docs/api-references/
- **Plotly Finance**: https://plotly.com/python/financial-charts/
- **Portfolio-Lab**: https://github.com/dcajasn/Portfolio-Lab
- **VectorBT**: https://polastr.github.io/vectorbt/

---

**Last Updated**: December 28, 2025
**Estimated Implementation Time**: 8-10 hours total
**Difficulty Level**: Medium (mostly copy/paste + configuration)
