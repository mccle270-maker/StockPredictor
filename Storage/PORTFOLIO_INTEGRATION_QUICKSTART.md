# PORTFOLIO_INTEGRATION_QUICKSTART.md

## Quick Start: Add Portfolio to Your Streamlit App

This guide shows you how to add portfolio tracking to your existing `app.py` in **30 minutes**.

---

## ✅ Step 1: Test the Portfolio Manager (5 min)

```bash
# Test the new portfolio_manager.py module
python portfolio_manager.py
```

Expected output:
```
Testing portfolio_manager.py...

1. Adding position...

2. Loading portfolio...
  ticker  quantity  purchase_price purchase_date
0   AAPL        10          180.50     2024-12-28
1   NVDA         5          120.00     2024-12-28

3. Getting portfolio with prices...
  [includes current prices]

4. Getting summary...
  total_value: 50000.50
  total_pl: 1234.56
  ...

✅ All tests passed!
```

---

## ✅ Step 2: Add Portfolio Tab to `app.py`

Find this section in `app.py` (around line 763):

```python
tab_dash, tab_backtests, tab_port = st.tabs(
    ["🔮 Dashboard", "📊 Backtests", "💼 Portfolio"]
)
```

**REPLACE** with:

```python
tab_dash, tab_backtests, tab_port, tab_recommend = st.tabs(
    ["🔮 Dashboard", "📊 Backtests", "💼 Portfolio", "🎯 Recommendations"]
)
```

---

## ✅ Step 3: Add Portfolio Tab Implementation

**Find the end of your `app.py`** (around line 1800) and add:

```python
# ============ PORTFOLIO TAB ============
with tab_port:
    from portfolio_manager import (
        load_portfolio_data,
        get_portfolio_with_prices,
        get_portfolio_summary,
        get_allocation,
        add_position,
        remove_position,
        log_execution,
        get_recent_executions,
    )
    
    st.header("💼 Portfolio Overview")
    
    # Get portfolio data
    portfolio = get_portfolio_with_prices()
    summary = get_portfolio_summary()
    
    # ===== Summary Metrics =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${summary['total_value']:,.0f}",
            f"${summary['total_pl']:,.0f}"
        )
    with col2:
        color = "🟢" if summary['total_pl_pct'] >= 0 else "🔴"
        st.metric(
            "Return %",
            f"{color} {summary['total_pl_pct']:+.2f}%"
        )
    with col3:
        st.metric(
            "Cash Available",
            f"${summary['cash']:,.0f}"
        )
    with col4:
        st.metric(
            "Positions",
            summary['positions_count']
        )
    
    st.divider()
    
    if not portfolio.empty:
        # ===== Holdings Table =====
        st.subheader("Your Holdings")
        
        display_cols = ['ticker', 'quantity', 'purchase_price', 'current_price', 'current_value', 'unrealized_pl_pct']
        portfolio_display = portfolio[display_cols].copy()
        portfolio_display.columns = ['Ticker', 'Qty', 'Entry $', 'Current $', 'Value $', 'Return %']
        portfolio_display = portfolio_display.sort_values('Value $', ascending=False)
        
        st.dataframe(
            portfolio_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry $": st.column_config.NumberColumn(format="$%.2f"),
                "Current $": st.column_config.NumberColumn(format="$%.2f"),
                "Value $": st.column_config.NumberColumn(format="$%.2f"),
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
        
        # ===== Charts =====
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Allocation")
            allocation = get_allocation()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(allocation.keys()) + ['Cash'],
                values=[d['value'] for d in allocation.values()] + [summary['cash']],
            )])
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.subheader("P&L Breakdown")
            
            fig_bar = go.Figure(data=[go.Bar(
                x=portfolio['ticker'].tolist(),
                y=portfolio['unrealized_pl'].tolist(),
                marker_color=['green' if x > 0 else 'red' for x in portfolio['unrealized_pl']],
            )])
            fig_bar.update_layout(
                height=400,
                xaxis_title="Ticker",
                yaxis_title="Unrealized P&L ($)",
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("📊 No positions yet. Add your first position below!")
    
    st.divider()
    
    # ===== Add Position =====
    st.subheader("Manage Positions")
    col_add, col_remove = st.columns(2)
    
    with col_add:
        st.write("**➕ Add Position**")
        with st.form("add_position_form"):
            add_ticker = st.text_input("Ticker", placeholder="e.g., AAPL")
            add_qty = st.number_input("Quantity", min_value=1, value=1)
            add_price = st.number_input("Entry Price $", min_value=0.01, value=100.0, format="%.2f")
            
            if st.form_submit_button("Add", use_container_width=True):
                if add_ticker:
                    if add_position(add_ticker, add_qty, add_price):
                        st.success(f"✅ Added {add_qty} shares of {add_ticker}")
                        st.rerun()
                    else:
                        st.error("❌ Error adding position")
                else:
                    st.error("Enter a ticker")
    
    with col_remove:
        st.write("**➖ Remove Position**")
        if not portfolio.empty:
            remove_ticker = st.selectbox("Select ticker to remove", portfolio['ticker'].tolist())
            if st.button("Remove", use_container_width=True, key="remove_btn"):
                if remove_position(remove_ticker):
                    st.success(f"✅ Removed {remove_ticker}")
                    st.rerun()
                else:
                    st.error("❌ Error removing position")
        else:
            st.info("Add a position first")
    
    st.divider()
    
    # ===== Recent Executions =====
    st.subheader("Recent Executions")
    executions = get_recent_executions(limit=10)
    
    if executions:
        exec_df = pd.DataFrame(executions)
        exec_df['timestamp'] = pd.to_datetime(exec_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        exec_df['total_value'] = exec_df['total_value'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(
            exec_df[['timestamp', 'ticker', 'action', 'quantity', 'price', 'total_value']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No executions yet")
```

---

## ✅ Step 4: Add Recommendations Tab (Optional but Recommended)

Add this **after the Portfolio tab code**:

```python
# ============ RECOMMENDATIONS TAB ============
with tab_recommend:
    st.header("🎯 AI-Generated Recommendations")
    
    # Get all tickers (portfolio + watchlist)
    portfolio = load_portfolio_data()
    watchlist = ['PLTR', 'SMCI', 'TSLA', 'NVDA', 'AAPL']
    
    all_tickers = list(set(portfolio['ticker'].tolist() + watchlist))
    
    st.info(f"Analyzing {len(all_tickers)} tickers for AI signals...")
    
    recommendations = []
    
    with st.spinner("Computing predictions..."):
        for tk in all_tickers:
            try:
                pred = predict_next_for_ticker(tk, period="5y", model_type="rf", horizon=1)
                
                expected_ret = pred.get('pred_next_ret', 0)
                prob_up = pred.get('prob_up', 0.5)
                sharpe = pred.get('backtest_sharpe', 0)
                
                # Simple signal logic
                if expected_ret > 0.02 and prob_up > 0.65:
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
                    'return': expected_ret,
                    'prob_up': prob_up,
                    'sharpe': sharpe,
                    'strength': strength,
                    'in_portfolio': tk in portfolio['ticker'].values
                })
            except Exception as e:
                pass
    
    rec_df = pd.DataFrame(recommendations)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_buys = st.checkbox("Show BUY signals", value=True)
    with col2:
        show_sells = st.checkbox("Show SELL signals", value=True)
    with col3:
        show_portfolio_only = st.checkbox("Portfolio only", value=False)
    
    # Apply filters
    filtered = rec_df.copy()
    
    if show_portfolio_only:
        filtered = filtered[filtered['in_portfolio']]
    
    signals_to_show = []
    if show_buys:
        signals_to_show.extend(['🟢 STRONG BUY', '🟡 BUY'])
    if show_sells:
        signals_to_show.extend(['🔴 STRONG SELL', '🟠 SELL'])
    
    if signals_to_show:
        filtered = filtered[filtered['signal'].isin(signals_to_show)]
    
    # Sort by strength
    filtered = filtered.sort_values('strength', ascending=False)
    
    if not filtered.empty:
        st.subheader("Top Opportunities")
        
        for idx, row in filtered.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                indicator = "📊" if row['in_portfolio'] else "🎯"
                st.write(f"**{indicator} {row['ticker']}**")
            with col2:
                st.write(row['signal'])
            with col3:
                color = "🟢" if row['return'] > 0 else "🔴"
                st.write(f"{color} {row['return']:+.2%}")
            with col4:
                st.write(f"P↑ {row['prob_up']:.0%}")
    else:
        st.info("No matching signals")
```

---

## ✅ Step 5: Test Your Implementation

1. **Save `app.py`** with your changes
2. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```
3. **Add a test position**:
   - Go to 💼 Portfolio tab
   - Click "Add Position"
   - Enter: AAPL, 10 shares, $180
   - Click "Add"
4. **Verify Portfolio displays**:
   - Should show AAPL in holdings table
   - Should show allocation pie chart
   - Should show P&L

---

## 🎯 Optional: Connect to Alpaca (Real Trading)

Once your UI is working, you can add this to an **Account Integration** tab:

```python
# In your .streamlit/secrets.toml
alpaca_api_key = "your_paper_key_here"
alpaca_secret_key = "your_paper_secret_here"

# Then in app.py:
import alpaca_trade_api as tradeapi

# Connect to Alpaca
api = tradeapi.REST(
    st.secrets["alpaca_api_key"],
    st.secrets["alpaca_secret_key"],
    base_url="https://paper-api.alpaca.markets"  # Paper trading
)

# Get account info
account = api.get_account()
st.metric("Account Value", f"${float(account.portfolio_value):,.0f}")
```

---

## 📦 What You Get

✅ **Portfolio tracking** with automatic P&L calculations
✅ **Real-time prices** from yfinance
✅ **Position management** (add/remove)
✅ **Allocation charts** showing your breakdown
✅ **Execution history** logging all trades
✅ **AI recommendations** from your prediction model
✅ **Ready for Alpaca integration** when you're ready for real trading

---

## 🚀 Next Steps

1. **Test with paper trading** for 2-4 weeks
2. **Validate signals** are accurate
3. **Connect real Alpaca account** when confident
4. **Set risk limits** in risk rules
5. **Enable automated trading** with schedule

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io
- **Alpaca API**: https://alpaca.markets/docs/
- **Plotly Charts**: https://plotly.com/python/

---

**Time to implement**: ~30 minutes
**Difficulty**: Easy (mostly copy/paste)
**Payoff**: Full portfolio management + AI signals ready for real trading!
