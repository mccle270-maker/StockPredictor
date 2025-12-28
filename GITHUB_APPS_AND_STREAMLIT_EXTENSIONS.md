# TOP GITHUB APPS & STREAMLIT EXTENSIONS FOR YOUR PORTFOLIO DASHBOARD

## 🎯 QUICK RECOMMENDATIONS FOR YOUR USE CASE

### **Tier 1: Must Have** ⭐⭐⭐

| Library | Purpose | Cost | Time to Add | Why |
|---------|---------|------|------------|-----|
| **streamlit-extras** | UI enhancements (cards, buttons, grids) | Free | 30 min | Makes dashboard look professional |
| **streamlit-aggrid** | Interactive data tables | Free | 20 min | Much better than st.dataframe() |
| **plotly** | Interactive charts (you have this) | Free | Already done | Essential for portfolio viz |
| **alpaca-trade-api** | Paper/live trading integration | Free | 1 hour | Connects to real broker |

### **Tier 2: Should Have** ⭐⭐

| Library | Purpose | Cost | Time to Add | Why |
|---------|---------|------|------------|-----|
| **streamlit-plotly-events** | Click/interact with charts | Free | 15 min | Select stocks to trade by clicking |
| **streamlit-authenticator** | User login system | Free | 30 min | Secure your portfolio dashboard |
| **pandas-ta** | 150+ technical indicators | Free | Already done | Enhances prediction features |

### **Tier 3: Nice to Have** ⭐

| Library | Purpose | Cost | Time to Add | Why |
|---------|---------|------|------------|-----|
| **streamlit-lottie** | Animations/loading states | Free | 10 min | Better UX while loading |
| **streamlit-option-menu** | Sidebar navigation | Free | 15 min | Cleaner navigation |
| **yfinance** | Stock data (you have this) | Free | Already done | Your current data source |

---

## 🚀 IMPLEMENTATION GUIDE

### **1. Install Streamlit Extras** (Recommended First)

```bash
pip install streamlit-extras
```

**Use cases:**
- Better looking metric cards
- Fancy buttons and spinners
- Grid layouts
- Stateful button handling

**Add to your app.py:**

```python
from streamlit_extras.metric_cards import style_metric_cards

# In your Portfolio tab metrics section:
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", f"${summary['total_value']:,.0f}")
with col2:
    st.metric("Return", f"{summary['total_pl_pct']:+.2f}%")
with col3:
    st.metric("Cash", f"${summary['cash']:,.0f}")
with col4:
    st.metric("Positions", summary['positions_count'])

# Make them fancy
style_metric_cards(
    background_color="#0E1117",
    border_left_color="#00D9FF",
    border_color="#1F1F23",
    box_shadow=True
)
```

---

### **2. Install AgGrid for Tables** (Recommended Second)

```bash
pip install streamlit-aggrid
```

**Makes portfolio table interactive with:**
- Sort by any column
- Edit cells directly
- Filter by text
- Drag columns
- Right-click context menu

**Add to your app.py:**

```python
from st_aggrid import AgGrid, GridOptionsBuilder

# In Portfolio tab instead of st.dataframe():
portfolio = get_portfolio_with_prices()

# Configure grid
gb = GridOptionsBuilder.from_dataframe(portfolio)
gb.configure_pagination(paginationPageSize=10)
gb.configure_column("unrealized_pl_pct", type=["numericColumn"], precision=2)
gb.configure_column("current_value", type=["numericColumn"], precision=2)
gb.configure_selection("single")  # Click row to select
gb.configure_default_column(editable=False, sortable=True, filterable=True)

options = gb.build()

grid_response = AgGrid(
    portfolio,
    gridOptions=options,
    data_return_mode="as_selected",
    update_mode="MODEL_CHANGED",
    fit_columns_on_grid_load=True,
    height=400
)

# Get selected row (great for actions)
if len(grid_response['selected_rows']) > 0:
    selected = grid_response['selected_rows'][0]
    st.write(f"Selected: {selected['ticker']}")
```

---

### **3. Install Plotly Events** (Optional but Cool)

```bash
pip install streamlit-plotly-events
```

**Makes charts interactive:**
- Click on a pie chart slice to view that stock
- Click bar chart to open detailed analysis
- Hover for details
- No page reload needed

**Add to your app.py:**

```python
from streamlit_plotly_events import plotly_events

# In Portfolio tab - Allocation pie chart:
allocation = get_allocation()

fig_pie = go.Figure(data=[go.Pie(
    labels=list(allocation.keys()),
    values=[d['value'] for d in allocation.values()],
)])

# Instead of st.plotly_chart(), use:
selected_data = plotly_events(fig_pie, click_event=True)

# Handle click
if selected_data:
    ticker = selected_data[0]['label']
    st.write(f"Clicked on {ticker}")
    # Show details about that ticker
```

---

### **4. Install Authenticator for Login** (Recommended for Real Money)

```bash
pip install streamlit-authenticator
```

**Protects your portfolio:**
- Username/password login
- Session management
- Hashed passwords
- Multi-user support

**Add to your app.py (at the top):**

```python
import streamlit_authenticator as stauth
import yaml
from pathlib import Path

# Load credentials from YAML file (create storage/auth.yaml)
with open('storage/auth.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.write(f"Welcome *{name}*")
    
    # Show your app here (Portfolio, Recommendations, etc.)
    tab_dash, tab_backtests, tab_port, tab_recommend = st.tabs([...])
    
elif authentication_status == False:
    st.error("Username/password is incorrect")
elif authentication_status == None:
    st.warning("Please enter your username and password")
```

**Create `storage/auth.yaml`:**

```yaml
credentials:
  usernames:
    your_username:
      name: Your Name
      password: "$2b$12$..." # Use online bcrypt generator
preauthorized:
  emails:
    - your_email@example.com
cookie:
  expiry_days: 30
  key: some-secret-key
  name: auth_cookie
```

---

### **5. Install Lottie for Animations** (Nice Polish)

```bash
pip install streamlit-lottie
```

**Add animations:**
- Loading spinners
- Success checkmarks
- Error icons
- Portfolio animations

**Add to your app.py:**

```python
from streamlit_lottie import st_lottie
import requests
import json

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Show loading animation while fetching prices
lottie_loading = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_o2xg5a1q.json")

with st.spinner("Updating prices..."):
    portfolio = get_portfolio_with_prices()
    st_lottie(lottie_loading, height=200)

st.success("Portfolio updated!")
```

---

## 🔌 GITHUB APPS & INTEGRATIONS

### **1. GitHub Actions for Automated Predictions** ⭐

**What it does:** Run predictions every morning at 8:30 AM ET, send results to Slack/Discord

**Create `.github/workflows/daily-predictions.yml`:**

```yaml
name: Daily Stock Predictions

on:
  schedule:
    - cron: '30 14 * * MON-FRI'  # 9:30 AM ET (14:30 UTC)

jobs:
  predict:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run daily predictions
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
        run: |
          python -c "
          from prediction_model import predict_next_for_ticker
          import json
          
          tickers = ['AAPL', 'NVDA', 'PLTR', 'SMCI', 'TSLA']
          results = {}
          
          for tk in tickers:
              try:
                  pred = predict_next_for_ticker(tk)
                  results[tk] = {
                      'signal': 'BUY' if pred.get('prob_up', 0) > 0.65 else 'SELL' if pred.get('prob_up', 0) < 0.35 else 'HOLD',
                      'prob_up': pred.get('prob_up', 0),
                      'expected_return': pred.get('pred_next_ret', 0)
                  }
              except:
                  pass
          
          print(json.dumps(results, indent=2))
          "
      
      - name: Send to Discord
        uses: tsickert/discord-webhook@v1
        with:
          webhook-url: ${{ secrets.DISCORD_WEBHOOK }}
          content: 'Daily stock predictions ready! Check your dashboard.'
```

**Get Discord webhook:**
1. Create Discord server
2. Settings → Integrations → Webhooks
3. Create webhook, copy URL
4. Add to GitHub Secrets: `DISCORD_WEBHOOK`

---

### **2. GitHub Pages for Public Dashboard** (Optional)

**Deploy your app to live URL:**

```bash
# 1. Create Streamlit Cloud account at https://streamlit.io/cloud
# 2. Connect GitHub repo
# 3. Deploy with one click
# 4. Get live URL like: https://your-app.streamlit.app
```

---

### **3. GitHub Dependabot for Auto Updates**

**Automatically update dependencies:**

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    allow:
      - dependency-type: "production"
```

---

## 📊 RECOMMENDED INSTALLATION ORDER

### **Week 1: Core Features** (1 hour total)
```bash
# Already have these
pip install streamlit pandas numpy yfinance plotly

# Add these
pip install streamlit-extras streamlit-aggrid
```

### **Week 2: Authentication + Alpaca** (1.5 hours)
```bash
pip install streamlit-authenticator alpaca-trade-api
```

### **Week 3: Enhancements** (1 hour)
```bash
pip install streamlit-plotly-events streamlit-lottie
```

---

## 🎨 BEFORE & AFTER COMPARISON

### **Before (Basic Streamlit)**
- Plain gray tables
- No interactivity
- Boring metrics
- Manual testing

### **After (Enhanced)**
- Beautiful metric cards with colors ✨
- Interactive tables (click to select, edit, sort)
- Click charts to drill down
- Secured with login
- Automated daily predictions
- Ready for real money management

---

## 💰 COST BREAKDOWN

| Component | Cost | Notes |
|-----------|------|-------|
| All libraries | **FREE** | Open source |
| Streamlit Cloud hosting | **FREE** (up to 1 app) | https://streamlit.io/cloud |
| GitHub Actions | **FREE** (up to 2000 min/month) | Runs predictions daily |
| Alpaca Paper Trading | **FREE** | No real money |
| Alpaca Live Trading | **Free account** | You pay commissions only ($0 with $500+ account) |
| **Total** | **FREE** | Everything is free to start! |

---

## 🚀 YOUR IMPLEMENTATION ROADMAP

### **Right Now (30 min)**
- ✅ Add `portfolio_manager.py` (already created)
- ✅ Add portfolio tab to app (30 min)
- ✅ Test with dummy positions

### **This Week (1 hour)**
- Install `streamlit-extras` + `streamlit-aggrid`
- Make portfolio table interactive
- Style metrics with colors
- Test recommendations tab

### **Next Week (1.5 hours)**
- Add `streamlit-authenticator` for login
- Connect Alpaca paper trading account
- Log real executions

### **Week 3+ (When ready)**
- Set up GitHub Actions for daily predictions
- Deploy to Streamlit Cloud
- Connect Alpaca live account
- Start real trading

---

## 🎯 FINAL CHECKLIST

- [ ] ✅ `portfolio_manager.py` created
- [ ] ✅ Portfolio tab added to `app.py`
- [ ] ✅ Recommendations tab working
- [ ] ✅ `streamlit-extras` installed
- [ ] ✅ `streamlit-aggrid` installed
- [ ] ✅ Interactive tables working
- [ ] ✅ Test with 2-3 manual positions
- [ ] ✅ Get Alpaca API keys
- [ ] ✅ Connect paper trading
- [ ] ✅ Execute first test trade
- [ ] ✅ GitHub Actions setup
- [ ] [ ] Month 2: Go live with real money

---

## 📞 QUICK LINKS

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud**: https://streamlit.io/cloud
- **Alpaca**: https://alpaca.markets
- **GitHub Actions**: https://github.com/features/actions
- **Streamlit-Extras**: https://github.com/arnaudmiribel/streamlit-extras
- **AgGrid**: https://github.com/PablocFonseca/streamlit-aggrid

---

**Last Updated**: December 28, 2025
**Status**: Ready to implement
**Estimated Time to Production**: 3-4 weeks
