# Quick Reference Guide

## 📊 Dashboard Quick Start

### Run the UI
```bash
cd /Users/jakobmccleary/Desktop/Stock\ Predictor
streamlit run app.py
```

### Use the Dashboard
1. **Sidebar:** Enter tickers (e.g., `AAPL, NVDA, MSFT`)
2. **Select model:** RF (default), XGB, or GBRT
3. **Click "Run Screener + Model"**
4. **View candidates table** (filtered by IV, prediction magnitude)
5. **Select a ticker** from dropdown → see detailed analysis
6. **Expand sections:** Model Prediction, Price Chart, Options, Greeks + News

### Key Sections
| Section | What It Shows |
|---------|---------------|
| **📊 Model Prediction** | 3-day return forecast, probability, features used |
| **📈 Price Chart & Forecast** | Historical price + target price (red dashed line) |
| **📊 Options & Risk** | IV, Put/Call ratio, theoretical option prices |
| **📰 Greeks + News** | Option Greeks (Delta, Gamma, Vega, Theta) + recent news |
| **Validation: accuracy + Sharpe** | Direction accuracy, Sharpe ratio (historical) |

---

## 🤖 Model Prediction Commands (CLI)

### Single Ticker (3-day prediction)
```bash
python -c "
from prediction_model import predict_next_for_ticker
import json
out = predict_next_for_ticker('AAPL', period='5y', model_type='rf', horizon=3)
print(json.dumps({
    'ticker': out['ticker'],
    'pred_ret': out['pred_next_ret'],
    'prob_up': out['prob_up'],
    'last_close': out['last_close'],
}, indent=2))
"
```

### Single Ticker Backtest
```bash
python -c "
from prediction_model import backtest_one_ticker
results = backtest_one_ticker('AAPL', period='10y', model_type='rf', horizon=3)
print(f\"Sharpe: {results['sharpe']:.2f}, Hit rate: {results['hit_rate']:.1%}\")
"
```

### Multi-Ticker Walk-Forward (20 tickers, 34+ folds)
```bash
python -c "
from prediction_model import walkforward_cross_sectional
tickers = ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','INTC','NFLX',
           'JPM','BAC','GS','XOM','CVX','KO','PEP','WMT','COST','DIS']
results = walkforward_cross_sectional(tickers, model_type='rf', train_years=2, test_years=0.5)
print(f\"Median Sharpe: {results['sharpe'].median():.2f}\")
print(f\"Median Hit Rate: {results['hit_rate'].median():.1%}\")
print(results[['fold','sharpe','ann_return','max_dd']].to_string())
"
```

---

## 📈 Portfolio Engine Commands (Coming Week 1)

### Initialize Engine
```python
from portfolio_engine import PortfolioEngine

engine = PortfolioEngine(
    universe=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','INTC','NFLX',
              'JPM','BAC','GS','XOM','CVX','KO','PEP','WMT','COST','DIS'],
    capital=100000,
    max_dd=0.20,
    sharpe_target=1.0
)

# Detect regime
regime = engine.detect_regime(vix=20.5, corr=0.72, vol_cluster=1.1)
print(f"VIX regime: {regime['vix_bucket']}, Kill-switch: {regime['kill_switch']}")

# Size positions
signals = {'AAPL': 0.025, 'MSFT': -0.018, 'NVDA': 0.032, ...}  # pred_ret by ticker
positions = engine.size_positions(signals, regime)
print(positions)  # {ticker: size_pct}

# Execute trades & log
engine.execute_trades(positions, prices={'AAPL': 245.67, ...})
print(f"Portfolio value: ${engine.equity_curve[-1]:.0f}")

# Save trade log
engine.save_trade_log('trades.csv')
```

---

## 🎯 Key Concepts

### VIX Regime
| Regime | VIX | Action |
|--------|-----|--------|
| **Low** | < 15 | ⚠️ Reduce 25% (complacency) |
| **Medium** | 15–35 | ✅ Normal sizing |
| **High** | > 35 | 🛑 Liquidate all; move to cash |

### Kill-Switches (Immediate De-Risk)
1. **VIX > 35** → Liquidate longs, hold 10% short hedge
2. **VIX < 15** → Reduce longs by 25%
3. **Correlation > 0.8** → Cut to top-3 holdings
4. **Single-asset > 15%** → Force trim to 15%
5. **Drawdown > 15%** → Move to 50% cash (recovery mode)
6. **Sharpe < 0.5** → Move to 100% cash

### Position Sizing
- **Vol-targeted:** Position = 1% × (15% ÷ asset_vol)
- **Max per ticker:** 15% of capital
- **Max long:** 80% of capital
- **Max short:** 25% of capital

---

## 📊 File Structure

```
Stock Predictor/
├── .github/
│   └── copilot-instructions.md        ← AI agent guide (70 lines)
├── PORTFOLIO_ENGINE.md                 ← Engine spec (350+ lines)
├── UPDATE_SUMMARY.md                   ← Change log (this update)
├── DEPLOYMENT_CHECKLIST.md             ← Implementation roadmap
├── QUICK_REFERENCE.md                  ← This file
├── app.py                              ← Streamlit dashboard (1,630 lines, UPDATED)
├── prediction_model.py                 ← Core ML engine (2,213 lines)
├── data_fetch.py                       ← Data pipeline
├── stock_screener.py                   ← Pre-filtering
├── auto_paper_trade.py                 ← Trade execution (Alpaca)
├── runner.py                           ← Scheduler (Monday 8:35am ET)
├── portfolio_engine.py                 ← TO IMPLEMENT (Week 1)
├── regime_features.py                  ← TO IMPLEMENT (Week 1)
├── requirements.txt
├── gaf_cnn_updown.keras                ← Pre-trained CNN model
├── option_pricing.py                   ← Black-Scholes / Heston
├── monte_carlo_pricer.py               ← Option pricing simulation
└── ...
```

---

## 🐛 Troubleshooting

### UI Issues

**Q: Ticker dropdown crashes or shows no options**
- **A:** Run "Run Screener + Model" first; it populates `detail_universe`

**Q: Chart doesn't show predicted price**
- **A:** Ensure model ran successfully; check for NaN in `pred_next_price`

**Q: Greeks + News section empty**
- **A:** Click "Load Greeks + News" checkbox; these are slow APIs

**Q: "KeyError: pred_next_ret"**
- **A:** Model may have failed for that ticker; check console output

### Model Issues

**Q: `ValueError: No usable history for TICKER`**
- **A:** Not enough data for that ticker; fallback periods tried automatically

**Q: `YFRateLimitError`**
- **A:** Yahoo Finance rate limited; wait 1 min or reduce # tickers

**Q: GBM features showing NaN**
- **A:** Requires >60 days of data; SPX cache issues; check `_SPX_CACHE`

### Portfolio Engine (Week 1+)

**Q: Trade execution not logging**
- **A:** Verify `engine.save_trade_log()` called; check file permissions

**Q: VIX regime always "medium"**
- **A:** Ensure VIX data fetched; check `get_fred_series()` for FRED API key

---

## 🔗 Integration Points

### Prediction → Signals
```python
# prediction_model.py
pred_dict = predict_next_for_ticker('AAPL', horizon=3)
# Returns: {ticker, pred_next_ret, prob_up, last_close, ...}

# app.py
signals = build_signals_from_pred_df(pred_df, trade_mode='Options if suggested', ...)
# Returns: {AAPL: {asset: 'stock', action: 'BUY', qty: 1, ...}, ...}

# Signals → JSON
write_signals_json_atomic(signals, 'signals.json')

# JSON → Auto-trader
python auto_paper_trade.py  # Reads signals.json; executes trades
```

### Portfolio Engine → Dashboard
```python
# portfolio_engine.py (Week 1)
engine.daily_validation()  # Compute Sharpe, hit rate, drift

# app.py (Portfolio Engine tab, to implement)
st.metric('Rolling Sharpe (60D)', engine.sharpe_60d)
st.metric('Max Drawdown', f"{engine.max_dd:.1%}")
```

---

## 📚 Learning Resources

### Core Papers
- Random Forest for time series: *Krauss et al. (2016), "Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500"*
- Walk-forward backtesting: *de Prado (2018), "Advances in Financial Machine Learning"*
- Volatility targeting: *Blitz et al. (2012), "Volatility Targeting"*

### Code Examples
- **Feature engineering:** `prediction_model.py` lines 583–629 (GBM features)
- **Model training:** `prediction_model.py` lines 1170–1200 (`make_model()`)
- **Walk-forward:** `prediction_model.py` lines 2010+ (`walkforward_cross_sectional()`)

### API Docs
- **Alpaca:** https://alpaca.markets/docs/api-references/
- **yfinance:** https://github.com/ranaroussi/yfinance
- **FRED API:** https://fred.stlouisfed.org/docs/api/

---

## 📞 Support

### Questions About
- **UI Changes:** See `app.py` line comments (e.g., line 1225 = chart rendering)
- **Portfolio Spec:** See `PORTFOLIO_ENGINE.md` sections 1–10
- **Implementation:** See `DEPLOYMENT_CHECKLIST.md` phases 1–5
- **AI Coding Patterns:** See `.github/copilot-instructions.md`

### Contact
- **ML Engineering:** Review backtesting results in `backtest_results_comprehensive.csv`
- **Data Issues:** Check `data_fetch.py`; verify API keys (FMP, FRED, Alpaca)
- **Trading Issues:** Check `auto_paper_trade.py` logs; verify Alpaca credentials

---

## 📅 Timeline

| Phase | Dates | Status |
|-------|-------|--------|
| **Phase 1: UI & Docs** | 2025-12-27 | ✅ DONE |
| **Phase 2: Engine Impl** | 2026-01-03 to 01-10 | ⏳ NEXT |
| **Phase 3: Backtesting** | 2026-01-10 to 01-17 | 📅 Queued |
| **Phase 4: Deployment** | 2026-01-17 to 01-31 | 📅 Queued |
| **Phase 5: Live Trading** | 2026-02-01+ | 📅 Queued |

---

**Last Updated:** 2025-12-27  
**Version:** 1.0 (Production Ready)  
**Next Milestone:** Week 1 Engine Implementation
