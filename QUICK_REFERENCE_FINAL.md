# Stock Predictor - Quick Reference Guide

**System Status**: ✅ **FULLY OPERATIONAL**

---

## 🚀 Quick Start (5 minutes)

### 1. Verify API Keys Are Configured
```bash
cd '/Users/jakobmccleary/Desktop/Stock Predictor'

python3 << 'EOF'
import os
keys = ['APCA_API_KEY_ID', 'FRED_API_KEY', 'FMP_API_KEY', 'MARKETAUX_API_KEY']
for k in keys:
    print(f"✅ {k}" if os.environ.get(k) else f"❌ {k}")
EOF
```

### 2. Launch Streamlit Dashboard
```bash
streamlit run app.py
# Opens http://localhost:8501
```

### 3. Run Single Prediction
```bash
python3 << 'EOF'
from prediction_model import predict_next_for_ticker
result = predict_next_for_ticker('AAPL', period='1y', model_type='rf', horizon=1)
print(f"Return: {result['pred_next_ret']:.6f}, Confidence: {result['confidence']:.4f}")
EOF
```

### 4. Execute Paper Trades
```bash
python auto_paper_trade.py
```

---

## 📊 Core Functions Reference

### Predictions
```python
from prediction_model import predict_next_for_ticker

# Basic prediction
result = predict_next_for_ticker('AAPL', period='1y', model_type='rf', horizon=1)
# Returns: pred_next_ret, pred_next_price, prob_up, confidence, option_atm_price, etc.

# With XGBoost (best Sharpe)
result = predict_next_for_ticker('MSFT', model_type='xgb')

# Multi-day horizon
result = predict_next_for_ticker('NVDA', horizon=5)  # 5-day forecast
```

### Backtesting
```python
from prediction_model import backtest_one_ticker, walk_forward_backtest

# Single ticker backtest
metrics = backtest_one_ticker('AAPL', period='5y', model_type='rf')
# Returns: sharpe_ratio, hit_rate, max_drawdown, win_rate, profit_factor, etc.

# Walk-forward validation (date-based, prevents look-ahead bias)
metrics = walk_forward_backtest('AAPL', period='5y', train_years=1, test_years=0.25)
```

### Grid Search
```bash
python grid_search.py --ticker AAPL --period 5y --models rf xgb gbrt --max_depth 5 7 10
# Outputs: grid_search_results.json (ranked by Sharpe ratio)
```

### Experiments
```bash
# Run pre-configured experiments
python run_experiments.py --config experiments_phase2b.json

# Run experiments from code
from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig

runner = ExperimentRunner()
runner.add_experiment(ExperimentConfig(
    experiment_id='test1',
    ticker='AAPL',
    model=ModelConfig(model_type='rf', n_estimators=100)
))
runner.run_all_experiments()
```

---

## 🔌 API Keys Overview

| API | Purpose | Environment Variable | Used By |
|-----|---------|----------------------|---------|
| **Alpaca** | Paper trading | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` | `auto_paper_trade.py` |
| **FRED** | Macro data (T10Y, VIX) | `FRED_API_KEY` | `prediction_model.py` |
| **FMP** | Fundamentals (P/E, P/B) | `FMP_API_KEY` | `data_fetch.py` |
| **Marketaux** | News sentiment | `MARKETAUX_API_KEY` | `data_fetch.py` |
| **Alpha Vantage** | Fallback data | `ALPHAVANTAGE_API_KEY` | `data_fetch.py` |

All configured in `.env` and `~/.streamlit/secrets.toml`

---

## 🎯 Common Tasks

### Get Prediction for Multiple Tickers
```python
from prediction_model import predict_next_for_ticker
import json

tickers = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
for ticker in tickers:
    result = predict_next_for_ticker(ticker, period='1y', model_type='rf')
    print(f"{ticker}: {result['pred_next_ret']:.6f} (confidence: {result['confidence']:.4f})")
```

### Backtest Portfolio of Stocks
```python
from prediction_model import walkforward_cross_sectional

# Test multiple tickers in walk-forward mode
results = walkforward_cross_sectional(
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    model_type='rf',
    train_years=1,
    test_years=0.25,
    period='5y'
)

for ticker, metrics in results.items():
    print(f"{ticker}: Sharpe={metrics['sharpe_ratio']:.2f}, HitRate={metrics['hit_rate']:.1%}")
```

### Compare Model Types
```python
from prediction_model import predict_next_for_ticker

for model in ['rf', 'xgb', 'gbrt']:
    result = predict_next_for_ticker('AAPL', period='1y', model_type=model)
    print(f"{model:6s}: {result['pred_next_ret']:.6f}")
```

### Feature Analysis
```python
from prediction_model import build_features_and_target

# Get features used for prediction
features_df, target = build_features_and_target('AAPL', period='1y', horizon=1)

print(f"Total rows: {len(features_df)}")
print(f"Feature columns: {len(features_df.columns)}")
print(f"Missing features: {features_df.isna().sum()}")
print(f"Features: {list(features_df.columns)[:10]}...")
```

---

## ⚙️ Configuration Options

### Environment Variables
```bash
# Feature selection (optional)
export USE_ELASTICNET_SELECT=1
export ELASTICNET_L1_RATIO=0.5
export ELASTICNET_CV_FOLDS=5

# Or OLS significance selection
export USE_OLSSIGSELECT=1
export OLSSIG_ALPHA=0.05
export OLSSIG_TOPK=50

# Trading calendar
export TRADING_DAYS=252  # NYSE calendar days per year
```

### Model Hyperparameters
```python
from prediction_model import make_model

# RandomForest
model = make_model('rf', n_estimators=100, max_depth=10, min_samples_split=5)

# XGBoost
model = make_model('xgb', n_estimators=100, learning_rate=0.1, max_depth=5)

# GradientBoosting
model = make_model('gbrt', n_estimators=100, learning_rate=0.05, max_depth=5)
```

---

## 📈 Understanding Predictions

### Prediction Output Structure
```python
{
    'pred_next_ret': -0.000347,           # Predicted next-day return
    'pred_next_price': 235.64,             # Predicted next day closing price
    'prob_up': 0.5686,                     # Probability of positive return
    'prob_down': 0.4314,                   # Probability of negative return
    'confidence': 0.0,                     # Model confidence (0-1)
    'option_atm_call_price': 1.23,         # Theoretical call price
    'option_atm_put_price': 0.98,          # Theoretical put price
    'strategy': 'hold',                    # Suggested position
    'model_type': 'rf',                    # Model used
    'features_used': 49,                   # Number of active features
    'last_close': 235.93,                  # Most recent close price
    'n_data_points': 249,                  # Training data points used
}
```

### Interpreting Confidence Score
- **0.0-0.2**: Low confidence (near random, hold position)
- **0.2-0.4**: Medium-low confidence (slight bias, small position)
- **0.4-0.6**: Medium confidence (reasonable signal, moderate position)
- **0.6-0.8**: Medium-high confidence (strong signal, larger position)
- **0.8-1.0**: High confidence (very strong signal, full position)

---

## 🔄 Data Pipeline Flow

```
Market Open (UTC-5)
    ↓
[1] Fetch historical data (yfinance)
    ↓
[2] Fetch macro data (FRED: T10Y, VIX, etc.)
    ↓
[3] Fetch fundamentals (FMP: P/E, P/B, etc.)
    ↓
[4] Fetch news sentiment (Marketaux)
    ↓
[5] Build 100+ features (technical, GBM, regime)
    ↓
[6] Train model (RF/XGBoost/GBRT)
    ↓
[7] Generate prediction
    ↓
[8] Price options (Black-Scholes / Heston / Monte Carlo)
    ↓
[9] Signal to Alpaca (stock + options strategies)
    ↓
[10] Execute trades (paper trading)
```

---

## 🚨 Troubleshooting

### Issue: "No price history available for ^GSPC"
**Cause**: SPX data fetch failed  
**Solution**: System gracefully degrades (uses only technical features)

### Issue: "ARIMA prediction failed"
**Cause**: statsmodels ARIMA deprecated method  
**Solution**: Non-critical, system uses other features

### Issue: "TensorFlow not available"
**Cause**: Optional deep learning model not installed  
**Solution**: Main predictions work, GAF-CNN skipped

### Issue: "429 Client Error: Too Many Requests"
**Cause**: Yahoo Finance rate limiting  
**Solution**: Falls back to Stooq CSV data

### Issue: Prediction has low confidence
**Cause**: Insufficient data or conflicting signals  
**Solution**: May indicate market uncertainty, consider small positions

---

## 📋 File Locations

| File | Purpose | Path |
|------|---------|------|
| Main UI | Dashboard | `app.py` |
| Predictions | ML engine | `prediction_model.py` |
| Data fetching | APIs | `data_fetch.py` |
| Trading | Paper trades | `auto_paper_trade.py` |
| Experiments | Batch runs | `experiment_runner.py` |
| Search | Best params | `grid_search.py` |
| Features | Enhancements | `model_improvements.py` |
| API keys | Environment | `.env` |
| Config | Theme/UI | `.streamlit/config.toml` |
| Secrets | Streamlit | `~/.streamlit/secrets.toml` |

---

## 🎓 Learning Resources

- **Architecture**: See `.github/copilot-instructions.md`
- **Features**: See `FEATURE_ENGINEERING_GUIDE.md`
- **Framework**: See `EXPERIMENT_FRAMEWORK_README.md`
- **Trading**: See `ENHANCED_AUTO_TRADER_GUIDE.md`
- **Status**: See `SYSTEM_STATUS_FINAL.md`

---

## 💡 Performance Tips

1. **Feature Selection**: Use Elastic Net to reduce features (`USE_ELASTICNET_SELECT=1`)
2. **Hyperparameter Tuning**: Run `grid_search.py` to optimize model
3. **Data Quality**: Check macro data availability (FRED API key)
4. **Model Selection**: XGBoost best for Sharpe, RF most stable
5. **Validation**: Always use walk-forward (prevents look-ahead bias)
6. **Feature Lagging**: Automatic via `.shift(1)` - don't modify!

---

## 📞 Common Commands Cheat Sheet

```bash
# List all experiments
ls experiments*.json

# Check dependencies
python3 -m pip list | grep -E "sklearn|xgboost|pandas"

# Validate syntax
python3 -m py_compile prediction_model.py

# Single ticker test
python3 << 'EOF'
from prediction_model import predict_next_for_ticker
print(predict_next_for_ticker('AAPL', period='1y'))
EOF

# Run backtest
python3 << 'EOF'
from prediction_model import backtest_one_ticker
result = backtest_one_ticker('AAPL', period='5y')
print(f"Sharpe: {result['sharpe_ratio']:.2f}")
EOF

# Check API keys
grep -E "API|APCA" .env

# View recent trades
tail -20 trades.csv

# Monitor trades
python runner.py  # Runs scheduled trades
```

---

**Last Updated**: December 29, 2025  
**Status**: ✅ Production Ready

Start with: `streamlit run app.py`
