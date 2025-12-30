# Stock Predictor - Quick Start Guide

## ✅ System Status: FULLY OPERATIONAL

Your Stock Predictor is ready to use. All core systems tested and working.

---

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys (Optional)
Edit `.env` file with your API keys for enhanced features:
```bash
FRED_API_KEY=your_key           # Macro data
FMP_API_KEY=your_key            # Fundamentals
ALPACA_API_KEY=your_key         # Paper trading
```

Without these: System works with reduced features but core predictions still function.

### Step 3: Choose Your Use Case

---

## 📊 Option A: Interactive Dashboard

Run Streamlit UI for real-time predictions and backtests:

```bash
streamlit run app.py
```

Then:
1. Go to `http://localhost:8501`
2. Enter a ticker (e.g., "AAPL")
3. Select model type (RandomForest, XGBoost, GradientBoosting)
4. Click "Generate Prediction" to get return forecast
5. Click "Run Backtest" to validate strategy on historical data
6. View options analytics and suggested trading strategies

---

## 💻 Option B: Command Line (CLI)

### Get a Single Prediction
```bash
python3 -c "
from prediction_model import predict_next_for_ticker
import json
result = predict_next_for_ticker('AAPL', period='1y', model_type='rf', horizon=1)
print(json.dumps(result, default=str, indent=2))
"
```

Returns:
```json
{
  "ticker": "AAPL",
  "pred_next_ret": 0.001264,
  "pred_next_price": 234.56,
  "prob_up": 0.52,
  "prob_down": 0.48,
  "confidence_score": 0.001264,
  "model_type": "rf",
  "horizon": 1
}
```

### Run a Backtest
```bash
python3 -c "
from prediction_model import backtest_one_ticker
result = backtest_one_ticker('SPY', period='2y', model_type='rf', horizon=1)
print(f\"Sharpe: {result['sharpe']:.4f}\")
print(f\"Hit Rate: {result.get('hitrate', 0):.2%}\")
print(f\"Total Return: {result.get('total_return', 0):.2%}\")
"
```

### Run Walk-Forward Backtest (Prevents Look-Ahead Bias)
```bash
python3 -c "
from prediction_model import walk_forward_backtest
result = walk_forward_backtest(
    ticker='SPY',
    period='5y',
    train_years=2,
    test_years=0.5,
    model_type='rf',
    horizon=1
)
print(f\"Folds: {len(result)}\")
print(f\"Avg Sharpe: {result[0]['sharpe']:.4f}\")
"
```

---

## 🔬 Option C: Run Pre-Built Experiments

10 experiments pre-configured in `experiments_phase2b.json`:

```bash
python run_experiments.py --config experiments_phase2b.json --max_experiments 3
```

Results saved to `./results/` directory with:
- Sharpe ratios
- Hit rates
- Drawdowns
- Trade statistics

View leaderboard:
```bash
python -c "
from experiment_runner import ExperimentRunner
runner = ExperimentRunner()
runner.run_all_experiments()
leaderboard = runner.get_leaderboard()
print(leaderboard)
"
```

---

## ⚙️ Option D: Find Optimal Hyperparameters (Grid Search)

Test combinations of models and parameters:

```bash
python grid_search.py \
  --ticker SPY \
  --period 2y \
  --models rf xgb gbrt \
  --n_estimators 50 100 200 \
  --max_depth 5 10 15
```

Returns best parameters ranked by Sharpe ratio.

---

## 🤖 Option E: Automated Paper Trading

Run auto-trader to execute trading signals on Alpaca (paper account):

```bash
python auto_paper_trade.py
```

Or schedule it to run periodically:
```bash
python runner.py  # Runs every Monday 08:35 ET (via schedule module)
```

Requires:
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env`
- Active Alpaca account (free paper trading available)

Trading signals auto-generated from predictions, then:
1. Filter for US stocks only
2. Check option availability (if applicable)
3. Place market or limit orders
4. Track trade results in `paper_trading_tracker.csv`

---

## 📈 Model Types

Choose from 4 different models:

| Model | Speed | Accuracy | Stability | Best For |
|-------|-------|----------|-----------|----------|
| `rf` (RandomForest) | Fast | Good | Very Stable | Default, most reliable |
| `xgb` (XGBoost) | Medium | Excellent | Sensitive | Fine-tuning, best Sharpe |
| `gbrt` (GradientBoosting) | Slow | Good | Stable | Long backtests |
| `linreg` (Linear Regression) | Fastest | Poor | Unstable | Baseline only |

---

## 🎯 Prediction Horizons

Forecast returns for different time periods:

| Horizon | Use Case | Typical Accuracy |
|---------|----------|------------------|
| `horizon=1` | Next-day trading | ~40-50% |
| `horizon=5` | 1-week swing trades | ~45-55% |
| `horizon=20` | 1-month positions | ~50-60% |

---

## 📊 Features Included

### ✅ Always Available
- Technical indicators (RSI, MACD, ATR, ADX, Bollinger Bands, KAMA, etc.)
- Momentum & trend (SMA, EMA, volume profile)
- GBM-derived probabilities (probability of up move, expected return)
- ARIMA time series forecasts
- Price-based features (returns, volatility, log returns)
- Volume indicators (OBV, MFI, VP, VPT)

### ⚠️ Available with API Keys
- **FRED_API_KEY**: Macro data (VIX, 10-year yield, term spread, unemployment, CPI, OAS, Fed funds)
- **FMP_API_KEY**: Fundamentals (P/E ratio, P/B ratio, market cap)
- **MARKETAUX_API_KEY**: News sentiment

### 🔧 Optional (Can Be Installed)
- **TensorFlow**: GAF-CNN deep learning for up/down classification (`pip install tensorflow`)
- **TA-Lib**: Advanced technical indicators (`pip install ta-lib`)

Without these: System degrades gracefully, core predictions still work.

---

## ⚡ Example Scripts

### Example 1: Batch Prediction for Multiple Tickers
```python
from prediction_model import predict_next_for_ticker
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
results = []

for ticker in tickers:
    pred = predict_next_for_ticker(ticker, period="1y", model_type="rf")
    results.append({
        "Ticker": pred["ticker"],
        "Predicted Return": pred["pred_next_ret"],
        "Confidence": pred["confidence_score"],
        "Prob Up": pred["prob_up"]
    })

df = pd.DataFrame(results)
print(df.to_string())
df.to_csv("predictions.csv", index=False)
```

### Example 2: Backtest Across Multiple Models
```python
from prediction_model import backtest_one_ticker

ticker = "SPY"
models = ["rf", "xgb", "gbrt"]
results = {}

for model in models:
    result = backtest_one_ticker(ticker, period="2y", model_type=model)
    results[model] = {
        "Sharpe": result["sharpe"],
        "Hit Rate": result.get("hitrate", 0),
        "Return": result.get("total_return", 0)
    }

for model, metrics in results.items():
    print(f"{model:10s} | Sharpe: {metrics['Sharpe']:+.4f} | Hit Rate: {metrics['Hit Rate']:.2%}")
```

### Example 3: Compare Models with Walk-Forward Validation
```python
from prediction_model import walk_forward_backtest

ticker = "SPY"
print(f"\nWalk-Forward Backtest: {ticker}")
print("-" * 60)

for model in ["rf", "xgb", "gbrt"]:
    results = walk_forward_backtest(
        ticker=ticker,
        period="5y",
        train_years=2,
        test_years=0.5,
        model_type=model,
        horizon=1
    )
    
    avg_sharpe = sum(r.get("sharpe", 0) for r in results) / len(results)
    avg_hitrate = sum(r.get("hitrate", 0) for r in results) / len(results)
    
    print(f"{model:10s} | Folds: {len(results):2d} | Sharpe: {avg_sharpe:+.4f} | Hit Rate: {avg_hitrate:.2%}")
```

---

## 📋 Files & Directories

| File/Folder | Purpose |
|-------------|---------|
| `prediction_model.py` | Core ML engine (2,922 lines) |
| `data_fetch.py` | Data pipeline with multi-source fallbacks |
| `app.py` | Streamlit dashboard |
| `auto_paper_trade.py` | Alpaca paper trading |
| `experiment_runner.py` | Experiment orchestration |
| `grid_search.py` | Hyperparameter optimization |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (create and fill as needed) |
| `.streamlit/config.toml` | Streamlit settings |
| `experiments_phase2b.json` | Pre-configured experiments |
| `results/` | Experiment outputs |
| `.github/copilot-instructions.md` | AI coding guidance |

---

## 🔍 Troubleshooting

### "No price history available"
**Cause**: yfinance API failed, fallback sources also failed  
**Fix**: Try again (API rate limits are temporary) or use longer period

### "Insufficient data (60+ rows required)"
**Cause**: Not enough historical data for the requested period  
**Fix**: Use a longer period (e.g., "5y" instead of "6mo")

### Predictions all return 0
**Cause**: No features available after filtering NaNs  
**Fix**: Check console warnings; may need FRED/FMP API keys for features

### Streamlit won't start
**Cause**: Config file syntax error  
**Fix**: Check `.streamlit/config.toml` TOML syntax

### TensorFlow warnings
**Info**: GAF-CNN disabled due to missing TensorFlow  
**Fix**: `pip install tensorflow` or ignore (system continues without it)

---

## 📚 Next Steps

1. **Run Dashboard**: `streamlit run app.py` and test with a ticker
2. **Try CLI**: Use command-line examples to generate predictions
3. **Run Experiment**: Execute `python run_experiments.py --config experiments_phase2b.json --max_experiments 1`
4. **Grid Search**: Find optimal parameters with `python grid_search.py --ticker SPY`
5. **Setup Alpaca**: Configure `.env` and run `python auto_paper_trade.py` for paper trading

---

## 📖 Documentation

- **This File**: Quick start and examples
- **SYSTEM_VALIDATION_COMPLETE.md**: Detailed validation results and architecture
- **.github/copilot-instructions.md**: AI coding guidance (for future development)
- **Code Comments**: Each module has inline documentation

---

## ✨ You're All Set!

Your Stock Predictor system is fully operational. Choose your preferred way to use it:
- 🎨 **Streamlit UI** for interactive exploration
- 💻 **Python CLI** for automation
- 🔬 **Experiments** for reproducible research
- 🤖 **Auto-Trader** for live paper trading

Questions? Check the troubleshooting section or review the inline code comments.

**Happy predicting! 📈**

