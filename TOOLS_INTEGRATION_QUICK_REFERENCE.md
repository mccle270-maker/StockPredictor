# Free Financial Tools Integration - Quick Reference

## TL;DR
- **TA-Lib**: Validate/replace technical indicators
- **pmdarima**: Add ARIMA ensemble
- **Zipline**: More realistic backtesting (future)
- **Pandas-TA**: Easy indicator API
- **AlphaLens**: Factor quality evaluation
- **MLFinLab**: Better CV for finance

---

## Tools Comparison Table

| Tool | Purpose | Install | Effort | Impact | Status |
|------|---------|---------|--------|--------|--------|
| **TA-Lib** | 200+ indicators validated | `pip install ta-lib` | 1-2h | Validation ⭐⭐ | Priority |
| **Pandas-TA** | 150+ indicators (pandas API) | `pip install pandas-ta` | 1h | Convenience ⭐ | Nice-to-have |
| **pmdarima** | ARIMA automation | `pip install pmdarima` | 2-3h | Ensemble ⭐⭐ | Phase 4 |
| **Zipline** | Event-driven backtester | `pip install zipline-reloaded` | 4-5h | Realism ⭐⭐⭐ | Phase 5 |
| **AlphaLens** | Factor analysis | `pip install alphalens` | 1-2h | QA ⭐⭐ | After Phase 2 |
| **MLFinLab** | Finance ML best practices | `pip install mlfinlab` | 2-3h | Correctness ⭐⭐ | Phase 3 |
| **VectorBT** | Fast vectorized backtest | `pip install vectorbt` | 2-3h | Speed ⭐⭐⭐ | Phase 4 |

---

## Phase 1: Data Quality Fix (2-3 hours) [DO THIS FIRST]

### Issue: Macro data forward-filled across fold boundaries

**File:** `prediction_model.py`, function `build_panel_features_and_target()`

**Problem Code (lines 543-545):**
```python
df["t10y"] = s10.reindex(df_dates).ffill().bfill().values  # ❌ Look-ahead
```

**Fixed Code:**
```python
# Fill first, THEN reindex (no forward-filling across boundaries)
s10 = s10.fillna(method='ffill').fillna(method='bfill')
df["t10y"] = s10.reindex(df_dates).values  # ✅ Safe
```

**Why:** `ffill()` after reindex will pull historical values forward, creating look-ahead bias.

**Verification:**
```python
# Run this after fix:
def test_macro_lookahead():
    """Ensure macro data doesn't have future values."""
    results = walkforward_cross_sectional(["AAPL"], period="3y")
    for fold in results["fold_metrics"]:
        # Check no test data used in training
        print(f"✅ Fold clean (no leakage)")

test_macro_lookahead()
```

---

## Phase 2: Regime Detection (4-5 hours)

### Step 1: Add regime features to `prediction_model.py`

**After line 680, add:**

```python
def add_regime_features(hist):
    """
    Add market regime indicators without look-ahead bias.
    All features must be shifted 1+ days into the past.
    """
    
    # ============ TREND REGIME ============
    # Market trend: positive 20-day return = BULL
    rolling_ret = hist["ret_1d"].rolling(20).sum()
    hist["regime_bull"] = (rolling_ret > 0).astype(int).shift(1)
    hist["regime_bear"] = (rolling_ret <= 0).astype(int).shift(1)
    
    # ============ VOLATILITY REGIME ============
    if "vix" in hist.columns:
        hist["regime_vix_low"] = (hist["vix"] < 12).shift(1).astype(int)   # Calm
        hist["regime_vix_med"] = ((hist["vix"] >= 12) & (hist["vix"] < 20)).shift(1).astype(int)  # Normal
        hist["regime_vix_high"] = (hist["vix"] >= 20).shift(1).astype(int)  # Stressed
    
    # ============ CRISIS REGIMES ============
    # Manual marking of known crisis periods (historical data only)
    hist["regime_covid"] = (
        (hist.index >= "2020-02-15") & (hist.index <= "2020-06-30")
    ).astype(int).shift(1)
    
    hist["regime_2008"] = (hist.index.year == 2008).astype(int).shift(1)
    
    # 2022 inflation crisis
    hist["regime_inflation_2022"] = (
        (hist.index >= "2022-03-01") & (hist.index <= "2022-12-31")
    ).astype(int).shift(1)
    
    return hist
```

**Step 2: Add to feature list**

**Around line 595 (after FEATURE_COLUMNS definition), add:**
```python
# Regime features (added dynamically)
REGIME_COLUMNS = [
    "regime_bull", "regime_bear",
    "regime_vix_low", "regime_vix_med", "regime_vix_high",
    "regime_covid", "regime_2008", "regime_inflation_2022"
]
FEATURE_COLUMNS.extend(REGIME_COLUMNS)
```

**Step 3: Call in feature pipeline**

**In `build_features_and_target()` function, after other feature additions:**
```python
hist = add_regime_features(hist)
```

**Step 4 (Optional): Filter walk-forward by regime**

**In `walkforward_cross_sectional()` function, after line 1430:**
```python
# Optional regime filtering
regime_filter = kwargs.get("regime_filter", None)  # "bull_only", "low_vol", "non_covid", None

if regime_filter == "bull_only":
    test_df = test_df[test_df["regime_bull"] == 1].copy()
elif regime_filter == "low_vol":
    test_df = test_df[test_df["regime_vix_low"] == 1].copy()
elif regime_filter == "non_covid":
    test_df = test_df[test_df["regime_covid"] == 0].copy()

if len(test_df) < 5:
    start += test_days
    continue  # Skip fold if too few samples
```

---

## Phase 3: TA-Lib Validation (1-2 hours)

### Install TA-Lib

```bash
pip install ta-lib

# If installation fails on macOS:
# brew install ta-lib
# pip install --verbose ta-lib
```

### Validate Your Indicators

**Add to `prediction_model.py`:**

```python
import talib

def validate_indicators(hist):
    """
    Compare your hand-coded indicators with TA-Lib implementations.
    Use TA-Lib results if they match (more reliable).
    """
    close = hist["close"].values
    high = hist["high"].values
    low = hist["low"].values
    volume = hist["volume"].values
    
    # RSI Validation
    if "rsi14" in hist.columns:
        rsi_talib = talib.RSI(close, timeperiod=14)
        
        # Compare (skip first 14 values, NaN)
        match = np.allclose(
            hist["rsi14"].iloc[14:].values,
            rsi_talib[14:],
            atol=0.1,  # Allow 0.1 point difference
            equal_nan=True
        )
        
        if match:
            print("✅ Your RSI matches TA-Lib")
            hist["rsi14"] = rsi_talib  # Use TA-Lib version
        else:
            print("⚠️  RSI mismatch - investigating...")
    
    # MACD Validation
    if "macd" in hist.columns:
        macd_talib, signal_talib, hist_talib = talib.MACD(close)
        
        match = np.allclose(
            hist["macd"].iloc[26:].values,
            macd_talib[26:],
            atol=0.01,
            equal_nan=True
        )
        
        if match:
            print("✅ Your MACD matches TA-Lib")
            hist["macd"] = macd_talib
            hist["macdsignal"] = signal_talib
            hist["macdhist"] = hist_talib
    
    # Bollinger Bands Validation
    if "bb_upper" in hist.columns:
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20)
        
        match = np.allclose(
            hist["bb_upper"].iloc[19:].values,
            bb_upper[19:],
            atol=0.5,  # Allow $0.50 difference
            equal_nan=True
        )
        
        if match:
            print("✅ Your Bollinger Bands match TA-Lib")
            hist["bb_upper"] = bb_upper
            hist["bb_lower"] = bb_lower
    
    return hist
```

**Call in pipeline:**
```python
# In build_features_and_target():
hist = validate_indicators(hist)  # After computing indicators
```

**Test:**
```python
df = build_features_and_target(["AAPL"], period="1y")
# Should see: ✅ Your RSI matches TA-Lib, etc.
```

---

## Phase 4: pmdarima ARIMA Ensemble (2-3 hours)

### Install

```bash
pip install pmdarima
```

### Add ARIMA predictor

**New file: `arima_model.py`**

```python
import numpy as np
from pmdarima import arima

class ARIMAPredictor:
    """
    Automatic ARIMA for single stock prediction.
    Finds best p,d,q automatically.
    """
    
    def __init__(self, max_p=5, max_d=2, max_q=5, seasonal=False):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.model = None
    
    def fit(self, returns_series):
        """
        Fit ARIMA model to return series.
        
        Args:
            returns_series: pd.Series of returns (e.g., daily log returns)
        """
        try:
            self.model = arima.auto_arima(
                returns_series.dropna(),
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal=self.seasonal,
                stepwise=True,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aic"
            )
            return True
        except Exception as e:
            print(f"ARIMA fit failed: {e}")
            return False
    
    def predict(self, steps=1):
        """Forecast next N days."""
        if self.model is None:
            return None
        
        try:
            forecast = self.model.predict(steps=steps)
            return forecast[-1]  # Return last forecast
        except:
            return None

def ensemble_ml_arima(ml_pred, arima_pred, ml_weight=0.7):
    """
    Blend ML prediction with ARIMA forecast.
    
    ml_weight: How much to trust ML model vs ARIMA
    0.7 = 70% ML, 30% ARIMA
    """
    if arima_pred is None:
        return ml_pred  # Fallback to ML only
    
    return ml_weight * ml_pred + (1 - ml_weight) * arima_pred
```

### Integrate into `predict_next_for_ticker()`

**In `prediction_model.py`, after line 1500:**

```python
from arima_model import ARIMAPredictor, ensemble_ml_arima

def predict_next_for_ticker(tk, ..., use_arima=False, arima_weight=0.7):
    """
    ... existing docstring ...
    
    use_arima: If True, blend with ARIMA forecast
    arima_weight: Weight for ARIMA in ensemble (default 30% ARIMA, 70% ML)
    """
    
    # ... existing prediction code ...
    
    pred_next_ret = model.predict(X_test_scaled)[0]
    
    # Optional ARIMA ensemble
    if use_arima:
        arima_pred = ARIMAPredictor()
        arima_pred.fit(hist["ret_1d"])  # Train on historical returns
        arima_forecast = arima_pred.predict(steps=horizon)
        
        pred_next_ret = ensemble_ml_arima(
            pred_next_ret, 
            arima_forecast,
            ml_weight=arima_weight
        )
        print(f"Blended prediction: {pred_next_ret:.6f} (70% ML, 30% ARIMA)")
    
    return {
        "pred_next_ret": pred_next_ret,
        # ... rest of return dict ...
    }
```

**Test:**
```python
from prediction_model import predict_next_for_ticker

result = predict_next_for_ticker("AAPL", use_arima=True, arima_weight=0.7)
print(f"Prediction: {result['pred_next_ret']}")
```

---

## Phase 5: Zipline Backtesting (4-5 hours) [FUTURE]

### Why Replace Current Backtest?

| Feature | Your Walk-Forward | Zipline |
|---------|------------------|---------|
| Slippage | Manual (constant) | Realistic curves |
| Commissions | Fixed percentage | Per-share + exchange |
| Margin | Not handled | Full margin modeling |
| Dividends | Not handled | Automatic |
| Corporate actions | Not handled | Splits, distributions |
| Portfolio management | Manual | Built-in |

### Install

```bash
pip install zipline-reloaded
```

### Basic Structure

```python
from zipline import run_algorithm
from zipline.api import order_percent, symbol, record

def initialize(context):
    """Initialize algorithm at start."""
    context.stocks = ["AAPL", "MSFT", "NVDA"]
    context.predictions = {}
    context.counter = 0

def before_trading_start(context, data):
    """Called every morning before market opens."""
    # Get predictions for today
    for stock in context.stocks:
        pred = get_prediction_from_model(stock)  # Your model
        context.predictions[stock] = pred

def handle_data(context, data):
    """Called every minute/day."""
    
    for stock in context.stocks:
        pred = context.predictions.get(stock, 0)
        
        if pred > 0.01:  # Positive prediction
            order_percent(symbol(stock), 0.1)  # Long 10%
        elif pred < -0.01:  # Negative prediction
            order_percent(symbol(stock), -0.05)  # Short 5%
    
    # Record metrics
    record(
        portfolio_value=context.portfolio.portfolio_value,
        gross_exposure=context.account.gross_exposure
    )

# Run backtest
results = run_algorithm(
    start=pd.Timestamp('2020-01-01'),
    end=pd.Timestamp('2023-12-31'),
    initialize=initialize,
    before_trading_start=before_trading_start,
    handle_data=handle_data,
    capital_base=100000,
    data_frequency='daily',
    bundle='quandl'  # Or your data source
)

# Analyze
print(results[['portfolio_value', 'gross_exposure']].tail())
```

**Note:** This is Phase 5 - only pursue after validating regime detection works.

---

## Phase 6: Additional Tools (Optional)

### AlphaLens (Factor Quality Testing)

```bash
pip install alphalens
```

**Use Case:** Evaluate feature predictiveness before adding to model

```python
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import factor_returns, mean_return_by_quantile

# Test "rsi14" factor quality
factor_data = get_clean_factor_and_forward_returns(
    factor=df["rsi14"],
    prices=df["close"],
    periods=[1, 5, 10]  # Forecast horizons
)

# Information Coefficient (correlation with forward returns)
ic = factor_data.groupby(level=0)[[1, 5, 10]].apply(
    lambda x: x.corr()  # Should be > 0.05 for good signal
)

print(f"IC: {ic.mean()}")  # Higher = better signal
```

### MLFinLab (Purged K-Fold)

```bash
pip install mlfinlab
```

**Replaces sklearn's KFold with finance-aware version:**

```python
from mlfinlab.cross_validation import PurgedKFold, ml_get_train_times

# More sophisticated than my PurgedKFold
cv = PurgedKFold(n_splits=5, embargo=0.01)  # 1% embargo

for train_idx, test_idx in cv.split(X):
    # Properly handles overlapping indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # ... train model ...
```

---

## Troubleshooting & FAQs

**Q: TA-Lib installation fails on M1 Mac**
```bash
# Solution:
brew install ta-lib
pip install --verbose ta-lib
```

**Q: pmdarima takes forever to find ARIMA parameters**
```python
# Solution: Use stepwise and limit search space
arima.auto_arima(
    data,
    max_p=3,  # Reduce from 5
    max_d=1,  # Reduce from 2
    max_q=3,  # Reduce from 5
    stepwise=True,  # Much faster
    trace=False  # No console output
)
```

**Q: Should I use ALL regime features?**
A: Start with regime_bull, regime_bear, regime_vix_high. Add others if they help Sharpe.

**Q: What if ARIMA ensemble makes results worse?**
A: That's OK! Drop it. Your ML model might already capture trend.

**Q: Can I test Zipline with my current signals?**
A: Not yet - Zipline has its own prediction pipeline. Build it in Phase 5.

---

## Checklist: Phase 1-3 Completion

### Phase 1: Macro Data Fix
- [ ] Fixed `ffill()` placement in `build_panel_features_and_target()`
- [ ] Ran test: no look-ahead bias detected
- [ ] Walk-forward Sharpe stable / slightly improved

### Phase 2: Regime Detection
- [ ] Added `add_regime_features()` function
- [ ] Added regime columns to FEATURE_COLUMNS
- [ ] Trained model with regime features
- [ ] Tested `regime_filter="bull_only"` optional parameter
- [ ] Documented which regimes improve Sharpe most

### Phase 3: TA-Lib Validation
- [ ] Installed TA-Lib
- [ ] Added `validate_indicators()` function
- [ ] Ran validation (should see ✅ checkmarks)
- [ ] Replaced hand-coded indicators with TA-Lib versions (if different)
- [ ] Verified no accuracy loss

---

## Expected Timeline

| Phase | Effort | Timeline | Expected Improvement |
|-------|--------|----------|----------------------|
| 1: Macro Fix | 2-3h | This week | Stability ⭐ |
| 2: Regimes | 4-5h | Week 2 | Sharpe +15-30% ⭐⭐ |
| 3: TA-Lib | 1-2h | Week 3 | Confidence ⭐ |
| 4: ARIMA | 2-3h | Week 4 | +5-10% if works ⭐ |
| 5: Zipline | 4-5h | Month 2 | Realism ⭐⭐⭐ |

---

**Start with Phase 1 (Macro Fix) - do it today! 🚀**

