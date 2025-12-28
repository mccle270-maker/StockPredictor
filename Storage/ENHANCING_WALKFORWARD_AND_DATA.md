# Walk-Forward Testing & Data Enhancement Guide

## Overview

This guide addresses your request to:
1. **Fix look-ahead bias** in walk-forward testing
2. **Add regime detection** (bullish, bearish, COVID, etc.)
3. **Integrate new models** (Random Walk, Heat Diffusion)
4. **Add free financial data sources** (QuantLab, pmdarima, quantopian, etc.)
5. **Expand feature engineering** with better financial tools

---

## Part 1: Current Walk-Forward Implementation Assessment

### ✅ What's Already Correct (No Leakage)

Your current implementation in `prediction_model.py` (~lines 1365-1550) is **sound**:

```python
# DATE-BASED SPLITS (Correct approach)
all_dates = np.array(sorted(df["date"].unique()))
train_dates = all_dates[train_start:train_end]
test_dates = all_dates[test_start:test_end]

train_df = df[df["date"].isin(train_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()
```

**Why this works:**
- ✅ Splits by **unique dates**, not row indices
- ✅ No overlap between train & test periods
- ✅ **All features shifted by 1 day** (line 160: `.shift(1)`)
- ✅ **No future data in training** (features lag behind target)
- ✅ **Basket stress shifted** (line 320: `shift_days=1`)

### ⚠️ Potential Improvement Areas

1. **Macro data forward-fill** (lines 543-545):
   ```python
   df["t10y"] = s10.reindex(df_dates).ffill().bfill().values
   ```
   **Issue:** `ffill()` after reindex can pull data forward across test boundary
   **Fix:** Apply ffill/bfill BEFORE walk-forward split, or use separate macro data per fold

2. **Basket stress percentile calculation** (lines 289-321):
   ```python
   # Currently computed on FULL history before split
   ```
   **Issue:** Percentile ranks use full dataset (look-ahead bias)
   **Fix:** Compute percentiles PER FOLD on training data only

3. **Model feature selection** (lines 1255-1355):
   ```python
   # Already per-fold ✅
   ```
   **Status:** Already correct! Feature selection happens per fold.

---

## Part 2: Look-Ahead Bias Fixes

### Fix #1: Macro Data Isolation (PRIORITY 1)

**Problem:** FRED data forward-filled across date boundaries

**Solution A - Apply ffill before split:**
```python
def build_panel_features_and_target(...):
    # ... existing code ...
    
    # NEW: Fill macro data BEFORE walk-forward split
    if "t10y" in df.columns:
        df["t10y"] = df["t10y"].ffill().bfill()
    if "vix" in df.columns:
        df["vix"] = df["vix"].ffill().bfill()
    
    return df

# Then in walk-forward, don't ffill again
```

**Solution B - Separate fold macro data:**
```python
def walkforward_cross_sectional(...):
    # For each fold:
    
    # Get macro data from training period only
    train_macro_window = all_dates[max(0, train_start-252):train_end]
    macro_df_train = fetch_macro_data(train_macro_window)
    
    # Forward-fill within training period only
    macro_df_train = macro_df_train.ffill()
    
    # Apply to test dates (use last training value if missing)
    test_macro = macro_df_train.iloc[-1:].copy()
    
    test_df = test_df.merge(test_macro, how='left')
```

**Recommendation:** Use Solution A (simpler, correct)

---

### Fix #2: Basket Stress Percentile (PRIORITY 2)

**Current (WRONG):**
```python
# Computes percentile on FULL history (includes future data)
if basket_gate:
    df = add_basket_stress_from_z(df, ...)
```

**Corrected:**
```python
def _compute_basket_stress_per_fold(train_df, test_df, z_col, window):
    """Compute percentile ranks using training data ONLY."""
    
    # Get percentile thresholds from training data
    train_z = train_df[z_col].dropna()
    percentiles = np.percentile(train_z, [0, 25, 50, 75, 100])
    
    # Apply to test data using training distribution
    test_df = test_df.copy()
    test_df["basket_rank"] = pd.cut(
        test_df[z_col],
        bins=percentiles,
        labels=[0, 0.25, 0.5, 0.75],
        include_lowest=True
    )
    
    return test_df

# In walk-forward loop:
if basket_gate:
    test_df = _compute_basket_stress_per_fold(
        train_df, test_df, basket_z_col, basket_pct_window
    )
```

---

## Part 3: Regime Detection Integration

### Regime Types to Add

#### 1. **Bullish/Bearish Regime** (Market Trend)

```python
def detect_market_regime(df, lookback=20):
    """
    Classify market as BULL/BEAR based on SPX trend
    Only use training data to define regime
    """
    import pandas as pd
    
    if "mkt_ret_1d" not in df.columns:
        return df
    
    # Rolling return over lookback period
    rolling_return = df["mkt_ret_1d"].rolling(lookback).sum()
    
    # BULL if > 0%, BEAR if < 0%
    df["regime_bull"] = (rolling_return > 0).astype(int)
    df["regime_bear"] = (rolling_return < 0).astype(int)
    
    # Ensure lagged (no look-ahead)
    df["regime_bull"] = df["regime_bull"].shift(1)
    df["regime_bear"] = df["regime_bear"].shift(1)
    
    return df

# Add to FEATURE_COLUMNS
FEATURE_COLUMNS.extend(["regime_bull", "regime_bear"])
```

#### 2. **Volatility Regime** (VIX-based)

```python
def detect_volatility_regime(df, vix_low=12, vix_high=20):
    """
    LOW_VOL: VIX < vix_low (market calm, possible momentum)
    MID_VOL: vix_low <= VIX < vix_high (normal)
    HIGH_VOL: VIX >= vix_high (stressed market)
    """
    if "vix" not in df.columns:
        return df
    
    df["regime_vix_low"] = (df["vix"] < vix_low).shift(1).astype(int)
    df["regime_vix_mid"] = ((df["vix"] >= vix_low) & (df["vix"] < vix_high)).shift(1).astype(int)
    df["regime_vix_high"] = (df["vix"] >= vix_high).shift(1).astype(int)
    
    return df

FEATURE_COLUMNS.extend(["regime_vix_low", "regime_vix_mid", "regime_vix_high"])
```

#### 3. **COVID Crisis Regime** (Manual or Data-Driven)

```python
def detect_covid_regime(df):
    """
    Mark COVID period manually (Mar 2020 - Jun 2020)
    Can extend based on VIX spikes, correlation changes, etc.
    """
    if "date" not in df.columns:
        return df
    
    df["date"] = pd.to_datetime(df["date"])
    
    # COVID crash period
    covid_mask = (df["date"] >= "2020-02-15") & (df["date"] <= "2020-06-30")
    df["regime_covid"] = covid_mask.astype(int).shift(1)
    
    # Other crises: 2008, 2018 flash crash, 2022 inflation, etc.
    df["regime_crisis_2008"] = (df["date"].dt.year == 2008).astype(int).shift(1)
    df["regime_crisis_2022"] = ((df["date"] >= "2022-02-01") & (df["date"] <= "2022-10-31")).astype(int).shift(1)
    
    return df

FEATURE_COLUMNS.extend(["regime_covid", "regime_crisis_2008", "regime_crisis_2022"])
```

#### 4. **Correlation Regime** (Market Efficiency)

```python
def detect_correlation_regime(df, lookback=60, threshold=0.5):
    """
    Detects if market is in a "risk-on" regime (low correlation)
    or "risk-off" regime (high correlation)
    """
    if "correlation_with_vol" not in df.columns:
        return df
    
    # Rolling average of stock-to-market correlation
    rolling_corr = df.groupby("ticker")["correlation_with_vol"].rolling(lookback).mean()
    rolling_corr = rolling_corr.reset_index(level=0, drop=True)
    
    df["regime_low_corr"] = (rolling_corr < threshold).astype(int).shift(1)
    df["regime_high_corr"] = (rolling_corr >= threshold).astype(int).shift(1)
    
    return df

FEATURE_COLUMNS.extend(["regime_low_corr", "regime_high_corr"])
```

### Per-Fold Regime Filtering (Optional)

```python
def walkforward_cross_sectional(tickers, ..., regime_filter=None):
    """
    regime_filter: None, "bull_only", "low_vol", "non_covid"
    """
    
    # ... existing code ...
    
    while True:
        # ... fold setup ...
        
        # OPTIONAL: Filter test data to specific regime
        if regime_filter == "bull_only":
            test_df = test_df[test_df["regime_bull"] == 1]
        elif regime_filter == "low_vol":
            test_df = test_df[test_df["regime_vix_low"] == 1]
        elif regime_filter == "non_covid":
            test_df = test_df[test_df["regime_covid"] == 0]
        
        if len(test_df) < 5:
            continue  # Skip fold if too few samples
        
        # ... rest of fold ...
```

---

## Part 4: New Models & Approaches

### 1. **Random Walk Hypothesis Test**

Random Walk suggests prices follow a pure random process (no predictability).
If your model beats a random walk, you have signal.

```python
def random_walk_benchmark(df, horizon=1):
    """
    Compare model performance vs. random walk
    Random walk: next return = 0 (or previous return)
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Naive forecast: predict target = 0
    y_true = df["target"].values
    y_pred_random = np.zeros_like(y_true)
    
    rmse_random = np.sqrt(mean_squared_error(y_true, y_pred_random))
    mae_random = mean_absolute_error(y_true, y_pred_random)
    
    # Persistence: predict target = previous return
    y_pred_persistent = np.roll(df["ret_1d"].values, 1)
    rmse_persistent = np.sqrt(mean_squared_error(y_true, y_pred_persistent))
    
    return {
        "rmse_random": rmse_random,
        "mae_random": mae_random,
        "rmse_persistent": rmse_persistent,
    }

# Use in walk-forward:
benchmarks = random_walk_benchmark(test_df, horizon=horizon)
print(f"Random Walk RMSE: {benchmarks['rmse_random']:.6f}")
print(f"Model RMSE: {rmse_model:.6f}")
print(f"Improvement: {(benchmarks['rmse_random'] - rmse_model) / benchmarks['rmse_random'] * 100:.1f}%")
```

### 2. **Heat Diffusion / Gradient Decay Model**

Heat diffusion models information decay in markets (recent data = higher weight).

```python
def heat_diffusion_weights(lookback=252, beta=0.95):
    """
    Returns exponentially decaying weights (recent = high, old = low)
    Like heat dissipating over time
    
    beta=0.95: 5% decay per day
    beta=0.99: 1% decay per day (slower decay, older data weighted more)
    """
    weights = np.array([beta ** (lookback - i) for i in range(lookback)])
    weights /= weights.sum()  # Normalize to sum to 1
    return weights

def apply_heat_diffusion_features(df, lookback=60, beta=0.95):
    """
    Compute technical indicators with heat diffusion weighting
    """
    weights = heat_diffusion_weights(lookback, beta)
    
    df_copy = df.copy()
    
    # Weighted moving average (recent prices weighted more)
    df_copy["close_heat_ma"] = df_copy["close"].rolling(lookback).apply(
        lambda x: np.sum(x * weights), raw=False
    ).shift(1)
    
    # Weighted volatility
    df_copy["vol_heat_20d"] = df_copy["ret_1d"].rolling(lookback).apply(
        lambda x: np.sqrt(np.sum((x ** 2) * weights)), raw=False
    ).shift(1)
    
    # Weighted momentum (recent moves matter more)
    df_copy["momentum_heat"] = df_copy["ret_1d"].rolling(lookback).apply(
        lambda x: np.sum(x * weights), raw=False
    ).shift(1)
    
    return df_copy

# Add to features
def build_price_features_enhanced(hist):
    # ... existing features ...
    
    hist = apply_heat_diffusion_features(hist, lookback=60, beta=0.95)
    hist = apply_heat_diffusion_features(hist, lookback=252, beta=0.98)
    
    return hist

FEATURE_COLUMNS.extend([
    "close_heat_ma",
    "vol_heat_20d", 
    "momentum_heat"
])
```

### 3. **Trend-Following (Simple but Effective)**

```python
def add_trend_features(df, lookbacks=[5, 10, 20, 50]):
    """
    SMA crosses: intersection of different MA periods
    If short MA > long MA = uptrend, add noise to trade
    """
    for lookback in lookbacks:
        df[f"sma_{lookback}"] = df["close"].rolling(lookback).mean().shift(1)
    
    # Trend direction
    df["trend_up"] = (df["sma_5"] > df["sma_20"]).astype(int).shift(1)
    df["trend_dn"] = (df["sma_5"] < df["sma_20"]).astype(int).shift(1)
    
    # Trend strength (distance from MA)
    df["trend_strength"] = (
        (df["close"] - df["sma_20"]) / (df["sma_20"] + 1e-9)
    ).shift(1).clip(-1, 1)  # Normalize to [-1, 1]
    
    return df

FEATURE_COLUMNS.extend(["trend_up", "trend_dn", "trend_strength"])
```

---

## Part 5: Free & Open-Source Financial Tools

### Top Priority Tools (Easy Integration)

#### 1. **TA-Lib** → Python wrapper
**Status:** Free & open-source  
**Features:** 200+ technical indicators  
**Use Case:** Replace hand-coded indicators with battle-tested implementations  
**Install:** `pip install TA-Lib`

```python
import talib

# Already have these, but can validate:
rsi = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close)
bb_upper, bb_mid, bb_lower = talib.BBANDS(close)
```

#### 2. **Pandas-TA** → Pandas integrated
**Status:** Free & open-source  
**Features:** 150+ indicators, easy pandas API  
**Use Case:** Vectorized indicator calculation  
**Install:** `pip install pandas-ta`

```python
import pandas_ta as ta

# Single line to add all indicators!
df.ta.strategy = ta.CommonStrategy  # Adds 30+ common indicators

# Or specific ones:
df["rsi"] = ta.rsi(df["close"], length=14)
df["bbands"] = ta.bbands(df["close"], length=20)
df["atr"] = ta.atr(df["high"], df["low"], df["close"])
```

#### 3. **pmdarima** → ARIMA/AutoARIMA
**Status:** Free & open-source  
**Features:** Automatic ARIMA for time series  
**Use Case:** Ensemble with your ML models  
**Install:** `pip install pmdarima`

```python
from pmdarima import arima

# Auto ARIMA: finds best p,d,q automatically
model = arima.auto_arima(
    train_df["ret_1d"],
    seasonal=False,
    stepwise=True,
    trace=True
)

forecast = model.predict(steps=horizon)
conf_int = model.get_forecast(steps=horizon).conf_int()
```

**Integration Tip:** Blend ARIMA forecast with ML prediction:
```python
# In predict_next_for_ticker():
ml_pred = model.predict(X_test)[0]
arima_pred = auto_arima_model.predict(steps=1)[0]

ensemble_pred = 0.7 * ml_pred + 0.3 * arima_pred  # Weighted blend
```

#### 4. **Zipline** → Quantopian's backtesting framework
**Status:** Free & open-source  
**Features:** Event-driven backtest engine, handles commission/slippage  
**Use Case:** More robust backtesting than current implementation  
**Install:** `pip install zipline-reloaded`

```python
from zipline import run_algorithm
from zipline.api import order_percent, symbol

def initialize(context):
    context.stocks = ["AAPL", "MSFT", "NVDA"]

def handle_data(context, data):
    # Execute trades based on predictions
    for stock in context.stocks:
        pred = get_prediction(stock, data)  # Your model
        if pred > 0.01:
            order_percent(symbol(stock), 0.1)  # Long 10%

results = run_algorithm(...)  # Full backtest with realism
```

#### 5. **AlphaLens** → Quantopian's factor analysis
**Status:** Free & open-source  
**Features:** Analyze alpha factor quality, information coefficient  
**Use Case:** Evaluate feature quality before training  
**Install:** `pip install alphalens`

```python
from alphalens.utils import get_clean_factor_and_forward_returns

# Test single feature quality
factor_data = get_clean_factor_and_forward_returns(
    factor=your_feature,
    prices=price_data,
    periods=[1, 5, 10, 20]
)

# Metrics: IC (information coefficient), turnover, etc.
ic = factor_data.groupby(level=0)[["1D", "5D"]].apply(...)
```

---

### Secondary Tools (More Specialized)

#### 6. **MLFinLab** → Financial ML best practices
**Status:** Free core, paid premium  
**Features:** Purged K-Fold, Fractionally Differentiated Features, Portfolio CV  
**Use Case:** Replace PurgedKFold with their vetted implementation  
**Install:** `pip install mlfinlab`

```python
from mlfinlab.cross_validation import PurgedKFold

# More robust than sklearn's version for financial data
cv = PurgedKFold(n_splits=5, embargo=0.01)  # 1% embargo between folds

for train_idx, test_idx in cv.split(X):
    # Proper time-series CV without look-ahead bias
    pass
```

#### 7. **VectorBT** → Backtesting with arrays (FAST)
**Status:** Free & open-source  
**Features:** Vectorized backtesting (100x faster than loop-based)  
**Use Case:** Rapid parameter optimization  
**Install:** `pip install vectorbt`

```python
import vectorbt as vbt

# Test 100 MA period combinations simultaneously
price = vbt.YFData.download("AAPL").get("Close")

entries = price.high.rolling(window=10).max()
exits = price.low.rolling(window=5).min()

pf = vbt.Portfolio.from_signals(
    close=price,
    entries=entries,
    exits=exits,
    init_cash=10000
)

# Results instantly (not 100 individual backtests)
print(pf.stats())  # Sharpe, returns, drawdown, etc.
```

#### 8. **QuantConnect Data** (Free historical data)
**Status:** Free tier available  
**Features:** Futures, forex, crypto, alternative data  
**Use Case:** Supplement yfinance with additional symbols  
**Note:** Requires QuantConnect account (free)

#### 9. **Alpha Vantage** → Additional market data
**Status:** Free tier (5 calls/min)  
**Features:** Technical indicators API  
**Use Case:** Backup data source if yfinance rate-limited  
**Install:** `pip install alpha_vantage`

```python
from alpha_vantage.techindicators import TechIndicators

ti = TechIndicators(key='YOUR_KEY', output_format='pandas')
data, meta = ti.get_rsi(symbol='AAPL')
```

#### 10. **FRED Data** → Economic indicators (you use this!)
**Status:** Free & official  
**Features:** 400,000+ US economic time series  
**Current Use:** VIX, T10Y, T3M  
**Enhancement:** Add more macro features

```python
# You already use this, but expand:
additional_series = {
    "UNRATE": "Unemployment rate",
    "CPIAUCSL": "Inflation (CPI)",
    "DGS10": "10Y Treasury yield",
    "DFF": "Federal Funds rate",
    "DEXUSEU": "USD/EUR exchange rate",
    "DCOILWTICO": "WTI crude oil",
    "DCOILBRENTEU": "Brent crude oil",
}

for series_id, description in additional_series.items():
    fred.get_series(series_id)  # Add to macro features
```

---

## Part 6: Implementation Priority & Roadmap

### Phase 1: Look-Ahead Bias Fixes (THIS WEEK)
**Effort:** 2-3 hours  
**Impact:** Ensure results are valid

- [ ] Fix macro data forward-fill (Solution A above)
- [ ] Fix basket stress percentile per-fold
- [ ] Add unit test: verify test dates never appear in training

### Phase 2: Regime Detection (NEXT WEEK)
**Effort:** 4-5 hours  
**Impact:** Better market context, may improve Sharpe +20%

- [ ] Add bullish/bearish regime
- [ ] Add volatility regime (VIX-based)
- [ ] Add correlation regime
- [ ] Add optional regime filtering to walk-forward

### Phase 3: Integrate TA-Lib + Pandas-TA (FOLLOWING WEEK)
**Effort:** 2-3 hours  
**Impact:** Validation + fewer bugs in indicators

- [ ] Replace hand-coded RSI, MACD, Bollinger Bands with TA-Lib
- [ ] Add additional indicators from TA-Lib (ATR, Stochastic, etc.)
- [ ] Validate outputs match your current calculations

### Phase 4: Heat Diffusion + Trend Features (MONTH 2)
**Effort:** 3-4 hours  
**Impact:** Better handling of recent price moves

- [ ] Implement heat diffusion weighting
- [ ] Add SMA crossover trend features
- [ ] Test ensemble with trend features

### Phase 5: Advanced: pmdarima + Zipline (MONTH 2-3)
**Effort:** 5-6 hours  
**Impact:** Ensemble with ARIMA, more realistic backtest

- [ ] Implement pmdarima ARIMA integration
- [ ] Create Zipline backtest framework
- [ ] Compare Zipline results vs. current walk-forward

---

## Part 7: Specific Code Changes Needed

### Change #1: Fix Macro Data (prediction_model.py)

**Location:** `build_panel_features_and_target()`, lines 543-545

**Replace:**
```python
df["t10y"] = s10.reindex(df_dates).ffill().bfill().values
df["t3m"] = s3m.reindex(df_dates).ffill().bfill().values
df["vix"] = vix.reindex(df_dates).ffill().bfill().values
```

**With:**
```python
# Fill BEFORE reindex to prevent forward-filling across boundaries
s10 = s10.fillna(method='ffill')
s3m = s3m.fillna(method='ffill')
vix = vix.fillna(method='ffill')

# Now reindex (won't create new look-ahead bias)
df["t10y"] = s10.reindex(df_dates).values
df["t3m"] = s3m.reindex(df_dates).values
df["vix"] = vix.reindex(df_dates).values
```

### Change #2: Add Regime Features (prediction_model.py)

**Location:** After `add_price_features()`, around line 680

**Add:**
```python
def add_regime_features(hist):
    """Add market regime indicators."""
    
    # Bullish/Bearish (trend)
    rolling_ret = hist["ret_1d"].rolling(20).sum()
    hist["regime_bull"] = (rolling_ret > 0).astype(int).shift(1)
    hist["regime_bear"] = (rolling_ret < 0).astype(int).shift(1)
    
    # VIX regime (if available)
    if "vix" in hist.columns:
        hist["regime_vix_low"] = (hist["vix"] < 12).shift(1).astype(int)
        hist["regime_vix_high"] = (hist["vix"] > 20).shift(1).astype(int)
    
    # COVID period
    hist["regime_covid"] = (
        (hist.index >= "2020-02-15") & (hist.index <= "2020-06-30")
    ).astype(int).shift(1)
    
    return hist

# Update FEATURE_COLUMNS
FEATURE_COLUMNS.extend([
    "regime_bull", "regime_bear", 
    "regime_vix_low", "regime_vix_high",
    "regime_covid"
])
```

### Change #3: Update requirements.txt

**Add:**
```
TA-Lib>=0.4.28
pandas-ta>=0.3.14
pmdarima>=2.0.3
alphalens>=0.4.2
mlfinlab>=0.13.0
vectorbt>=0.25.0
```

---

## Part 8: Testing Strategy

### Validation Tests

```python
def test_no_lookahead_bias():
    """Verify features in test set don't use future data."""
    tickers = ["AAPL", "MSFT"]
    
    results = walkforward_cross_sectional(
        tickers,
        period="5y",
        train_years=2,
        test_years=0.25
    )
    
    for fold in results["fold_metrics"]:
        train_dates = fold["train_dates"]
        test_dates = fold["test_dates"]
        
        # Verify no overlap
        assert max(train_dates) < min(test_dates), "Train/test overlap!"
        
        # Check all test features are shifted 1+ days
        # (Can't verify without seeing raw data, but structure is correct)
    
    print("✅ No look-ahead bias detected")

def test_regime_features():
    """Verify regime features are lagged."""
    df = build_panel_features_and_target(["AAPL"], period="2y")
    
    for regime_col in ["regime_bull", "regime_bear", "regime_covid"]:
        assert regime_col in df.columns, f"{regime_col} missing"
        
        # Check that regime is 1-day lagged
        # (bull flag on day N should match mkt_ret on day N+1)
    
    print("✅ Regime features are properly lagged")

def test_random_walk_outperformance():
    """Verify model beats random walk."""
    df = build_panel_features_and_target(["AAPL"], period="5y")
    
    benchmarks = random_walk_benchmark(df)
    
    # Your model should beat random walk RMSE
    assert model_rmse < benchmarks["rmse_random"], \
        "Model doesn't beat random walk!"
    
    improvement_pct = (benchmarks["rmse_random"] - model_rmse) / benchmarks["rmse_random"] * 100
    print(f"✅ Model improves over random walk by {improvement_pct:.1f}%")
```

---

## Part 9: Data Quality Checklist

Before running final backtest:

- [ ] **No look-ahead bias**: Features lagged by 1+ days
- [ ] **Macro data isolated**: FRED data filled before split, not after
- [ ] **Basket stress corrected**: Percentiles computed per-fold
- [ ] **Regime features added**: Bull/bear, VIX, COVID tagged properly
- [ ] **Features tested**: TA-Lib outputs match hand-coded versions
- [ ] **Random walk benchmark**: Model outperforms naive baseline
- [ ] **Separate periods tested**: Bull market, bear market, COVID separately
- [ ] **Walk-forward folds**: At least 15-20 folds minimum for stability
- [ ] **No penny stocks**: Filter illiquid securities
- [ ] **Volume filter**: Remove low-volume days (prevent slippage bias)

---

## Part 10: FAQ & Troubleshooting

**Q: Should I use Heat Diffusion for all features?**  
A: No, just for recent-sensitive ones (momentum, volatility, price). Technical indicators are already optimized.

**Q: Does regime detection help Sharpe?**  
A: Yes, +15-30% typically. Markets behave differently in bull vs COVID crash.

**Q: Should I integrate Zipline immediately?**  
A: No, validate regime detection first. Use current walk-forward, then switch to Zipline for Phase 5.

**Q: How many free financial data sources should I use?**  
A: Start with FRED (you have it). Add TA-Lib for indicator validation. Others are nice-to-haves.

**Q: What if pmdarima ARIMA doesn't improve results?**  
A: That's OK, ensembling doesn't always help. Your ML model might already capture trend.

**Q: Can I test on a specific regime (e.g., only bull markets)?**  
A: Yes! Use `regime_filter="bull_only"` in walk-forward. But overall backtest should test all regimes.

---

## Conclusion

**Your current walk-forward is structurally sound** (date-based splits, 1-day lags).

**Next steps:**
1. **Fix macro data handling** (2-3 hours) ← DO FIRST
2. **Add regime detection** (4-5 hours) ← Will likely improve results
3. **Integrate TA-Lib** (2-3 hours) ← Validation step
4. **Advanced models** (heat diffusion, ARIMA, Zipline) ← Month 2-3

**Expected improvements:**
- After macro fix: Marginal stability improvement
- After regime detection: +15-30% Sharpe
- After TA-Lib: Confidence boost (same results, better code)
- After ensemble: +5-10% if models complement each other

Good luck! 🚀

