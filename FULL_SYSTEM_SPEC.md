# Stock Predictor - Complete System Specification

> **Purpose**: This document contains the complete specification and implementation details for a stock prediction and automated trading system. It can be used to recreate or build upon this system in any programming language or platform.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Feature Engineering](#feature-engineering)
4. [Short-Term Prediction Model](#short-term-prediction-model)
5. [Long-Horizon Prediction Model](#long-horizon-prediction-model)
6. [Options Pricing](#options-pricing)
7. [Signal Generation](#signal-generation)
8. [Trading Execution](#trading-execution)
9. [Web Dashboard](#web-dashboard)
10. [API Endpoints](#api-endpoints)
11. [Configuration](#configuration)
12. [Complete Algorithm Pseudocode](#complete-algorithm-pseudocode)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STOCK PREDICTOR SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ Data Sources │───▶│   Features   │───▶│  ML Models   │                  │
│  │              │    │   (100+)     │    │              │                  │
│  │ • Yahoo      │    │ • Technical  │    │ • RF         │                  │
│  │ • FRED       │    │ • Macro      │    │ • XGBoost    │                  │
│  │ • FMP        │    │ • Sentiment  │    │ • GBRT       │                  │
│  │ • Marketaux  │    │ • GBM Probs  │    │ • Analog     │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                  │                          │
│                                                  ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Alpaca     │◀───│   Signals    │◀───│ Predictions  │                  │
│  │   Broker     │    │  (JSON)      │    │              │                  │
│  │              │    │              │    │ • Direction  │                  │
│  │ • Stocks     │    │ • BUY/SELL   │    │ • Magnitude  │                  │
│  │ • Options    │    │ • Strategy   │    │ • Confidence │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                  │                          │
│                                                  ▼                          │
│                                          ┌──────────────┐                  │
│                                          │  Streamlit   │                  │
│                                          │  Dashboard   │                  │
│                                          └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Short-Term Prediction** | 1-5 day return forecasts using ensemble ML |
| **Long-Term Outlook** | 20-30 day direction using historical analog matching |
| **Options Strategies** | Automated strategy suggestions (spreads, iron condors) |
| **Paper Trading** | Alpaca integration for automated execution |
| **Risk Metrics** | Sharpe, drawdown, VaR calculations |
| **Live Dashboard** | Streamlit web interface |

---

## Data Pipeline

### Data Sources

| Source | API Key Variable | Data Provided | Endpoint |
|--------|------------------|---------------|----------|
| **Yahoo Finance** | None | OHLCV prices | `yfinance` library |
| **FRED** | `FRED_API_KEY` | VIX, T10Y, unemployment, CPI, Fed Funds | `api.stlouisfed.org` |
| **FMP** | `FMP_API_KEY` | P/E, P/B, market cap | `financialmodelingprep.com/stable` |
| **Marketaux** | `MARKETAUX_API_KEY` | News sentiment | `api.marketaux.com/v1/news` |
| **Alpha Vantage** | `ALPHAVANTAGE_API_KEY` | Backup data | `alphavantage.co/query` |

### Price History Fetching (with fallbacks)

```
FUNCTION get_price_history(ticker, period, interval):
    
    # Try 1: Yahoo Finance via yfinance
    TRY:
        data = yfinance.Ticker(ticker).history(period, interval)
        IF data is not empty:
            RETURN data
    CATCH error:
        log("Yahoo failed: " + error)
    
    # Try 2: Stooq CSV (daily only)
    IF interval == "1d":
        TRY:
            url = "https://stooq.com/q/d/l/?s={ticker}.us&i=d"
            data = download_csv(url)
            data = filter_by_period(data, period)
            IF data is not empty:
                RETURN data
        CATCH error:
            log("Stooq failed: " + error)
    
    # Try 3: Raw Yahoo download
    TRY:
        data = yfinance.download(ticker, period=period)
        IF data is not empty:
            RETURN data
    CATCH error:
        log("All sources failed")
    
    THROW "No price history available"
```

### FRED Macro Data Fetching

```
FUNCTION get_fred_series(series_id, start_date, end_date):
    
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }
    
    response = HTTP_GET(url, params)
    observations = response.json()["observations"]
    
    dates = []
    values = []
    FOR each obs IN observations:
        IF obs["value"] != ".":
            dates.append(obs["date"])
            values.append(float(obs["value"]))
    
    RETURN Series(values, index=dates)
```

### FRED Series Used

| Series ID | Feature Name | Description |
|-----------|--------------|-------------|
| `VIXCLS` | `vix` | CBOE Volatility Index |
| `DGS10` | `t10y` | 10-Year Treasury Yield |
| `DGS3MO` | (for term_spread) | 3-Month Treasury Yield |
| `UNRATE` | `unrate` | Unemployment Rate |
| `CPIAUCSL` | `cpi` | Consumer Price Index |
| `BAMLH0A0HYM2` | `oas` | High Yield OAS Spread |
| `FEDFUNDS` | `fed_funds` | Federal Funds Rate |

### Macro DataFrame Construction

```
FUNCTION get_macro_df(period):
    
    end_date = today()
    start_date = end_date - period_to_days(period)
    
    # Fetch each series
    vix = get_fred_series("VIXCLS", start_date, end_date)
    t10y = get_fred_series("DGS10", start_date, end_date)
    t3m = get_fred_series("DGS3MO", start_date, end_date)
    unrate = get_fred_series("UNRATE", start_date, end_date)
    cpi = get_fred_series("CPIAUCSL", start_date, end_date)
    oas = get_fred_series("BAMLH0A0HYM2", start_date, end_date)
    fed_funds = get_fred_series("FEDFUNDS", start_date, end_date)
    
    # Get SPX for market return
    spx = get_price_history("^GSPC", period, "1d")
    mkt_ret_1d = spx["Close"].pct_change()
    
    # Combine into DataFrame
    macro = DataFrame({
        "mkt_ret_1d": mkt_ret_1d,
        "vix": vix,
        "t10y": t10y,
        "term_spread": t10y - t3m,
        "unrate": unrate,
        "cpi": cpi,
        "oas": oas,
        "fed_funds": fed_funds
    })
    
    # Forward fill missing values, then backward fill
    macro = macro.ffill().bfill()
    
    RETURN macro
```

---

## Feature Engineering

### Feature Categories

The system generates 100+ features organized into logical groups:

#### 1. Price Returns
```
ret_1d = Close.pct_change(1).shift(1)
ret_3d = Close.pct_change(3).shift(1)
ret_5d = Close.pct_change(5).shift(1)
ret_20d = Close.pct_change(20).shift(1)
cumret_3d = (1 + ret_1d).rolling(3).apply(prod) - 1
cumret_5d = (1 + ret_1d).rolling(5).apply(prod) - 1
```

#### 2. Volatility
```
vol_10d = ret_1d.rolling(10).std().shift(1)
vol_20d = ret_1d.rolling(20).std().shift(1)
vol_60d = ret_1d.rolling(60).std().shift(1)
vol_ratio_10_60 = vol_10d / vol_60d

# ATR (Average True Range)
true_range = max(High - Low, abs(High - prev_Close), abs(Low - prev_Close))
atr_14 = true_range.ewm(span=14).mean().shift(1)
```

#### 3. Technical Indicators

**RSI (Relative Strength Index)**
```
delta = Close.diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
rs = avg_gain / avg_loss
rsi14 = 100 - (100 / (1 + rs))
```

**MACD**
```
ema_12 = Close.ewm(span=12).mean()
ema_26 = Close.ewm(span=26).mean()
macd = ema_12 - ema_26
macdsignal = macd.ewm(span=9).mean()
macdhist = macd - macdsignal
```

**Bollinger Bands**
```
ma_20 = Close.rolling(20).mean()
std_20 = Close.rolling(20).std()
bb_upper = ma_20 + 2 * std_20
bb_lower = ma_20 - 2 * std_20
bb_pctb = (Close - bb_lower) / (bb_upper - bb_lower)
```

**ADX (Average Directional Index)**
```
up_move = High.diff()
down_move = -Low.diff()
plus_dm = where((up_move > down_move) AND (up_move > 0), up_move, 0)
minus_dm = where((down_move > up_move) AND (down_move > 0), down_move, 0)
plus_di = 100 * ewm(plus_dm, 14) / atr_14
minus_di = 100 * ewm(minus_dm, 14) / atr_14
dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
adx_14 = dx.ewm(span=14).mean().shift(1)
```

**MFI (Money Flow Index)**
```
typical_price = (High + Low + Close) / 3
raw_mf = typical_price * Volume
positive_mf = where(typical_price > typical_price.shift(1), raw_mf, 0)
negative_mf = where(typical_price < typical_price.shift(1), raw_mf, 0)
mfr = positive_mf.rolling(14).sum() / negative_mf.rolling(14).sum()
mfi14 = 100 - (100 / (1 + mfr))
```

#### 4. Volume Features
```
vol_ma_20 = Volume.rolling(20).mean()
vol_spike_20 = Volume / vol_ma_20
volume_zscore = (Volume - vol_ma_20) / Volume.rolling(20).std()
dollar_volume = Close * Volume
dollar_volume_20d_avg = dollar_volume.rolling(20).mean()
```

#### 5. Price Patterns
```
daily_range = (High - Low) / Close
high_low_ratio = High / Low
close_position = (Close - Low) / (High - Low)  # Where in day's range
body_to_range = abs(Close - Open) / (High - Low)
upper_wick_to_range = (High - max(Close, Open)) / (High - Low)
lower_wick_to_range = (min(Close, Open) - Low) / (High - Low)
```

#### 6. GBM (Geometric Brownian Motion) Probabilities
```
# Estimate drift and volatility from 60-day history
gbm_mu_60d = ret_1d.rolling(60).mean() * 252  # Annualized
gbm_sig_60d = ret_1d.rolling(60).std() * sqrt(252)  # Annualized

# Log-normal distribution parameters
log_drift = (gbm_mu_60d - 0.5 * gbm_sig_60d^2) / 252
log_vol = gbm_sig_60d / sqrt(252)

# Probability of positive return (1-day)
z_threshold = -log_drift / log_vol
gbm_prob_up_1d = 1 - normal_cdf(z_threshold)

# Expected return
gbm_exp_ret_1d = exp(gbm_mu_60d / 252) - 1

# 5th and 95th percentile returns
gbm_p05_ret_1d = exp(log_drift + log_vol * normal_ppf(0.05)) - 1
gbm_p95_ret_1d = exp(log_drift + log_vol * normal_ppf(0.95)) - 1
```

#### 7. Relative Strength (vs SPX)
```
spx_ret_1d = SPX_Close.pct_change()
rel_strength_1d = ret_1d - spx_ret_1d
rel_momentum_5d = ret_5d - spx_ret_5d

# Rolling beta to SPX
cov_60 = ret_1d.rolling(60).cov(spx_ret_1d)
var_60 = spx_ret_1d.rolling(60).var()
beta_60_spx = (cov_60 / var_60).shift(1)

# Correlation
corr_20_spx = ret_1d.rolling(20).corr(spx_ret_1d).shift(1)
```

#### 8. Regime Detection
```
# Bull/Bear regime
regime_bull = 1 IF Close > sma_200 ELSE 0
regime_bear = 1 IF Close < sma_200 ELSE 0

# VIX regime
regime_vix_low = 1 IF vix < 15 ELSE 0
regime_vix_medium = 1 IF 15 <= vix < 25 ELSE 0
regime_vix_high = 1 IF vix >= 25 ELSE 0

# Streak counting
bull_streak = consecutive_days(Close > Close.shift(1))
bear_streak = consecutive_days(Close < Close.shift(1))
```

### Critical Rule: Look-Ahead Bias Prevention

**ALL features must be lagged by 1 day using `.shift(1)` to prevent look-ahead bias.**

```
# CORRECT - uses only past data
rsi14 = calculate_rsi(Close).shift(1)

# WRONG - uses current day's data in prediction
rsi14 = calculate_rsi(Close)  # DO NOT DO THIS
```

### Target Variable

```
# Forward return (what we're predicting)
# This gets shifted FORWARD, not backward
target = Close.pct_change(horizon).shift(-horizon)

# Example for 5-day horizon:
# On day T, target = (Close[T+5] - Close[T]) / Close[T]
```

---

## Short-Term Prediction Model

### Model Types

| Model | Code | Best For | Key Parameters |
|-------|------|----------|----------------|
| Random Forest | `rf` | Stability, interpretability | `n_estimators=300, max_depth=8, min_samples_leaf=50` |
| XGBoost | `xgb` | Performance (best Sharpe) | `n_estimators=300, learning_rate=0.05, max_depth=4` |
| Gradient Boosting | `gbrt` | Balance | `n_estimators=300, learning_rate=0.05, max_depth=4` |

### Training Pipeline

```
FUNCTION train_model(ticker, period, model_type, horizon):
    
    # 1. Get price history
    hist = get_price_history(ticker, period, "1d")
    
    # 2. Build features
    hist = add_price_features(hist)      # Returns, volatility, technicals
    hist = add_regime_features(hist)     # Bull/bear, VIX regime
    hist = add_macro_features(hist)      # FRED data
    hist = add_fundamental_features(hist, ticker)  # P/E, P/B
    
    # 3. Build target
    hist["target"] = hist["Close"].pct_change(horizon).shift(-horizon)
    
    # 4. Select features (only those with <50% NaN)
    feature_cols = get_available_features(hist)
    
    # 5. Drop NaN rows
    df = hist[feature_cols + ["target"]].dropna()
    
    # 6. Split chronologically (80/20)
    split_idx = int(len(df) * 0.8)
    X_train = df[feature_cols][:split_idx]
    y_train = df["target"][:split_idx]
    X_test = df[feature_cols][split_idx:]
    y_test = df["target"][split_idx:]
    
    # 7. Optional: Feature selection via Elastic Net
    IF use_elasticnet:
        selected_features = elasticnet_select(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    
    # 8. Train model
    model = create_model(model_type)
    model.fit(X_train, y_train)
    
    # 9. Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    direction_accuracy = mean(sign(y_pred) == sign(y_test))
    
    RETURN model, metrics
```

### Prediction Function

```
FUNCTION predict_next_for_ticker(ticker, period, model_type, horizon):
    
    # 1. Build features for full history
    X, y, last_row_features, last_close, vol_20d, dates = 
        build_features_and_target(ticker, period, horizon)
    
    # 2. Train model on all but last row
    model = create_model(model_type)
    model.fit(X, y)
    
    # 3. Predict using last row's features
    pred_ret = model.predict(last_row_features.reshape(1, -1))[0]
    
    # 4. Calculate predicted price
    pred_price = last_close * (1 + pred_ret)
    
    # 5. Get probability estimates (classification model)
    y_direction = (y > 0).astype(int)
    clf = create_classifier(model_type)
    clf.fit(X, y_direction)
    
    IF clf has predict_proba:
        proba = clf.predict_proba(last_row_features)[0]
        prob_up = proba[class_1_index]
        prob_down = 1 - prob_up
    ELSE:
        prob_up = 1.0 IF pred_ret > 0 ELSE 0.0
        prob_down = 1 - prob_up
    
    # 6. Confidence score (magnitude of prediction)
    confidence = abs(pred_ret)
    
    RETURN {
        "ticker": ticker,
        "pred_next_ret": pred_ret,
        "pred_next_price": pred_price,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "confidence_score": confidence,
        "last_close": last_close,
        "vol_20d": vol_20d,
        "model_type": model_type,
        "horizon": horizon
    }
```

### Feature Selection (Elastic Net)

```
FUNCTION elasticnet_select(X, y, l1_ratio=0.5, min_features=12):
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Elastic Net with cross-validation
    enet = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=logspace(-4, 1, 100),
        cv=5,
        max_iter=10000
    )
    enet.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    mask = abs(enet.coef_) > 1e-10
    
    # Ensure minimum number of features
    IF sum(mask) < min_features:
        top_indices = argsort(abs(enet.coef_))[::-1][:min_features]
        mask = zeros_like(mask, dtype=bool)
        mask[top_indices] = True
    
    selected_features = feature_names[mask]
    RETURN selected_features
```

---

## Long-Horizon Prediction Model

### Overview

The long-horizon model predicts 20-30 day direction using historical analog matching. It finds similar market regimes in history and uses their forward returns to estimate probabilities.

### Feature Engineering (Slow/Structural Features)

```
FUNCTION build_long_features(hist):
    
    n = len(hist)
    Close = hist["Close"]
    Volume = hist["Volume"]
    
    # Momentum (adaptive windows for short histories)
    mom_20_window = min(20, max(5, n // 4))
    mom_50_window = min(50, max(5, n // 3))
    mom_100_window = min(100, max(5, n // 2))
    
    mom_20d = Close.pct_change(mom_20_window)
    mom_50d = Close.pct_change(mom_50_window)
    mom_100d = Close.pct_change(mom_100_window)
    
    # Trend slope (linear regression)
    slope_50d, slope_r2_50d = trend_slope_r2(Close, 50)
    slope_100d, slope_r2_100d = trend_slope_r2(Close, 100)
    
    # Distance from moving averages
    ma50 = Close.rolling(50).mean()
    ma100 = Close.rolling(100).mean()
    dist_ma50 = Close / ma50 - 1
    dist_ma100 = Close / ma100 - 1
    
    # Volatility regime
    ret_1d = Close.pct_change()
    hv_20 = ret_1d.rolling(20).std()
    hv_60 = ret_1d.rolling(60).std()
    vol_of_vol = hv_20.rolling(20).std()
    hv_20_pct_1y = hv_20.rolling(252).rank(pct=True)
    
    # Volume trends
    vol_ma20 = Volume.rolling(20).mean()
    vol_ma60 = Volume.rolling(60).mean()
    vol_trend_20_60 = vol_ma20 / vol_ma60
    
    # Z-score of 20-day return
    ret_20d = Close.pct_change(20)
    z_ret_20d = (ret_20d - ret_20d.rolling(252).mean()) / ret_20d.rolling(252).std()
    
    # Forward labels (for training analog weights)
    fwd_ret_30d = Close.pct_change(30).shift(-30)
    hv_20_fwd = hv_20.shift(-20)
    vol_expanded = (hv_20_fwd > hv_20).astype(float)
    
    RETURN DataFrame with all features
```

### Regime Tagging

```
FUNCTION tag_regime(df):
    
    # Trend state
    trend_state = "up" IF slope_50d > 0 ELSE "down"
    
    # Volatility state (tertiles)
    TRY:
        vol_state = qcut(hv_20, q=3, labels=["low", "med", "high"])
    CATCH (not enough unique values):
        hv_med = median(hv_20)
        hv_q75 = quantile(hv_20, 0.75)
        vol_state = "low" IF hv_20 <= hv_med
                    ELSE "med" IF hv_20 <= hv_q75
                    ELSE "high"
    
    regime = trend_state + "_" + vol_state
    # Examples: "up_low", "up_med", "up_high", "down_low", etc.
    
    RETURN regime
```

### Analog Inference

```
FUNCTION analog_infer(df, as_of_date, k=200, decay_half_life=252):
    
    # Get historical data up to (but not including) as_of_date
    hist = df.loc[:as_of_date].iloc[:-1]
    
    IF len(hist) < 30:
        RETURN None
    
    # Feature columns for similarity
    feat_cols = [
        "mom_20d", "mom_50d", "mom_100d",
        "slope_50d", "slope_r2_50d", "slope_100d", "slope_r2_100d",
        "dist_ma50", "dist_ma100",
        "hv_20", "hv_60", "vol_of_vol", "hv_20_pct_1y",
        "vol_trend_20_60", "dollar_vol_20_trend", "z_ret_20d"
    ]
    
    # Standardize features
    hist_feat = hist[feat_cols].fillna(0)
    means = hist_feat.mean()
    stds = hist_feat.std().replace(0, 1)
    hist_z = (hist_feat - means) / stds
    
    current = df.loc[as_of_date, feat_cols].fillna(0)
    current_z = (current - means) / stds
    
    # Filter to same regime
    current_regime = tag_regime(df.loc[[as_of_date]])[0]
    same_regime_mask = tag_regime(hist) == current_regime
    candidates = hist_z[same_regime_mask]
    
    IF candidates is empty:
        candidates = hist_z
    
    # Cosine similarity
    curr_vec = current_z.values
    cand_mat = candidates.values
    cos_sim = (cand_mat @ curr_vec) / (norm(cand_mat) * norm(curr_vec))
    distance = 1 - cos_sim
    
    # Time decay weighting
    ages_days = (as_of_date - candidates.index).days
    lambda = log(2) / decay_half_life
    decay = exp(-lambda * ages_days)
    
    # Combined weights
    weights = (1 / (distance + 1e-6)) * decay
    
    # Select top k analogs
    top_k_idx = argsort(-weights)[:k]
    top_weights = weights[top_k_idx]
    top_dates = candidates.index[top_k_idx]
    
    # Get forward returns for analogs
    fwd_returns = df.loc[top_dates, "fwd_ret_30d"].fillna(0)
    vol_expanded = df.loc[top_dates, "vol_expanded"].fillna(0)
    
    # Weighted statistics
    p_up = weighted_average((fwd_returns > 0).astype(float), top_weights)
    
    p10, p50, p90 = weighted_quantiles(fwd_returns, top_weights, [0.1, 0.5, 0.9])
    
    vol_exp_prob = weighted_average(vol_expanded, top_weights)
    
    # Effective sample size
    ess = sum(top_weights)^2 / sum(top_weights^2)
    
    # Flags
    flags = {
        "low_confidence": ess < 20,
        "high_drawdown_risk": p10 < -0.05,
        "regime_shift_risk": vol_exp_prob > 0.6
    }
    
    RETURN {
        "p_up_30d": p_up,
        "ret_p10_30d": p10,
        "ret_p50_30d": p50,
        "ret_p90_30d": p90,
        "vol_expansion_prob": vol_exp_prob,
        "flags": flags,
        "effective_sample_size": int(ess),
        "analog_count": min(k, len(candidates))
    }
```

---

## Options Pricing

### Black-Scholes Model

```
FUNCTION black_scholes_price(flag, S, K, T, r, sigma):
    """
    flag: 'c' for call, 'p' for put
    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate (e.g., 0.04)
    sigma: implied volatility (e.g., 0.35)
    """
    
    IF T <= 0 OR sigma <= 0 OR S <= 0 OR K <= 0:
        RETURN None
    
    d1 = (log(S/K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    IF flag == 'c':
        price = S * N(d1) - K * exp(-r * T) * N(d2)
    ELSE:
        price = K * exp(-r * T) * N(-d2) - S * N(-d1)
    
    RETURN price


FUNCTION black_scholes_greeks(flag, S, K, T, r, sigma):
    
    d1 = (log(S/K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    # Delta
    IF flag == 'c':
        delta = N(d1)
    ELSE:
        delta = N(d1) - 1
    
    # Gamma (same for call and put)
    gamma = n(d1) / (S * sigma * sqrt(T))
    
    # Vega (same for call and put)
    vega = S * n(d1) * sqrt(T)
    
    # Theta
    IF flag == 'c':
        theta = -S * n(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r*T) * N(d2)
    ELSE:
        theta = -S * n(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r*T) * N(-d2)
    
    RETURN {delta, gamma, vega, theta}
```

### Strategy Suggestion Logic

```
FUNCTION suggest_options_strategy(pred_ret, put_call_ratio, atm_iv, horizon):
    
    # Horizon multiplier (longer horizon = higher threshold)
    threshold = 0.005 * horizon  # 0.5% per day
    
    # Strong directional signal
    IF abs(pred_ret) > threshold:
        IF pred_ret > 0:
            IF put_call_ratio > 1.2:
                RETURN "BULLISH: Buy Calls (high put OI suggests squeeze)"
            RETURN "BULLISH: Bull Call Spread"
        ELSE:
            IF put_call_ratio < 0.8:
                RETURN "BEARISH: Buy Puts (low protection)"
            RETURN "BEARISH: Bear Put Spread"
    
    # High IV + neutral = sell premium
    IF abs(pred_ret) < 0.5 * threshold AND atm_iv > 0.35:
        RETURN "NEUTRAL: Iron Condor (high IV, harvest premium)"
    
    RETURN "NEUTRAL: No trade"
```

---

## Signal Generation

### Signal JSON Format

```json
{
  "AAPL": {
    "asset": "option",
    "strategy": "BULL_CALL_SPREAD",
    "dte_min": 3,
    "dte_max": 45,
    "max_strike": 200,
    "max_premium": 500,
    "width_pct": 0.05,
    "qty": 2,
    "pred_next_ret": 0.0234,
    "last_close": 193.50,
    "execution": {
      "delay_days": 1,
      "half_spread_bps": 2.0,
      "slippage_bps": 3.0,
      "fee_bps": 0.0
    }
  },
  "MSFT": {
    "asset": "stock",
    "action": "BUY",
    "qty": 10,
    "pred_next_ret": 0.0156,
    "execution": {
      "delay_days": 1,
      "half_spread_bps": 2.0,
      "slippage_bps": 3.0,
      "fee_bps": 0.0
    }
  }
}
```

### Signal Generation Logic

```
FUNCTION build_signals_from_predictions(pred_df, trade_mode, options_params):
    
    signals = {}
    
    FOR each row IN pred_df:
        ticker = row["ticker"]
        pred_ret = row["pred_next_ret"]
        confidence = row["confidence_score"]
        
        # Skip weak signals
        IF abs(pred_ret) < 0.0025:
            CONTINUE
        
        # Confidence-based quantity (1-5 based on confidence percentile)
        qty = min(5, max(1, int(confidence * 100)))
        
        # Get options strategy suggestion
        strategy, bias = suggest_options_strategy(
            pred_ret,
            row.get("put_call_oi_ratio"),
            row.get("atm_iv"),
            horizon
        )
        
        # Determine stock action
        IF pred_ret > 0:
            stock_action = "BUY"
        ELSE IF pred_ret < 0:
            stock_action = "SHORT"
        ELSE:
            stock_action = "HOLD"
        
        # Build signal based on trade mode
        IF trade_mode == "Options only" OR (trade_mode == "Options if suggested" AND strategy):
            signals[ticker] = {
                "asset": "option",
                "strategy": normalize_strategy(strategy),
                "dte_min": options_params["dte_min"],
                "dte_max": options_params["dte_max"],
                "max_strike": options_params["max_strike"],
                "max_premium": options_params["max_premium"],
                "width_pct": options_params["width_pct"],
                "qty": qty,
                "pred_next_ret": pred_ret,
                "last_close": row["last_close"]
            }
        ELSE:
            signals[ticker] = {
                "asset": "stock",
                "action": stock_action,
                "qty": qty,
                "pred_next_ret": pred_ret
            }
    
    RETURN signals
```

---

## Trading Execution

### Alpaca Integration

```
FUNCTION execute_signals(signals_file):
    
    # Initialize Alpaca client
    api = alpaca.REST(
        key_id=APCA_API_KEY_ID,
        secret_key=APCA_API_SECRET_KEY,
        base_url="https://paper-api.alpaca.markets"  # Paper trading
    )
    
    signals = load_json(signals_file)
    
    FOR ticker, signal IN signals:
        
        # Skip non-US symbols
        IF "." IN ticker:
            log(f"Skipping non-US symbol: {ticker}")
            CONTINUE
        
        IF signal["asset"] == "stock":
            execute_stock_order(api, ticker, signal)
        ELSE:
            execute_option_order(api, ticker, signal)


FUNCTION execute_stock_order(api, ticker, signal):
    
    action = signal["action"]
    qty = signal["qty"]
    
    IF action == "BUY":
        side = "buy"
    ELSE IF action == "SHORT":
        side = "sell"
    ELSE:
        RETURN  # HOLD = no action
    
    order = api.submit_order(
        symbol=ticker,
        qty=qty,
        side=side,
        type="market",
        time_in_force="day"
    )
    
    log(f"Submitted {side} order for {qty} shares of {ticker}")


FUNCTION execute_option_order(api, ticker, signal):
    
    strategy = signal["strategy"]
    dte_min = signal["dte_min"]
    dte_max = signal["dte_max"]
    max_premium = signal["max_premium"]
    
    # Get option chain
    contracts = api.get_option_contracts(
        underlying_symbol=ticker,
        expiration_date_gte=today() + dte_min days,
        expiration_date_lte=today() + dte_max days
    )
    
    # Filter and select appropriate contract
    IF strategy == "BUY_CALL":
        contract = select_atm_call(contracts, signal["last_close"], max_premium)
    ELSE IF strategy == "BUY_PUT":
        contract = select_atm_put(contracts, signal["last_close"], max_premium)
    ELSE IF strategy == "BULL_CALL_SPREAD":
        contracts = select_bull_call_spread(contracts, signal)
    # ... etc
    
    # Submit order
    order = api.submit_order(
        symbol=contract.symbol,
        qty=signal["qty"],
        side="buy",
        type="limit",
        limit_price=contract.ask,
        time_in_force="day"
    )
```

---

## Web Dashboard

### Streamlit App Structure

```
FUNCTION main():
    
    # Initialize session state
    session_state.pred_df = None
    session_state.signals = None
    
    # Sidebar controls
    tickers = sidebar.text_input("Tickers", "AAPL, NVDA")
    period = sidebar.selectbox("Period", ["2y", "5y", "10y"])
    horizon = sidebar.selectbox("Horizon", [1, 2, 3, 4, 5])
    model_type = sidebar.selectbox("Model", ["rf", "gbrt", "xgb"])
    
    # Main tabs
    tab_dashboard, tab_backtests, tab_portfolio = tabs(["Dashboard", "Backtests", "Portfolio"])
    
    WITH tab_dashboard:
        
        IF button("Run Screener + Model"):
            
            # Screen stocks
            screener_df = screen_stocks(tickers)
            
            # Run predictions
            results = []
            FOR ticker IN screener_df["ticker"]:
                pred = predict_next_for_ticker(ticker, period, model_type, horizon)
                results.append(pred)
            
            pred_df = DataFrame(results)
            session_state.pred_df = pred_df
            
            # Generate signals
            signals = build_signals_from_predictions(pred_df)
            write_json(signals, "signals.json")
            session_state.signals = signals
        
        # Display results
        IF session_state.pred_df is not None:
            display_predictions_table(session_state.pred_df)
            
            # Ticker detail view
            selected_ticker = selectbox("Select ticker", tickers)
            display_ticker_detail(selected_ticker, session_state.pred_df)
            display_long_horizon_panel(selected_ticker)
            display_price_chart(selected_ticker)
            display_risk_metrics(selected_ticker)
```

### Ticker Detail Panel

```
FUNCTION display_ticker_detail(ticker, pred_df):
    
    row = pred_df[pred_df["ticker"] == ticker].iloc[0]
    
    # Short-term prediction metrics
    expander("📊 Model Prediction"):
        columns(3):
            metric("Predicted Return", format_pct(row["pred_next_ret"]))
            metric("Prob Up", format_pct(row["prob_up"]))
            metric("Features Used", row["num_features"])
        
        columns(4):
            metric("Last Close", format_usd(row["last_close"]))
            metric("Pred Price", format_usd(row["pred_next_price"]))
            metric("Vol 20D", format_pct(row["vol_20d"]))
            metric("Signal", row["signal_alignment"])
    
    # Long-horizon prediction
    expander("📅 30-Day Outlook"):
        lh = predict_long_horizon_for_ticker(ticker)
        
        IF "error" IN lh:
            warning(lh["error"])
        ELSE:
            columns(4):
                metric("30d Up Prob", format_pct(lh["p_up_30d"]))
                metric("Return P50", format_pct(lh["ret_p50_30d"]))
                metric("Return P10", format_pct(lh["ret_p10_30d"]))
                metric("Return P90", format_pct(lh["ret_p90_30d"]))
            
            metric("Vol Expansion Risk", format_pct(lh["vol_expansion_prob"]))
            metric("Sample Size", lh["effective_sample_size"])
            
            IF lh["flags"]["low_confidence"]:
                warning("⚠️ Low confidence")
    
    # Options panel
    expander("📊 Options & Risk"):
        strategy, bias = suggest_options_strategy(row)
        
        columns(4):
            metric("ATM IV", row["atm_iv"])
            metric("Put/Call OI", row["put_call_oi_ratio"])
            metric("IV vs Realized", row["iv_minus_realized"])
            metric("Theo ATM Call", row["theo_atm_call_price"])
        
        info(f"Strategy: {strategy}")
```

---

## API Endpoints

For mobile app integration, expose these REST endpoints:

### GET /predict/{ticker}

```json
Request: GET /predict/AAPL?horizon=5&model=rf

Response:
{
  "short_term": {
    "ticker": "AAPL",
    "pred_next_ret": 0.0234,
    "pred_next_price": 197.53,
    "prob_up": 0.68,
    "prob_down": 0.32,
    "confidence_score": 0.0234,
    "last_close": 193.00,
    "vol_20d": 0.18
  },
  "long_term": {
    "p_up_30d": 0.62,
    "ret_p50_30d": 0.035,
    "ret_p10_30d": -0.05,
    "ret_p90_30d": 0.12,
    "vol_expansion_prob": 0.28,
    "effective_sample_size": 45
  }
}
```

### POST /analyze

```json
Request: POST /analyze
{
  "tickers": ["AAPL", "NVDA", "MSFT"],
  "horizon": 5,
  "model": "xgb"
}

Response:
{
  "predictions": [
    {"ticker": "NVDA", "pred_next_ret": 0.045, "prob_up": 0.72, ...},
    {"ticker": "AAPL", "pred_next_ret": 0.023, "prob_up": 0.68, ...},
    {"ticker": "MSFT", "pred_next_ret": 0.018, "prob_up": 0.61, ...}
  ],
  "top_pick": "NVDA",
  "generated_at": "2026-01-04T12:00:00Z"
}
```

### GET /signals

```json
Request: GET /signals

Response:
{
  "AAPL": {"asset": "option", "strategy": "BULL_CALL_SPREAD", "qty": 2, ...},
  "NVDA": {"asset": "stock", "action": "BUY", "qty": 5, ...}
}
```

---

## Configuration

### Environment Variables

```bash
# Trading API (Alpaca)
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret

# Data APIs
FRED_API_KEY=your_fred_key
FMP_API_KEY=your_fmp_key
MARKETAUX_API_KEY=your_marketaux_key
ALPHAVANTAGE_API_KEY=your_alphavantage_key

# System
TRADING_DAYS=252
```

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 300 | Number of trees |
| `max_depth` | 8 (RF), 4 (XGB) | Tree depth |
| `min_samples_leaf` | 50 | Minimum samples per leaf |
| `learning_rate` | 0.05 | XGB/GBRT learning rate |
| `subsample` | 0.8 | Row sampling rate |

### Execution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delay_days` | 1 | Days between signal and execution |
| `half_spread_bps` | 2.0 | Half bid-ask spread in basis points |
| `slippage_bps` | 3.0 | Slippage in basis points |
| `fee_bps` | 0.0 | Trading fees in basis points |

---

## Complete Algorithm Pseudocode

### Main Prediction Flow

```
FUNCTION run_full_prediction_pipeline(tickers, period, model_type, horizon):
    
    results = []
    
    FOR ticker IN tickers:
        
        # ===== DATA FETCHING =====
        hist = get_price_history(ticker, period, "1d")
        macro = get_macro_df(period)
        fundamentals = get_fmp_fundamentals(ticker)
        sentiment = get_news_sentiment(ticker)
        
        # ===== FEATURE ENGINEERING =====
        # Add all features with .shift(1) for look-ahead prevention
        hist = add_returns(hist)           # ret_1d, ret_5d, etc.
        hist = add_volatility(hist)        # vol_10d, vol_20d, atr_14
        hist = add_technicals(hist)        # rsi14, macd, mfi14, adx_14
        hist = add_gbm_features(hist)      # gbm_prob_up_1d, etc.
        hist = add_relative_strength(hist) # rel_strength_1d, beta_60_spx
        hist = add_regime_flags(hist)      # regime_bull, regime_bear
        
        # Join external data
        hist = hist.join(macro, how="left")
        hist["fund_pe"] = fundamentals["pe_trailing"]
        hist["fund_pb"] = fundamentals["pb"]
        hist["news_sentiment"] = sentiment["score"]
        
        # ===== TARGET =====
        hist["target"] = hist["Close"].pct_change(horizon).shift(-horizon)
        
        # ===== CLEAN DATA =====
        feature_cols = get_valid_features(hist, max_nan_pct=0.5)
        df = hist[feature_cols + ["target"]].dropna()
        
        IF len(df) < 60:
            log(f"Skipping {ticker}: insufficient data")
            CONTINUE
        
        # ===== FEATURE SELECTION =====
        IF use_elasticnet:
            selected = elasticnet_select(df[feature_cols], df["target"])
            feature_cols = selected
        
        # ===== TRAIN MODEL =====
        X = df[feature_cols].values
        y = df["target"].values
        
        model = create_model(model_type)
        model.fit(X, y)
        
        # ===== PREDICT =====
        last_features = df[feature_cols].iloc[-1].values
        pred_ret = model.predict(last_features.reshape(1, -1))[0]
        
        # ===== CLASSIFICATION (for probabilities) =====
        y_dir = (y > 0).astype(int)
        clf = create_classifier(model_type)
        clf.fit(X, y_dir)
        prob_up = clf.predict_proba(last_features.reshape(1, -1))[0][1]
        
        # ===== LONG HORIZON =====
        long_horizon = predict_long_horizon(hist)
        
        # ===== OPTIONS =====
        options = get_option_snapshot_features(ticker)
        strategy = suggest_options_strategy(pred_ret, options)
        
        # ===== COMPILE RESULT =====
        results.append({
            "ticker": ticker,
            "pred_next_ret": pred_ret,
            "pred_next_price": hist["Close"].iloc[-1] * (1 + pred_ret),
            "prob_up": prob_up,
            "confidence": abs(pred_ret),
            "long_horizon": long_horizon,
            "strategy": strategy,
            "options": options
        })
    
    # ===== GENERATE SIGNALS =====
    signals = build_signals(results)
    write_json(signals, "signals.json")
    
    RETURN results, signals
```

---

## Dependencies

```
# Core
python >= 3.10
pandas >= 2.0.0
numpy >= 1.24.0
scipy >= 1.11.0

# ML
scikit-learn >= 1.3.0
xgboost >= 2.0.0
statsmodels >= 0.14.0

# Data
yfinance >= 0.2.30
requests >= 2.31.0
python-dotenv >= 1.0.0

# Web
streamlit >= 1.28.0
plotly >= 5.18.0

# Trading
alpaca-trade-api >= 3.0.0

# Optional
ta-lib  # Technical analysis (C library required)
tensorflow >= 2.15.0  # For GAF-CNN
```

---

*Document Version: 1.0*  
*Generated: January 2026*  
*For use with app builders and code generation tools*
