# Stock Predictor AI Coding Instructions

## Architecture Overview

This is a **stock prediction system** with three integrated layers:

1. **Core Prediction Engine** (`prediction_model.py`): ML models (Random Forest, XGBoost, Gradient Boosting) predicting next-day/multi-day returns with extensive feature engineering
2. **Data Pipeline** (`data_fetch.py`, `stock_screener.py`): Historical data (yfinance), fundamentals (FMP API), macro data (FRED API), news sentiment (Marketaux)
3. **UI + Trading Integration** (`app.py` = Streamlit): Interactive dashboard + automated trading via Alpaca (`auto_paper_trade.py`)

**Key data flows:**
- `app.py` → `predict_next_for_ticker()` → returns predictions + option pricing → signals JSON → `auto_paper_trade.py` executes trades
- SPX is cached globally (`_SPX_CACHE`) to compute relative strength/beta across all tickers without repeated downloads

## Critical Components & Patterns

### Feature Engineering (Massive Attack Surface)
- **FEATURE_COLUMNS** (70+ features in `prediction_model.py` ~line 583): Technical (RSI, MACD, Bollinger Bands), volume, momentum, GBM-derived probabilities, SPX relative strength
- **Macro columns** (FRED API): T10Y, VIX, term spread
- **Fundamental columns**: P/E, P/B, market cap
- **GBM features**: `gbm_prob_up_1d`, `gbm_exp_ret_1d`, percentiles — built from log-return distribution (lines 616-629)
- **Time-series handling**: All features are **lagged by 1 day** via `.shift(1)` to prevent look-ahead bias (critical!)
- **Nulls**: Heavy `.fillna()` / `.ffill()` / `.bfill()` usage — when modifying features, preserve this pattern

### Model Training Architecture
- **make_model()** (~line 1170): Unified factory for XGBoost/RandomForest/GradientBoosting (regressor or classifier)
- **train_model()** (~line 1594): Standard 80/20 train/test split
- **Feature selection** (optional via env vars): 
  - `USE_ELASTICNET_SELECT=1` + `ELASTICNET_L1_RATIO/CV_FOLDS`: Elastic Net p-value pruning (lines 1062–1134)
  - `USE_OLSSIGSELECT=1` + `OLSSIG_ALPHA/TOPK/MINFEATURES`: OLS significance filter (lines 1135–1160)
- **Walk-forward backtesting**: `walk_forward_backtest()` (line 2010) splits by DATE, not row index, to prevent data leakage

### Prediction Entrypoint
**`predict_next_for_ticker(tk, period="5y", model_type="rf", horizon=1, ...)`** (~line 1359):
- Builds features + target from raw history
- Optionally auto-selects best `model_type` via grid search
- Returns dict: `pred_next_ret`, `pred_next_price`, `prob_up`, `prob_down`, `prob_up_gaf` (CNN-based), plus all metadata
- **GAF-CNN layer**: Gramian Angular Field images → Conv2D model (`gaf_cnn_updown.keras`) for up/down classification
- **Option pricing overlay**: Theo ATM call price (Black-Scholes or Heston), Monte Carlo expected value, IV vs realized vol

### Data Dependencies
- **Cache layers** (avoid re-fetching):
  - `_SPX_CACHE` (global module dict, line ~8)
  - `get_history_cached()` → streamlit cache (30s TTL)
  - Intraday data: separate 1-minute cache (30s), daily cache (10m)
- **API keys** (env / streamlit secrets): `FMP_API_KEY`, `FRED_API_KEY`, `MARKETAUX_API_KEY`, `ALPHAVANTAGE_API_KEY`
- **Fallback data sources**: yfinance → Stooq CSV → raw Yahoo (handles rate limits gracefully)

### Trading Signals System
- **build_signals_from_pred_df()** (~line 465): Converts model predictions → JSON signals (stock/option strategies)
- **suggest_options_strategy()** (~line 384): Heuristic rules (return threshold × horizon multiplier, IV rank, put/call OI ratio)
- **Execution costs** (`ExecutionModel` dataclass): Delay (1d default), spread (2bps), slippage (3bps), fees
- **Alpaca integration** (`auto_paper_trade.py`): Market/limit orders, option contract filtering (DTE min/max, strike, premium caps), bid/ask handling

## Developer Workflows

### Running Predictions (CLI)
```bash
# Predict next return for AAPL (5y history, random forest, 1-day horizon)
python -c "from prediction_model import predict_next_for_ticker; import json; print(json.dumps(predict_next_for_ticker('AAPL'), default=str))"

# Backtest single ticker
python -c "from prediction_model import backtest_one_ticker; print(backtest_one_ticker('AAPL', period='10y', model_type='xgb'))"

# Walk-forward backtest (cross-sectional)
python -c "from prediction_model import walkforward_cross_sectional; print(walkforward_cross_sectional(['AAPL','MSFT','NVDA'], model_type='rf', train_years=1))"
```

### Running UI
```bash
streamlit run app.py
```

### Running Auto-Trader (Scheduled)
```bash
python runner.py  # Runs trades every Monday 08:35 ET via schedule.schedule
# or direct:
python auto_paper_trade.py  # One execution
```

### Testing & Diagnostics
- **Check feature availability**: `build_features_and_target('AAPL')` will raise if data incomplete
- **Verify GBM features**: Check for `gbm_prob_up_1d`, `gbm_exp_ret_1d` in output DataFrame
- **Debug feature selection**: Set `USE_ELASTICNET_SELECT=1` in shell, run any model → console prints selected features
- **Macro data**: Set `FRED_API_KEY` env var or macro pipeline silently degrades to just `mkt_ret_1d`

## Project-Specific Conventions

### Environment Variables (in `prediction_model.py`)
- `USE_ELASTICNET_SELECT`, `ELASTICNET_L1_RATIO`, `ELASTICNET_CV_FOLDS`
- `USE_OLSSIGSELECT`, `OLSSIG_ALPHA`, `OLSSIG_TOPK`, `OLSSIG_MINFEATURES`
- `FRED_API_KEY`, `TRADING_DAYS=252`
- Accessed via `env_bool()`, `float(os.environ.get(..., default))`

### Column Naming Conventions
- **Returns**: `ret_Xd` (1d, 5d, 20d), `cumret_Xd`
- **Volatility**: `vol_Xd` (10d, 20d, 60d)
- **Technical**: `rsi14`, `macd`, `macdsignal`, `macdhist`, `mfi14`
- **GBM**: `gbm_*` (mu, sig, prob_up, exp_ret, p05/p95)
- **Target**: Always `ftarget_ret_horizon_ahead` in training DataFrames
- **Relative strength**: `rel_strength_1d`, `rel_momentum_5d` (vs SPX)
- **Fundamentals**: `fund_*` (pe_trailing, pb, marketcap)
- **Macro**: `mkt_ret_1d`, `vix`, `t10y`, `term_spread`

### Model Type Constants
- `"rf"` = RandomForestRegressor (default)
- `"xgb"` = XGBRegressor
- `"gbrt"` = GradientBoostingRegressor
- `"linreg"` = LinearRegression (rarely used)
- For classification: append `clf` task (e.g., RandomForestClassifier)

### Data Quality Guardrails
- **Minimum rows**: 60 after NaN drop (lines ~1308–1310)
- **Fallback periods**: ['5y', '3y', '2y', '1y', '6mo', '3mo'] (auto-retry if current period insufficient)
- **Shift operations**: All features shifted 1 day back; target shifted forward `horizon` days
- **NaN handling**: Strict `.dropna()` after feature build; macro data uses `.ffill().bfill()` for forward-fill

### Known Gotchas
1. **SPX index mismatch**: `_get_spx()` normalizes timezone and uses `.reindex(..., method="ffill")` → ensure all dates align
2. **Walk-forward by dates**: `walkforward_cross_sectional()` uses unique dates, NOT row indices, to prevent leakage
3. **GAF-CNN requires 30-day window**: `predict_up_gafcnn_from_rets()` needs at least 30 return values
4. **Heston pricing**: Only 5 tickers hardcoded (AAPL, NVDA) → falls back to Black-Scholes
5. **Streamlit caching**: `@st.cache_data(ttl=...)` invalidates with new arguments → pass immutable types (tuples, strings)

## Integration Points

### Adding a New Data Source
1. Add fetch function in `data_fetch.py` (e.g., `get_crypto_prices()`)
2. Join to historical DataFrame in `build_features_and_target()` → `hist.join(new_source, how="left")`
3. Add column to `FEATURE_COLUMNS` or `MACRO_COLUMNS`
4. Handle NaN fill (`.ffill().bfill()` or scalar default)

### Adding a Feature
1. Implement in `add_price_features()` or new helper
2. **Shift by 1**: `df["new_feature"] = calculation.shift(1)`
3. Add to `FEATURE_COLUMNS` list
4. Test: `build_features_and_target('TEST_TICKER')` should include your column

### Modifying Prediction Output
- Return dict keys in `predict_next_for_ticker()` (lines 1359–1550) flow directly to `app.py` cache + signals
- Ensure JSON-serializable (use `float()`, `str()`, avoid numpy types)
- Update `_build_display_df()` rename_map if new columns should appear in UI

### Extending Options Strategies
- Edit `suggest_options_strategy()` logic (lines 384–410)
- Add new strategy names to `normalize_model_option_strategy()` (lines 412–422)
- Mapping flows into `build_signals_from_pred_df()` → Alpaca execution

---

**Last Updated**: 2025-12-27 | **Scope**: Stock prediction, backtesting, options pricing, automated paper trading
