# Stock Predictor AI Coding Instructions

## Architecture Overview

This is a **stock prediction system** with three integrated layers:

1. **Core Prediction Engine** (`prediction_model.py`): ML models (Random Forest, XGBoost, Gradient Boosting) predicting next-day/multi-day returns with extensive feature engineering (100+ features)
2. **Data Pipeline** (`data_fetch.py`, `stock_screener.py`): Historical data (yfinance), fundamentals (FMP API), macro data (FRED API), news sentiment (Marketaux)
3. **UI + Trading Integration** (`app.py` = Streamlit): Interactive dashboard + automated trading via Alpaca (`auto_paper_trade.py`)
4. **Experiment Orchestration** (`experiment_runner.py`, `grid_search.py`): Reproducible walk-forward backtesting across model types, hyperparameters, and feature configurations

**Key data flows:**
- `app.py` → `predict_next_for_ticker()` → returns predictions + option pricing → signals JSON → `auto_paper_trade.py` executes trades
- `experiment_runner.py` → `walk_forward_backtest()` → evaluates model performance with proper date-based train/test splits
- SPX is cached globally (`_SPX_CACHE`) to compute relative strength/beta across all tickers without repeated downloads

## Critical Components & Patterns

### Feature Engineering (100+ features in `FEATURE_COLUMNS` list)
- **Technical indicators** (RSI, MACD, Bollinger Bands, ATR, ADX): Lines ~440-650 in `prediction_model.py`
- **GBM-derived probabilities** (`gbm_prob_up_1d`, `gbm_exp_ret_1d`, percentiles): Built from log-return distribution via geometric Brownian motion
- **Macro data** (FRED API): T10Y, VIX, term spread, unemployment, CPI, OAS, Fed Funds Rate
- **Relative strength vs SPX**: `rel_strength_1d`, `rel_momentum_5d` (computed from SPX cache)
- **Fundamentals**: P/E trailing, P/B, market cap (from FMP API)
- **Enhanced features** (`model_improvements.py`): Volatility-adjusted returns, regime detection, momentum confirmation, mean reversion signals
- **Critical Rule**: All features are **lagged by 1 day** via `.shift(1)` to prevent look-ahead bias. This is enforced everywhere.
- **NaN handling**: Heavy `.fillna()` / `.ffill()` / `.bfill()` usage with fallback to 0. When modifying features, preserve this pattern.

### Model Training Architecture
- **make_model()** (~line 1170): Unified factory for XGBoost/RandomForest/GradientBoosting (regressor or classifier)
- **train_model()** (~line 1594): Standard 80/20 train/test split with option for cross-validation
- **Feature selection** (optional via env vars):
  - `USE_ELASTICNET_SELECT=1` + `ELASTICNET_L1_RATIO/CV_FOLDS`: Elastic Net p-value pruning (lines 1062–1134)
  - `USE_OLSSIGSELECT=1` + `OLSSIG_ALPHA/TOPK/MINFEATURES`: OLS significance filter (lines 1135–1160)
- **Walk-forward backtesting**: `walk_forward_backtest()` splits by DATE, not row index, to prevent data leakage
  - Key: Uses unique date boundaries, retrains model on each fold, evaluates on test fold
  - Returns: Dictionary with metrics (Sharpe, accuracy, drawdown, win rate, etc.)

### Prediction Entrypoint
**`predict_next_for_ticker(tk, period="5y", model_type="rf", horizon=1, ...)`** (~line 1359):
- Builds features + target from raw history via `build_features_and_target()`
- Optionally auto-selects best `model_type` via grid search (`grid_search.py`)
- Returns dict with: `pred_next_ret`, `pred_next_price`, `prob_up`, `prob_down`, `prob_up_gaf`, plus confidence score and all metadata
- **GAF-CNN layer** (optional): Gramian Angular Field images → Conv2D model (`gaf_cnn_updown.keras`) for up/down classification
- **Option pricing overlay**: Theo ATM call price (Black-Scholes or Heston), Monte Carlo expected value, IV vs realized vol
- **Robustness**: Gracefully handles missing data, falls back to shorter periods if insufficient history

### Data Dependencies & Fallbacks
- **Cache layers** (avoid re-fetching):
  - `_SPX_CACHE` (global module dict): tz-aware index matching caller's timezone
  - `get_history_cached()` → streamlit cache (30s TTL for intraday, 10m for daily)
  - `macro_cache` (module-level dict): FRED data cached per symbol/period
- **API keys** (env / streamlit secrets): `FMP_API_KEY`, `FRED_API_KEY`, `MARKETAUX_API_KEY`, `ALPHAVANTAGE_API_KEY`
- **Fallback chain**: yfinance (primary) → Stooq CSV → raw Yahoo download → error with clear message
- **Graceful degradation**: Missing macro data → uses just `mkt_ret_1d`; missing fundamentals → sets to 0; missing sentiment → skipped

### Trading Signals System
- **build_signals_from_pred_df()** (~line 465): Converts model predictions → JSON signals (stock/option strategies)
  - US-only filtering: Non-US stocks filtered out before Alpaca submission (see `auto_paper_trade.py`)
- **suggest_options_strategy()** (~line 384): Heuristic rules (return threshold × horizon multiplier, IV rank, put/call OI ratio)
- **Execution costs** (`ExecutionModel` dataclass): Delay (1d default), spread (2bps), slippage (3bps), fees
- **Alpaca integration** (`auto_paper_trade.py`): Market/limit orders, option contract filtering (DTE min/max, strike, premium caps), bid/ask handling

## Experiment Framework (`experiment_runner.py`)

### Configuration Classes
- **ModelConfig**: Model type + hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- **BacktestConfig**: period, horizon, train_years, test_years, step_days, threshold
- **FeatureConfig**: Which feature categories to include (price, volume, technical, macro, sentiment, fundamentals)
- **ExperimentConfig**: Combines above + defines experiment_id, ticker, optimization objective

### Running Experiments
```bash
# Run single experiment (interactive)
python -c "from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig; runner = ExperimentRunner(); runner.add_experiment(ExperimentConfig(experiment_id='test1', ticker='AAPL', model=ModelConfig(model_type='rf'))); runner.run_all_experiments()"

# Run batch from JSON config
python run_experiments.py --config experiments_phase2b.json

# Validate config
python validate_framework.py
```

### Key Methods
- `add_experiment()`: Queue experiment with config
- `run_all_experiments()`: Execute all queued experiments, track metrics
- `get_leaderboard()`: Rank experiments by Sharpe/accuracy
- Results exported to CSV/JSON for dashboard integration

## Developer Workflows

### Prediction & Backtest (CLI)
```bash
# Single prediction
python -c "from prediction_model import predict_next_for_ticker; import json; print(json.dumps(predict_next_for_ticker('AAPL'), default=str))"

# Backtest one ticker
python -c "from prediction_model import backtest_one_ticker; print(backtest_one_ticker('AAPL', period='10y', model_type='xgb'))"

# Walk-forward cross-sectional (multiple tickers)
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
- **Syntax validation**: `python3 -m py_compile prediction_model.py data_fetch.py`

## Project-Specific Conventions

### Environment Variables (in `prediction_model.py`)
- `USE_ELASTICNET_SELECT`, `ELASTICNET_L1_RATIO`, `ELASTICNET_CV_FOLDS`
- `USE_OLSSIGSELECT`, `OLSSIG_ALPHA`, `OLSSIG_TOPK`, `OLSSIG_MINFEATURES`
- `FRED_API_KEY`, `TRADING_DAYS=252`
- Accessed via `env_bool()`, `float(os.environ.get(..., default))`

### Column Naming Conventions
- **Returns**: `ret_Xd` (1d, 5d, 20d), `cumret_Xd`
- **Volatility**: `vol_Xd` (10d, 20d, 60d)
- **Technical**: `rsi14`, `macd`, `macdsignal`, `macdhist`, `mfi14`, `atr_14`, `adx_14`
- **GBM**: `gbm_*` (mu, sig, prob_up, exp_ret, p05/p95)
- **Target**: Always `ftarget_ret_horizon_ahead` in training DataFrames
- **Relative strength**: `rel_strength_1d`, `rel_momentum_5d` (vs SPX)
- **Fundamentals**: `fund_*` (pe_trailing, pb, marketcap)
- **Macro**: `mkt_ret_1d`, `vix`, `t10y`, `term_spread`, `unrate`, `cpi`, `oas`, `fed_funds`
- **Regime**: `regime_*` (bull, bear, vix_low/medium/high, covid, high/low_corr, bull_streak, bear_streak)

### Model Type Constants
- `"rf"` = RandomForestRegressor (default, most stable)
- `"xgb"` = XGBRegressor (best Sharpe, sensitive to hyperparams)
- `"gbrt"` = GradientBoostingRegressor (middle ground)
- `"linreg"` = LinearRegression (rarely used baseline)
- For classification: append `_clf` or use task="clf" (e.g., RandomForestClassifier)

### Data Quality Guardrails
- **Minimum rows**: 60 after NaN drop (lines ~1308–1310)
- **Fallback periods**: ['5y', '3y', '2y', '1y', '6mo', '3mo'] (auto-retry if current period insufficient)
- **Shift operations**: All features shifted 1 day back; target shifted forward `horizon` days
- **NaN handling**: Strict `.dropna()` after feature build; macro data uses `.ffill().bfill()` for forward-fill before reindex

### Known Gotchas & Common Errors
1. **SPX index mismatch**: `_get_spx()` normalizes timezone; use `.reindex(..., method="ffill")` on aligned dates
2. **Walk-forward by dates**: `walkforward_cross_sectional()` uses unique dates, NOT row indices, to prevent leakage
3. **GAF-CNN requires 30-day window**: `predict_up_gafcnn_from_rets()` needs at least 30 return values; will fail otherwise
4. **Heston pricing**: Only AAPL, NVDA hardcoded; others fall back to Black-Scholes
5. **Streamlit caching**: `@st.cache_data(ttl=...)` invalidates with new arguments → pass immutable types (tuples, strings)
6. **Feature lagging**: Missing `.shift(1)` on ANY feature → look-ahead bias → inflated backtest results
7. **Macro fill order**: Fill NaN BEFORE reindex to prevent forward-fill at fold boundaries (Phase 1 fix)
8. **Non-US stock trading**: Alpaca paper trading rejects non-US symbols; filter in `build_signals_from_pred_df()`

## Integration Points & Extensions

### Adding a New Data Source
1. Add fetch function in `data_fetch.py` (e.g., `get_crypto_prices()`)
2. Join to historical DataFrame in `build_features_and_target()` → `hist.join(new_source, how="left")`
3. Add column to `FEATURE_COLUMNS` or `MACRO_COLUMNS` list (~line 593 & 453)
4. Handle NaN fill: Use `.ffill().bfill()` or scalar default (e.g., 0)

### Adding a Feature
1. Implement calculation in `add_price_features()` (~line 700) or new helper function
2. **Critical**: `df["new_feature"] = calculation.shift(1)` (lag by 1 day)
3. Add to `FEATURE_COLUMNS` list (~line 593)
4. Test: `build_features_and_target('TEST_TICKER')` should include your column without NaNs

### Modifying Prediction Output
- Return dict keys in `predict_next_for_ticker()` (lines 1359–1550) flow directly to `app.py` cache + signals
- Ensure JSON-serializable: use `float()`, `str()`, avoid numpy types (`np.float64` → `float()`)
- Update `_build_display_df()` rename_map if new columns should appear in UI

### Extending Options Strategies
- Edit `suggest_options_strategy()` logic (lines 384–410)
- Add new strategy names to `normalize_model_option_strategy()` (lines 412–422)
- Mapping flows into `build_signals_from_pred_df()` → Alpaca execution in `auto_paper_trade.py`

### Running Grid Search
```bash
python grid_search.py --ticker AAPL --period 5y --models rf xgb gbrt --max_depth 5 7 10
```
- Returns best hyperparameters ranked by Sharpe ratio
- Results saved to `grid_search_results.json`

## Critical Look-Ahead Bias Safeguards

All of these are implemented; maintain them:
1. Features computed from past data only (via `.shift(1)`)
2. Macro data forward-filled BEFORE reindex (not after)
3. Walk-forward uses unique date boundaries (not row indices)
4. Each fold retrains model (not reused across folds)
5. Target shifted forward by `horizon` days (not overlapping with features)

---

**Last Updated**: 2025-12-29 | **Scope**: Stock prediction, backtesting, options pricing, automated paper trading, experiment orchestration
