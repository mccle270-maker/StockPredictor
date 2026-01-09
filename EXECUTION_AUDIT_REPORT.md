# Stock Predictor - Execution & Wiring Audit Report
**Generated**: 2026-01-07  
**Scope**: Full execution flow audit from entry points to outputs
**Last Updated**: 2026-01-07 - Fixed P&L table wiring

---

## 1. ENTRY POINTS

### ✅ Main UI Entry
- **File**: `app_new.py`
- **Invocation**: `streamlit run app_new.py`
- **Status**: ✅ Wired correctly
- **Imports from**:
  - `src.services.prediction` → `predict_next_for_ticker`, `predict_long_horizon_for_ticker`
  - `src.services.backtest` → `track_predictions`, `backtest_one_ticker`, `walk_forward_backtest`
  - `src.services.signals` → `build_signals_from_pred_df`, `is_us_tradeable_symbol`, `suggest_options_strategy`
  - `src.core.pricing` → `PricingModel`
  - `src.core.metrics` → `compute_sharpe`, `compute_drawdown`, `summarize_risk`
  - `src.data.*` → market, options, news data

### ✅ Auto Paper Trader Entry
- **File**: `auto_paper_trade.py` (line 852: `def main()`)
- **Scheduler**: `runner.py` → runs `trade_once()` every Monday 08:35 ET
- **Invocation**: `python auto_paper_trade.py` or `python runner.py`
- **Status**: ✅ Wired correctly
- **Features**: 
  - Max 10 open positions
  - 2% max risk per trade
  - Z-score soft filtering (env `ZSCORE_HARD_FILTER=1` for hard)
  - Take-profit/stop-loss (env `TAKE_PROFIT_PCT`, `STOP_LOSS_PCT`)
  - TradeLog persistence to `trade_log.json`

### ✅ Experiment CLI Entry
- **File**: `run_experiments.py` (line 117: `if __name__ == "__main__"`)
- **Invocation**: `python run_experiments.py --config experiments_phase2b.json`
- **Status**: ✅ Wired correctly to `experiment_runner.py`

### ✅ Grid Search CLI Entry
- **File**: `grid_search.py` (line 339: `if __name__ == "__main__"`)
- **Invocation**: `python grid_search.py --ticker AAPL --period 5y --models rf xgb`
- **Status**: ✅ Wired correctly

### ✅ FastAPI Entry
- **File**: `fastapi_app.py`
- **Invocation**: `uvicorn fastapi_app:app --reload --port 8000`
- **Status**: ✅ Wired correctly, imports from `prediction_model.py` (legacy)
- **Note**: Uses legacy `prediction_model.py` not refactored `src/services/prediction.py`

---

## 2. MODEL EXECUTION TRACE

### ✅ Model Factory (`src/core/models.py`)
- `make_model(model_type, task, **kwargs)` → Creates RF, XGBoost, GBRT, LinReg
- **Supported model_types**: `"rf"`, `"xgb"`, `"gbrt"`, `"linreg"`
- **Status**: ✅ Working correctly

### ❌ ISSUE: Ensemble Model Type NOT IMPLEMENTED
- **Location**: `app_new.py` line 2771 exposes `model_type_options = ["rf", "gbrt", "xgb", "ensemble"]`
- **Problem**: `make_model()` does NOT handle `model_type="ensemble"`
- **Result**: Selecting "ensemble" in UI will raise `ValueError: Unknown model_type: ensemble`
- **Severity**: 🔴 **BROKEN PATH**
- **Fix Required**: Either implement ensemble logic or remove from UI options

### ✅ GAF-CNN Model (`src/services/prediction.py`)
- Optional loading of `gaf_cnn_updown.keras`
- Activated via `run_gaf=True` parameter
- Graceful fallback if TensorFlow/model not available
- **Status**: ✅ Working correctly (optional feature)

### ✅ Feature Selection
- ElasticNet: Activated via `USE_ELASTICNET_SELECT=1` env var
- OLS p-value: Activated via `USE_OLSSIGSELECT=1` env var
- Auto-pruning: Removes features with importance < 0.001
- **Status**: ✅ All selection methods wired correctly

---

## 3. FEATURE PIPELINE TRACE

### ✅ Feature Engineering (`src/core/features.py`)
- `build_all_features()` orchestrates all feature additions:
  - `add_returns()` → ret_1d/3d/5d/10d/20d, cumret_*
  - `add_volatility()` → vol_10d/20d/60d, atr_14
  - `add_all_technicals()` → RSI, MACD, BB, MFI, ADX
  - `add_gbm_features()` → GBM probabilities
  - `add_relative_strength()` → vs SPX (beta, correlation)
  - `add_regime_features()` → bull/bear regime detection
- **Critical**: All features use `.shift(1)` for look-ahead prevention
- **Status**: ✅ Properly wired

### ✅ Feature Columns (`src/config.py`)
- `FEATURE_COLUMNS` = 57 features total
- `MACRO_COLUMNS` = 8 FRED-sourced columns
- All categories combined: Price, Technical, GBM, Macro, Regime, Fundamental, Relative Strength
- **Status**: ✅ Config matches implementation

---

## 4. SIGNALS & DECISION LOGIC TRACE

### ✅ Z-Score Filtering (`src/core/zscore_filter.py`)
- `ZScoreFilter.evaluate()` computes rolling z-score
- Default threshold: 1.0 (from `ZSCORE_GATING_CONFIG`)
- Default mode: **SOFT** (tags weak signals but doesn't exclude)
- Hard mode: Set `hard_filter=True` in config or `ZSCORE_HARD_FILTER=1` env
- Logs weak signals to `CACHE_DIR/weak_signals.jsonl`
- **Status**: ✅ Fully implemented

### ✅ Trade Limiting (`src/core/trade_limiter.py`)
- `TradeLimiter` limits trades per ticker per period
- Default: 1 trade per ticker per day
- Ranking: by z-score, confidence, or return
- Logs skipped signals to `CACHE_DIR/skipped_signals.jsonl`
- **Status**: ✅ Fully implemented

### ✅ Regime Filter (`src/core/regime_filter.py`)
- `RegimeFilter` checks SPY 200DMA, VIX, RSI
- Blocks longs in strong bear/crash, shorts in strong bull
- High-conviction signals (z-score > 2.0) can override
- Logs blocked trades to `CACHE_DIR/blocked_trades.jsonl`
- **Status**: ✅ Fully implemented

### ✅ Position Sizing (`src/core/position_sizing.py`)
- Volatility-scaled sizing (target 1% daily vol)
- Max leverage: 2.0x
- Min position: 25% of base
- Configurable via `POSITION_SIZING_CONFIG`
- **Status**: ✅ Fully implemented

### ⚠️ ISSUE: Ensemble Prediction Columns Placeholder Only
- **Location**: `app_new.py` lines 2477-2480
- **Code**: `cand_df["ensemble_prediction"] = cand_df.get("ensemble_prediction", np.nan)`
- **Problem**: Ensemble columns are initialized to NaN if not present; no actual ensemble logic populates them
- **Severity**: 🟡 **UNUSED** (displays empty column)
- **Recommendation**: Either implement ensemble averaging or remove columns from display

---

## 5. OPTIONS PIPELINE TRACE

### ✅ Black-Scholes Pricing (`src/core/pricing.py`)
- `black_scholes_price()` for option valuation
- `black_scholes_greeks()` for Delta/Gamma/Vega/Theta/Rho
- Heston model parameters available for AAPL, NVDA
- **Status**: ✅ Working correctly

### ✅ Option Strategy Generator (`src/core/option_strategy.py`)
- `determine_strategy()` → rules-based strategy selection
- `generate_option_strategy()` → batch processing
- Strategies: Buy Call, Buy Put, Bull Call Spread, Bear Put Spread, Iron Condor
- **Status**: ✅ Working correctly

### ✅ Options Data (`src/data/options.py`)
- `get_option_chain()` → fetches via yfinance
- `get_atm_options()` → finds near-ATM strikes
- `get_atm_greeks()` → computes ATM option Greeks
- **Status**: ✅ Working correctly

---

## 6. AUTO TRADER TRACE

### ✅ Signal Loading
- Reads from `signals.json` (built by UI)
- Falls back to hardcoded `WATCHLIST` if no signals
- **Status**: ✅ Wired correctly

### ✅ Risk Controls
- Max 10 open positions (hard limit)
- 2% max portfolio risk per trade
- Confidence filtering (ticker-specific thresholds)
- Z-score filtering (soft by default)
- Trade limits per ticker per period
- Take-profit/stop-loss auto-exit
- **Status**: ✅ All controls implemented

### ✅ Order Execution
- Stock: Market orders via Alpaca
- Options: Single-leg and multi-leg (spreads)
- Spread strategies: Bull Call, Bear Put, Iron Condor
- **Status**: ✅ Working correctly

### ✅ P&L Logging
- `TradeLog` class persists to `trade_log.json`
- Tracks entry/exit, P&L, holding period
- Stats: win rate, total P&L
- **Status**: ✅ Working correctly

---

## 7. UI CONSUMPTION TRACE

### ✅ Prediction Display
- `_cached_prediction()` → 15-minute TTL
- Results flow to signal table with Z-SCORE column
- **Status**: ✅ Working correctly

### ✅ Backtest Display
- `_cached_backtest_one_ticker()` → 30-minute TTL
- Walk-forward backtest results displayed
- **Status**: ✅ Working correctly

### ✅ Z-Score in UI
- Computed at line 2445-2456 (rolling window=60, min_periods=20)
- Displayed as "Z-SCORE" column in signals table
- **Status**: ✅ Working correctly (requires 20+ data points per ticker)

### ⚠️ ISSUE: Ensemble Columns Show NaN
- Columns exist in display but never populated
- **Status**: 🟡 **UNUSED** - consider removing or implementing

---

## 8. CONFIG & SNAPSHOT TRACE

### ✅ Centralized Config (`src/config.py`)
- All constants, thresholds, presets in one place
- Model versions locked with metrics
- Ticker eligibility thresholds
- **Status**: ✅ Well-organized

### ✅ Config Snapshots (`src/core/versioning.py`)
- `create_config_snapshot()` captures all key params
- `save_config_snapshot()` writes to `snapshots/` dir
- **Status**: ✅ Implemented

### ⚠️ Versioning NOT Used in UI
- Snapshots only created during backtest runs
- No snapshot capture on prediction runs in UI
- **Severity**: 🟡 **GAP** - reduces reproducibility
- **Recommendation**: Add snapshot capture when generating predictions

---

## SUMMARY

### ✅ WORKING CORRECTLY (23 components)
| Component | Location | Notes |
|-----------|----------|-------|
| Main UI Entry | `app_new.py` | Streamlit app |
| Auto Trader Entry | `auto_paper_trade.py` | Alpaca integration |
| Experiment CLI | `run_experiments.py` | Batch experiments |
| Grid Search CLI | `grid_search.py` | Hyperparameter tuning |
| FastAPI | `fastapi_app.py` | REST API |
| Model Factory | `src/core/models.py` | RF, XGB, GBRT |
| GAF-CNN | `src/services/prediction.py` | Optional deep learning |
| Feature Engineering | `src/core/features.py` | 57 features |
| Z-Score Filter | `src/core/zscore_filter.py` | Soft/hard gating |
| Trade Limiter | `src/core/trade_limiter.py` | Overtrading prevention |
| Regime Filter | `src/core/regime_filter.py` | Market regime |
| Position Sizing | `src/core/position_sizing.py` | Vol-scaled sizing |
| Options Pricing | `src/core/pricing.py` | Black-Scholes |
| Options Strategy | `src/core/option_strategy.py` | Strategy generation |
| Options Data | `src/data/options.py` | Chain fetching |
| Signal Building | `src/services/signals.py` | Trade signals |
| Prediction Service | `src/services/prediction.py` | Core predictions |
| Backtest Service | `src/services/backtest.py` | Backtesting |
| Risk Controls | `auto_paper_trade.py` | Max positions, risk |
| P&L Logging | `auto_paper_trade.py` | TradeLog class |
| Config | `src/config.py` | Centralized |
| Versioning | `src/core/versioning.py` | Snapshots |
| UI Caching | `app_new.py` | Streamlit cache |

### ❌ BROKEN (1 issue)
| Issue | Location | Severity | Fix |
|-------|----------|----------|-----|
| Ensemble model type not implemented | `src/core/models.py` | 🔴 HIGH | Remove from UI or implement |

### ⚠️ UNUSED/GAPS (3 issues)
| Issue | Location | Severity | Recommendation |
|-------|----------|----------|----------------|
| Ensemble columns show NaN | `app_new.py` L2477-2480 | 🟡 LOW | Remove or implement |
| Versioning not in UI predictions | `app_new.py` | 🟡 MEDIUM | Add snapshot on predict |
| FastAPI uses legacy module | `fastapi_app.py` | 🟡 LOW | Update to use `src/` modules |

### 🧠 RISKS
1. **Z-Score NaN for few predictions**: Rolling window=60, min_periods=20 means new tickers show NaN until 20+ predictions
2. **Rate limiting**: Yahoo Finance 429 errors on rapid requests
3. **Alpaca non-US rejection**: Non-US symbols correctly filtered but could confuse users

---

## RECOMMENDED FIXES

### 1. Fix Ensemble Model Type (🔴 HIGH PRIORITY)
**Option A**: Remove from UI
```python
# app_new.py line 2771
model_type_options = ["rf", "gbrt", "xgb"]  # Remove "ensemble"
```

**Option B**: Implement ensemble averaging
```python
# src/core/models.py - add ensemble support
elif model_type == "ensemble":
    # Train all base models and average predictions
    base_models = [make_model("rf"), make_model("xgb"), make_model("gbrt")]
    return EnsembleModel(base_models)
```

### 2. Remove Unused Ensemble Columns (🟡 MEDIUM)
```python
# app_new.py - remove lines 2476-2480 and columns from display list
cols = [c for c in cols if c not in ["ensemble_prediction", "ensemble_confidence"]]
```

### 3. Add Versioning to UI Predictions (🟡 MEDIUM)
```python
# app_new.py - after prediction call
from src.core.versioning import create_config_snapshot
snapshot = create_config_snapshot(model_type, feature_set)
```

---

*Audit complete. All major execution paths verified.*
