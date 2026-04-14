# Stock Predictor AI Coding Instructions

## Architecture Overview

This is a **stock prediction system** with four integrated layers:

1. **Core Prediction Engine** (`prediction_model.py`, `src/core/`): ML models (Random Forest, XGBoost) predicting next-day/multi-day returns with extensive feature engineering (100+ features)
2. **Data Pipeline** (`src/data/`): Multi-source data with automatic fallbacks - yfinance, Tiingo, Finnhub, SEC EDGAR, Alpha Vantage, FRED, FMP, Marketaux
3. **UI + Trading Integration** (`app.py` = Streamlit): Interactive dashboard + automated trading via Alpaca (`auto_paper_trade.py`)
4. **Experiment Orchestration** (`experiment_runner.py`, `grid_search.py`): Reproducible walk-forward backtesting across model types, hyperparameters, and feature configurations

---

## ⚠️ CURRENT KNOWN ISSUES

**Update this section as issues are resolved:**

| Issue | Severity | Location | Status |
|-------|----------|----------|--------|
| 12 features with >5% NaN rate | 🟡 MEDIUM | Feature warmup period | Expected (warmup-related) |
| GBRT model severe overfitting | ✅ RESOLVED | `model_improvements.py` | **Removed from ensemble** |
| RF model poor performance | ✅ RESOLVED | `experiments/optimize_rf.py` | **Optimized via Optuna - Sharpe 11.32** |
| Yahoo Finance 429 rate limits | ✅ RESOLVED | `src/data/aggregator.py` | Multi-source fallback implemented |
| Finnhub sentiment 403 errors | ✅ RESOLVED | `src/data/providers/finnhub_provider.py` | Using free endpoints |
| META/SPY poor performance | 🟡 MEDIUM | Ticker selection | Excluded in BASELINE_005 |

**When modifying features or models, always:**
1. Check NaN rates after changes
2. Run diagnostic: `python run_diagnostic_baseline.py`
3. Validate all 10 tickers pass data quality checks

---

## Module Structure

```
src/
├── config.py                  # All configuration constants, API keys
├── core/
│   ├── models.py             # Model factory (make_model) - RF + XGB only
│   ├── production_predictor.py  # NEW: Adaptive model with trading modes
│   ├── regime_filter.py      # Market regime detection
│   └── zscore_filter.py      # Signal strength filtering
├── data/
│   ├── __init__.py           # Unified data access layer
│   ├── market.py             # Price data (get_price_history)
│   ├── macro.py              # FRED data (get_macro_df)
│   ├── fundamentals.py       # FMP data (get_fundamental_features)
│   ├── news.py               # News & sentiment (Marketaux, Alpha Vantage)
│   ├── options.py            # Options chain data
│   ├── cache_manager.py      # Aggressive file-based caching
│   ├── aggregator.py         # Multi-source with automatic fallback
│   └── providers/            # Individual data providers
│       ├── base.py           # BaseProvider abstract class
│       ├── yfinance_provider.py   # Primary - free, fast
│       ├── tiingo_provider.py     # Backup - good quality
│       ├── finnhub_provider.py    # Sentiment, earnings, analyst recs
│       ├── sec_edgar_provider.py  # Official SEC filings (free)
│       └── alphavantage_provider.py # Last resort
├── monitoring/
│   ├── __init__.py           # Performance monitoring exports
│   └── performance_monitor.py # PerformanceMonitor, alerts, daily summaries
├── risk/
│   ├── __init__.py           # Risk management exports
│   └── circuit_breaker.py    # CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
├── services/
│   └── backtest.py           # Backtesting logic
└── ui/
    └── components.py         # Streamlit UI components

# Root-level key files
prediction_model.py           # Main prediction engine
model_improvements.py         # ModelEnsemble (RF + XGB), enhanced features
app.py                        # Streamlit dashboard
auto_paper_trade.py           # Alpaca paper trading execution
```

---

## Production Configuration (BASELINE_006 - Adaptive Model)

**NEW: The production baseline is now the Adaptive Model with trading modes.**

Walk-forward validated (24mo train, 6mo test, 12 folds from 2018-2025):
- 75% positive Sharpe periods (9/12 folds)
- Beats B&H in bear markets (2022), conservative in bulls
- No data leakage - proper out-of-sample validation

### Trading Modes

| Mode | Sharpe | Positive Periods | Long Conf | Short Conf | Use Case |
|------|--------|------------------|-----------|------------|----------|
| **conservative** | 0.68 | 83% | 45% | 70% | Capital preservation |
| **balanced** (default) | 1.10 | 83% | 42% | 55% | Best risk/reward |
| **aggressive** | 1.17 | 75% | 38% | 45% | Maximum returns |

**Recommendation**: Use BALANCED mode for optimal risk/reward tradeoff.

### Usage

```python
from src.core.production_predictor import ProductionPredictor, quick_predict

# Quick prediction
result = quick_predict("AAPL", mode="balanced")
print(f"{result.signal} @ {result.confidence:.1%} confidence")

# Full usage
predictor = ProductionPredictor(mode="balanced")
result = predictor.predict("AAPL")
print(f"Signal: {result.signal}")
print(f"Position size: {result.position_size:.0%}")
print(f"Predicted price: ${result.predicted_price:.2f}")
```

### Legacy Configuration (BASELINE_005)

For backward compatibility, the old regression-based models are still available:

```python
LEGACY_CONFIG = {
    # Signal Generation
    "z_score_threshold": 2.0,
    "regime_filter_enabled": True,
    "regime_bear_scale": 0.5,
    "regime_neutral_scale": 0.75,
    
    # Risk Management
    "max_position_size": 0.5,
    "daily_loss_limit": -0.03,
    "weekly_loss_limit": -0.05,
    "consecutive_loss_limit": 3,
    
    # Ticker Selection
    "allowed_tickers": ["AAPL", "MSFT", "AMZN"],
    "excluded_tickers": ["SPY", "META"],
    
    # Ensemble (GBRT removed due to overfitting)
    "ensemble_mode": "majority",
    "active_models": ["rf", "xgb"],
}
```

---

## Data Sources & API Keys

### Multi-Source Data Pipeline (Implemented 2026-01-07)

| Source | API Key | Best For | Rate Limit | Status |
|--------|---------|----------|------------|--------|
| yfinance | No | Prices, fundamentals, options | Generous | ✅ Primary |
| Tiingo | Yes | Prices, fundamentals backup | 500/hr | ✅ Fallback |
| Finnhub | Yes | News sentiment, insider trades, analyst recs | 60/min | ✅ Free endpoints |
| SEC EDGAR | No | Official filings, fundamentals | 10/sec | ✅ Free |
| Alpha Vantage | Yes | Last resort backup | 5/min | ✅ Fallback |
| FRED | Yes | Macro/economic data | 120/min | ✅ Working |
| FMP | Yes | Fundamentals | 250/day | ⚠️ Rate limited |
| Marketaux | Yes | News articles with sentiment | 100/day | ✅ Working |

### Environment Variables (in `.streamlit/secrets.toml`)
```toml
FRED_API_KEY = "your_key"
TIINGO_API_KEY = "your_key"
FINNHUB_API_KEY = "your_key"
FINNHUB_SECRET = "your_key"
ALPHAVANTAGE_API_KEY = "your_key"
FMP_API_KEY = "your_key"
MARKETAUX_API_KEY = "your_key"
```

### Data Fallback Chains (Automatic)
```
Prices:       yfinance → Tiingo → Alpha Vantage
Fundamentals: yfinance → Tiingo → SEC EDGAR → Alpha Vantage
Macro:        FRED → cached fallback → zeros
Sentiment:    Finnhub (free endpoints) → Marketaux
News:         Marketaux → Alpha Vantage
```

### Cache TTLs
| Data Type | TTL | Location |
|-----------|-----|----------|
| Prices | 1 hour | `.cache/data/price/` |
| Fundamentals | 24 hours | `.cache/data/fundamentals/` |
| Macro | 6 hours | `.cache/data/macro/` |
| Sentiment | 2 hours | `.cache/data/sentiment/` |

### Using the Data Pipeline
```python
# New multi-source approach (recommended)
from src.data import fetch_prices, fetch_fundamentals, fetch_sentiment

df = fetch_prices("AAPL", period="2y")        # Auto-fallback
funds = fetch_fundamentals("AAPL")            # Merged from multiple sources
sentiment = fetch_sentiment("AAPL")           # Finnhub free endpoints

# Check provider health
from src.data.aggregator import get_aggregator
health = get_aggregator().get_provider_health()
```

---

## Feature Engineering

### Required Pattern for New Features
```python
def add_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Always follow this pattern."""
    # 1. Calculate feature
    df["new_feature"] = some_calculation(df)
    
    # 2. CRITICAL: Lag by 1 day to prevent look-ahead bias
    df["new_feature"] = df["new_feature"].shift(1)
    
    # 3. Handle NaNs
    df["new_feature"] = df["new_feature"].ffill().fillna(0)
    
    # 4. Validate NaN rate
    nan_rate = df["new_feature"].isna().mean()
    if nan_rate > 0.05:
        warnings.warn(f"new_feature has {nan_rate:.1%} NaN rate")
    
    return df
```

### Problem Features (High NaN Risk)
- **Warmup-related** (expected): `ma_200`, `vol_60d`, `rsi14` - need lookback period
- **Macro**: `vix`, `t10y`, `term_spread` - require FRED API
- **Fundamentals**: `fund_pe`, `fund_pb` - require FMP API (rate limited)
- **Relative**: `rel_strength_1d` - depends on SPX cache

### Column Naming Conventions

| Category | Pattern | Examples |
|----------|---------|----------|
| Returns | `ret_Xd` | `ret_1d`, `ret_5d`, `ret_20d` |
| Volatility | `vol_Xd` | `vol_10d`, `vol_20d`, `vol_60d` |
| Technical | lowercase | `rsi14`, `macd`, `atr_14`, `adx_14` |
| GBM | `gbm_*` | `gbm_prob_up`, `gbm_exp_ret` |
| Fundamentals | `fund_*` | `fund_pe`, `fund_pb`, `fund_marketcap` |
| Macro | lowercase | `vix`, `t10y`, `term_spread` |
| Regime | `regime_*` | `regime_bull`, `regime_bear` |
| Temporal | descriptive | `momentum_consistency_20d`, `trend_strength_60d` |
| Target | `ftarget_*` | `ftarget_ret_horizon_ahead` |

---

## Model Reference

### Model Types
| Type | Class | Status | Avg Sharpe | Notes |
|------|-------|--------|------------|-------|
| `rf` | RandomForestRegressor | ✅ Active | +11.32 | **Optimized 2026-01-08** via Optuna |
| `xgb` | XGBRegressor | ✅ Primary | +2.12 | Best performance |
| `gbrt` | GradientBoostingRegressor | ❌ **Removed** | -0.540 | Severe overfitting (Train R²=0.98, Test R²=-0.20) |
| `linreg` | LinearRegression | ⚪ Baseline | N/A | Testing only |

### Optimized RF Config (2026-01-08)
Via Optuna (50 trials). See `experiments/RF_OPTIMIZATION_REPORT.md`:
```python
OPTIMIZED_RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,          # No limit is optimal
    "min_samples_split": 2,
    "min_samples_leaf": 4,
    "max_features": 0.7,        # 70% of features
    "bootstrap": True,
    "random_state": 42,
}
```

### Why GBRT Was Removed (2026-01-07)
Investigation found severe overfitting:
- Train R² = 0.976 vs Test R² = -0.204 (gap of 1.18!)
- GBRT lacks XGBoost's regularization (subsample, colsample_bytree)
- 2-model ensemble (RF + XGB) now achieves Sharpe +0.70

### Ensemble Configuration (Optimized 2026-01-08)
Experiment 4 found: `xgb_rf_equal` (50/50 weights) is best by Sharpe.
`rf_only` is most stable (best worst-3-month Sharpe: 7.26).
```python
# In model_improvements.py - uses optimized configs
class ModelEnsemble:
    """Combine Random Forest and XGBoost predictions."""
    # Uses get_optimized_rf_config() and get_optimized_config() from src/config.py
    # Default: equal weights (50/50)
    # Alternative: xgb_heavy (70/30)
```

---

## Finnhub Free Endpoints (Implemented 2026-01-07)

The premium `social-sentiment` endpoint returns 403. We now use these **free** endpoints:

| Endpoint | Data | Use Case |
|----------|------|----------|
| `company-news` | Article count, headlines | News buzz score |
| `insider-sentiment` | MSPR (Monthly Share Purchase Ratio) | Insider buying/selling |
| `stock/recommendation` | Analyst buy/hold/sell counts | Analyst sentiment |
| `stock/earnings` | Historical EPS, surprise % | Earnings data |

### Example Output
```python
{
    "buzz": {"articlesInLastWeek": 140, "buzz": 1.0},
    "sentiment": {"bullishPercent": 32.9, "bearishPercent": 67.1},
    "insiderSentiment": {"mspr": -34.25, "change": -492344},
    "analystRecommendations": {"buy": 21, "hold": 16, "sell": 2}
}
```

---

## Stability Thresholds

| Metric | Target | Warning | Hard Fail |
|--------|--------|---------|-----------|
| Mean Sharpe | > +0.3 | < +0.1 | < 0 |
| Worst 3-month Sharpe | > -0.5 | < -1.0 | < -2.0 |
| Max Drawdown | > -15% | < -20% | < -25% |
| Win Rate | > 50% | < 48% | < 45% |
| Data Quality Pass | 10/10 | < 9/10 | < 8/10 |

---

## Circuit Breakers

**Required for all production trading:**

```python
CIRCUIT_BREAKER_CONFIG = {
    "daily_loss_limit": -0.03,       # -3% daily
    "weekly_loss_limit": -0.05,      # -5% weekly
    "consecutive_loss_limit": 3,     # 3 losses in a row
}
```
```

### Integration
```python
from src.risk.circuit_breaker import CircuitBreaker

cb = CircuitBreaker()

def execute_trade(signal):
    if not cb.can_trade():
        return {"status": "rejected", "reason": cb.halt_reason}
    
    result = submit_order(signal)
    cb.update(result["pnl"])
    return result
```

---

## Testing Requirements

### Before Any PR/Merge
1. **Syntax**: `python3 -m py_compile <files>`
2. **Data quality**: `python run_diagnostic_baseline.py` → 10/10 pass
3. **Backtest**: Run on AAPL, MSFT, AMZN with 6-month window
4. **Sharpe check**: Must not decrease by >0.1

### Quick Commands
```bash
# Full diagnostic
python run_diagnostic_baseline.py

# Test data pipeline
python test_data_pipeline.py

# Test news/sentiment features
python test_news_features.py

# Feature NaN check
python -c "
from prediction_model import build_features_and_target
result = build_features_and_target('AAPL', period='1y')
print('Features built successfully')
"

# Single ticker backtest
python -c "
from prediction_model import backtest_one_ticker
print(backtest_one_ticker('AAPL', model_type='xgb'))
"

# Test sentiment providers
python -c "
from src.data.aggregator import fetch_sentiment
print(fetch_sentiment('AAPL'))
"

# Start UI
streamlit run app.py

# Paper trading
python auto_paper_trade.py
```

---

## Look-Ahead Bias Safeguards

**Critical rules - never violate:**

1. ✅ All features lagged via `.shift(1)`
2. ✅ Macro data forward-filled BEFORE reindex
3. ✅ Walk-forward uses DATE boundaries, not row indices
4. ✅ Each fold retrains model from scratch
5. ✅ Target shifted forward by `horizon` days
6. ✅ Validate NaN rates after feature engineering

---

## Known Gotchas

| Issue | Solution |
|-------|----------|
| SPX index timezone mismatch | Use `.reindex(..., method="ffill")` |
| Walk-forward data leakage | Use unique dates, not row indices |
| GAF-CNN fails | Requires minimum 30 return values |
| Heston pricing fails | Falls back to Black-Scholes (only AAPL, NVDA supported) |
| Streamlit cache issues | Pass immutable types (tuples, strings) |
| Inflated backtest results | Check for missing `.shift(1)` on features |
| Forward-fill at fold boundaries | Fill NaN BEFORE reindex, not after |
| Alpaca rejects orders | Filter non-US symbols before submission |
| High feature NaN rates | Check API rate limits, use multi-source fallbacks |
| Yahoo Finance 429 errors | Auto-fallback to Tiingo/Alpha Vantage |
| Finnhub 403 errors | Use free endpoints (news, insider, recommendations) |
| GBRT hurts ensemble | **Removed** - use RF + XGB only |

---

## Experiment Results Reference

### BASELINE Evolution

| Version | Sharpe | Max DD | Key Change |
|---------|--------|--------|------------|
| BASELINE_001 | -0.094 | -23.66% | Original |
| BASELINE_002 | +0.128 | -12.85% | z-score=1.6 |
| BASELINE_003 | +0.178 | -11.28% | +regime filter |
| BASELINE_004 | +0.129 | -8.40% | +vol×conf sizing |
| BASELINE_005 | +0.55 | -13.14% | Ticker filter (AAPL, MSFT, AMZN only) |

### Key Findings
- ✅ Z-score filtering (1.6-2.0) dramatically improves Sharpe
- ✅ Regime filter reduces drawdown without hurting returns
- ✅ Vol × confidence sizing smooths equity curve
- ✅ Multi-source data pipeline eliminates 429 errors
- ✅ 2-model ensemble (RF + XGB) achieves Sharpe +0.70
- ❌ SPY and META are systematically unprofitable - exclude
- ❌ Trade frequency limits hurt performance - don't use
- ❌ GBRT has severe overfitting - removed from ensemble

---

## Implementation Status

### ✅ Completed (2026-01-07)
- [x] Multi-source data pipeline with fallback chains
- [x] Aggressive file-based caching (1hr prices, 24hr fundamentals)
- [x] Finnhub free endpoints (news, insider, analyst)
- [x] GBRT removed from ensemble
- [x] News/sentiment integration working
- [x] SEC EDGAR provider for free fundamentals
- [x] Circuit breaker implementation (`src/risk/circuit_breaker.py`)
- [x] Trade history tracking (`trade_log.json`)
- [x] Stop loss / take profit auto-exits
- [x] Alpaca API verified working (paper mode)
- [x] Performance monitoring (`src/monitoring/performance_monitor.py`)

### ✅ Completed (2026-01-09)
- [x] Target winsorization (clips extreme returns at 1%/99%)
- [x] Tested: +0.44 Sharpe improvement across 5 tickers
- [x] Enabled by default in `build_target(winsorize=True)`

### 🔄 In Progress
- [ ] BASELINE_005 deployment to paper trading

### 📋 Planned
- [ ] Ensemble voting (majority mode)
- [ ] Enhanced regime detection
- [ ] Ticker health scoring
- [ ] Walk-forward optimization

---

## Performance Monitoring

The `src/monitoring/` module provides real-time performance tracking:

### Metrics Tracked
- **Rolling Sharpe Ratios**: 21-day and 63-day
- **Drawdown**: Current drawdown from peak equity
- **Win Rate**: Over last 20 trades and all-time
- **P&L**: Daily, weekly, monthly, total

### Alert Thresholds
| Metric | WARNING | CRITICAL |
|--------|---------|----------|
| Rolling Sharpe (21d) | < 0.0 | < -0.5 |
| Drawdown | > 5% | > 8% |
| Win Rate (20 trades) | < 45% | < 35% |
| Consecutive Losses | ≥ 3 | ≥ 5 |

### Daily Summary Reports
Saved to `.monitoring/daily_summaries/summary_YYYY-MM-DD.json` with:
- All metrics snapshot
- Per-ticker performance breakdown
- Active alerts
- Today's trades

### Slack Notifications
Set `SLACK_WEBHOOK_URL` environment variable to receive critical alerts.

### Usage
```python
from src.monitoring import PerformanceMonitor

monitor = PerformanceMonitor(starting_capital=50000.0)
monitor.record_trade("AAPL", "BUY", 10, 150.0, 155.0, pnl=50.0)
alerts = monitor.check_alerts()
summary = monitor.generate_daily_summary()
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_diagnostic_baseline.py` | System health check |
| `test_data_pipeline.py` | Data provider tests |
| `test_news_features.py` | News/sentiment tests |
| `test_alpaca_api.py` | Alpaca API connectivity test |
| `src/config.py` | All configuration constants |
| `src/data/aggregator.py` | Multi-source data with fallbacks |
| `src/data/cache_manager.py` | File-based caching |
| `src/data/providers/` | Individual data providers |
| `src/monitoring/performance_monitor.py` | Performance tracking and alerts |
| `src/risk/circuit_breaker.py` | Risk management circuit breaker |
| `prediction_model.py` | Main prediction engine |
| `model_improvements.py` | ModelEnsemble (RF + XGB) |
| `auto_paper_trade.py` | Live paper trading |
| `app.py` | Streamlit UI |

---

**Last Updated**: 2026-01-07
**Maintainer**: @mccle270-maker
**Scope**: Stock prediction, backtesting, options pricing, automated paper trading