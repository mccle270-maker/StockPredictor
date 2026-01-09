# Problem Features Analysis

**Generated:** 2026-01-07T16:06:38.859328

---

## Executive Summary

- **Tickers analyzed:** 10
- **Features checked:** 57
- **Problem features (>5.0% NaN):** 12

---

## Problem Features Table

| # | Feature | Avg NaN% | Max NaN% | Worst Ticker | Category | Likely Cause | Fix |
|---|---------|----------|----------|--------------|----------|--------------|-----|
| 1 | `vol_60d` | 12.4% | 12.4% | SPY | ROLLING | 60-day rolling window warmup period | A |
| 2 | `vol_ratio_10_60` | 12.4% | 12.4% | SPY | ROLLING | 60-day rolling window warmup period | A |
| 3 | `gbm_prob_up_1d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 4 | `gbm_exp_ret_1d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 5 | `gbm_p05_ret_1d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 6 | `gbm_p95_ret_1d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 7 | `gbm_prob_up_5d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 8 | `gbm_exp_ret_5d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 9 | `gbm_p05_ret_5d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 10 | `gbm_p95_ret_5d` | 12.4% | 12.4% | SPY | GBM | 60-day rolling window warmup period | A |
| 11 | `gbm_mu_60d` | 12.2% | 12.2% | SPY | GBM | 60-day rolling window warmup period | A |
| 12 | `gbm_sig_60d` | 12.2% | 12.2% | SPY | GBM | 60-day rolling window warmup period | A |

---

## Fix Categories

| Category | Description | Features Affected |
|----------|-------------|-------------------|
| **A** | Needs more historical data (increase lookback or drop warmup rows) | 12 |
| **B** | External API issue (add caching/fallback values) | 0 |
| **C** | Calculation bug (fix formula or handling) | 0 |
| **D** | Feature not applicable to all tickers (make optional) | 0 |

---

## Detailed Analysis by Category

### Category A: Warmup Period Issues

These features have NaN at the beginning of the dataset due to rolling window calculations.

**Solution:** Drop the first N rows where N = max lookback period (typically 60-62 days).

**Features:**

- `vol_60d`: 12.4% NaN — 60-day rolling window warmup period
- `vol_ratio_10_60`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_mu_60d`: 12.2% NaN — 60-day rolling window warmup period
- `gbm_sig_60d`: 12.2% NaN — 60-day rolling window warmup period
- `gbm_prob_up_1d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_exp_ret_1d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_p05_ret_1d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_p95_ret_1d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_prob_up_5d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_exp_ret_5d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_p05_ret_5d`: 12.4% NaN — 60-day rolling window warmup period
- `gbm_p95_ret_5d`: 12.4% NaN — 60-day rolling window warmup period

---

## By Source Category

### GBM (10 features)

- `gbm_mu_60d`: 12.2% NaN
- `gbm_sig_60d`: 12.2% NaN
- `gbm_prob_up_1d`: 12.4% NaN
- `gbm_exp_ret_1d`: 12.4% NaN
- `gbm_p05_ret_1d`: 12.4% NaN
- `gbm_p95_ret_1d`: 12.4% NaN
- `gbm_prob_up_5d`: 12.4% NaN
- `gbm_exp_ret_5d`: 12.4% NaN
- `gbm_p05_ret_5d`: 12.4% NaN
- `gbm_p95_ret_5d`: 12.4% NaN

### ROLLING (2 features)

- `vol_60d`: 12.4% NaN
- `vol_ratio_10_60`: 12.4% NaN

---

## Recommended Fix Order

1. **Category A (Warmup)** — Already handled by `validate_features()` dropping first 62 rows
2. **Category B (API)** — Highest priority: fix FRED/FMP API connections or add fallbacks
3. **Category C (Bugs)** — Fix any calculation errors
4. **Category D (Optional)** — Low priority: make features conditional

---

## Per-Ticker Summary

| Ticker | Rows | Features with NaN | Problem Features |
|--------|------|-------------------|------------------|
| SPY | 501 | 39 | 12 |
| AAPL | 501 | 39 | 12 |
| MSFT | 501 | 39 | 12 |
| NVDA | 501 | 39 | 12 |
| GOOGL | 501 | 39 | 12 |
| AMZN | 501 | 39 | 12 |
| META | 501 | 39 | 12 |
| JPM | 501 | 39 | 12 |
| XOM | 501 | 39 | 12 |
| JNJ | 501 | 39 | 12 |
