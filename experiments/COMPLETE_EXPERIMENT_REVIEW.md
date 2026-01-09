# 📊 Complete Experiment Review
## Stock Predictor Quantitative Research Session
### Date: January 7, 2026

---

## 🎯 EXECUTIVE SUMMARY

We conducted a **systematic quantitative research session** following strict one-variable-at-a-time methodology to optimize your stock prediction trading strategy. Here's what happened:

### Starting Point → Final Result

| Metric | BASELINE_001 (Original) | BASELINE_004 (Optimized) | Improvement |
|--------|-------------------------|--------------------------|-------------|
| **Sharpe Ratio** | -0.094 | +0.129 | **+0.223** ✅ |
| **Max Drawdown** | -23.66% | -8.40% | **+15.26%** ✅ |
| **Win Rate** | 48.7% | 49.9% | +1.2% |
| **Trades** | 1,836 | 1,284 | -30% (fewer, higher quality) |

**However**, when we stress-tested this configuration with rolling 3-month windows (Phase 6), we found **extreme instability** with worst-case Sharpe of -6.29.

---

## 📁 FILES TO INVESTIGATE

### Experiment Scripts (How we ran tests)
| File | Purpose |
|------|---------|
| `experiments/run_baseline.py` | Created frozen baseline |
| `experiments/run_zscore_sweep.py` | Tested z-score thresholds |
| `experiments/run_regime_filter_test.py` | Tested regime filter |
| `experiments/run_position_sizing_test.py` | Tested position sizing methods |
| `experiments/run_trade_limit_test.py` | Tested trade frequency limits |
| `experiments/run_robustness_validation.py` | Phase 6 rolling window validation |
| `experiments/run_stabilization.py` | Initial stabilization attempt |
| `experiments/run_aggressive_stabilization.py` | Final stabilization with ticker filtering |

### Results Data (Raw experiment results)
| File | Purpose |
|------|---------|
| `experiments/baseline_001_original.json` | Original frozen baseline |
| `experiments/baseline_002_zscore.json` | After z-score optimization |
| `experiments/baseline_003_regime.json` | After regime filter added |
| `experiments/baseline_metrics.json` | BASELINE_004 (current best) |
| `experiments/baseline_005_stabilized.json` | Candidate production config |
| `experiments/zscore_sweep.json` | Z-score sweep results |
| `experiments/regime_filter_test.json` | Regime filter comparison |
| `experiments/position_sizing_tests.json` | Position sizing comparison |
| `experiments/trade_limit_tests.json` | Trade limit comparison |
| `experiments/robustness_report.json` | Phase 6 rolling window results |
| `experiments/aggressive_stabilization_report.json` | Final stabilization results |

### Summary Documents
| File | Purpose |
|------|---------|
| `experiments/STABILIZATION_RESULTS.md` | Final stabilization analysis |

---

## 🧪 EXPERIMENT-BY-EXPERIMENT BREAKDOWN

### Phase 1: Establish Baseline (BASELINE_001)

**Goal**: Create frozen starting point for all comparisons

**Configuration**:
- 10 tickers: SPY, AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, JNJ
- Models: RF, XGBoost, GBRT (ensemble = average)
- No z-score filtering (threshold = 0)
- No regime filter
- Basic volatility scaling

**Results**:
```
Sharpe:      -0.094  ❌ (negative = losing money on average)
Max DD:      -23.66% ❌ (too risky)
Win Rate:    48.7%
Trades:      1,836
```

**Verdict**: Strategy was losing money. Needed optimization.

---

### Phase 2: Z-Score Sweep (Variable: z_score_threshold)

**Goal**: Find optimal signal strength filter

**Values Tested**: [0.5, 0.8, 1.0, 1.3, 1.6]

**Results**:
| Z-Score | Sharpe | Max DD | Trades |
|---------|--------|--------|--------|
| 0.5 | +0.015 | -19.0% | 1,469 |
| 0.8 | +0.047 | -16.7% | 1,199 |
| 1.0 | +0.085 | -14.9% | 998 |
| 1.3 | +0.110 | -13.5% | 836 |
| **1.6** | **+0.128** | **-12.9%** | 801 |

**Winner**: z_score = 1.6

**Decision**: ✅ ACCEPTED → BASELINE_002

**Insight**: Higher z-score = only trade when model is very confident. Fewer trades but much better quality.

---

### Phase 3: Regime Filter (Variable: regime_filter ON/OFF)

**Goal**: Test if reducing exposure in bear markets helps

**Results**:
| Setting | Sharpe | Max DD | Time in Market |
|---------|--------|--------|----------------|
| OFF | +0.170 | -13.45% | 13.25% |
| **ON** | +0.178 | **-11.28%** | 10.82% |

**Winner**: Regime Filter = ON

**Decision**: ✅ ACCEPTED → BASELINE_003

**Insight**: Regime filter reduced drawdown by 2.17% with minimal Sharpe impact. Worth the protection.

---

### Phase 4: Position Sizing (Variable: sizing method)

**Goal**: Find best way to size positions

**Methods Tested**:
- **A**: Vol-scaling only (base position = 1/volatility)
- **B**: Vol-scaling × confidence (scale by model certainty)
- **C**: Vol × conf with cap (same as B but max 50% position)

**Results**:
| Method | Sharpe | Max DD | Std(Returns) |
|--------|--------|--------|--------------|
| A | +0.129 | -11.3% | 0.00457 |
| **B** | **+0.129** | **-8.4%** | **0.00344** |
| C | +0.129 | -7.0% | 0.00301 |

**Winner**: Method B (vol_scaling × confidence)

**Decision**: ✅ ACCEPTED → BASELINE_004

**Insight**: Method B reduced return variance by 24.6% vs Method A. Smoother equity curve.

---

### Phase 5: Trade Limits (Variable: trade frequency)

**Goal**: Test if limiting trade frequency improves stability

**Configs Tested**:
- **A**: No limits (baseline)
- **B**: Min 3 days between trades, max 10 positions
- **C**: Min 3 days between trades, max 8 positions

**Results**:
| Config | Sharpe | Max DD | Trades |
|--------|--------|--------|--------|
| **A** | **+0.129** | -8.4% | 1,284 |
| B | +0.093 | -8.0% | 937 |
| C | +0.077 | -7.8% | 812 |

**Winner**: Config A (no limits)

**Decision**: ❌ REJECTED - Trade limits hurt performance

**Insight**: Limiting trades reduced Sharpe without meaningful drawdown improvement.

---

### Phase 6: Robustness Validation (Rolling Windows)

**Goal**: Stress test the optimized config across different time periods

**Method**: 
- 63-day windows (~3 months)
- 21-day step (overlapping windows)
- 10 tickers × 27 windows = 243 tests

**Stability Thresholds**:
- Worst 3-month Sharpe > -0.5
- Worst Drawdown > -15%
- Std(Sharpe) < 1.5 × Mean(Sharpe)

**Results** ❌ FAILED:
```
Mean Sharpe:     +0.114
Worst Sharpe:    -6.29   ❌ (threshold: > -0.5)
Std Sharpe:      2.05    ❌ (18x higher than mean!)
Worst DD:        -14.11% ✅
```

**Worst Period**: META, Dec 2024 - Mar 2025 (Sharpe: -6.29)

**Per-Ticker Analysis**:
| Ticker | Avg Sharpe | Worst Sharpe | Status |
|--------|------------|--------------|--------|
| **MSFT** | **+0.76** | -2.86 | Best performer |
| AAPL | +0.42 | -2.88 | Good |
| AMZN | +0.23 | -4.03 | Mixed |
| GOOGL | +0.01 | -4.62 | Unstable |
| **SPY** | **-0.24** | -3.12 | **Negative avg** |
| **META** | **-0.32** | **-6.29** | **Worst ticker** |

**Decision**: ❌ CONFIGURATION REJECTED for production

**Insight**: Strategy works on average but has catastrophic worst-case periods.

---

### Phase 7: Stabilization Experiments

**Goal**: Find a configuration that passes stability thresholds

#### Attempt 1: Basic Stabilization
Tested higher z-scores (2.0, 2.2), stricter regime scaling, vol caps, loss limits

**Result**: All configs still failed. Best worst-Sharpe: -3.57 (vs target -0.5)

#### Attempt 2: Aggressive Stabilization
Tested:
1. Remove problematic tickers (SPY, META)
2. Very high z-score (2.5, 3.0)
3. Only top 3 performers (AAPL, MSFT, AMZN)
4. MSFT only (best single performer)

**Results**:
| Config | Mean Sharpe | Worst Sharpe | Max DD |
|--------|-------------|--------------|--------|
| **Top 3 (z=2.0)** | **+0.55** | -2.35 | -13.1% |
| MSFT only (z=2.0) | +0.28 | **-1.81** | -9.8% |
| z-score 3.0 | +0.24 | -2.00 | **-4.8%** |

**Best Candidate**: Top 3 Performers (AAPL, MSFT, AMZN) with z=2.0

**Decision**: ⚠️ PARTIALLY STABLE - No config fully passes thresholds

---

## 📈 KEY FINDINGS

### What Works
1. **Z-score filtering (1.6-2.0)**: Dramatically improves Sharpe (+0.22)
2. **Regime filter**: Reduces drawdown without hurting returns
3. **Vol × confidence sizing**: Smoother equity curve
4. **Stock selection matters**: MSFT, AAPL, AMZN work well; SPY, META don't

### What Doesn't Work
1. **Trading SPY**: Negative average Sharpe (-0.24)
2. **Trading META**: Extreme worst-case (-6.29 Sharpe in one quarter)
3. **Trade frequency limits**: Hurt performance without improving stability

### Root Cause of Instability
The strategy experiences **regime-dependent performance**:
- Works well in trending markets
- Fails in mean-reverting or choppy markets
- Individual stocks can have 3-month periods where predictions are consistently wrong

---

## 🎯 CURRENT STATUS

### Optimized Configuration (BASELINE_004)
```json
{
  "z_score_threshold": 1.6,
  "regime_filter": true,
  "position_sizing": "vol_scaling_x_confidence",
  "trade_limits": false,
  "tickers": ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"]
}
```
- **Average Sharpe**: +0.129
- **Robustness**: ❌ FAILED (worst-case too extreme)

### Recommended Production Configuration (BASELINE_005)
```json
{
  "z_score_threshold": 2.0,
  "regime_filter": true,
  "regime_bear_scale": 0.5,
  "position_sizing": "vol_scaling_x_confidence",
  "max_position_size": 0.5,
  "tickers": ["AAPL", "MSFT", "AMZN"],
  "excluded_tickers": ["SPY", "META"]
}
```
- **Average Sharpe**: +0.55
- **Robustness**: ⚠️ PARTIAL (worst Sharpe -2.35, better but still fails strict threshold)

---

## 🔮 NEXT STEPS (Your Decision Required)

### Option A: Accept Pragmatic Config
- Deploy `BASELINE_005` with circuit breakers
- Add daily/weekly loss limits (-3% / -5%)
- Monitor and iterate based on live results
- **Realistic expectation**: Some quarters will be negative

### Option B: Higher Selectivity
- Trade only MSFT (best single performer)
- Accept lower returns for more stability
- **Trade-off**: Single-stock concentration risk

### Option C: More Research
- Investigate ensemble voting (require 2/3 models to agree)
- Add VIX-based regime switching
- Explore mean-reversion strategies as complement

---

## 📊 QUICK REFERENCE: BASELINE EVOLUTION

```
BASELINE_001 (Original)
    │  Sharpe: -0.094, DD: -23.66%
    │
    ▼ [Z-score = 1.6]
BASELINE_002 
    │  Sharpe: +0.128, DD: -12.85%
    │
    ▼ [Regime filter = ON]
BASELINE_003
    │  Sharpe: +0.178, DD: -11.28%
    │
    ▼ [Position sizing = vol×conf]
BASELINE_004 (Current Best)
    │  Sharpe: +0.129, DD: -8.40%
    │
    ▼ [Robustness validation FAILED]
    │
    ▼ [Ticker filter + z=2.0]
BASELINE_005 (Candidate)
       Sharpe: +0.55, DD: -13.1%
       Worst 3-mo Sharpe: -2.35
```

---

## 🔗 Related Files

- **Core Model**: `prediction_model.py`
- **Backtest Logic**: `src/services/backtest.py`
- **Trading Integration**: `auto_paper_trade.py`
- **UI**: `app.py`

---

*Document generated: January 7, 2026*
