# Production Model Baseline - January 2026

## Overview

The new production model is an **Adaptive Classifier** with three trading modes, tested via proper walk-forward validation from 2018-2025.

**BASELINE_006** - Integrated into `app_new.py` with UI controls.

## App Integration (app_new.py)

The new adaptive model is integrated into the Streamlit app:

1. **Trading Mode Selector** in sidebar under "🎯 Trading Mode (NEW)"
2. **Adaptive Model Toggle** - enabled by default
3. **Mode Selection** - Conservative, Balanced (default), or Aggressive
4. **Header Display** - shows "ADAPTIVE-BALANCED" etc. when active

### Key Implementation Files
- `src/core/production_predictor.py` - Main predictor class
- `app_new.py` - UI integration with sidebar controls
- `src/core/__init__.py` - Exports ProductionPredictor

## Performance Summary

| Mode | Avg Sharpe | Positive % | Beat B&H % | Total Return |
|------|-----------|------------|------------|--------------|
| **Conservative** | 0.68 | **83%** | 25% | 1,783% |
| **Balanced** | 1.10 | **83%** | 25% | 4,207% |
| **Aggressive** | 1.17 | 75% | **42%** | 5,020% |

*B&H comparison: Sharpe 2.08, Return 10,178%*

## Key Features

### Three Trading Modes

1. **CONSERVATIVE** (Default for paper trading)
   - Long threshold: 45% UP probability
   - Short threshold: 70% DOWN probability  
   - Best for: Capital preservation, minimal drawdown
   - Trades rarely, stays flat most of the time

2. **BALANCED** (Recommended)
   - Long threshold: 42% UP probability
   - Short threshold: 55% DOWN probability
   - Best for: Optimal risk/reward tradeoff
   - Same 83% positive rate as conservative, but 2x the Sharpe!

3. **AGGRESSIVE** (For risk-takers)
   - Long threshold: 38% UP probability
   - Short threshold: 45% DOWN probability
   - Best for: Maximum returns, beating B&H
   - Trades more frequently, higher volatility

### Model Architecture

- **Classifier**: XGBoost 3-class (UP/DOWN/NEUTRAL)
- **Calibration**: Isotonic regression for accurate probabilities
- **Features**: 57 technical + regime features
- **Training**: Walk-forward (2-year train, 6-month test)
- **Horizon**: 5-day prediction

### Regime-Aware Features

The model includes regime detection:
- `golden_cross` / `death_cross`: MA50 vs MA200
- `in_correction`: Drawdown > 10%
- `in_bear_market`: Drawdown > 20%
- `momentum_20d_zscore`: Momentum relative to history
- `vol_percentile`: Volatility rank over past year

## Usage

```python
from src.core.production_predictor import ProductionPredictor, quick_predict

# Quick prediction
result = quick_predict("AAPL", mode="balanced")
print(f"Signal: {result.signal}")  # BUY, SELL, or HOLD
print(f"Confidence: {result.confidence:.1%}")
print(f"Position Size: {result.position_size:.0%}")

# Full predictor
predictor = ProductionPredictor(mode="conservative")
result = predictor.predict("AAPL")

# Batch prediction
results = predictor.predict_batch(["AAPL", "MSFT", "GOOGL"])
for r in results:
    print(f"{r.ticker}: {r.signal} @ {r.confidence:.1%}")
```

## Why This Model?

### Previous Issues

1. **Regression models couldn't short**: Always predicted positive returns
2. **Data leakage**: Training on 2018-2023 then "testing" on 2022 was invalid
3. **Regime transitions**: Shorting hurt during 2023 H1 recovery

### Solutions

1. **Classification**: Model predicts UP/DOWN/NEUTRAL, not returns
2. **Walk-forward validation**: Proper out-of-sample testing
3. **Asymmetric thresholds**: Higher bar for shorts than longs
4. **Three modes**: User can choose risk tolerance

## Comparison to Previous Baseline

| Metric | Old (BASELINE_005) | New (Adaptive) |
|--------|-------------------|----------------|
| Avg Sharpe | 0.55 | 0.68 - 1.17 |
| Positive % | 75% | 75% - 83% |
| Shorting | No | Yes (controlled) |
| Modes | 1 | 3 |
| Walk-forward | No | Yes |

## Files

- `src/core/production_predictor.py`: Main predictor class
- `train_adaptive_model.py`: Training script
- `run_comprehensive_backtest.py`: Validation script
- `models/adaptive_model_config.pkl`: Mode configurations

## Recommendations

1. **Paper Trading**: Use CONSERVATIVE mode first
2. **Live Trading**: BALANCED mode after confidence
3. **Aggressive**: Only if you can handle 25% drawdowns

## Future Improvements

- [ ] Per-ticker mode optimization
- [ ] Dynamic mode switching based on regime
- [ ] Ensemble with multiple horizons
- [ ] Options overlay for hedging
