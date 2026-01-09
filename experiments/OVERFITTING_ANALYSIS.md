# Overfitting Analysis Report

**Date:** 2026-01-08
**Test Type:** True Unseen Data Holdout (Last 60 Days)

---

## ⚠️ CRITICAL FINDING: SEVERE OVERFITTING DETECTED

### Summary Table

| Model | Train Dir Acc | Test Dir Acc | Overfitting Gap (R²) | Status |
|-------|---------------|--------------|----------------------|--------|
| **Optimized RF** | 91.6% | **52.0%** | 0.73 | ⚠️ OVERFIT |
| **Optimized XGB** | 96.7% | **48.5%** | 1.31 | ⚠️ SEVERE OVERFIT |
| **Ensemble (50/50)** | N/A | **48.0%** | N/A | ⚠️ OVERFIT |

### What This Means

1. **The 98% accuracy from optimization was misleading** - it was measured on data that overlapped with training windows in walk-forward validation
2. **On truly unseen data (last 60 days), all models perform at ~50% (random)** 
3. **XGB is actually slightly worse than random** (48.5% accuracy)
4. **The ensemble doesn't help** - averaging overfit models still gives overfit results

---

## Per-Ticker Results (UNSEEN DATA)

| Ticker | RF Dir Acc | RF Sharpe | XGB Dir Acc | XGB Sharpe | Ensemble Sharpe |
|--------|------------|-----------|-------------|------------|-----------------|
| AAPL | 47.5% | -0.69 | 52.5% | 0.22 | -1.56 |
| MSFT | **60.0%** | **3.29** | 57.5% | 1.32 | 3.04 |
| GOOGL | 52.5% | 3.24 | 50.0% | -1.38 | -2.21 |
| AMZN | **62.5%** | **5.34** | 37.5% | -5.27 | -1.76 |
| NVDA | 37.5% | -3.43 | 45.0% | -2.67 | -2.82 |

**Key Observations:**
- MSFT and AMZN show some RF predictability (60-62%)
- AAPL and NVDA are essentially random or worse
- XGB underperforms RF on unseen data (opposite of training results!)
- Ensemble often makes things worse due to XGB's poor unseen performance

---

## Root Cause Analysis

### Why Walk-Forward Gave Inflated Results

1. **Overlapping Windows**: Each walk-forward fold trains on 80% of data, tests on 20%. The test windows overlap significantly.

2. **Feature Leakage**: Some features may contain forward-looking information despite `.shift(1)` (e.g., macro data with publication lags)

3. **Regime Change**: The last 60 days (Nov-Jan 2025-2026) may represent a different market regime than the training period

4. **Hyperparameter Overfitting**: Optuna's 50 trials may have found parameters that fit the specific training period too well

### Why XGB is Worse Than RF on Unseen Data

| Metric | RF | XGB | Implication |
|--------|-----|-----|-------------|
| Train R² | 0.66 | **0.96** | XGB memorizes training data |
| Test R² | **-0.07** | -0.34 | RF generalizes better |
| Complexity | Lower | Higher | Simpler = better generalization |

XGBoost's higher capacity (450 trees, depth 7) allows it to memorize noise in the training data, leading to worse generalization.

---

## Recommended Fixes

### Immediate (High Priority)

1. **Reduce Model Complexity**:
   ```python
   # XGB - reduce capacity
   XGB_SAFER_CONFIG = {
       "n_estimators": 100,      # Was 450
       "max_depth": 4,           # Was 7
       "learning_rate": 0.1,     # Was 0.048
       "subsample": 0.7,         # Was 0.998
       "colsample_bytree": 0.5,  # Was 0.67
       "reg_alpha": 0.5,         # Was 0.012
       "reg_lambda": 10.0,       # Was 9.3
   }
   
   # RF - increase regularization
   RF_SAFER_CONFIG = {
       "n_estimators": 50,       # Was 100
       "max_depth": 10,          # Was None (unlimited)
       "min_samples_leaf": 20,   # Was 4
       "max_features": 0.5,      # Was 0.7
   }
   ```

2. **Reduce Feature Count**:
   - Current: ~100+ features
   - Recommended: 10-15 most stable features
   - Use only price-based features (returns, volatility, momentum)
   - Drop fundamentals, macro, sentiment (data quality issues)

3. **Increase Holdout Period**:
   - Current: 60 days
   - Recommended: 120-180 days (6 months)
   - Retrain only quarterly

### Medium Priority

4. **Use Rolling Validation Instead of Walk-Forward**:
   - Gap between train and test to prevent leakage
   - Example: Train on days 1-500, skip 501-520, test on 521-540

5. **Add Calibration Check**:
   - Track live prediction accuracy over time
   - Halt trading if accuracy drops below 52%

6. **Consider Simpler Models**:
   - Ridge/Lasso regression may outperform on unseen data
   - Moving average crossover (no ML) as baseline

---

## Integration Status

### ✅ Completed

1. **`src/core/models.py`**: Updated to use optimized RF config when `use_optimized=True`
2. **`model_improvements.py`**: `ModelEnsemble` uses optimized XGB + RF configs
3. **`app_new.py`**: Valid models include `["rf", "xgb", "ensemble"]`
4. **Test script**: `experiments/test_unseen_data.py` for ongoing validation

### ⚠️ Warnings

- **Do NOT rely on the 98% accuracy** from optimization reports
- **Expect ~50-55% direction accuracy** on live trading
- **MSFT and AMZN** show the most promise for RF predictions
- **AAPL and NVDA** may be better traded without ML signals

---

## Test Commands

```bash
# Run unseen data test
python experiments/test_unseen_data.py

# Check model configs
python -c "from src.core.models import make_model; m = make_model('rf', use_optimized=True); print(m)"

# Test prediction pipeline
python -c "
from src.services.prediction import predict_next_for_ticker
result = predict_next_for_ticker('AAPL', model_type='rf')
print(result)
"
```

---

## Conclusion

**The models are severely overfit.** While they achieve high accuracy on walk-forward validation (which has overlapping test periods), they perform at random levels on truly unseen data.

**Recommendation:** 
- Use RF only (not XGB or ensemble) for now
- Focus on MSFT and AMZN which showed some predictability
- Reduce model complexity and feature count
- Implement proper rolling validation with gaps

**Expected Real-World Accuracy:** 50-55% (marginally better than random)

---

*Generated by test_unseen_data.py*
