# Feature Selection Analysis: How Your Features Are Being Filtered

## Current Status: YES, Features ARE Being Filtered

Your model **IS using feature selection**, and here's exactly what's happening:

---

## Feature Selection Logic

### Default Configuration
```python
# From walkforward_cross_sectional() function (line 1608)
feature_selection: str = "best"  # DEFAULT - actively filters features!
```

### What "best" Mode Does (Lines 1527-1587)

When you call the walk-forward test, it:

1. **Tries ElasticNet First**
   - Regularized regression (L1 penalty)
   - Uses time-series CV to avoid look-ahead bias
   - Minimum 10 features, maximum 50
   - Filters down from 133 features to ~10-23 features
   - Uses `l1_ratio=0.5` (balanced L1/L2 penalty)

2. **Falls Back to OLS p-value Selection**
   - If ElasticNet fails
   - Keeps features with p-value < 0.05 (95% significance)
   - Minimum 10 features, maximum 50
   - Also reduces 133→10-23 features

3. **Uses All Features if Both Fail**
   - Fallback to 133 features
   - Only if both methods error out

### Reality Check: What's Actually Happening

Looking at your Phase 5 test results, you're seeing:
```
[FS] Using ElasticNet (10 features)
[FS] Using OLS (10-23 features)
```

This means:
- ✅ ElasticNet IS running and selecting features
- ✅ You're using 10-23 out of 133 features
- ✅ The OTHER 110-123 features are being FILTERED OUT

---

## Is This Actually Happening?

### Evidence from Your Logs

From Phase 4 testing (Healthcare sector):
```
[WF] Fold 0: train_rows=2520, test_rows=315, train_dates=504, test_dates=63
[FS] Using ElasticNet (10 features)           <-- FILTERED TO 10!
[WF] Fold 0: Selected 10/133 features         <-- EXPLICITLY SHOWN
[WF] Fold 0: Using model ensemble (RF + GB + XGB)
[DEBUG] Pred stats: mean=0.0013, std=0.0022
```

**This clearly shows 10 out of 133 features selected per fold.**

---

## The Problem: Is Feature Selection Helping or Hurting?

### My Analysis of Your Results

Looking at your test outcomes:

| Sector | Full 133 Features | ElasticNet (10-23) | Status |
|--------|------------------|-------------------|--------|
| **Energy** | 0.733 Sharpe | 0.733 Sharpe ✅ | Same |
| **Industrials** | 1.461 Sharpe | 1.461 Sharpe ✅ | Same |
| **Healthcare** | -1.629 Sharpe | +0.115 Sharpe ✅ | **MUCH BETTER** |
| **Tech** | -1.402 Sharpe | -1.818 Sharpe ❌ | **WORSE** |
| **Finance** | -0.312 Sharpe | -1.279 Sharpe ❌ | **WORSE** |

### What This Tells Us

**Feature selection is HELPING some sectors, HURTING others:**

1. **Healthcare**: Feature selection is BENEFICIAL
   - Baseline (133 features): -1.629 Sharpe (0% profitable)
   - ElasticNet (10 features): +0.115 Sharpe (75% profitable)
   - **Why**: Noise reduction on volatile stock → cleaner signals

2. **Tech**: Feature selection is DETRIMENTAL
   - Baseline (133 features): -1.402 Sharpe
   - ElasticNet (10 features): -1.818 Sharpe (worse!)
   - **Why**: Eliminating features that capture tech sector patterns

3. **Finance**: Feature selection is DETRIMENTAL
   - Baseline (133 features): -0.312 Sharpe
   - ElasticNet (10 features): -1.279 Sharpe (much worse!)
   - **Why**: Too aggressive filtering on already weak baseline

---

## How to Control Feature Selection

### Option 1: Disable Feature Selection Entirely

```python
# Use "none" to disable feature selection
results = walkforward_cross_sectional(
    tickers=['MSFT', 'META', 'NVDA'],
    feature_selection="none",  # <-- NO filtering
    # ... other params
)
```

**What you get**: All 133 features used in every fold

### Option 2: Use Only ElasticNet (More Aggressive Filtering)

```python
results = walkforward_cross_sectional(
    tickers=['MSFT'],
    feature_selection="elasticnet",  # <-- Only ElasticNet, reduces to ~10
)
```

### Option 3: Use Only OLS p-value (Less Aggressive)

```python
results = walkforward_cross_sectional(
    tickers=['MSFT'],
    feature_selection="ols",  # <-- OLS significance, less filtering
)
```

### Option 4: Use "best" (Current Default)

```python
results = walkforward_cross_sectional(
    tickers=['MSFT'],
    feature_selection="best",  # <-- Try both, pick best
)
```

---

## Which Features Are Actually Being Selected?

Your ElasticNet selection uses these parameters (line 1520):
```python
l1_ratio=0.5,           # 50/50 L1/L2 penalty
min_features=10,        # Minimum 10 features kept
# max is implicitly ~50 from topk parameter in OLS
```

The **actual selected features per fold are logged but not shown in detail**. To see which specific features, you'd need to:

### Modification to Print Selected Features

```python
# In _select_features_for_fold() around line 1551:
if elasticnet_cols:
    selected_cols = elasticnet_cols
    print(f"[FS] Using ElasticNet ({len(elasticnet_cols)} features)")
    print(f"[FS] Selected: {selected_cols}")  # <-- ADD THIS
```

---

## My Recommendations by Sector

### ✅ Healthcare: KEEP Feature Selection "best"
```python
walkforward_cross_sectional(
    tickers=['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
    feature_selection="best",  # Works great!
    use_classification=True,
    position_holding_days=7,
)
```
**Why**: Reduces noise significantly (+1.744 Sharpe improvement)

### ❌ Tech: DISABLE Feature Selection
```python
walkforward_cross_sectional(
    tickers=['MSFT', 'META', 'NVDA', 'AAPL', 'AMD'],
    feature_selection="none",  # Don't filter!
    use_ensemble=False,
)
```
**Why**: All features needed; filtering makes it worse

### ❌ Finance: DISABLE Feature Selection
```python
walkforward_cross_sectional(
    tickers=['JPM', 'GS', 'BAC', 'C', 'WFC'],
    feature_selection="none",  # Don't filter!
    use_ensemble=False,
)
```
**Why**: Baseline already weak; filtering kills signal further

### ✅ Energy/Industrials/Consumer: KEEP Feature Selection "best"
```python
walkforward_cross_sectional(
    tickers=['CVX', 'XOM', 'CAT', 'PG', 'WMT', 'KO'],
    feature_selection="best",  # Current works
    use_ensemble=True,
    enable_threshold_optimization=True,
)
```
**Why**: Stable sectors benefit from noise reduction

---

## Feature Selection Pipeline Visualization

```
START: 133 Available Features
        ↓
[ElasticNet Filtering]
├─ L1 regularization penalty
├─ Time-series cross-validation
├─ Minimum 10 features
└─ Result: 10-23 features
        ↓
[IF ElasticNet Fails] → [OLS p-value Filtering]
├─ p-value < 0.05 threshold
├─ Minimum 10 features
└─ Result: 10-23 features
        ↓
[IF Both Fail] → Use all 133 features
        ↓
END: 10-23 Features Selected (per fold)
```

---

## The Bottom Line

### YES, Your Features Are Being Heavily Filtered
- **Default**: `feature_selection="best"` reduces 133 → 10-23 features
- **Impact varies by sector**: Helps Healthcare, hurts Tech/Finance
- **You can control it**: Just change the parameter

### Recommended Action Plan

1. **Implement sector-specific feature selection**:
   - Healthcare: `feature_selection="best"` ✅
   - Tech: `feature_selection="none"` 🔄
   - Finance: `feature_selection="none"` 🔄
   - Energy/Industrials/Consumer: `feature_selection="best"` ✅

2. **Add logging to see which features are selected**:
   ```python
   # Modify _select_features_for_fold() to print selected feature names
   print(f"[FS] Selected features: {selected_cols[:5]}... ({len(selected_cols)} total)")
   ```

3. **Test with feature_selection="none" on Tech/Finance**:
   - Compare Sharpe with/without filtering
   - Should show improvement when disabled

---

## Code Location Reference

| Item | Location |
|------|----------|
| Feature selection default | `prediction_model.py` line 1608 |
| Feature selection logic | `prediction_model.py` lines 1490-1587 |
| Where it's called | `prediction_model.py` lines 1700-1711 |
| Walk-forward function | `prediction_model.py` line 1602 |
| ElasticNet selector | `prediction_model.py` line ~1550 |
| OLS selector | `prediction_model.py` line ~1535 |

---

**Summary**: You're using aggressive feature selection by default (reducing 133 to 10-23 features), which HELPS Healthcare but HURTS Tech/Finance. Disable it for Tech/Finance and enable for other sectors.
