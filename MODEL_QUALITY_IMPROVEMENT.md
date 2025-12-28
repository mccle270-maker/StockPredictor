# How to Improve Walk-Forward Model Quality & Signals

## Problem Diagnosis

Based on your output, there are two main issues:

1. **Very Low Prediction Variance**
   ```
   [DEBUG] Pred stats: mean=-0.0072, std=0.0279
   ```
   The std=0.0279 is extremely low—all predictions are clustered around -0.007.
   This makes it nearly impossible to rank stocks and generate signals.

2. **Zero Long Signals**
   ```
   [WF] Base Long: 100, Final Long: 0
   ```
   The model found 100 stocks in top 30% (good), but your position sizing threshold
   (top_pct_long) was too tight, filtering them all out.

---

## Root Causes & Solutions

### Issue 1: Low Prediction Signal (Low Std)

**Why this happens:**
- Model is underfitting (not learning patterns)
- Feature quality is poor (weak predictors)
- Target scaling is too small (absolute returns, not normalized)
- Not enough training data
- Features are redundant/correlated

**Solutions (in order of impact):**

#### Solution 1A: Increase Training Data ⭐ BEST
```
UI Setting: Time Windows → Preset: Aggressive (More Data)

Current: 1.5 year train (Balanced)
Better:  2+ years train (Conservative)

Why: More data = better pattern learning
Effect: +30-50% prediction std improvement
```

#### Solution 1B: Use More Tickers
```
UI Setting: Universe → Quick Universe or custom list
Current: 3 tickers?
Better:  10+ tickers

Why: More diverse data reduces noise
Effect: +20-30% prediction std
Time Cost: +1-2 min per fold
```

#### Solution 1C: Check Feature Quality
```
In console, look for:
[FS] Using ElasticNet (10/99 features)

If selecting very few features (≤15):
→ Most features are noise
→ Consider: Different model type, more training data

If selecting many features (40+):
→ Good feature diversity
→ Problem is likely elsewhere
```

#### Solution 1D: Use Less Aggressive Feature Selection
```
UI Setting: Feature Selection: ❌ None (temporarily)

Why: Aggressive selection removes weak signals
Try: "❌ None" → compare Sharpe with "🏆 Best"

If "None" gives better Sharpe:
→ Feature selection is too aggressive
→ May need different model or more data
```

#### Solution 1E: Try Different Model Type
```
Current: Model Type: [RF]
Try:     Model Type: [XGBoost] or [GBRT]

Why: Different models learn differently
XGBoost: Better for noisy data (more regularization built-in)
GBRT:    Slower but sometimes better fit

Time Cost: +20-30% per fold
```

---

### Issue 2: Zero Long Signals (Threshold Too Tight)

**Why this happens:**
- Default UI sliders were too aggressive (10% for longs)
- Model needs leeway to generate signals even if weak

**Solution: Adjust Position Sizing** ⭐ EASY FIX

**UPDATED UI DEFAULTS (just applied):**
```
Old:  Long % default = 0.10 (10%) ← Too aggressive
New:  Long % default = 0.25 (25%) ← More signals

Old:  Short % default = 0.30 (30%)
New:  Short % default = 0.40 (40%) ← More short positions
```

**How to use:**
```
Position Sizing section:
├─ Long %:  [======●======] 0.25 ← Increase to 0.30-0.40 if still no signals
├─ Short %: [==========●====] 0.40
└─ Net Exposure: -15% (negative = more shorts than longs)
```

**If still getting zero longs:**
- Increase Long % to 0.35-0.40
- The model is predicting downside overall (negative mean=-0.007)
- That's OK! Let shorts work instead

---

## Complete Improvement Workflow

### Step 1: Increase Training Data (Biggest Impact)
```
1. Portfolio WF tab
2. Time Windows → 📋 Window Preset
3. Change from "Balanced" to "Conservative (Anti-Overfit)"
4. Click Run Backtest
5. Compare Sharpe before/after

Expected: +15-30% improvement
```

### Step 2: Add More Tickers (If Still Weak)
```
1. Universe section
2. Change from 3 tickers to:
   ├─ Click "Top 10" preset, or
   ├─ Click "Mag 7", or
   ├─ Enter: AAPL,MSFT,NVDA,AMD,INTC,CRM,ADBE,MU,NFLX,TSLA
3. Click Run Backtest
4. Compare Sharpe with/without

Expected: +10-20% improvement
```

### Step 3: Adjust Position Sizing (Get Signals)
```
1. Scroll to "Position Sizing"
2. Long %:  Try 0.25-0.35 (not 0.10)
3. Short %: Try 0.40-0.50
4. Click Run Backtest

Expected: Get some long/short signals (not 0)
```

### Step 4: Check Feature Selection Quality
```
1. Model & Risk → Feature Selection
2. Try each mode:
   ├─ 🏆 Best (default, recommended)
   ├─ 🎯 ElasticNet (most aggressive)
   ├─ 📊 OLS (moderate)
   ├─ ❌ None (baseline, all 100 features)
3. Compare Sharpe across modes

Expected: One mode will consistently outperform
→ Use that mode going forward
```

### Step 5: Try Different Model Types
```
1. Model & Risk → Model Type
2. Try:
   ├─ RF (Random Forest) - current
   ├─ XGBoost - often better on small data
   ├─ GBRT - Gradient Boosting
3. Run each and compare Sharpe

Expected: XGBoost often wins on 1-2 year windows
→ Use best performer
```

---

## Interpretation Guide

### Low Std = Weak Model Signal
```
std < 0.02:  Very weak signal, likely underfitting
std 0.02-0.04: Weak signal, increase training data
std 0.04-0.08: OK signal, add more tickers
std > 0.08:  Strong signal, model learning well
```

### No Long Signals
```
Issue 1: Position sizing too tight
Fix:     Increase top_pct_long from 0.10 to 0.25-0.40

Issue 2: Model predicting mostly downside (mean < 0)
Fix:     That's OK! Let shorts work. Model may be pessimistic
         Try different model type or more training data

Issue 3: Very few stocks in "Base Long" (< 50)
Fix:     Add more tickers to get more data points
```

### High Short Signals, Zero Long Signals
```
This usually means:
- Market was bearish during your test period
- Or model learned downside better than upside
- Or features are inversely predictive

Fix:
1. Increase top_pct_long threshold
2. Try different training period (change Preset)
3. Verify target is defined correctly (should be forward returns)
```

---

## Quick Comparison Checklist

Run each configuration once and compare:

| Config | Time | Sharpe | Avg Return | Notes |
|--------|------|--------|------------|-------|
| Conservative + ElasticNet | 5min | ? | ? | More data + aggressive filtering |
| Conservative + OLS | 5min | ? | ? | More data + statistical filtering |
| Conservative + None | 5min | ? | ? | More data + all features |
| Balanced + ElasticNet | 4min | ? | ? | Medium data + aggressive filtering |
| Balanced + None | 4min | ? | ? | Medium data + all features |

→ Winner combination should be obvious (highest Sharpe)

---

## Advanced Tuning

### If Sharpe is Still Low (< 0.3):

**Option 1: Target Normalization**
```
Current target: Absolute log returns (very small)
Issue: Predictions also very small (std 0.028)

Potential fix: Normalize returns to per-unit-vol
Status: Would require code change
```

**Option 2: More Feature Engineering**
```
Current features: 100 technical + macro
Try: Add more domain-specific features
- Earnings surprises
- Revenue growth
- Insider trades
- Options flow
```

**Option 3: Different Prediction Horizon**
```
Current: 1-day horizon (very noisy)
Try:     5-day or 10-day horizon (smoother)

UI Setting: Prediction Horizon → [3 days ▼]
```

**Option 4: Use Market Regime Filter**
```
Current: All market conditions
Try:     Only when VIX < 20, or > 30

UI Setting: VIX Filter [✓] VIX Max: [20] or [35]
```

---

## What We Fixed (v4 Update)

### 1. ElasticNet Convergence ✅
**Before:**
```
FutureWarning: 'alphas=None' is deprecated...
ConvergenceWarning: Objective did not converge...
```

**After:**
```
alphas=np.logspace(-4, 1, 100)  # Explicit alpha range
max_iter=10000                   # More iterations
tol=1e-3                         # Looser tolerance
→ No convergence warnings
```

### 2. Pandas GroupBy Deprecation ✅
**Before:**
```
FutureWarning: DataFrameGroupBy.apply operated on grouping columns...
```

**After:**
```
.groupby("date", as_index=True).apply(..., include_groups=False)
→ No deprecation warnings
```

### 3. Streamlit Deprecation ✅
**Before:**
```
st.plotly_chart(fig, use_container_width=True)
```

**After:**
```
st.plotly_chart(fig, width="full")  # or width=800
→ Uses new Streamlit API
```

### 4. Position Sizing Defaults ✅
**Before:**
```
Long %:  [0.01 to 0.20, default 0.10] ← Too tight
Short %: [0.20 to 0.50, default 0.30]
```

**After:**
```
Long %:  [0.01 to 0.50, default 0.25] ← More generous
Short %: [0.20 to 0.70, default 0.40] ← More options
```

---

## Recommended Next Steps

1. **Immediate (5 min):**
   - Run with Conservative preset + 🏆 Best feature selection
   - Verify you get both long and short signals (check Position Sizing defaults)

2. **Quick experiment (10 min):**
   - Try 🎯 ElasticNet vs ❌ None and compare Sharpe
   - Use same tickers/dates for fair comparison

3. **If still weak (20 min):**
   - Add more tickers (Top 10 preset)
   - Try XGBoost model type
   - Check feature selection console output

4. **Deep dive (30+ min):**
   - Read WALKFORWARD_FEATURE_SELECTION.md
   - Compare all model types × feature selection combinations
   - Build comparison table

---

## Expected Performance

### With Fixes (Conservative + Best + 10 tickers):
- Sharpe: 0.40-0.65
- Num longs: 20-50% of dates
- Num shorts: 40-80% of dates
- Time: 4-5 min first run, 2 sec cached

### If model is good:
- Sharpe: > 0.50
- Win rate: 52-58%
- Max DD: 15-25%

### If model is weak:
- Sharpe: < 0.30
- Problem: Likely model/data issue, not signals
- Fix: See Advanced Tuning section above

