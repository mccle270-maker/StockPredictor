# Portfolio Walk-Forward Presets Guide

## Overview

The Portfolio Walk-Forward backtest now includes **preset configurations** designed to prevent overfitting while validating whether your model genuinely works in real market conditions.

---

## The Three Presets

### 1. **Conservative (Anti-Overfit)** ← DEFAULT ⭐

**Settings:**
- Training Period: **2 years**
- Test Period: **~38 trading days (~0.15 years)**
- Expected Folds: **~25**

**Best For:**
- Validating if your model actually works (not memorizing)
- Getting high statistical confidence in signal quality
- Detecting overfitting early

**Why This Works:**
- **2 years is enough data** to learn real patterns without noise memorization
- **~38 day test windows** = realistic out-of-sample testing
- **25 folds** = statistically significant (high confidence in results)
- **Low risk:** If Sharpe is good here, your model is probably real

**Signal Interpretation:**
- Sharpe > 1.2 across all folds? **Your model works.** Deploy with confidence.
- Sharpe 0.5-1.2? **Promising but risky.** Refine before live trading.
- Sharpe < 0.5? **Weak signal.** Go back to drawing board.

---

### 2. **Balanced**

**Settings:**
- Training Period: **1.5 years**
- Test Period: **~50 trading days (~0.2 years)**
- Expected Folds: **~18**

**Best For:**
- Medium balance between reliability and data efficiency
- When you have limited historical data
- Quick iteration on model improvements

**Trade-off:**
- Slightly more generous training data
- Fewer folds (18 vs 25), so slightly less statistical power
- Good middle ground if Conservative seems too strict

---

### 3. **Aggressive (More Data)**

**Settings:**
- Training Period: **1 year**
- Test Period: **~25 trading days (~0.1 years)**
- Expected Folds: **~40**

**Best For:**
- Extracting maximum signal from limited data
- When you want maximum folds (highest precision)
- After validating with Conservative preset

**⚠️ Risk:** Shorter training windows can lead to overfitting if not careful.

**When To Use:**
- Only AFTER your model passes Conservative preset
- For fine-tuning, not initial validation
- When exploring parameter sensitivity

---

## How To Use: Step-by-Step

### First Time Validation (No Overfitting):

1. **Select:** "Conservative (Anti-Overfit)" preset ← RECOMMENDED
2. **Run:** Backtest with your current model
3. **Interpret Results:**
   - If **Sharpe > 1.2:** Your model is solid. Can try Aggressive preset for more insight.
   - If **Sharpe 0.5-1.2:** Model needs work. Refine features, add data, improve signals.
   - If **Sharpe < 0.5:** Model is likely overfitted or lacks signal. Start over.

### Iterative Improvement:

1. Keep using **Conservative** as your validation standard
2. Try different:
   - Feature combinations
   - Model types (RF vs XGBoost vs GBRT)
   - Position sizing rules
3. When Conservative gives Sharpe > 0.8, try **Balanced** or **Aggressive** for detail
4. Use Aggressive only after Conservative validates concept

---

## Override Option (Advanced Users)

If you know what you're doing, you can override defaults:

1. Click **"⚡ Override Defaults"** in the Time Windows section
2. Adjust Train/Test periods manually
3. Same anti-overfitting rules apply (don't be too aggressive)

**Safe Override Rules:**
- Training ≥ 1 year (prevents overfitting)
- Test ≥ 20 days (meaningful sample)
- Total folds ≥ 15 (statistical power)

---

## Interpreting Results: Model Quality

### Quality Indicators (From Results Panel)

**Sharpe Ratio:**
| Sharpe | Interpretation | Action |
|--------|---|---|
| > 1.5 | Excellent | Deploy immediately |
| 1.2-1.5 | Strong | Deploy with monitoring |
| 0.5-1.2 | Promising | Refine before deployment |
| < 0.5 | Weak | Fundamental issues, restart |

**Consistency (Sharpe Std Dev):**
- σ < 0.5 → Very stable (best)
- σ 0.5-1.0 → Moderate variation (acceptable)
- σ > 1.0 → Highly volatile (concerning)

**Hit Rate:**
- > 55% → Better than coin flip
- 50-55% → Barely above random
- < 50% → Worse than random (error in model)

**Stability Check (New):**
- ✅ **Stable** (σ < 0.5): Consistent across folds = real signal
- ⚡ **Moderate** (σ < 1.0): Some variation but acceptable
- ⚠️ **Volatile** (σ > 1.0): Inconsistent = likely overfitted

---

## Key Insights: Why These Defaults Prevent Overfitting

### Problem: Bad Models Can Fake Good Sharpe
A model can memorize noise on a single 1-month test window and show amazing Sharpe. But it will fail in real trading.

### Solution: Walk-Forward with Multiple Folds
- **Conservative (25 folds):** Must be good across 25 different 38-day periods
- **Harder to fake:** Overfitting breaks down across time periods
- **Real validation:** If it works in 25 windows, it probably works

### Example:
```
Overfitted Model on Single Window: Sharpe = 3.5 (Fake!)
Same Model on 25 Windows (Conservative): Sharpe = 0.2 (Exposed!)

Real Model on 25 Windows (Conservative): Sharpe = 1.3 (Reliable)
```

---

## Recommended Workflow

### Week 1: Initial Validation
```
1. Build model + features
2. Run Portfolio WF with CONSERVATIVE preset
3. If Sharpe < 1.0, refine features
4. Repeat until Sharpe > 0.8
```

### Week 2: Refinement
```
1. Keep Conservative as baseline
2. Try different model types (RF, XGBoost, GBRT)
3. Use Balanced preset for faster iteration
4. Ensure Conservative still validates
```

### Week 3: Fine-Tuning
```
1. Once Conservative Sharpe > 1.2
2. Try Aggressive preset for maximum folds
3. Validate position sizing rules
4. Ready for paper trading
```

---

## FAQ

**Q: Why 2 years training for Conservative?**
A: Sweet spot between data sufficiency and preventing overfitting. Less = noise, more = stale patterns.

**Q: Can I override to 1 year training?**
A: Only in Balanced/Aggressive presets. Conservative is designed to be strict.

**Q: What if I have less than 5 years of data?**
A: Conservative will auto-adjust down but keep same ratio. Example: 2y data → ~7 folds instead of 25.

**Q: Should I trust a Sharpe of 0.8 with Conservative?**
A: It's borderline. Good enough for paper trading, but monitor carefully before live trading.

**Q: Can I mix presets?**
A: Yes! Use Conservative to validate, then try Aggressive to dig deeper. Just don't trust Aggressive alone.

**Q: What about bad tickers ruining results?**
A: Bad tickers are automatically excluded from P&L calculation (only missing-data rows dropped). Your results reflect only valid data.

---

## Summary

✅ **Use Conservative** for initial validation (prevents overfitting)  
✅ **Check Stability** (Sharpe σ) alongside Sharpe median  
✅ **Override only if** you understand the trade-offs  
✅ **Walk-forward beats** single-window backtests for real validation  
✅ **Multiple folds** = high confidence in model quality  

**Remember:** A model that works in 25 different time windows is real. One that works in one window is probably luck.
