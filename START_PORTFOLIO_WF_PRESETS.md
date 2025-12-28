# START HERE: Portfolio WF Presets Quick Start 🚀

## 30-Second Overview

You now have **3 preset button options** for Portfolio Walk-Forward testing:

| Preset | Default? | Training | Test | Folds | Use When |
|--------|----------|----------|------|-------|----------|
| **Conservative** | ✅ YES | 2 years | ~38d | ~25 | **Validate model (prevent overfitting)** |
| **Balanced** | ❌ | 1.5 years | ~50d | ~18 | After Conservative works |
| **Aggressive** | ❌ | 1 year | ~25d | ~40 | Fine-tuning only |

**Plus:** You get a quality card after each backtest telling you if your model is real or overfitted.

---

## 60-Second Tutorial

### Step 1: Open Portfolio WF
Go to **Portfolio WF** tab in the app

### Step 2: See Presets
```
📋 Window Preset: [Conservative (Anti-Overfit) ▼]
🎯 2y train, ~38d test → ~25 folds
```
Conservative is already selected ✅

### Step 3: Run It
1. Enter tickers: `AAPL,NVDA,MSFT`
2. Click `▶️ Run Backtest`
3. Wait for results

### Step 4: Read Results
After backtest, see two new cards:

```
🎯 Model Quality Check          📊 Stability Check
├─ ✅ EXCELLENT/⚡ GOOD/⏸️ WEAK   ├─ ✅ Stable/⚡ Moderate/⚠️ Volatile
├─ Sharpe interpretation         ├─ Is performance consistent?
└─ Recommendation: Deploy/Refine └─ Is this overfitted?
```

### Step 5: Decide
- **✅ EXCELLENT + Stable?** → Model works! Paper trading ready.
- **⚡ GOOD + Moderate?** → Promise but needs refinement.
- **⏸️ WEAK or Volatile?** → Model has issues, redesign features.

**Done.** No complex settings to understand.

---

## Why This Matters

### The Problem It Solves

**Before:**
- You set 3 months training + 1 month test
- Backtest shows Sharpe = 5.0 (amazing!)
- Go live, model fails in 2 weeks 💥

**After:**
- Conservative preset forces 2 years training
- Test on ~25 different 38-day windows
- Must work in all 25 to pass
- Can't fake, only real models survive ✅

### The Magic: Why 25 Folds?

```
Good models work consistently across time:
├─ Fold 1: ✓ Sharpe 1.2
├─ Fold 2: ✓ Sharpe 1.3
├─ ... (25 total)
└─ Average: Sharpe 1.2 = Proven winner ✅

Bad models only work sometimes:
├─ Fold 1: ✓ Sharpe 2.0 (lucky)
├─ Fold 2: ✗ Sharpe 0.1 (breaks)
├─ Fold 3: ✗ Sharpe -0.5 (fails)
└─ Average: Sharpe 0.5 = Overfitted ❌
```

---

## Quality Card Interpretation

### Model Quality Check

| Sharpe | Meaning | Your Action |
|--------|---------|------------|
| > 1.2 | ✅ EXCELLENT | Deploy to paper trading |
| 0.5-1.2 | ⚡ GOOD | Refine features, then retest |
| < 0.5 | ⏸️ WEAK | Go back to drawing board |

### Stability Check

| Sharpe σ | Meaning | Your Action |
|----------|---------|------------|
| < 0.5 | ✅ Stable | Real signal, consistent |
| 0.5-1.0 | ⚡ Moderate | Acceptable variation |
| > 1.0 | ⚠️ Volatile | Likely overfitted, redesign |

---

## Common Scenarios

### Scenario 1: First Time
```
1. Run with Conservative (default)
2. See ✅ EXCELLENT + ✅ Stable
3. Read: "Model works in real conditions"
4. → Ready for paper trading!
```

### Scenario 2: Decent But Not Great
```
1. Run with Conservative
2. See ⚡ GOOD + ⚡ Moderate  
3. Read: "Model shows promise"
4. → Refine features, try again
```

### Scenario 3: Bad Model
```
1. Run with Conservative
2. See ⏸️ WEAK or ⚠️ Volatile
3. Read: "Model overfitted or lacks signal"
4. → Redesign, add features, start over
```

---

## Pro Tips

✅ **Always start with Conservative** — It's the real test  
✅ **Check both Quality AND Stability** — One number isn't enough  
✅ **Keep Conservative as validation standard** — Don't switch presets midway  
✅ **Only try Balanced/Aggressive AFTER Conservative validates** — Know what you're doing  
✅ **Sharpe > 1.2 is "ready to trade"** — That's the bar  
✅ **σ < 0.5 means real signal** — Consistency matters  

---

## Override Option (Advanced)

Click `⚡ Override Defaults` if you want custom settings.

**But know what you're doing:**
- Less training = easier to overfit
- Fewer folds = lower confidence
- Only override if you understand trade-offs

Safe override ranges:
```
✅ Train: 1-2 years (less = risky)
✅ Test: 25-50 days (less = unreliable)
✅ Folds: 15+ (less = no statistical power)
```

---

## FAQ

**Q: Is Conservative always best?**  
A: Yes, for validation. Once it passes, try others for detail.

**Q: What if I have < 5 years data?**  
A: System auto-adjusts, fewer folds than expected, still works.

**Q: Can I override Conservative?**  
A: Yes, but know the overfit risk increases.

**Q: What's a good Sharpe?**  
A: > 1.2 is excellent, 0.8-1.2 is good, < 0.5 is weak.

**Q: What does σ = 1.2 mean?**  
A: Volatile performance across folds → Likely overfitted.

---

## Deployment Checklist

Before going live:

```
[ ] Conservative Sharpe > 1.2
[ ] Stability σ < 0.5
[ ] Hit Rate > 55%
[ ] Recent Sharpe ≈ Median (no trend)
[ ] Understand what the model does
```

If all pass → Paper trading ready!

---

## Key Files

| File | What | Read Time |
|------|------|-----------|
| **This file** | Quick start | 2 min |
| `PORTFOLIO_WF_QUICK_REFERENCE.md` | One-page cheat sheet | 5 min |
| `PORTFOLIO_WF_PRESETS_GUIDE.md` | Complete technical guide | 15 min |
| `PORTFOLIO_WF_VISUAL_GUIDE.md` | UI before/after, examples | 10 min |

---

## The Bottom Line

✅ **Default:** Conservative (prevents overfitting)  
✅ **Simple:** One button to select  
✅ **Safe:** 25 folds validate across time  
✅ **Smart:** Quality card tells you if model works  
✅ **Flexible:** Can override if needed  

**Result:** You can now validate models with high confidence they'll work in real trading.

---

## Next Step

1. **Read:** `PORTFOLIO_WF_QUICK_REFERENCE.md` (if you want more detail)
2. **Try:** Open Portfolio WF, run backtest with default preset
3. **Check:** Model Quality Assessment card
4. **Iterate:** Improve model while keeping Conservative as validation standard

**You're ready to go.** 🚀
