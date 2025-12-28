# Portfolio WF Presets - Quick Reference Card

## One-Page Cheat Sheet

### The 3 Presets at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. CONSERVATIVE (Anti-Overfit) ⭐ DEFAULT                           │
├─────────────────────────────────────────────────────────────────────┤
│ Train: 2 years    │ Test: ~38 days    │ Folds: ~25                  │
│ ✅ Best for validation (prevents overfitting)                       │
│ Use first, always. If Sharpe > 1.2 here, model is real.            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. BALANCED                                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Train: 1.5 years  │ Test: ~50 days    │ Folds: ~18                  │
│ ⚡ Good for faster iteration (after Conservative validates)        │
│ Use after model passes Conservative test.                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. AGGRESSIVE (More Data)                                            │
├─────────────────────────────────────────────────────────────────────┤
│ Train: 1 year     │ Test: ~25 days    │ Folds: ~40                  │
│ 🚀 Fine-tuning only (easier to overfit with 1y training)          │
│ Use only AFTER Conservative shows Sharpe > 1.2                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How To Use

### Default (Recommended)
```
1. Open Portfolio WF
2. See "Conservative (Anti-Overfit)" ← Already selected!
3. Click "▶️ Run Backtest"
4. Done - no config needed
```

### Override (Advanced)
```
1. Click "⚡ Override Defaults"
2. Adjust sliders
3. Run backtest
```

---

## Interpreting Results

### Quality Card

| Sharpe | Status | Action |
|--------|--------|--------|
| > 1.2 | ✅ EXCELLENT | Deploy! Model works. |
| 0.5-1.2 | ⚡ GOOD | Refine. Shows promise. |
| < 0.5 | ⏸️ WEAK | Redesign. Overfitted likely. |

### Stability Card

| Sharpe σ | Status | Action |
|----------|--------|--------|
| < 0.5 | ✅ Stable | Real signal. Consistent. |
| 0.5-1.0 | ⚡ Moderate | Acceptable variation. |
| > 1.0 | ⚠️ Volatile | Overfitted likely. Refine. |

---

## Decision Tree

```
                    Run Backtest (Conservative)
                            ↓
                    Check Sharpe Median
                        ↙   ↓   ↘
                    < 0.5  0.5-1.2  > 1.2
                      ↓       ↓       ↓
                   WEAK    GOOD   EXCELLENT
                      ↓       ↓       ↓
                  Redesign  Refine  Deploy!
                  features  more    Ready for
                  Add data   folds   paper trade
                           Try Balanced
                           Then Aggressive
```

---

## The Why (Anti-Overfitting Math)

```
❌ BAD SETUP (Easy to overfit):
   Train: 3 months
   Test: 1 month
   Folds: 5
   → Model can fake good results → Fails in real trading

✅ GOOD SETUP (Conservative):
   Train: 2 years
   Test: ~38 days
   Folds: ~25
   → Must work in 25 windows → Real patterns only
   → Overfitting breaks down after 5 windows
```

---

## Deployment Criteria (Checklist)

```
Before going live, verify:

[ ] Conservative Sharpe > 1.2
[ ] Stability σ < 0.5 (consistent)
[ ] Hit Rate > 55% (better than random)
[ ] Recent Sharpe ≈ Median Sharpe (no degradation)
[ ] Understand what the model does

If any fail:
→ Not ready yet, refine features
```

---

## Common Scenarios

### "My Sharpe is 0.8, can I trade?"
```
Answer: Paper trading yes, live trading no.
→ Run again with same model
→ If stays 0.7-0.9, it's real
→ If jumps around, it's overfitted
→ Try Balanced preset for more insight
```

### "Conservative gave Sharpe 1.5, then I ran Aggressive and got 0.5"
```
Answer: Model is overfitted.
→ The 1y training in Aggressive couldn't generalize
→ Stick with Conservative (it's the real test)
→ Refine your features/model
```

### "I have a bad ticker, how do I exclude it?"
```
Answer: It's automatic.
→ Bad tickers are dropped from P&L calc
→ No need to manually remove
→ Results only include valid data
```

### "How many folds do I need?"
```
Rule of thumb:
→ < 10 folds: Too few (lucky/unlucky)
→ 15+ folds: Good (statistical power)
→ 25+ folds: Excellent (Conservative default)
→ 40+ folds: Overkill (Aggressive)
```

---

## Settings Interpretation

### Horizon
```
[1 day] = Next day prediction (fastest feedback)
[3 day] = 3-day prediction (more stable)
[5 day] = 5-day prediction (most stable, slower signal)
```

### Training Period
```
0.5y = Too short, overfitting risk
1y = Minimum acceptable
1.5y = Good balance
2y = Conservative, less overfitting (DEFAULT)
3y+ = Overkill, stale patterns
```

### Test Period
```
0.05y (~13 days) = Too short
0.1y (~25 days) = Minimum (Aggressive)
0.15y (~38 days) = Balanced (Conservative)
0.2y (~50 days) = Slow (fewer folds)
> 0.25y = Way too slow
```

---

## Preset Comparison

```
Feature          Conservative    Balanced      Aggressive
─────────────────────────────────────────────
Training         2 years        1.5 years     1 year
Testing          ~38 days       ~50 days      ~25 days
Folds            ~25            ~18           ~40
Overfit Risk     LOW ✅         MEDIUM ⚡     HIGH ⚠️
Statistical Power HIGH ✅        GOOD ⚡       VERY HIGH 🚀
Use Case         VALIDATION     Iteration    Fine-tuning
Signal Quality   MOST REAL      GOOD         DETAILED
When To Use      FIRST          AFTER        ONLY AFTER
                 ALWAYS         CONSERVATIVE VALIDATES
                 DEFAULT        VALIDATES
```

---

## Key Takeaways

1. **Use Conservative by Default** - Prevents overfitting
2. **~25 Folds = High Confidence** - Hard to fake across time periods
3. **Sharpe > 1.2 + Stable σ = Ready** - Deploy with confidence
4. **Multiple Presets** - Validate with Conservative, explore with others
5. **Override Optional** - For advanced users who understand trade-offs

---

## Pro Tips

✅ Always start with Conservative  
✅ Keep Conservative as validation standard  
✅ Check Stability (σ) alongside Sharpe  
✅ Use Balanced to iterate faster (after Conservative validates)  
✅ Never deploy on Aggressive results alone  
✅ Sharpe > 1.2 is "good", > 1.5 is "excellent"  
✅ Hit rate > 55% beats random  
✅ σ < 0.5 = real signal, > 1.0 = overfitted  

---

## File Guide (For More Info)

```
PORTFOLIO_WF_PRESETS_GUIDE.md
  → Complete guide, why presets work, anti-overfitting

PORTFOLIO_WF_VISUAL_GUIDE.md
  → UI before/after, how to use, interpretation

PORTFOLIO_WF_PRESETS_COMPLETE.md
  → Full implementation details, checklist, FAQ
```

---

## TL;DR

```
🎯 What: 3 presets to prevent overfitting
🎯 Which: Conservative (default), Balanced, Aggressive
🎯 Why: 25 folds = hard to fake, real signal validation
🎯 How: Select preset dropdown, hit Run, read quality assessment
🎯 Go Live: When Conservative Sharpe > 1.2 + Stable σ < 0.5
```

**That's it. You're ready to go.**
