# Portfolio Walk-Forward Performance Optimization Guide

## Why Is It Slow?

The Portfolio Walk-Forward backtest runs multiple folds, each training a new model and calculating metrics. This is **computationally expensive** but **necessary for real validation**.

```
Time Breakdown (rough estimates):
├─ Data downloading (first run): ~20-30 seconds
├─ Feature engineering: ~10-20 seconds per ticker
├─ Model training per fold: ~5-10 seconds per fold
├─ Calculations: ~5 seconds per fold
└─ TOTAL: 2-5 minutes for first run (then cached 30 min)
```

---

## Quick Fixes (Fastest First)

### 1. **REDUCE TICKERS** (Biggest Impact)
```
❌ 10 tickers × 25 folds = 250 model trainings
✅ 3 tickers × 25 folds = 75 model trainings → 3x faster
```

**Use these quick presets:**
- **Quick Start:** `AAPL,MSFT,NVDA` (3 tickers, ~2 min)
- **Small Cap Check:** `SPY,QQQ,IWM` (3 tickers, ~2 min)
- **Tech Heavy:** `AAPL,MSFT,NVDA,AMD,INTC` (5 tickers, ~3 min)

### 2. **USE BALANCED PRESET** (2nd Fastest)
```
Conservative: ~25 folds, ~4-5 min
Balanced: ~18 folds, ~3-4 min ← Try this
Aggressive: ~40 folds, ~6-7 min
```

### 3. **DISABLE VIX FILTER**
```
With VIX filter: +30 seconds (fetches VIX data)
Without VIX: Save 30 seconds
```

### 4. **SHORTER TRAINING PERIOD** (Override)
```
2 years (Conservative): ~4-5 min
1.5 years (Balanced): ~3-4 min
1 year (Aggressive): ~5-6 min (but fewer folds)
```

---

## Why First Run Is Slow, Then Fast

### First Run (2-5 minutes)
```
1. Download historical data for all tickers (API calls)
   └─ ~20-30 sec

2. Calculate features for entire history
   └─ ~10-20 sec

3. Build panel (combine all tickers)
   └─ ~5 sec

4. Train N folds (N=25 for Conservative)
   └─ ~25 × (5-10 sec) = 2-4 minutes

TOTAL: 2-5 minutes
```

### Second+ Runs (2-3 seconds)
```
Streamlit caches results for 30 minutes.
If you use same settings → Instant results!

Results expire and need refresh only if:
├─ You change tickers
├─ You change preset
├─ You change model type
├─ You change position sizing
└─ 30 minutes pass
```

---

## Recommended Approach

### For Initial Validation (Fast)
```
1. Use Quick Start: AAPL,MSFT,NVDA
2. Use Balanced preset
3. Disable VIX filter
4. Run → 2-3 minutes
5. Check results
```

### For Full Validation (Thorough)
```
1. Start with 3-5 tickers
2. Use Conservative preset
3. Keep VIX filter on
4. Run → 3-5 minutes
5. If good, add more tickers
6. Use Balanced for speed on iteration
```

### For Fine-Tuning (Fast)
```
1. Results cached 30 minutes
2. Run multiple times with same settings → instant
3. Only re-run when changing settings
4. Balanced preset + 5 tickers = 3-4 min refresh
```

---

## What's Happening During Wait

**If you see a long wait, the system is:**

1. **Downloading data** (first 30 sec) - API calls to get prices
2. **Calculating features** (next 20 sec) - RSI, MACD, Bollinger Bands, etc.
3. **Building models** (rest of time) - Training RandomForest/XGBoost on each fold

**Progress bar jumps to 25% after setup, stays there during model training** (this is normal).

---

## Optimization Checklist

For fastest results:

```
[ ] Use 3-5 tickers (not 10+)
[ ] Use Balanced preset (not Conservative)
[ ] Disable VIX filter (if not needed)
[ ] Close other apps (free up CPU)
[ ] Wait for first run to complete (then cached)
[ ] Don't change settings between runs (stay cached)
```

---

## Cache Behavior

### Cache Time-To-Live: 30 Minutes

```
Time 0:00 → Run backtest with AAPL,MSFT,NVDA
            Takes: 2-3 minutes
            
Time 0:05 → Run same backtest
            Takes: 2 seconds (cached!)
            
Time 0:10 → Change to AAPL,MSFT,NVDA,AMD
            Takes: 2-3 minutes (new config)
            
Time 0:15 → Back to AAPL,MSFT,NVDA
            Takes: 2 seconds (still cached!)
            
Time 30:05 → Run AAPL,MSFT,NVDA again
            Takes: 2-3 minutes (cache expired)
```

### How to Clear Cache (if needed)

Click **"🔄 Clear Results"** button in the Portfolio WF tab.

---

## Expected Performance

### By Ticker Count

| Tickers | Folds | Time (First Run) | Time (Cached) | Recommendation |
|---------|-------|-----------------|---------------|---|
| 3 | 25 | 2-3 min | 2 sec | ⭐ Start here |
| 5 | 25 | 3-4 min | 2 sec | Good balance |
| 7 | 25 | 4-5 min | 2 sec | Thorough |
| 10 | 25 | 5-7 min | 2 sec | Slow but complete |

### By Preset

| Preset | Folds | Time (First Run) | Confidence |
|--------|-------|-----------------|-----------|
| Conservative | ~25 | 4-5 min | Very high |
| Balanced | ~18 | 3-4 min | High |
| Aggressive | ~40 | 6-7 min | Highest |

---

## Pro Tips

✅ **Validate once per day**  
- Run once in morning, cached all day
- Results refresh daily

✅ **Use Balanced for iteration**  
- Faster: ~3-4 min first run
- Still reliable: 18 folds
- Good for feature testing

✅ **Conservative for deployment decisions**  
- More folds (25) = higher confidence
- Takes 4-5 min but worth it
- Use before going live

✅ **Small universe for experimentation**  
- 3 tickers: 2-3 min
- Test ideas fast
- Expand later

✅ **Close other apps**  
- CPU-intensive computation
- Every % of CPU helps
- Saves 10-20 seconds

---

## FAQ

**Q: Why is first run so slow?**  
A: Downloads data, calculates 100+ features, trains 25 models. Normal & necessary.

**Q: How long should it take?**  
A: 2-5 min (3 tickers) to 5-7 min (10 tickers). First run only.

**Q: Can I speed it up?**  
A: Reduce tickers (biggest impact) → use Balanced preset → disable VIX filter.

**Q: Why is second run so fast?**  
A: Streamlit caches results for 30 minutes. No computation needed.

**Q: What if it's still slow after caching?**  
A: You probably changed a setting. Cache invalidated, need fresh run.

**Q: How long is cache valid?**  
A: 30 minutes. Same settings = cached. Different settings = new run.

**Q: Can I clear cache?**  
A: Yes - click "🔄 Clear Results" button.

**Q: What's the minimum time?**  
A: ~2 minutes (3 tickers, Balanced preset, no VIX filter).

**Q: What if I need faster?**  
A: Use 2-3 tickers + Balanced + no VIX = ~2 min minimum.

---

## Summary

| Action | Time Saved | Difficulty |
|--------|-----------|-----------|
| Use 3 tickers instead of 10 | ~3-4 min | Easy |
| Use Balanced instead of Conservative | ~1 min | Easy |
| Disable VIX filter | ~30 sec | Easy |
| Close other apps | ~10-20 sec | Easy |
| **Recommended combo** | **~4-5 min saved** | **Easy** |

**Result:** First run down from 5-7 min → 2-3 min  
Subsequent runs: 2 seconds (cached)
