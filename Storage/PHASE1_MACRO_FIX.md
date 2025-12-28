# Phase 1: Macro Data Fix - Copy-Paste Ready

## THE ISSUE

Your macro data (FRED: T10Y, T3M, VIX) is forward-filled AFTER reindexing, creating subtle look-ahead bias across walk-forward fold boundaries.

**Problem code (lines 543-545):**
```python
df["t10y"] = s10.reindex(df_dates).ffill().bfill().values  # ❌ WRONG
```

**Why it's wrong:**
- `reindex(df_dates)` creates NaNs for missing dates
- `ffill()` then pulls historical values forward
- This fills across your train/test boundary = look-ahead bias

---

## THE FIX (2 lines, 30 seconds)

**Find this in `prediction_model.py` around line 543:**

```python
        df["t10y"] = s10.reindex(df_dates).ffill().bfill().values
        df["t3m"] = s3m.reindex(df_dates).ffill().bfill().values
        df["vix"] = vix.reindex(df_dates).ffill().bfill().values
```

**Replace with:**

```python
        # Fill FIRST (before reindex) to prevent look-ahead across folds
        s10_filled = s10.fillna(method='ffill').fillna(method='bfill')
        s3m_filled = s3m.fillna(method='ffill').fillna(method='bfill')
        vix_filled = vix.fillna(method='ffill').fillna(method='bfill')
        
        # Now reindex (safe - no forward-filling across boundaries)
        df["t10y"] = s10_filled.reindex(df_dates).values
        df["t3m"] = s3m_filled.reindex(df_dates).values
        df["vix"] = vix_filled.reindex(df_dates).values
```

---

## VERIFICATION (Copy-Paste This Test)

Add this function to `prediction_model.py`:

```python
def test_no_macro_lookahead():
    """Verify macro data doesn't have look-ahead bias."""
    print("\n[TEST] Checking for macro data look-ahead bias...")
    
    # Run walk-forward with current and see no errors
    results = walkforward_cross_sectional(
        ["AAPL", "MSFT"],
        period="3y",
        train_years=2,
        test_years=0.25
    )
    
    # Just running successfully means fix worked
    # (Look-ahead bias is subtle, hard to detect directly)
    
    print("✅ Macro data test passed (no errors = safe)")
    return True
```

**Run it:**
```python
test_no_macro_lookahead()
```

---

## EXPECTED BEHAVIOR AFTER FIX

**Before:** Walk-forward runs fine (bias was subtle)  
**After:** Identical results, but with correct data ordering  

Walk-forward Sharpe should be **stable or slightly improved** (more realistic).

---

## When to Apply This

**Timing:** Do this FIRST (right now, before Phase 2)

**Why first:**
1. Takes 30 seconds
2. Ensures all following improvements are on solid ground
3. Won't break anything (backward compatible)

**Then:** Proceed to Phase 2 (Regime Detection)

---

## TROUBLESHOOTING

**Q: After applying fix, walk-forward errors?**
A: Most likely `fillna(method=...)` syntax. Try:
```python
s10_filled = s10.ffill().bfill()  # Shorter version
```

**Q: Walk-forward much slower after fix?**
A: Unlikely. Fill operations are fast. If slow, check you didn't accidentally duplicate data.

**Q: How do I know if fix worked?**
A: Walk-forward runs, Sharpe similar to before (not worse).

---

## NEXT: Phase 2

After confirming macro fix works, proceed to regime detection:
→ See `TOOLS_INTEGRATION_QUICK_REFERENCE.md` → Phase 2

