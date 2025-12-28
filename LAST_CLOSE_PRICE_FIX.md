# Last Close Price Bug Fix

**Status**: ✅ FIXED  
**Date**: December 28, 2025  
**Commit**: 48229cf

## The Problem

Your Streamlit app was showing an incorrect **Last Close** price that didn't match:
- The actual current price from yfinance
- The prices shown in your charts/graphs

**Example for NVDA:**
```
❌ App displayed: $174.14 (WRONG)
✅ Actual price: $190.53 (CORRECT)
✅ Graph showed: $190.53 (CORRECT)
```

## Root Cause

The bug was in `build_features_and_target()` function (Line 1388 of prediction_model.py).

### How the data flows:

1. **Load history**: `hist = yf.download(...)` → 502 rows (all available data)
2. **Create target**: `hist["ftarget_ret_horizon_ahead"] = raw_target.shift(-horizon)` → last ~2 rows get NaN
3. **Filter for training**: `df = hist[cols_needed].dropna()` → 501 rows (removes NaN targets)
4. **Get last price**: ❌ `last_close = hist.loc[df.index[-1], "Close"]` → WRONG!

### The Problem:

When we do `.dropna()` on df, we lose the most recent rows (because the target is lagged forward by 1 day). So `df.index[-1]` points to **2025-12-24**, not **2025-12-26** (today).

When we then do `hist.loc[df.index[-1], "Close"]`, we're getting the close price from 2025-12-24, which is 1-2 days old!

## The Fix

Capture the actual last close **BEFORE** the dropna() operation:

```python
# BEFORE dropna() - hist still has all raw data
actual_last_close = hist["Close"].iloc[-1]  # 2025-12-26 close price
actual_last_date = hist.index[-1]           # 2025-12-26 date

cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
df = hist[cols_needed].dropna().copy()      # Training data (501 rows)

# ... later ...

# Use the actual most recent close price
last_close = float(actual_last_close)        # ✅ NOW CORRECT
```

## What Changed

**File**: `prediction_model.py`  
**Lines**: 1357-1397

### Before (WRONG):
```python
cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
df = hist[cols_needed].dropna().copy()

print("hist rows:", len(hist), "range:", hist.index.min(), "-", hist.index.max())
print("df rows:", len(df), "range:", df.index.min(), "-", df.index.max())

# ... validation ...

last_row = df.iloc[-1]
last_row_features = last_row[feat_cols].values
last_close = hist.loc[df.index[-1], "Close"]  # ❌ GETS OLD PRICE
last_vol_20d = last_row["vol_20d"]
```

### After (CORRECT):
```python
# IMPORTANT: Get the actual last close BEFORE dropna
# This is the most recent price in the raw data
actual_last_close = hist["Close"].iloc[-1]
actual_last_date = hist.index[-1]

cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
df = hist[cols_needed].dropna().copy()

print("hist rows:", len(hist), "range:", hist.index.min(), "-", hist.index.max())
print("df rows:", len(df), "range:", df.index.min(), "-", df.index.max())

# ... validation ...

last_row = df.iloc[-1]
last_row_features = last_row[feat_cols].values
# Use the actual most recent close price from raw history, not df (which is dropna'd)
last_close = float(actual_last_close)  # ✅ NOW GETS CURRENT PRICE
last_vol_20d = last_row["vol_20d"]
```

## Verification

**Test**: NVDA prediction with 2-year history

```
✅ Actual Last Close (from yfinance): $190.53
✅ Predicted Last Close (from app):   $190.53
✅ Match: True ✓
```

## Impact

### What This Bug Affected:
- ❌ **Last Close metric display** (off by 1-2 days)

### What It Did NOT Affect:
- ✅ Predictions (calculated correctly)
- ✅ Graphs (using raw data correctly)
- ✅ Backtesting (uses df, not last_close)
- ✅ Trading signals (uses pred_price calculated from correct base)

The bug was **purely a display issue** in the metrics card. The actual predictions and signals were calculated correctly.

## Testing After Fix

To verify the fix is working:

```python
from prediction_model import predict_next_for_ticker
import yfinance as yf

# Get actual current price
ticker_data = yf.download('NVDA', period='5d', progress=False)
actual_last_close = float(ticker_data['Close'].iloc[-1])

# Get prediction
pred = predict_next_for_ticker('NVDA', period='2y')

# Check match
assert abs(actual_last_close - pred['last_close']) < 0.01, "Prices don't match!"
print(f"✅ Last Close matches: ${pred['last_close']:.2f}")
```

## Git History

```
48229cf - Fix: Use actual last close from raw history instead of dropna'd df
```

---

**Result**: Your app now shows the correct current price across all UI elements! 🎉
