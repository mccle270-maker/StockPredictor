# Fix Applied: Missing Macro Data Issue

## Problem
When running predictions on AAPL, NVDA in your Streamlit app, you got:
```
ERROR: No usable history for AAPL after trying periods=['5y', '3y', '2y', '1y', '6mo', '3mo']. 
Last error=Only 0 usable rows for AAPL with period=3mo
```

## Root Cause
The `build_features_and_target()` function was:
1. Building features (price, technical, GBM, regime, ARIMA, etc.)
2. Trying to fetch macro data (VIX, T10Y, term_spread, market return)
3. Trying to fetch fundamental data (P/E, P/B, market cap)
4. Calling `dropna()` on ALL features
5. Since macro/fundamental data was missing (NaN), `dropna()` removed ALL rows

## Solution Applied

### 1. Make macro data optional (lines 1300-1314)
```python
# Try to get macro data, but don't fail if unavailable
try:
    macro_df = get_macro_df(symbol="^GSPC", period=per)
    hist = hist.join(macro_df, how="left")
except Exception as e:
    print(f"[build_features_and_target] Warning: Could not fetch macro data: {e}")
```

### 2. Make fundamental data optional (lines 1316-1321)
```python
# Try to get fundamental data, but don't fail if unavailable
try:
    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v
except Exception as e:
    print(f"[build_features_and_target] Warning: Could not fetch fundamental data: {e}")
```

### 3. Filter features intelligently (lines 1328-1340)
```python
# Use only columns that actually exist AND have data
# First, only include feature columns that exist
feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
# Then, filter to only columns with < 50% NaN
data_quality = hist[feat_cols_available].isna().sum() / len(hist)
feat_cols_available = [c for c in feat_cols_available if data_quality[c] < 0.5]

# Fill remaining NaNs with forward fill, then backward fill
hist[feat_cols] = hist[feat_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
```

### 4. Applied same fix to other functions
- `track_predictions()` (line 1954)
- `build_panel_features_and_target()` (line 1379)
- `predict_next_for_ticker()` - removed hardcoded feature lists

## Result

✅ **Predictions now work with missing data**

```
🔍 Testing AAPL...
   ✅ SUCCESS!
      • Next-day return: -0.07%
      • Prediction price: $273.63
      • Probability up: 54.5%
      • Num features: 44

🔍 Testing NVDA...
   ✅ SUCCESS!
      • Next-day return: 0.04%
      • Prediction price: $188.69
      • Probability up: 55.3%
      • Num features: 49
```

## Key Changes in Files
- `prediction_model.py` (lines 1300-1340, 1379-1410, 1954-1980, 1787-1793)

## Graceful Degradation
- ✅ If macro data unavailable → Uses 44-49 features instead of 106
- ✅ If fundamental data unavailable → Gracefully skips P/E, P/B, market cap
- ✅ Remaining NaNs filled with forward/backward fill and zeros
- ✅ No more "0 usable rows" errors

## Testing
Run `python test_fix.py` to verify both AAPL and NVDA predict successfully
