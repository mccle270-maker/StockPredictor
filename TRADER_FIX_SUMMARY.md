# Auto Paper Trading Fix Summary

## Problem
The auto_paper_trade.py script crashed with:
```
alpaca.common.exceptions.APIError: {"code":42210000,"message":"asset \"LYC.AX\" not found"}
```

**Root Cause**: `LYC.AX` is an Australian stock (ticker ends in `.AX`) that is **not tradeable** on Alpaca's paper trading API. The script tried to submit an order for this non-US stock without validation.

---

## Solution Implemented

### 1. **app.py Changes** (Signal Generation Layer)
Added pre-filtering at the signal generation stage to prevent non-US stocks from being included:

#### New Validation Function (app.py ~line 423)
```python
def is_us_tradeable_symbol(ticker: str) -> bool:
    """
    Check if symbol is tradeable on Alpaca (US stocks only).
    Non-US symbols contain periods (e.g., LYC.AX for Australian stocks).
    """
    ticker_clean = str(ticker).upper().strip()
    # Filter out non-US markets (have periods like LSE.L, TSE, AX, etc.)
    if "." in ticker_clean:
        # Common non-US exchanges
        non_us_markers = [".AX", ".L", ".TO", ".V", ".NZ", ".AS", ".KL", ".SG", ".HK"]
        if any(ticker_clean.endswith(marker) for marker in non_us_markers):
            return False
    return True
```

#### Updated build_signals_from_pred_df() (app.py ~line 447)
```python
for _, row in pred_df.iterrows():
    tk = str(row.get("ticker", "")).upper().strip()
    if not tk:
        continue
    
    # Filter out non-US tradeable symbols
    if not is_us_tradeable_symbol(tk):
        print(f"{tk}: Non-US market symbol, skipping (not supported by Alpaca paper trading)")
        continue
```

**Impact**: Non-US stocks are excluded from signals.json, so the auto-trader never tries to trade them.

---

### 2. **auto_paper_trade.py Changes** (Execution Layer)
Added error handling for order submission to gracefully skip stocks that Alpaca cannot trade:

#### Updated Order Submission (auto_paper_trade.py ~line 618)
```python
# Try to submit order with error handling for non-tradeable assets
try:
    submitted = trade_client.submit_order(order_data=order)
except Exception as e:
    error_msg = str(e)
    if "not found" in error_msg.lower():
        print(f"{symbol}: Asset not found on Alpaca (likely non-US market) -> skipping")
    else:
        print(f"{symbol}: Order submission failed: {e}")
    continue
```

**Impact**: If a non-US stock somehow gets through (or if there's an API error), the trader logs it and continues instead of crashing.

---

## What Gets Filtered

### Non-US Stock Symbols (Excluded from signals.json)
These symbols will now be **skipped before trading**:
- `.AX` - Australian Exchange (e.g., LYC.AX)
- `.L` - London Stock Exchange (e.g., GLDRSH.L)
- `.TO` - Toronto Stock Exchange (e.g., SHOP.TO)
- `.V` - Venture Exchange (e.g., SYR.V)
- `.NZ` - New Zealand Exchange
- `.AS` - Amsterdam Stock Exchange
- `.KL` - Kuala Lumpur Stock Exchange
- `.SG` - Singapore Exchange
- `.HK` - Hong Kong Stock Exchange

### US Stocks (Included - Trade as Normal)
- AAPL, MSFT, NVDA (no period = US stocks)
- DUK, EXC (US utilities, no period)
- Any US-listed stock without a period in the ticker

---

## Testing the Fix

### To verify the fix works:
```bash
cd /Users/jakobmccleary/Desktop/Stock\ Predictor
source tf-env/bin/activate

# Run the auto-trader (should skip any non-US symbols)
python3 auto_paper_trade.py
```

**Expected Output**:
- Non-US stocks: `LYC.AX: Non-US market symbol, skipping (not supported by Alpaca paper trading)`
- US stocks: Successfully submitted orders with order IDs

### To verify signals.json filtering:
```bash
python3 << 'EOF'
import json
with open('signals.json', 'r') as f:
    signals = json.load(f)
    
for ticker in signals.keys():
    if '.' in ticker:
        print(f"WARNING: Non-US ticker in signals: {ticker}")
    else:
        print(f"OK: {ticker}")
EOF
```

All tickers should print `OK` (no period in ticker).

---

## Dual-Layer Protection

This fix implements **two levels of defense**:

| Layer | Location | Function | Benefit |
|-------|----------|----------|---------|
| **Prevention** | app.py (signals generation) | Filter before trading signals are created | Cleanest solution - stops problem at source |
| **Fallback** | auto_paper_trade.py (order submission) | Catch and skip if asset not found | Handles unexpected API errors gracefully |

If the first layer misses a symbol, the second layer catches it.

---

## Impact Summary

| Component | Before | After |
|-----------|--------|-------|
| **Auto Trader Crash** | YES (on non-US stocks) | NO |
| **Signals.json** | Contains non-US tickers | Only contains US-tradeable tickers |
| **Execution Flow** | Crashes → script stops | Logs warning → continues to next symbol |
| **Alpaca API Errors** | Fatal | Handled gracefully |

---

## Related Files Modified
1. `/Users/jakobmccleary/Desktop/Stock Predictor/app.py`
   - Added `is_us_tradeable_symbol()` function (line ~423)
   - Updated `build_signals_from_pred_df()` to filter non-US stocks (line ~447)

2. `/Users/jakobmccleary/Desktop/Stock Predictor/auto_paper_trade.py`
   - Added try-catch around `trade_client.submit_order()` (line ~618)
   - Added error logging for non-tradeable assets

---

## Next Steps

1. **Test the fix**: Run `python3 auto_paper_trade.py` with current signals.json
2. **Verify signals.json**: Check that no tickers contain a period
3. **Monitor trading**: Watch for any remaining "Asset not found" errors in logs
4. **Update screener**: If you're generating LYC.AX predictions, exclude it from your stock screener

---

## Related Issues
- Original Error: `alpaca.common.exceptions.APIError: asset "LYC.AX" not found`
- Affected Tickers: LYC.AX (Australian), and potentially other non-US symbols
- Date Fixed: 2025-12-29
