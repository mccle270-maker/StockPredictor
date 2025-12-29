# Quick Fix Guide: Non-US Stock Trading Error

## The Problem
```
alpaca.common.exceptions.APIError: asset "LYC.AX" not found
```

You can't trade non-US stocks on Alpaca paper trading (LYC.AX is Australian, DUK is US).

## The Solution (Already Applied ✅)

### File 1: `app.py` - Filter signals BEFORE trading
- Added function: `is_us_tradeable_symbol(ticker)` checks for periods in ticker
- Updated: `build_signals_from_pred_df()` skips non-US symbols
- Result: signals.json only contains US-tradeable stocks

### File 2: `auto_paper_trade.py` - Handle errors during trading
- Wrapped: `trade_client.submit_order()` in try-catch
- Result: If error occurs, logs "Asset not found" and continues

## Test It
```bash
cd /Users/jakobmccleary/Desktop/Stock\ Predictor
source tf-env/bin/activate
python3 auto_paper_trade.py
```

**Good output:**
```
2025-12-29T18:31:50 DUK BUY -> 123abc
2025-12-29T18:31:51 EXC BUY -> 456def
```

**Old bad output:**
```
alpaca.common.exceptions.APIError: asset "LYC.AX" not found
[CRASH]
```

## What Gets Blocked (Examples)
```
LYC.AX        ❌ Australian Stock Exchange (.AX)
GLDRSH.L      ❌ London Stock Exchange (.L)
SHOP.TO       ❌ Toronto Stock Exchange (.TO)
```

## What Gets Traded (Examples)
```
AAPL          ✅ US nasdaq
DUK           ✅ US NYSE
NVDA          ✅ US nasdaq
```

## Files Changed
1. `app.py` - Lines ~423 and ~447
2. `auto_paper_trade.py` - Lines ~618-625

Both changes are already applied. You're all set! 🎯
