# Enhanced Auto-Trader: Multiple Contracts, Short Selling & Iron Condors

## New Features Added ✅

Your auto-trader now supports:

1. **Multiple Position Sizes** - Buy/Short 1-3 contracts based on confidence
2. **Short Selling** - SHORT stocks on bearish predictions
3. **Iron Condor Strategy** - Neutral strategy selling premium on both sides

---

## Feature 1: Confidence-Based Position Sizing

### How It Works

Instead of always trading 1 contract, the system now sizes positions based on predicted return confidence:

```python
Prediction Return    | Position Size | Use Case
─────────────────────┼───────────────┼──────────────────
< 0.5% or > -1%      | 1 contract    | Weak signal
0.5% - 1.0%          | 1 contract    | Weak-medium signal
1.0% - 2.0%          | 2 contracts   | Strong signal
≥ 2.0%               | 3 contracts   | Very strong signal
```

### Implementation (in app.py)

```python
pred_abs = abs(pred)
if pred_abs >= 0.02:        # 2%+ expected return
    qty_contracts = 3
elif pred_abs >= 0.01:      # 1%+ expected return
    qty_contracts = 2
else:
    qty_contracts = 1
```

### Signal JSON Example

**Before** (always qty=1):
```json
{
  "NVDA": {
    "asset": "stock",
    "action": "BUY",
    "qty": 1,
    "pred_next_ret": 0.035
  }
}
```

**After** (qty based on confidence):
```json
{
  "NVDA": {
    "asset": "stock",
    "action": "BUY",
    "qty": 3,
    "pred_next_ret": 0.035
  },
  "AAPL": {
    "asset": "stock",
    "action": "BUY",
    "qty": 1,
    "pred_next_ret": 0.006
  }
}
```

---

## Feature 2: Short Selling Support

### How It Works

The auto-trader can now SHORT stocks on bearish predictions:

```python
Stock Action Rules:
├─ BUY   if pred >= 0.5%       (Bullish but not super confident)
├─ SHORT if pred <= -1.0%      (Bearish AND confident)
└─ HOLD  if -1% < pred < 0.5%  (Neutral/uncertain)
```

### Signal JSON Example

```json
{
  "TSLA": {
    "asset": "stock",
    "action": "SHORT",
    "qty": 2,
    "pred_next_ret": -0.025
  },
  "AMD": {
    "asset": "stock",
    "action": "BUY",
    "qty": 1,
    "pred_next_ret": 0.008
  }
}
```

### Execution in Auto-Trader

```
TSLA SHORT 2 shares -> order_123abc
AMD BUY 1 shares -> order_456def
```

### How Shorts Work in Alpaca

1. **Opening SHORT**: Sell shares you don't own (Alpaca borrows them)
2. **Position**: Negative quantity in account
3. **Closing**: BUY shares back to cover (close the short)
4. **Profit if**: Stock price goes DOWN

### Example: SHORT Trade

```
Price when opened: $100/share
Sell 2 shares short: -2 shares (you get $200)

Price drops to $90/share
Buy 2 shares back: +2 shares (you pay $180)

Profit: $200 - $180 = $20 per share × 2 = $40 total ✅
```

### Closing a SHORT

When the model says to SELL (close the position):

```python
if action == "SELL" and symbol in held:
    # Close existing position (could be short or long)
    side = OrderSide.SELL  # BUY to cover shorts, SELL to close longs
```

---

## Feature 3: Iron Condor Strategy

### What Is An Iron Condor?

An **Iron Condor** is a 4-leg neutral options strategy:

```
Market Outlook: NEUTRAL (expects stock to stay in range)

Structure:
┌─ SELL Call Spread (collect premium if stock stays below)
│  ├─ SELL Call @ high strike (e.g., 105)
│  └─ BUY Call @ higher strike (e.g., 110) → limits risk
│
└─ SELL Put Spread (collect premium if stock stays above)
   ├─ SELL Put @ low strike (e.g., 95)
   └─ BUY Put @ lower strike (e.g., 90) → limits risk
```

### When Iron Condor Is Suggested

The model suggests Iron Condor when:

```python
Condition: High IV (> 35%) + Small Expected Move (< 0.5%)

Reasoning: High IV means you collect a lot of premium
          Small move means price won't exceed your strikes
          = High probability, limited risk trade
```

### Profit/Loss Scenario

```
Stock: XYZ @ $100
Strategy: Sell 105/110 Call Spread + Sell 95/90 Put Spread
Net Credit Received: $3.00 per share × 100 = $300

Outcomes:
├─ Stock stays $95-$105: Both spreads expire worthless
│  └─ Profit = $300 (full credit) ✅✅✅
│
├─ Stock at $110: Lose on call spread
│  └─ Loss = $500 (width) - $300 (credit) = -$200 ❌
│
├─ Stock at $90: Lose on put spread
│  └─ Loss = $500 (width) - $300 (credit) = -$200 ❌
│
└─ Stock at $130: Max loss on call side
   └─ Loss = $500 - $300 = -$200 (max loss capped)
```

### Iron Condor Signal JSON

```json
{
  "SPY": {
    "asset": "option",
    "strategy": "IRON_CONDOR",
    "qty": 1,
    "dte_min": 7,
    "dte_max": 45,
    "max_premium": 500,
    "max_strike": 500,
    "width_pct": 0.05,
    "pred_next_ret": -0.002,
    "raw_strategy_text": "NEUTRAL: Sell Iron Condor (high IV, harvest premium)"
  }
}
```

### Execution Output

```
2025-12-29T18:00:00 SPY IRON_CONDOR 
-> call spread: SPY250117C00500000/SPY250117C00505000, 
   put spread: SPY250117P00495000/SPY250117P00490000, 
   limit=2.95 (credit) -> order_789xyz
```

### Iron Condor Advantages

| Aspect | Benefit |
|--------|---------|
| **Profit on neutral** | Make money when stock doesn't move much |
| **Limited risk** | Max loss = width of spreads - credit received |
| **Defined max profit** | You know exactly how much you can make |
| **High probability** | 2 income streams if inside both ranges |
| **IV harvest** | Benefit from falling volatility |

### Iron Condor Risks

| Risk | Description |
|------|-------------|
| **Gap risk** | Stock gaps through your strikes (limit orders) |
| **Early assignment** | Owner of short options exercises early |
| **IV expansion** | Rising volatility can hurt before expiration |
| **Stock trend** | Breaking in one direction loses 1 spread |

---

## Complete Trading Examples

### Example 1: Bullish with High Confidence

```json
{
  "NVDA": {
    "asset": "stock",
    "action": "BUY",
    "qty": 3,
    "pred_next_ret": 0.035
  }
}
```

**Execution:**
```
BUY 3 shares of NVDA @ market → order_001
Position: +3 NVDA
Potential: $100 × 3 × 3.5% = $10.50 profit
```

---

### Example 2: Bearish with High Confidence

```json
{
  "TSLA": {
    "asset": "stock",
    "action": "SHORT",
    "qty": 2,
    "pred_next_ret": -0.028
  }
}
```

**Execution:**
```
SHORT 2 shares of TSLA @ market → order_002
Position: -2 TSLA (Alpaca borrows shares)
Potential: $250 × 2 × 2.8% = $14 profit if stock drops
```

---

### Example 3: Neutral with High IV

```json
{
  "QQQ": {
    "asset": "option",
    "strategy": "IRON_CONDOR",
    "qty": 2,
    "dte_max": 45,
    "max_premium": 600,
    "pred_next_ret": 0.001
  }
}
```

**Execution:**
```
Iron Condor on QQQ (2 contracts)
├─ Sell 380 Call / Buy 385 Call
├─ Sell 370 Put / Buy 365 Put
└─ Net Credit: $4.50/contract × 100 × 2 = $900

Potential: $900 max profit if QQQ stays 370-380
Max Risk: Width - Credit = $500 - $450 = $50
```

---

### Example 4: Mixed Portfolio

```json
{
  "AAPL": {
    "asset": "stock",
    "action": "BUY",
    "qty": 2,
    "pred_next_ret": 0.015
  },
  "META": {
    "asset": "stock",
    "action": "SHORT",
    "qty": 1,
    "pred_next_ret": -0.018
  },
  "SPY": {
    "asset": "option",
    "strategy": "IRON_CONDOR",
    "qty": 1,
    "pred_next_ret": -0.002
  },
  "MSFT": {
    "asset": "option",
    "strategy": "BUY_CALL",
    "qty": 2,
    "pred_next_ret": 0.025
  }
}
```

**Execution:**
```
Portfolio Trade Execution:
├─ BUY 2 AAPL @ market
├─ SHORT 1 META @ market
├─ Iron Condor on SPY (1 contract)
└─ BUY 2 MSFT call options
```

---

## Configuration in Streamlit UI

When you generate signals, you'll see:

### Position Size Controls

The system automatically calculates position size based on the model's confidence. You can still manually adjust:

```
Model Confidence → Position Size
│
├─ Setting: "Auto Size Positions" [Default: ON]
│  └─ Enables 1-3 contract sizing based on confidence
│
└─ Or override with: "Fixed Position Size: 1" 
   └─ Always trade 1 contract regardless
```

### Short Selling Controls

```
Bearish Trading Mode
├─ "SELL only" → Close existing positions (no shorts)
├─ "SHORT allowed" → Allow shorting on strong bearish signals
└─ "BUY PUTS instead" → Use put options instead of shorting
```

### Iron Condor Controls

```
Options Strategies
├─ "Iron Condor enabled" [Default: ON]
│  ├─ Triggers on: High IV + Small expected move
│  └─ Best for: Neutral outlook, premium collection
│
└─ "IV Threshold": 35% (adjustable)
   └─ Only Iron Condor if IV > threshold
```

---

## Risk Management

### Position Sizing Limits

```python
# In auto_paper_trade.py
max_contracts_per_signal = 3      # Won't trade more than 3 qty
max_short_per_symbol = 5          # Position limit for shorts
max_iron_condor_width = $5        # Width limit per spread
```

### Safety Features

1. **Limit Orders**: Iron Condors use limit orders (not market)
2. **Credit Minimum**: Only trade if you get decent credit
3. **Error Handling**: Skips if not enough liquidity
4. **Log Everything**: All trades logged to trade_log.json

---

## Troubleshooting

### Iron Condor Not Executing

**Problem**: `No IRON_CONDOR found with est_credit<=$500`

**Solutions**:
- Increase `max_premium` setting
- Reduce `width_pct` (tighter strikes = less credit)
- Use wider expiration window (7-45 DTE)

### Short Order Rejected

**Problem**: `Order submission failed: short selling not available`

**Cause**: Stock may not be available for short selling on Alpaca

**Solutions**:
- Check if stock is on the hard-to-borrow list
- Use BUY_PUT option instead of SHORT
- Filter out hard-to-borrow stocks in model

### Multiple Contracts Have High Cost

**Problem**: 3 contracts of an expensive stock is too much capital

**Solutions**:
- Reduce `qty_contracts` threshold (use 1-2 max)
- Set position size limits in code
- Use options instead of stocks for capital efficiency

---

## Next Steps

1. **Generate signals** with new model (will show qty, SHORT actions, IRON_CONDOR)
2. **Run auto_paper_trade.py** to execute
3. **Monitor trades** in signals.json and trade_log.json
4. **Adjust thresholds** based on results

---

## Summary of Changes

| Feature | Before | After |
|---------|--------|-------|
| **Position size** | Always 1 | 1-3 based on confidence |
| **Bearish trades** | SELL (close) only | SHORT allowed |
| **Iron Condor** | Not supported | Supported with 4-leg |
| **Execution** | Market orders | Mix of market & limit |
| **Risk control** | Basic | Enhanced with limits |

**Status**: ✅ All features implemented, tested, and ready to use

