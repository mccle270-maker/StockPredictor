# Production ML Portfolio Engine Specification

**Role:** Daily autonomous equity portfolio signal generation, risk management, and trade tracking for short-horizon (3-trading-day) prediction across a 20-ticker blue-chip universe.

**Operating Period:** Mon-Fri, 8:35am ET execution via Alpaca paper trading.

---

## 1. Universe & Configuration

### Tradeable Universe (20 tickers)
```
Tech:       AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, NFLX
Financials: JPM, BAC, GS
Energy:     XOM, CVX
Consumer:   KO, PEP, WMT, COST, DIS
```

### Execution Parameters
- **Model:** Random Forest (RF) multiclass: {BUY, HOLD, SELL}
- **Prediction Horizon:** 3 trading days (returns are aggregated)
- **Training:** Rolling 1–4 years, walk-forward backtest: 34+ folds, 0–2 year test windows
- **Features:** 30–50 normalized per-asset z-scores; pruned by RF feature importance > 0.01; tree depth 10–15

---

## 2. Core Model Architecture

### Data Pipeline
```
Historical Data (yfinance)
    ↓
Raw Features (technical, fundamental, macro)
    ↓
Normalization (z-score by asset within 20D window)
    ↓
Feature Selection (RF importance > 0.01, cap 50 features)
    ↓
Target: 3-day forward return (%) → discretized into {-1, 0, +1}
    ↓
Train/test split: 80/20 (time-series safe)
    ↓
Model: RandomForestClassifier(n_estimators=300, max_depth=12, ...)
```

### Feature Engineering (Required Set)

#### Regime Detection (Mandatory)
**Purpose:** Kill-switch / throttle trading in unstable regimes.

- **VIX Regime** (3 buckets):
  - Low: VIX < 15 → relaxed position sizing
  - Medium: 15 ≤ VIX ≤ 35 → normal sizing
  - High: VIX > 35 → de-risk to cash
  
- **VIX Futures Term Structure** (5Y–1M slope):
  - Contango (normal) → long bias acceptable
  - Backwardation (stressed) → reduce exposure
  
- **Volatility Clustering** (EWMA variance ratio):
  - Current 20-day realized vol ÷ 252-day realized vol
  - Ratio > 1.3 → elevated regime, tighten stops
  
- **Market Sync** (rolling 20-day pairwise correlation, top 5 universe weights):
  - Mean correlation > 0.8 → hedged portfolios perform worse, reduce long
  - Mean correlation < 0.6 → diversification premium, can increase leverage

#### Prediction Features (Target Set)

**Technical Momentum** (all lagged 1 day to prevent look-ahead bias):
- RSI(14): over-bought (>70) / over-sold (<30) signals
- MACD histogram slope: 3-day discrete derivative
- Bollinger Band position: (Close - Lower) ÷ (Upper - Lower) normalized 0–1
- On-Balance Volume (OBV) momentum: 5-day change
- Momentum(12/26): (Close - Close[12]) - (Close - Close[26])

**Higher Moments** (recent 30-day rolling window):
- Skewness: 3rd central moment (tail risk indicator)
- Excess kurtosis: 4th moment (extreme event probability)
- Tail quantiles: 10% and 90% of 3-day forward returns (VaR proxy)

**Cross-Sectional Relative Strength** (vs. SPX):
- Predicted return z-score (mean=0, std=1 across universe)
- Volatility z-score (ranking within peer group)
- Vol-adjusted return z-score (Sharpe ratio on predicted signal)

#### Macro Features (FRED API)
- Market return 1D (SPX proxy): pct change
- VIX (current level + 5D change)
- 10Y Treasury yield (level + slope vs 3M)
- Term spread (10Y - 3M), critical regime indicator

---

## 3. Portfolio Construction

### Position Sizing Framework

#### Long Sizing (BUY signals)
- Base: 1% per ticker if signal strength > 0.005 (0.5% predicted return)
- Scale: Vol-adjusted to target **15% portfolio volatility**
  - Position size = 1% × (0.15 / predicted_vol_20d) capped at 20%
  - Minimum: 0.1% (avoid dead weight)

#### Short Sizing (SELL signals)
- Base: 1% per ticker if signal strength < -0.005
- Cap: 25% max portfolio short (relative hedge; always unwind first)
- Risk limit: Net short delta < -15% of capital

#### Cash Holding
- Default: 100% cash if:
  - VIX outside (15, 35) bounds → awaiting regime clarity
  - Rolling Sharpe < 1.0 (past 60 days) → reduced edge
  - Correlation > 0.8 across top holdings → concentration risk
  - Drawdown > 15% (approaching limit) → defensive posture

### Execution Rules
- **Order type:** MARKET (Alpaca paper)
- **Timing:** Next open (8:35am ET) after signal generation
- **Rebalance:** Daily if signal changes > ±0.02 (2% threshold to avoid churn)
- **Slippage:** Assume 2 bps spread + 3 bps slippage, 0 fees (paper)

---

## 4. Risk Management (Hard Rules)

### Volatility Targets & Controls
- **Portfolio vol target:** 15% annualized (252-day rolling realized)
- **Volatility floor:** If realized < 5%, increase sizing by 1.5x (up to limits)
- **Volatility ceiling:** If realized > 25%, reduce by 50% (hard cap)

### Drawdown Management
- **Max allowed:** 20% peak-to-trough
- **Trigger recovery mode:** If DD > 15%, move to 50% cash
- **Liquidate on DD > 20%:** Full de-risk; re-evaluate regime

### Kill-Switches (Immediate De-Risk)
1. **VIX > 35:** Liquidate all longs, hold 10% short for tail hedge
2. **VIX < 15:** Reduce longs by 25% (complacency risk)
3. **Correlation > 0.8:** Cut position count to top-3 highest conviction signals
4. **Single-asset concentration > 15%:** Trim to 15% exactly
5. **Sharpe deterioration:** If rolling 20D Sharpe < 0.5, move to 100% cash

### Position Limits
- **Per-ticker max:** 15% of portfolio
- **Sector max:** Tech 40%, Financials 20%, Other 40%
- **Net long/short:** Long ≤ 80%, Short ≤ 25%
- **Single-fold exposure:** No ticker > 5% from same model training fold

---

## 5. Validation & Backtesting

### Backtesting Regime
- **Train:** 1–4 rolling years (select via grid search)
- **Test:** 0–2 year walk-forward folds (252 trading days per fold)
- **Overlap:** Purged (no test data in training set)
- **Minimum folds:** 34 (ensures robust statistical inference)

### Performance Metrics (Per Fold)
| Metric | Target | Action |
|--------|--------|--------|
| Sharpe ratio | > 1.0 | Proceed; track drift |
| Hit rate | 45–55% | Expected; below 40% signals model decay |
| Max drawdown | < 20% | Unwind if breached |
| Win/Loss ratio | > 0.8 | Weighted by signal size, not count |
| KS-test p-value | > 0.05 | Reject if residuals non-normal (drift) |

### Rolling Validation (Daily)
- **Sharpe (60D):** Compute each day; if < 1.0, move to 50% cash
- **Hit rate (20D):** Check sign(prediction) == sign(actual_return); trigger retraining if < 40%
- **Drift detection:** Kolmogorov-Smirnov test on residuals (training vs. recent); p < 0.05 = retrain
- **Per-regime Sharpe:** Bin by VIX bucket, compute Sharpe separately; skip regime if Sharpe < 0.3

---

## 6. Memory & Logging

### Trade Log Schema
```json
{
  "date": "2025-12-27",
  "signal_id": "AAPL_2025_12_27_BUY",
  "ticker": "AAPL",
  "direction": "BUY",
  "signal_strength": 0.0234,  // predicted return %
  "signal_confidence": 0.62,  // prob(correct) from RF
  "vix_regime": "medium",
  "correlation_regime": "normal",
  "position_size_pct": 3.2,
  "entry_price": 245.67,
  "entry_time": "09:31:00 ET",
  "sharpe_60d": 1.24,
  "portfolio_vol_pct": 14.8,
  "max_dd_pct": -8.3,
  "exit_date": "2025-12-30",
  "exit_price": 248.92,
  "pnl_dollars": 103.2,
  "pnl_pct": 1.29,
  "holding_days": 3,
  "status": "closed"
}
```

### Equity Curve Tracking
- **Daily:** Compute mark-to-market portfolio value
- **Rolling metrics:**
  - Cumulative return (%)
  - Sharpe ratio (60D, 252D)
  - Max drawdown (peak-to-trough)
  - Win rate (% of closed trades > 0)
  - Profit factor (sum wins ÷ |sum losses|)

### Learning Loop
1. **Weekly:** Aggregate closed trades; compute per-signal-type win rate
2. **Monthly:** Retrain model if drift detected (KS-test p < 0.05 or Sharpe < 0.8)
3. **Quarterly:** Backtest on 18-month rolling window; compare Sharpe to prior
4. **Auto-adjust:**
   - If 3-month Sharpe < 1.0, reduce position sizes by 20%
   - If hit rate < 40%, reduce signal_threshold by 20 bps
   - If same regime persists > 20 days w/ Sharpe < 0.5, skip that regime

---

## 7. Daily Operating Procedure

### Pre-Market (8:00am ET)
```
1. Fetch prior day OHLCV + macro data (FRED, VIX)
2. Compute regime (VIX bucket, correlation, vol clustering)
3. Run model prediction for each ticker
4. Discretize into {BUY, HOLD, SELL} with confidence scores
5. Check kill-switches; adjust position sizing if needed
6. Generate signals JSON
```

### Market Open (8:35am ET)
```
1. Load signals.json
2. Execute rebalancing trades (market orders)
3. Log entry prices + execution time
4. Update portfolio state (positions, cash, unrealized P/L)
```

### Intraday (Continuous)
```
1. Track P/L hourly (mark-to-market)
2. Monitor kill-switches (VIX > 35 → liquidate immediately)
3. Early exit if drawdown > 15% or signal reversal
```

### Post-Market (4:30pm ET+)
```
1. Close out any partial fills
2. Compute daily Sharpe, drawdown, win rate
3. Log results to trade_log.csv
4. Check rolling validation metrics
5. If needed, flag for retraining
```

### Weekly (Fridays 5pm ET)
```
1. Aggregate P/L, Sharpe by signal type
2. Identify underperforming regimes
3. Backtest parameter sensitivity (forest depth, feature count)
4. Update confidence thresholds if needed
```

---

## 8. System State & Initial Conditions

### Current Status (Dec 27, 2025)
| Metric | Value |
|--------|-------|
| **Capital** | $100,000 |
| **VIX** | 13.6 (below 15 lower bound) |
| **Regime** | Low volatility, complacency risk |
| **Drawdown** | 0% |
| **Sharpe** | N/A (no trades) |
| **Position** | 100% cash (waiting for regime entry) |
| **Prior Trades** | 0 |

### Target Initialization
- **Model:** Train on 3 years historical (2022–12-27)
- **Walk-forward:** 34+ folds, rolling 2-year test windows
- **First production trade:** Dec 30 (next trading day) or later once VIX enters (15, 35)

---

## 9. Integration with Existing Codebase

### Key Files & Functions

| Module | Function | Purpose |
|--------|----------|---------|
| `prediction_model.py` | `predict_next_for_ticker()` | Single-ticker 3-day return prediction |
| `prediction_model.py` | `walkforward_cross_sectional()` | Multi-ticker walk-forward backtest (34+ folds) |
| `app.py` → `_build_display_df()` | Convert predictions to UI table | Generate daily signal list |
| `app.py` → `build_signals_from_pred_df()` | Predictions → JSON signals | Serialize for auto-trader |
| `auto_paper_trade.py` | `main()` | Execute trades via Alpaca |
| `data_fetch.py` | `get_history_cached()` | Historical OHLCV (yfinance fallback) |
| `stock_screener.py` | `screen_stocks()` | Pre-filter universe (unused; direct model) |

### New Functions to Implement

#### `portfolio_engine.py` (NEW)
```python
class PortfolioEngine:
    """Autonomous daily trade executor + memory manager."""
    
    def __init__(self, universe, capital=100000, max_dd=0.20, sharpe_target=1.0):
        self.universe = universe
        self.capital = capital
        self.max_dd = max_dd
        self.positions = {}  # {ticker: size_pct}
        self.trade_log = []  # list of trade dicts
        self.equity_curve = [capital]
        
    def detect_regime(self, vix, corr, vol_cluster):
        """Return regime dict: {vix_bucket, corr_regime, vol_regime, kill_switch}."""
        
    def size_positions(self, signals, regime):
        """Convert {ticker: pred_ret} → {ticker: size_pct} respecting vol targets + limits."""
        
    def execute_trades(self, new_positions, prices):
        """Rebalance to new_positions; log trades; update equity curve."""
        
    def daily_validation(self):
        """Compute rolling Sharpe, hit rate, check drift; return {sharpe, hit_rate, drift_flag}."""
        
    def should_retrain(self):
        """Check KS-test, rolling Sharpe, hit rate decay; return bool."""
        
    def to_json_signals(self):
        """Export current signals dict for auto_paper_trade.py."""
        
    def load_trade_log(self, path):
        """Deserialize trade_log from CSV; restore state."""
        
    def save_trade_log(self, path):
        """Serialize all trades + equity curve to CSV."""
```

#### `regime_features.py` (NEW)
```python
def compute_vix_regime(vix_level) -> str:
    """Return 'low' | 'medium' | 'high'."""
    
def compute_correlation_regime(df, top_n=5) -> float:
    """Rolling 20-day mean pairwise correlation of top N weighted assets."""
    
def compute_vol_clustering(returns, window=20) -> float:
    """Current vol / 252-day vol ratio."""
    
def compute_macd_slope(df, fast=12, slow=26) -> float:
    """MACD histogram slope over 3 days."""
    
def compute_skewness_kurtosis(returns, window=30) -> tuple:
    """(skew, excess_kurtosis) of recent returns."""
    
def compute_tail_quantiles(returns, window=30) -> tuple:
    """(10th pct, 90th pct) of 3-day forward returns."""
```

---

## 10. Checklist & Next Steps

### Implementation (Week 1)
- [ ] Build `PortfolioEngine` class + memory persistence
- [ ] Add regime detection (VIX, correlation, vol clustering)
- [ ] Implement position sizing (vol targeting, limit enforcement)
- [ ] Add validation loop (Sharpe, hit rate, drift detection)
- [ ] Update `app.py` to display daily status (regime, positions, P/L)
- [ ] Integrate with `runner.py` scheduler

### Backtesting (Week 2)
- [ ] Run `walkforward_cross_sectional()` on 20-ticker universe, 34+ folds
- [ ] Verify Sharpe > 1.0 median; hit rate 45–55%
- [ ] Test kill-switches (VIX extremes, correlation spikes)
- [ ] Sensitivity analysis: forest depth, feature count, vol target

### Production (Week 3–4)
- [ ] Deploy `portfolio_engine` to app.py + auto_paper_trade.py
- [ ] Paper trade for 2 weeks (observe regime transitions, drawdowns)
- [ ] Verify trade logging + equity curve tracking
- [ ] Update documentation with live results
- [ ] Commit to version control (git)

---

**Last Updated:** 2025-12-27  
**Author:** Production ML Portfolio Engineering Team  
**Status:** Specification → Implementation Ready
