# Production Deployment Checklist

**Project:** Stock Predictor → Production ML Portfolio Engine  
**Target Launch:** Q1 2026  
**Current Phase:** UI Refinement + Specification (✅ COMPLETE)

---

## Phase 1: UI & Documentation ✅ (DONE)

- [x] Fix `detail_universe` naming inconsistency
- [x] Fix ticker dropdown validation + sorting
- [x] Improve price chart rendering (dual traces, styling)
- [x] Add emoji headers to expanders
- [x] Refactor metrics layout (4-column grids)
- [x] Improve Greeks/News display (emoji, fallback)
- [x] Create `.github/copilot-instructions.md` (AI agent guide)
- [x] Create `PORTFOLIO_ENGINE.md` (350+ line spec)
- [x] Create `UPDATE_SUMMARY.md` (change log)

**Result:** ✅ Production-ready dashboard + complete spec

---

## Phase 2: Core Engine Implementation (Week 1)

**Status:** NOT STARTED | **Est. Hours:** 20 | **Blocker:** None

### 2.1 Create `portfolio_engine.py` (NEW)
```python
class PortfolioEngine:
    """Autonomous daily trade executor + memory manager."""
    
    def __init__(self, universe, capital=100000, max_dd=0.20):
        """Initialize portfolio state."""
        # - [ ] Initialize positions dict
        # - [ ] Initialize trade_log list
        # - [ ] Initialize equity_curve
        
    def detect_regime(self, vix, corr, vol_cluster):
        """Return regime dict."""
        # - [ ] Classify VIX bucket (low/med/high)
        # - [ ] Compute kill_switch flags
        
    def size_positions(self, signals, regime):
        """Convert {ticker: pred_ret} → {ticker: size_pct}."""
        # - [ ] Vol-target to 15%
        # - [ ] Enforce position limits
        # - [ ] Check kill-switches
        
    def execute_trades(self, new_positions, prices):
        """Rebalance + log."""
        # - [ ] Compute rebalance trades
        # - [ ] Update equity curve
        # - [ ] Write trade log
        
    def daily_validation(self):
        """Compute rolling Sharpe, hit rate, drift."""
        # - [ ] Compute 60D Sharpe
        # - [ ] Compute 20D hit rate
        # - [ ] KS-test residuals
        
    def should_retrain(self):
        """Check drift signals."""
        # - [ ] KS-test p < 0.05
        # - [ ] Rolling Sharpe < 0.8
        # - [ ] Hit rate decay
        
    def to_json_signals(self):
        """Export for auto_paper_trade.py."""
        # - [ ] Serialize signals
        
    def load_trade_log(self, path):
        """Restore state from CSV."""
        # - [ ] Deserialize trades
        
    def save_trade_log(self, path):
        """Persist trades + equity curve."""
        # - [ ] Write CSV
```

**Deliverable:** `portfolio_engine.py` (400–500 LOC)

### 2.2 Create `regime_features.py` (NEW)
```python
def compute_vix_regime(vix_level) -> str:
    # - [ ] Return 'low' | 'medium' | 'high'
    
def compute_correlation_regime(df, top_n=5) -> float:
    # - [ ] Rolling 20D mean pairwise correlation
    
def compute_vol_clustering(returns, window=20) -> float:
    # - [ ] Current vol / 252D vol ratio
    
def compute_macd_slope(df) -> float:
    # - [ ] 3-day MACD histogram slope
    
def compute_skewness_kurtosis(returns, window=30) -> tuple:
    # - [ ] (skew, excess_kurtosis)
    
def compute_tail_quantiles(returns, window=30) -> tuple:
    # - [ ] (10th pct, 90th pct)
```

**Deliverable:** `regime_features.py` (150–200 LOC)

### 2.3 Update `prediction_model.py`
- [ ] Add regime features to `FEATURE_COLUMNS` (30–50 total)
- [ ] Ensure GBM + cross-sectional features included
- [ ] Test with `build_features_and_target('AAPL')`

### 2.4 Update `app.py`
- [ ] Add "Portfolio Engine" tab (mock status display)
- [ ] Import `PortfolioEngine` class
- [ ] Display regime (VIX bucket, correlation, vol cluster)
- [ ] Show daily position sizes + constraints

**Milestone:** Core engine + regime detection functional

---

## Phase 3: Backtesting (Week 2)

**Status:** NOT STARTED | **Est. Hours:** 15 | **Blocker:** Phase 2

### 3.1 Backtest on 20-Ticker Universe
```bash
python -c "
from prediction_model import walkforward_cross_sectional
from portfolio_engine import PortfolioEngine

# 34+ folds, 1–4 year train, 0–2 year test
results = walkforward_cross_sectional(
    tickers=['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','AMD','INTC','NFLX',
             'JPM','BAC','GS','XOM','CVX','KO','PEP','WMT','COST','DIS'],
    period='5y',
    horizon=3,
    model_type='rf',
    train_years=2,
    test_years=0.5,
)

# Check: Sharpe > 1.0 median, hit_rate 45-55%, max_dd < 20%
"
```

**Checklist:**
- [ ] Generate 34+ folds
- [ ] Compute median Sharpe (target > 1.0)
- [ ] Compute median hit rate (target 45–55%)
- [ ] Compute max drawdown (verify < 20%)
- [ ] Check per-regime Sharpe (all > 0.3)
- [ ] Plot equity curve (smooth vs spiky)
- [ ] Validate trade logging (CSV output)

**Deliverable:** `backtest_results_20_tickers.csv` + `equity_curve.png`

### 3.2 Sensitivity Analysis
- [ ] Test forest depths: 8, 10, 12, 15
- [ ] Test feature counts: 20, 30, 40, 50
- [ ] Test vol targets: 10%, 15%, 20%
- [ ] Document optimal hyperparameters

**Milestone:** Confirmed edge on 20-ticker universe

---

## Phase 4: Production Deployment (Week 3–4)

**Status:** NOT STARTED | **Est. Hours:** 10 | **Blocker:** Phase 3

### 4.1 Integration
- [ ] Connect `PortfolioEngine` to `app.py` ("Portfolio Engine" tab)
- [ ] Update `runner.py` to call `engine.execute_trades()`
- [ ] Verify `auto_paper_trade.py` reads signals from engine
- [ ] Test end-to-end: Prediction → Signal → Trade → Log

### 4.2 Paper Trading (2 Weeks)
- [ ] Deploy to paper trading account (Alpaca)
- [ ] Monitor daily status (regime, positions, P/L)
- [ ] Track drawdown (verify < 20%)
- [ ] Verify trade logging (all trades logged with metadata)
- [ ] Check drift detection (rolling Sharpe, hit rate)
- [ ] Manual retraining if KS-test p < 0.05

**Success Criteria:**
- ✅ 0 crashes (robust error handling)
- ✅ Avg daily P/L: Positive or break-even (not < -0.5%)
- ✅ Sharpe (rolling 60D) > 0.8
- ✅ Drawdown < 10% (vs 20% limit)
- ✅ All trades logged + retrievable

### 4.3 Documentation Update
- [ ] Add deployment instructions to `PORTFOLIO_ENGINE.md`
- [ ] Document observed Sharpe/Drawdown/Hit-rate (2-week paper)
- [ ] Add troubleshooting section
- [ ] Commit to git with tag `v1.0-paper-trading`

**Milestone:** 2-week paper trading complete; ready for live (optional)

---

## Phase 5: Ongoing Operations

**Status:** NOT STARTED | **Est. Hours:** 5/week | **Blocker:** Phase 4

### 5.1 Daily (8:35am ET)
- [ ] Run model predictions for 20 tickers
- [ ] Execute trades via `PortfolioEngine`
- [ ] Monitor intraday P/L + drawdown
- [ ] Check kill-switches (VIX > 35 → liquidate)

### 5.2 Weekly (Fridays 5pm ET)
- [ ] Aggregate P/L by signal type
- [ ] Compute rolling Sharpe (60D, 252D)
- [ ] Check hit rate decay
- [ ] Identify underperforming regimes
- [ ] Backtest parameter sensitivity (optional)

### 5.3 Monthly
- [ ] Full backtesting run (validation)
- [ ] Compare Sharpe to prior month
- [ ] Update feature importance (prune weak features)
- [ ] Document lessons learned

### 5.4 Quarterly
- [ ] Comprehensive review (3-month live results)
- [ ] Backtest on 18-month rolling window
- [ ] Optimize hyperparameters (forest depth, vol target)
- [ ] Plan next iteration (Q2/Q3 improvements)

---

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|-----------|-------|
| VIX spike (>35) | Auto kill-switch in `detect_regime()` | Engine |
| Correlation spike | Reduce to top-3 holdings | Engine |
| Drawdown >20% | Full liquidation + 100% cash | Engine |
| Sharpe collapse | Pause trading; retrain model | ML Eng |
| Data pipeline failure | Fallback to Stooq / raw Yahoo | Data Eng |
| Rate limit (yfinance) | Cache + stagger requests | Data Eng |
| Trade execution failure | Log error; alert admin; retry | Trader |

---

## Success Metrics (End of Q1 2026)

### Portfolio Engine
- [ ] **Sharpe Ratio:** 1.0+ (annualized)
- [ ] **Hit Rate:** 45–55% (direction accuracy)
- [ ] **Max Drawdown:** < 20% (per spec)
- [ ] **Daily P/L:** Positive on 55%+ of trading days
- [ ] **Uptime:** 99.5% (negligible downtime)

### Operations
- [ ] **Trade Logging:** 100% of trades recorded with metadata
- [ ] **Drift Detection:** KS-test + Sharpe deterioration flagged in <1 day
- [ ] **Retraining:** Auto-retrain triggered within 1 week if needed
- [ ] **Documentation:** All procedures documented + tested

### Code Quality
- [ ] **Test Coverage:** 80%+ (unit + integration tests)
- [ ] **Code Review:** 0 critical bugs in production (2-week observation)
- [ ] **Git History:** Clean commits; tags for each phase
- [ ] **Reproducibility:** All results reproducible; no data leakage

---

## Sign-Off

- [ ] **ML Engineer:** Approves model + backtesting results
- [ ] **Software Engineer:** Approves code quality + testing
- [ ] **Product Manager:** Approves spec + user-facing changes
- [ ] **Risk Manager:** Approves risk limits + kill-switches

---

**Target Completion:** 2026-03-31  
**Status:** ON TRACK (Phase 1 complete, Phase 2 starting)  
**Last Updated:** 2025-12-27
