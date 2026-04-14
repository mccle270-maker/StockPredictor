"""
Bear Market Stress Test (2022)
==============================

Tests model performance during the 2022 bear market when:
- S&P 500 dropped ~20%
- NASDAQ dropped ~33%
- Most tech stocks got crushed

This is the TRUE test of model robustness.

Includes:
- Transaction costs (10 bps per trade)
- More tickers (10 stocks)
- Walk-forward with monthly retraining

Run: python run_bear_market_test.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
warnings.filterwarnings("ignore")

from src.core.models import make_model, select_features_elasticnet
from src.core.features import build_all_features, add_gbm_features
from src.core.metrics import compute_sharpe, compute_drawdown
from src.data.market import get_price_history
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS

# Test configuration - 2022 Bear Market
TEST_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Big tech
    "META", "TSLA", "AMD", "NFLX", "CRM",      # More volatile tech
]

# Date range for 2022 test
# Train: 2019-2021 (3 years before 2022)
# Test: Full year 2022
TRAIN_START = "2019-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2022-12-31"

HORIZON = 5
THRESHOLD = 0.002

# Transaction costs
COST_PER_TRADE_BPS = 10  # 10 basis points = 0.1%


def get_data_for_period(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and prepare data for a specific date range."""
    # Fetch extra data for feature warmup
    warmup_start = pd.to_datetime(start) - pd.Timedelta(days=400)
    
    hist = get_price_history(ticker, period="max", interval="1d")
    if hist is None or not isinstance(hist, pd.DataFrame):
        return None
    
    # Make index timezone-naive for consistent comparisons
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    # Filter to date range (with warmup for features)
    hist = hist.loc[hist.index >= warmup_start]
    
    # Build features
    hist = build_all_features(hist)
    if hist is None or not isinstance(hist, pd.DataFrame):
        return None
    
    hist = add_gbm_features(hist, horizons=(1, HORIZON))
    
    # Build target
    target_col = f"ftarget_ret_{HORIZON}d_ahead"
    hist[target_col] = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    
    # Winsorize
    lower = hist[target_col].quantile(0.01)
    upper = hist[target_col].quantile(0.99)
    hist[target_col] = hist[target_col].clip(lower=lower, upper=upper)
    
    return hist


def prepare_features(hist: pd.DataFrame, horizon: int = 5) -> tuple:
    """Extract features and target from prepared data."""
    target_col = f"ftarget_ret_{horizon}d_ahead"
    
    # Collect features
    feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    feat_cols = list(dict.fromkeys(feat_cols))
    
    # Quality filter
    nan_rates = hist[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    return hist, feat_cols, target_col


def apply_transaction_costs(pnl: np.ndarray, positions: np.ndarray, cost_bps: float = 10) -> np.ndarray:
    """Apply transaction costs based on position changes."""
    cost_per_trade = cost_bps / 10000  # Convert bps to decimal
    
    # Position changes (trades)
    pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    
    # Costs
    costs = pos_changes * cost_per_trade
    
    return pnl - costs


def run_single_period_test(
    ticker: str,
    train_start: str,
    train_end: str, 
    test_start: str,
    test_end: str,
    model_type: str = "xgb",
    use_elasticnet: bool = True,
    cost_bps: float = 10,
) -> dict:
    """Run backtest for a single train/test period."""
    
    # Get data
    hist = get_data_for_period(ticker, train_start, test_end)
    if hist is None:
        return {"error": "No data"}
    
    hist, feat_cols, target_col = prepare_features(hist, HORIZON)
    
    # Split by date
    train_mask = (hist.index >= train_start) & (hist.index <= train_end)
    test_mask = (hist.index >= test_start) & (hist.index <= test_end)
    
    train_df = hist.loc[train_mask].copy()
    test_df = hist.loc[test_mask].copy()
    
    # Drop NaN targets
    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])
    
    if len(train_df) < 100 or len(test_df) < 20:
        return {"error": f"Insufficient data: train={len(train_df)}, test={len(test_df)}"}
    
    X_train = train_df[feat_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feat_cols].values
    y_test = test_df[target_col].values
    
    n_features_orig = len(feat_cols)
    
    # ElasticNet feature selection
    if use_elasticnet:
        try:
            X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                X_train, y_train,
                feature_names=feat_cols,
                dates=train_df.index,
                horizon=HORIZON,
                n_splits=5,
                l1_ratio=0.5,
                min_features=10,
            )
            X_train = X_train_sel
            X_test = X_test[:, sel_mask]
            n_features = len(sel_names)
        except Exception:
            n_features = n_features_orig
    else:
        n_features = n_features_orig
    
    # Train model
    try:
        model = make_model(model_type=model_type, random_state=42, use_optimized=True)
        model.fit(X_train, y_train)
    except Exception as e:
        return {"error": str(e)}
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Trading simulation
    # Long if pred > threshold, Short if pred < -threshold, else flat
    positions = np.where(y_pred > THRESHOLD, 1, np.where(y_pred < -THRESHOLD, -1, 0))
    
    # Raw PnL (before costs)
    pnl_raw = positions * y_test
    
    # Apply transaction costs
    pnl_net = apply_transaction_costs(pnl_raw, positions, cost_bps)
    
    # Buy & Hold for comparison
    bh_returns = test_df["Close"].pct_change().dropna().values
    if len(bh_returns) < len(y_test):
        bh_returns = np.concatenate([[0], bh_returns])
    bh_returns = bh_returns[:len(y_test)]
    
    # Metrics
    accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean()) * 100
    sharpe_raw = compute_sharpe(pnl_raw)
    sharpe_net = compute_sharpe(pnl_net)
    sharpe_bh = compute_sharpe(bh_returns)
    
    total_return_raw = float(pnl_raw.sum()) * 100
    total_return_net = float(pnl_net.sum()) * 100
    total_return_bh = float((test_df["Close"].iloc[-1] / test_df["Close"].iloc[0] - 1)) * 100
    
    # Trade counts
    num_trades = int(np.count_nonzero(np.diff(positions)))
    long_days = int((positions == 1).sum())
    short_days = int((positions == -1).sum())
    flat_days = int((positions == 0).sum())
    
    # Max drawdown
    cumulative = np.cumsum(pnl_net)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
    
    return {
        "ticker": ticker,
        "model_type": model_type,
        "test_start": test_start,
        "test_end": test_end,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": n_features,
        "accuracy": accuracy,
        "sharpe_raw": sharpe_raw,
        "sharpe_net": sharpe_net,
        "sharpe_bh": sharpe_bh,
        "sharpe_vs_bh": (sharpe_net - sharpe_bh) if sharpe_bh else None,
        "return_raw": total_return_raw,
        "return_net": total_return_net,
        "return_bh": total_return_bh,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "long_days": long_days,
        "short_days": short_days,
        "flat_days": flat_days,
        "cost_bps": cost_bps,
    }


def run_walk_forward_monthly(
    ticker: str,
    model_type: str = "xgb",
    use_elasticnet: bool = True,
    cost_bps: float = 10,
) -> list:
    """
    Walk-forward backtest with monthly retraining throughout 2022.
    
    Each month:
    - Retrain on all data up to that month
    - Test on that month only
    """
    results = []
    
    # Monthly test periods in 2022
    test_periods = [
        ("2022-01-01", "2022-01-31"),
        ("2022-02-01", "2022-02-28"),
        ("2022-03-01", "2022-03-31"),
        ("2022-04-01", "2022-04-30"),
        ("2022-05-01", "2022-05-31"),
        ("2022-06-01", "2022-06-30"),
        ("2022-07-01", "2022-07-31"),
        ("2022-08-01", "2022-08-31"),
        ("2022-09-01", "2022-09-30"),
        ("2022-10-01", "2022-10-31"),
        ("2022-11-01", "2022-11-30"),
        ("2022-12-01", "2022-12-31"),
    ]
    
    for test_start, test_end in test_periods:
        # Train on 3 years before test start
        train_start = str((pd.to_datetime(test_start) - pd.Timedelta(days=365*3)).date())
        train_end = str((pd.to_datetime(test_start) - pd.Timedelta(days=1)).date())
        
        result = run_single_period_test(
            ticker=ticker,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            model_type=model_type,
            use_elasticnet=use_elasticnet,
            cost_bps=cost_bps,
        )
        
        if "error" not in result:
            results.append(result)
    
    return results


def run_bear_market_test():
    """Main test function."""
    
    print("=" * 80)
    print("🐻 BEAR MARKET STRESS TEST (2022)")
    print("=" * 80)
    print(f"📅 Test Period: {TEST_START} to {TEST_END}")
    print(f"📊 Tickers: {', '.join(TEST_TICKERS)}")
    print(f"💰 Transaction Cost: {COST_PER_TRADE_BPS} bps per trade")
    print(f"🎯 Horizon: {HORIZON} days")
    print("=" * 80)
    
    # Market context for 2022
    print("\n📉 2022 Market Context:")
    print("  - S&P 500: -19.4%")
    print("  - NASDAQ: -33.1%")
    print("  - Worst year since 2008")
    print("  - Fed raised rates aggressively")
    print("-" * 60)
    
    all_results = []
    ticker_summaries = []
    
    # Test each ticker with full 2022 period first
    print("\n" + "=" * 60)
    print("📊 FULL YEAR 2022 TEST (Single Train/Test Split)")
    print("=" * 60)
    
    for ticker in TEST_TICKERS:
        print(f"\n🏢 Testing {ticker}...")
        
        # Full year test
        result = run_single_period_test(
            ticker=ticker,
            train_start=TRAIN_START,
            train_end=TRAIN_END,
            test_start=TEST_START,
            test_end=TEST_END,
            model_type="xgb",
            use_elasticnet=True,
            cost_bps=COST_PER_TRADE_BPS,
        )
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            continue
        
        all_results.append(result)
        
        # Display results
        acc = result["accuracy"]
        sharpe_net = result["sharpe_net"]
        sharpe_bh = result["sharpe_bh"]
        ret_net = result["return_net"]
        ret_bh = result["return_bh"]
        trades = result["num_trades"]
        vs_bh = result["sharpe_vs_bh"] or 0
        
        status = "✅" if vs_bh > 0 else "❌"
        
        print(f"  Accuracy: {acc:.1f}%")
        print(f"  Sharpe (net): {sharpe_net:.3f} vs B&H: {sharpe_bh:.3f} {status}")
        print(f"  Return (net): {ret_net:+.2f}% vs B&H: {ret_bh:+.2f}%")
        print(f"  Trades: {trades} | Long: {result['long_days']}d | Short: {result['short_days']}d | Flat: {result['flat_days']}d")
        print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
        
        ticker_summaries.append({
            "ticker": ticker,
            "accuracy": acc,
            "sharpe_net": sharpe_net,
            "sharpe_bh": sharpe_bh,
            "return_net": ret_net,
            "return_bh": ret_bh,
            "beat_bh": vs_bh > 0,
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("📊 FULL YEAR 2022 SUMMARY")
    print("=" * 80)
    print(f"{'Ticker':<8} {'Acc':>7} {'Sharpe':>8} {'B&H Shp':>8} {'Return':>10} {'B&H Ret':>10} {'Status':>8}")
    print("-" * 80)
    
    for s in ticker_summaries:
        status = "✅ BEAT" if s["beat_bh"] else "❌ LOSE"
        print(f"{s['ticker']:<8} {s['accuracy']:>6.1f}% {s['sharpe_net']:>8.3f} {s['sharpe_bh']:>8.3f} "
              f"{s['return_net']:>+9.2f}% {s['return_bh']:>+9.2f}% {status:>8}")
    
    # Aggregate stats
    if ticker_summaries:
        avg_acc = np.mean([s["accuracy"] for s in ticker_summaries])
        avg_sharpe = np.mean([s["sharpe_net"] for s in ticker_summaries])
        avg_sharpe_bh = np.mean([s["sharpe_bh"] for s in ticker_summaries])
        win_rate = np.mean([s["beat_bh"] for s in ticker_summaries]) * 100
        avg_ret = np.mean([s["return_net"] for s in ticker_summaries])
        avg_ret_bh = np.mean([s["return_bh"] for s in ticker_summaries])
        
        print("-" * 80)
        print(f"{'AVERAGE':<8} {avg_acc:>6.1f}% {avg_sharpe:>8.3f} {avg_sharpe_bh:>8.3f} "
              f"{avg_ret:>+9.2f}% {avg_ret_bh:>+9.2f}% {win_rate:.0f}% win")
    
    # Walk-forward test
    print("\n" + "=" * 80)
    print("🔄 WALK-FORWARD TEST (Monthly Retraining)")
    print("=" * 80)
    
    wf_all_results = []
    
    for ticker in TEST_TICKERS[:5]:  # Test first 5 for speed
        print(f"\n🏢 Walk-forward {ticker}...")
        
        wf_results = run_walk_forward_monthly(
            ticker=ticker,
            model_type="xgb",
            use_elasticnet=True,
            cost_bps=COST_PER_TRADE_BPS,
        )
        
        if not wf_results:
            print(f"  ❌ No results")
            continue
        
        wf_all_results.extend(wf_results)
        
        # Monthly summary
        months_beat = sum(1 for r in wf_results if (r.get("sharpe_vs_bh") or 0) > 0)
        avg_sharpe = np.mean([r["sharpe_net"] for r in wf_results])
        avg_acc = np.mean([r["accuracy"] for r in wf_results])
        total_ret = sum(r["return_net"] for r in wf_results)
        
        print(f"  Months tested: {len(wf_results)}")
        print(f"  Months beat B&H: {months_beat}/{len(wf_results)} ({months_beat/len(wf_results)*100:.0f}%)")
        print(f"  Avg Monthly Sharpe: {avg_sharpe:.3f}")
        print(f"  Avg Accuracy: {avg_acc:.1f}%")
        print(f"  Cumulative Return: {total_ret:+.2f}%")
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🏆 FINAL BEAR MARKET ASSESSMENT")
    print("=" * 80)
    
    if ticker_summaries:
        beat_count = sum(1 for s in ticker_summaries if s["beat_bh"])
        total_count = len(ticker_summaries)
        
        avg_excess_sharpe = np.mean([s["sharpe_net"] - s["sharpe_bh"] for s in ticker_summaries])
        avg_excess_return = np.mean([s["return_net"] - s["return_bh"] for s in ticker_summaries])
        
        print(f"\n📊 Full Year 2022 Results:")
        print(f"  Tickers Beat B&H: {beat_count}/{total_count} ({beat_count/total_count*100:.0f}%)")
        print(f"  Avg Excess Sharpe: {avg_excess_sharpe:+.3f}")
        print(f"  Avg Excess Return: {avg_excess_return:+.2f}%")
        
        # Grade
        if beat_count >= total_count * 0.7 and avg_excess_sharpe > 0.5:
            grade = "A - Excellent bear market performance"
        elif beat_count >= total_count * 0.5 and avg_excess_sharpe > 0:
            grade = "B - Good, beats B&H majority of time"
        elif beat_count >= total_count * 0.3:
            grade = "C - Mixed results, some protection"
        else:
            grade = "D - Poor bear market performance"
        
        print(f"\n🎯 GRADE: {grade}")
        
        # Interpretation
        print("\n📝 Interpretation:")
        if avg_excess_sharpe > 0:
            print("  ✅ Model provides value even in bear markets")
            print("  ✅ Risk-adjusted returns beat buy-and-hold")
        else:
            print("  ⚠️ Model struggles in bear markets")
            print("  ⚠️ Consider adding short signals or reducing long bias")
        
        if avg_acc > 55:
            print(f"  ✅ Accuracy ({avg_acc:.1f}%) above random chance")
        else:
            print(f"  ⚠️ Accuracy ({avg_acc:.1f}%) near random")
    
    # Save results
    output_path = Path(__file__).parent / "bear_market_test_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "test_period": {"start": TEST_START, "end": TEST_END},
            "full_year_results": all_results,
            "walk_forward_results": wf_all_results,
            "summary": ticker_summaries,
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n" + "=" * 80)
    print("✅ BEAR MARKET TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_bear_market_test()
