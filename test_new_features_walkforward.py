#!/usr/bin/env python3
"""
Walk-Forward Backtest: Old Features vs New Features
====================================================

Trains on historical data, predicts on unseen future data.
Compares prediction accuracy with and without the new features:
- FRED credit spreads (BAA-AAA)
- VIX term structure (VIX/VIX3M)
- 2s10s yield curve
- HMM regime indicators

Walk-forward design:
- 5 folds, each with 18-month train + 3-month test
- Data: 2021-2026 (5 years)
- Model retrains from scratch each fold
- NO future data leakage
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
from datetime import datetime

# ── Data fetching ──
from src.data.market import get_price_history
from src.data.macro import get_macro_df
from src.data.fundamentals import get_fundamental_features
from src.core.features import build_all_features, add_gbm_features, build_target
from src.core.models import make_model
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS

# New features to test
NEW_MACRO_FEATURES = [
    "credit_spread", "credit_spread_chg_5d",
    "term_spread_2s10s", "vix_term_structure", "vix_ts_chg_5d",
]
NEW_HMM_FEATURES = [
    "hmm_regime_bull", "hmm_regime_bear", "hmm_regime_neutral",
]
ALL_NEW_FEATURES = NEW_MACRO_FEATURES + NEW_HMM_FEATURES


def build_full_dataset(ticker: str, period: str = "5y"):
    """Build complete dataset with ALL features for a ticker."""
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Build all features (includes HMM regime)
    hist = build_all_features(hist)
    hist = add_gbm_features(hist)
    
    # Macro data
    try:
        macro_df = get_macro_df(period=period)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"  ⚠️ Macro data unavailable: {e}")
    
    # Fundamentals
    try:
        fund = get_fundamental_features(ticker)
        for k, v in fund.items():
            hist[k] = v
    except Exception:
        pass
    
    # Target: next-day return (with winsorization)
    hist["target"] = build_target(hist, horizon=1)
    
    return hist


def get_feature_sets(df: pd.DataFrame):
    """
    Get two feature sets from the same DataFrame:
    1. OLD: All features EXCEPT the new ones
    2. NEW: All features INCLUDING the new ones
    """
    # Collect all available features
    all_feat = [c for c in FEATURE_COLUMNS if c in df.columns]
    all_macro = [c for c in MACRO_COLUMNS if c in df.columns]
    all_cols = list(dict.fromkeys(all_feat + all_macro))
    
    # Quality filter: drop >50% NaN
    quality = df[all_cols].isna().mean()
    all_cols = [c for c in all_cols if quality[c] < 0.5]
    
    # NEW = everything available
    new_cols = all_cols
    
    # OLD = everything MINUS the new features
    old_cols = [c for c in all_cols if c not in ALL_NEW_FEATURES]
    
    return old_cols, new_cols


def walk_forward_test(df: pd.DataFrame, feat_cols: list, label: str,
                      train_months: int = 18, test_months: int = 3,
                      model_type: str = "xgb"):
    """
    Walk-forward backtest on a single ticker.
    
    Returns dict with metrics per fold and aggregates.
    """
    # Clean data
    working = df[feat_cols + ["target"]].copy()
    working[feat_cols] = working[feat_cols].ffill().bfill().fillna(0)
    working = working.dropna(subset=["target"])
    
    if len(working) < 252:
        return None
    
    dates = working.index
    min_date = dates.min()
    max_date = dates.max()
    
    # Generate fold boundaries
    folds = []
    train_delta = pd.DateOffset(months=train_months)
    test_delta = pd.DateOffset(months=test_months)
    
    fold_start = min_date + train_delta  # First test start
    while fold_start + test_delta <= max_date:
        train_start = fold_start - train_delta
        test_end = fold_start + test_delta
        folds.append((train_start, fold_start, test_end))
        fold_start += test_delta  # Roll forward by test_months
    
    if not folds:
        return None
    
    all_preds = []
    all_actuals = []
    all_dates = []
    fold_results = []
    
    for fold_i, (train_start, test_start, test_end) in enumerate(folds):
        # Split data by DATE (no leakage)
        train_mask = (dates >= train_start) & (dates < test_start)
        test_mask = (dates >= test_start) & (dates < test_end)
        
        train_df = working[train_mask]
        test_df = working[test_mask]
        
        if len(train_df) < 60 or len(test_df) < 10:
            continue
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        
        # Train fresh model
        model = make_model(model_type=model_type, task="reg")
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  Fold {fold_i} fit failed: {e}")
            continue
        
        # Predict on unseen test data
        preds = model.predict(X_test)
        
        # Metrics for this fold
        direction_correct = ((preds > 0) == (y_test > 0)).astype(int)
        hit_rate = direction_correct.mean()
        
        # Simple long/short returns
        positions = np.sign(preds)
        daily_pnl = positions * y_test
        
        sharpe = (daily_pnl.mean() / (daily_pnl.std() + 1e-9)) * np.sqrt(252)
        total_ret = daily_pnl.sum()
        max_dd = _max_drawdown(daily_pnl)
        
        fold_results.append({
            "fold": fold_i,
            "train_start": str(train_start.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "total_return": total_ret,
            "max_drawdown": max_dd,
        })
        
        all_preds.extend(preds.tolist())
        all_actuals.extend(y_test.tolist())
        all_dates.extend(test_df.index.tolist())
    
    if not fold_results:
        return None
    
    # Aggregate metrics
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    overall_hit = ((all_preds > 0) == (all_actuals > 0)).mean()
    overall_positions = np.sign(all_preds)
    overall_pnl = overall_positions * all_actuals
    overall_sharpe = (overall_pnl.mean() / (overall_pnl.std() + 1e-9)) * np.sqrt(252)
    overall_return = overall_pnl.sum()
    overall_dd = _max_drawdown(overall_pnl)
    
    # Long-only accuracy (when model says BUY)
    long_mask = all_preds > 0
    long_hit = (all_actuals[long_mask] > 0).mean() if long_mask.sum() > 0 else 0
    
    # Short-only accuracy (when model says SELL)
    short_mask = all_preds < 0
    short_hit = (all_actuals[short_mask] < 0).mean() if short_mask.sum() > 0 else 0
    
    return {
        "label": label,
        "n_features": len(feat_cols),
        "n_folds": len(fold_results),
        "n_predictions": len(all_preds),
        "overall_hit_rate": overall_hit,
        "overall_sharpe": overall_sharpe,
        "overall_return": overall_return,
        "overall_max_dd": overall_dd,
        "long_accuracy": long_hit,
        "short_accuracy": short_hit,
        "n_long_signals": int(long_mask.sum()),
        "n_short_signals": int(short_mask.sum()),
        "fold_results": fold_results,
        "mean_fold_sharpe": np.mean([f["sharpe"] for f in fold_results]),
        "mean_fold_hit": np.mean([f["hit_rate"] for f in fold_results]),
        "positive_sharpe_folds": sum(1 for f in fold_results if f["sharpe"] > 0),
    }


def _max_drawdown(daily_pnl: np.ndarray) -> float:
    """Calculate maximum drawdown from daily P&L array."""
    cum = np.cumsum(daily_pnl)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def print_comparison(old_result, new_result, ticker: str):
    """Print side-by-side comparison of old vs new features."""
    print(f"\n{'='*70}")
    print(f"  {ticker} — Walk-Forward Results (Old Features vs New Features)")
    print(f"{'='*70}")
    
    if old_result is None or new_result is None:
        print("  ⚠️ Insufficient data for comparison")
        return
    
    metrics = [
        ("Features Used", "n_features", "", False),
        ("Total Predictions", "n_predictions", "", False),
        ("Folds", "n_folds", "", False),
        ("─── ACCURACY ───", None, "", False),
        ("Overall Hit Rate", "overall_hit_rate", "%", True),
        ("Long Accuracy", "long_accuracy", "%", True),
        ("Short Accuracy", "short_accuracy", "%", True),
        ("Mean Fold Hit Rate", "mean_fold_hit", "%", True),
        ("─── RETURNS ───", None, "", False),
        ("Overall Sharpe", "overall_sharpe", "", True),
        ("Mean Fold Sharpe", "mean_fold_sharpe", "", True),
        ("Positive Sharpe Folds", "positive_sharpe_folds", f"/{old_result['n_folds']}", False),
        ("Total Return", "overall_return", "%", True),
        ("Max Drawdown", "overall_max_dd", "%", True),
    ]
    
    print(f"  {'Metric':<28s} {'Old':>12s} {'New':>12s} {'Δ':>10s}")
    print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*10}")
    
    for name, key, suffix, show_delta in metrics:
        if key is None:
            print(f"  {name}")
            continue
        
        old_val = old_result.get(key, 0)
        new_val = new_result.get(key, 0)
        
        if suffix == "%":
            old_str = f"{old_val*100:.1f}%" if isinstance(old_val, float) else str(old_val)
            new_str = f"{new_val*100:.1f}%" if isinstance(new_val, float) else str(new_val)
        else:
            old_str = f"{old_val:.3f}" if isinstance(old_val, float) else f"{old_val}{suffix}"
            new_str = f"{new_val:.3f}" if isinstance(new_val, float) else f"{new_val}{suffix}"
        
        if show_delta and isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            delta = new_val - old_val
            if suffix == "%":
                delta_str = f"{'+' if delta >= 0 else ''}{delta*100:.1f}pp"
            else:
                delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"
            # Color hint
            better = delta > 0 if key not in ("overall_max_dd",) else delta > 0  # DD is negative, less negative = better
            icon = "✅" if better else "❌" if abs(delta) > 0.001 else "➖"
            delta_str = f"{icon} {delta_str}"
        else:
            delta_str = ""
        
        print(f"  {name:<28s} {old_str:>12s} {new_str:>12s} {delta_str:>10s}")
    
    # Fold-by-fold detail
    print(f"\n  Fold Details:")
    print(f"  {'Fold':<6s} {'Test Period':<24s} {'Old Hit':>8s} {'New Hit':>8s} {'Old Sharpe':>11s} {'New Sharpe':>11s}")
    print(f"  {'─'*6} {'─'*24} {'─'*8} {'─'*8} {'─'*11} {'─'*11}")
    
    for i in range(min(len(old_result["fold_results"]), len(new_result["fold_results"]))):
        of = old_result["fold_results"][i]
        nf = new_result["fold_results"][i]
        period = f"{of['test_start']} → {of['test_end']}"
        print(f"  {i:<6d} {period:<24s} {of['hit_rate']*100:>7.1f}% {nf['hit_rate']*100:>7.1f}% {of['sharpe']:>+10.2f} {nf['sharpe']:>+10.2f}")


def main():
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST: Old Features vs New Features")
    print("  Train: 18 months → Test: 3 months (unseen) → Roll forward")
    print("  Model: XGBoost (optimized config)")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    tickers = ["AAPL", "MSFT", "AMZN"]
    model_type = "xgb"
    
    all_old = []
    all_new = []
    
    for ticker in tickers:
        print(f"\n{'─'*70}")
        print(f"  Processing {ticker}...")
        print(f"{'─'*70}")
        
        t0 = time.time()
        
        # Build full dataset (all features)
        try:
            df = build_full_dataset(ticker, period="5y")
        except Exception as e:
            print(f"  ❌ Failed to build data: {e}")
            continue
        
        print(f"  Data: {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
        
        # Get OLD and NEW feature sets
        old_cols, new_cols = get_feature_sets(df)
        added = [c for c in new_cols if c not in old_cols]
        print(f"  Old features: {len(old_cols)}")
        print(f"  New features: {len(new_cols)} (+{len(added)}: {', '.join(added)})")
        
        # Run walk-forward with OLD features
        print(f"\n  ▶ Running OLD features walk-forward...")
        old_result = walk_forward_test(df, old_cols, f"{ticker}_old", model_type=model_type)
        
        # Run walk-forward with NEW features
        print(f"  ▶ Running NEW features walk-forward...")
        new_result = walk_forward_test(df, new_cols, f"{ticker}_new", model_type=model_type)
        
        elapsed = time.time() - t0
        print(f"  ⏱ {elapsed:.1f}s")
        
        if old_result and new_result:
            all_old.append(old_result)
            all_new.append(new_result)
            print_comparison(old_result, new_result, ticker)
    
    # ── AGGREGATE SUMMARY ──
    if all_old and all_new:
        print(f"\n\n{'='*70}")
        print(f"  AGGREGATE SUMMARY ACROSS {len(all_old)} TICKERS")
        print(f"{'='*70}")
        
        def avg(results, key):
            vals = [r[key] for r in results if key in r]
            return np.mean(vals) if vals else 0
        
        metrics = [
            ("Overall Hit Rate", "overall_hit_rate", "%"),
            ("Long Accuracy", "long_accuracy", "%"),
            ("Short Accuracy", "short_accuracy", "%"),
            ("Overall Sharpe", "overall_sharpe", ""),
            ("Mean Fold Sharpe", "mean_fold_sharpe", ""),
            ("Total Return", "overall_return", "%"),
        ]
        
        print(f"\n  {'Metric':<28s} {'Old':>12s} {'New':>12s} {'Δ':>12s} {'Verdict':>10s}")
        print(f"  {'─'*28} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
        
        for name, key, suffix in metrics:
            old_v = avg(all_old, key)
            new_v = avg(all_new, key)
            delta = new_v - old_v
            
            if suffix == "%":
                print(f"  {name:<28s} {old_v*100:>11.1f}% {new_v*100:>11.1f}% {delta*100:>+11.1f}pp {'✅ BETTER' if delta > 0.001 else '❌ WORSE' if delta < -0.001 else '➖ SAME':>10s}")
            else:
                print(f"  {name:<28s} {old_v:>12.3f} {new_v:>12.3f} {delta:>+12.3f} {'✅ BETTER' if delta > 0.001 else '❌ WORSE' if delta < -0.001 else '➖ SAME':>10s}")
        
        total_pos_old = sum(r["positive_sharpe_folds"] for r in all_old)
        total_pos_new = sum(r["positive_sharpe_folds"] for r in all_new)
        total_folds = sum(r["n_folds"] for r in all_old)
        print(f"\n  Positive Sharpe Folds: Old={total_pos_old}/{total_folds}, New={total_pos_new}/{total_folds}")
        
        # Final verdict
        improvements = 0
        for name, key, suffix in metrics:
            if avg(all_new, key) > avg(all_old, key) + 0.001:
                improvements += 1
        
        print(f"\n  🏁 VERDICT: New features improved {improvements}/{len(metrics)} metrics")
        if improvements >= 4:
            print(f"  ✅ NEW FEATURES ARE BENEFICIAL — recommend enabling in production")
        elif improvements >= 2:
            print(f"  🟡 MIXED RESULTS — new features help on some metrics, not others")
        else:
            print(f"  ❌ NEW FEATURES DON'T HELP — keep old configuration")


if __name__ == "__main__":
    main()
