#!/usr/bin/env python3
"""
Walk-Forward Accuracy Test
==========================
Tests XGB and RF models on AAPL, MSFT, AMZN with proper walk-forward validation.
Reports: Sharpe, hit rate, and fold-by-fold breakdown.
"""
import signal, sys, time, warnings
warnings.filterwarnings("ignore")

def _timeout(signum, frame):
    print("\n\n❌ TIMEOUT after 300s")
    sys.exit(1)
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(300)

import numpy as np
import pandas as pd

t0 = time.time()
print("=" * 80)
print("WALK-FORWARD ACCURACY TEST")
print("=" * 80)

from prediction_model import (
    walk_forward_backtest, make_model, FEATURE_COLUMNS, MACRO_COLUMNS,
    get_price_history, add_price_features, get_macro_df, get_fundamental_features,
)

# Test parameters
TICKERS = ["AAPL", "MSFT", "AMZN"]
MODEL_TYPES = ["xgb", "rf"]
PERIOD = "5y"
TRAIN_YEARS = 2
TEST_YEARS = 0.5  # 6 months
THRESHOLD = 0.002
COST = 0.0005

all_results = []

for ticker in TICKERS:
    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"  {ticker} / {model_type.upper()} — {TRAIN_YEARS}y train, {TEST_YEARS}y test")
        print(f"{'='*60}")
        
        try:
            t1 = time.time()
            folds = walk_forward_backtest(
                ticker=ticker,
                period=PERIOD,
                horizon=1,
                model_type=model_type,
                train_years=TRAIN_YEARS,
                test_years=TEST_YEARS,
                threshold=THRESHOLD,
                cost_per_trade=COST,
            )
            elapsed = time.time() - t1
            
            if not folds:
                print(f"  ❌ No folds returned ({elapsed:.1f}s)")
                continue
            
            print(f"\n  Fold-by-fold results ({len(folds)} folds, {elapsed:.1f}s):")
            print(f"  {'Fold':>4} {'Test Period':<25} {'Days':>5} {'Hit%':>6} {'Sharpe':>8} {'Trades':>7}")
            print(f"  {'-'*60}")
            
            sharpes = []
            hitrates = []
            for i, fold in enumerate(folds):
                ts = fold['test_start']
                te = fold['test_end']
                test_str = f"{ts.strftime('%Y-%m-%d')} → {te.strftime('%Y-%m-%d')}"
                print(f"  {i+1:>4} {test_str:<25} {fold['test_days']:>5} {fold['hitrate']:>5.1%} {fold['sharpe']:>8.2f} {fold['num_trades']:>7}")
                sharpes.append(fold['sharpe'])
                hitrates.append(fold['hitrate'])
            
            mean_sharpe = np.mean(sharpes)
            mean_hitrate = np.mean(hitrates)
            pos_sharpe_pct = sum(1 for s in sharpes if s > 0) / len(sharpes)
            
            print(f"\n  Summary:")
            print(f"    Mean Sharpe:     {mean_sharpe:+.3f}")
            print(f"    Mean Hit Rate:   {mean_hitrate:.1%}")
            print(f"    Positive Sharpe: {pos_sharpe_pct:.0%} ({sum(1 for s in sharpes if s > 0)}/{len(sharpes)} folds)")
            print(f"    Best fold:       {max(sharpes):+.3f}")
            print(f"    Worst fold:      {min(sharpes):+.3f}")
            
            all_results.append({
                "ticker": ticker,
                "model": model_type,
                "folds": len(folds),
                "mean_sharpe": mean_sharpe,
                "mean_hitrate": mean_hitrate,
                "pos_sharpe_pct": pos_sharpe_pct,
                "best_sharpe": max(sharpes),
                "worst_sharpe": min(sharpes),
            })
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

if all_results:
    print(f"\n{'Ticker':<8} {'Model':<6} {'Folds':>5} {'Sharpe':>8} {'Hit%':>6} {'Pos%':>5} {'Best':>8} {'Worst':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['ticker']:<8} {r['model']:<6} {r['folds']:>5} {r['mean_sharpe']:>+8.3f} {r['mean_hitrate']:>5.1%} {r['pos_sharpe_pct']:>4.0%} {r['best_sharpe']:>+8.3f} {r['worst_sharpe']:>+8.3f}")
    
    # Overall averages
    avg_sharpe = np.mean([r['mean_sharpe'] for r in all_results])
    avg_hitrate = np.mean([r['mean_hitrate'] for r in all_results])
    avg_pos = np.mean([r['pos_sharpe_pct'] for r in all_results])
    
    print("-" * 60)
    print(f"{'OVERALL':<8} {'avg':<6} {'':<5} {avg_sharpe:>+8.3f} {avg_hitrate:>5.1%} {avg_pos:>4.0%}")
    
    # Best model
    best = max(all_results, key=lambda x: x['mean_sharpe'])
    print(f"\n🏆 Best: {best['ticker']}/{best['model'].upper()} — Sharpe {best['mean_sharpe']:+.3f}, Hit {best['mean_hitrate']:.1%}")
else:
    print("❌ No results generated!")

print(f"\n⏱️  Total time: {time.time()-t0:.1f}s")
signal.alarm(0)
