#!/usr/bin/env python3
"""
Z-SCORE SWEEP EXPERIMENT
=========================
Experiment ID: ZSCORE_SWEEP_001
Date: 2026-01-07
Purpose: Test z-score gating thresholds against frozen baseline

Variable Changed: z_score_threshold
Values Tested: [0.5, 0.8, 1.0, 1.3, 1.6]

All other settings IDENTICAL to BASELINE_001.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.data.market import get_price_history
from src.core.features import build_all_features, add_gbm_features
from src.core.models import make_model
from src.core.metrics import compute_sharpe, compute_drawdown
from src.config import FEATURE_COLUMNS, TRADING_DAYS_PER_YEAR


# =============================================================================
# LOAD BASELINE CONFIG (FROZEN)
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

BASELINE_CONFIG = BASELINE["config"]
BASELINE_METRICS = BASELINE["aggregate_metrics"]

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
EXPERIMENT_ID = "ZSCORE_SWEEP_001"
ZSCORE_VALUES = [0.5, 0.8, 1.0, 1.3, 1.6]


def prepare_features(ticker: str, period: str = "10y") -> tuple[pd.DataFrame, list]:
    """
    Build feature matrix for a ticker.
    Returns (df with features and target, list of feature columns).
    """
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Build all features
    hist = build_all_features(hist)
    
    # Add GBM features if not present
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1,))
    
    # Target: forward 1-day return
    hist["target"] = hist["Close"].pct_change(1).shift(-1)
    
    # Available feature columns
    feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    # Drop rows with NaN target
    df = hist[feat_cols + ["target", "Close"]].dropna().copy()
    
    return df, feat_cols


def compute_zscore(predictions: np.ndarray, window: int = 60, min_periods: int = 20) -> np.ndarray:
    """
    Compute rolling z-score for predictions.
    """
    pred_series = pd.Series(predictions)
    rolling_mean = pred_series.rolling(window, min_periods=min_periods).mean()
    rolling_std = pred_series.rolling(window, min_periods=min_periods).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    zscore = (pred_series - rolling_mean) / rolling_std
    return zscore.fillna(0).values


def run_backtest_with_zscore(
    ticker: str,
    z_threshold: float,
    config: dict,
) -> dict:
    """
    Run backtest with z-score gating.
    Uses SAME data split as baseline.
    """
    try:
        df, feat_cols = prepare_features(ticker, period="10y")
        
        if len(df) < config["min_train_rows"] + 100:
            return {"ticker": ticker, "error": "Insufficient data"}
        
        # Split: IDENTICAL to baseline
        n = len(df)
        split_idx = int(n * (1 - config["test_fraction"]))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        
        # Train ensemble models (IDENTICAL to baseline)
        predictions = []
        for model_type in config["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Ensemble: average predictions (IDENTICAL to baseline)
        y_pred = np.mean(predictions, axis=0)
        
        # Volatility scaling (IDENTICAL to baseline)
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if vol > 0:
            target_vol = 0.01
            vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
        else:
            vol_scale = 1.0
        
        # === Z-SCORE GATING (THE ONLY CHANGE) ===
        z_scores = compute_zscore(y_pred)
        
        # Trading simulation with z-score filter
        threshold = config["trade_threshold"]
        
        # Base positions from predictions
        base_positions = np.where(y_pred > threshold, 1, 
                         np.where(y_pred < -threshold, -1, 0))
        
        # Apply z-score gate: only trade if |z-score| >= z_threshold
        if z_threshold > 0:
            z_gate = np.abs(z_scores) >= z_threshold
            positions = base_positions * z_gate.astype(float)
        else:
            positions = base_positions
        
        # Scale by volatility
        positions = positions * vol_scale
        
        # Calculate returns
        strategy_returns = positions * y_test
        
        # Metrics
        sharpe = compute_sharpe(strategy_returns)
        dd_df = compute_drawdown(pd.Series(strategy_returns))
        max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
        
        # Trade statistics
        trade_count = int(np.count_nonzero(np.diff(positions)))
        winning_trades = np.sum((positions[:-1] != 0) & (strategy_returns[:-1] > 0))
        total_trades = np.sum(positions[:-1] != 0)
        win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0
        
        # Cumulative return
        cum_return = float(np.sum(strategy_returns))
        
        return {
            "ticker": ticker,
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(cum_return, 4),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_zscore_sweep():
    """
    Run z-score sweep experiment across all thresholds.
    """
    print("=" * 70)
    print(f"Z-SCORE SWEEP EXPERIMENT: {EXPERIMENT_ID}")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Baseline Sharpe: {BASELINE_METRICS['avg_sharpe']:.4f}")
    print(f"Baseline Max DD: {BASELINE_METRICS['avg_max_drawdown']*100:.2f}%")
    print(f"Testing z_score_threshold: {ZSCORE_VALUES}")
    print("=" * 70)
    print()
    
    all_results = []
    
    for z_thresh in ZSCORE_VALUES:
        print(f"\n--- Testing z_score_threshold = {z_thresh} ---")
        
        ticker_results = []
        for ticker in BASELINE_CONFIG["tickers"]:
            print(f"  {ticker}...", end=" ")
            result = run_backtest_with_zscore(ticker, z_thresh, BASELINE_CONFIG)
            ticker_results.append(result)
            
            if "error" in result:
                print(f"ERROR")
            else:
                print(f"Sharpe={result['sharpe']:.2f}, Trades={result['trade_count']}")
        
        # Aggregate for this threshold
        valid = [r for r in ticker_results if "error" not in r]
        
        if valid:
            agg = {
                "z_score_threshold": z_thresh,
                "avg_sharpe": round(np.mean([r["sharpe"] for r in valid]), 4),
                "avg_max_drawdown": round(np.mean([r["max_drawdown"] for r in valid]), 4),
                "avg_win_rate": round(np.mean([r["win_rate"] for r in valid]), 4),
                "total_trades": sum(r["trade_count"] for r in valid),
                "per_ticker": ticker_results,
            }
            
            # Compare to baseline
            sharpe_delta = agg["avg_sharpe"] - BASELINE_METRICS["avg_sharpe"]
            dd_delta = agg["avg_max_drawdown"] - BASELINE_METRICS["avg_max_drawdown"]
            
            agg["sharpe_vs_baseline"] = round(sharpe_delta, 4)
            agg["maxdd_vs_baseline"] = round(dd_delta, 4)
            
            # Determine if acceptable
            sharpe_improved = sharpe_delta > 0
            dd_not_worse = dd_delta >= 0  # Less negative = better
            agg["acceptable"] = sharpe_improved and dd_not_worse
            
            all_results.append(agg)
            
            print(f"  => Avg Sharpe: {agg['avg_sharpe']:.4f} (Δ{sharpe_delta:+.4f})")
            print(f"  => Avg MaxDD:  {agg['avg_max_drawdown']*100:.2f}% (Δ{dd_delta*100:+.2f}%)")
            print(f"  => Trades:     {agg['total_trades']}")
            print(f"  => Acceptable: {'✅ YES' if agg['acceptable'] else '❌ NO'}")
    
    # Sort by Sharpe DESC
    all_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    
    # Find best acceptable result
    best_acceptable = None
    for r in all_results:
        if r["acceptable"]:
            best_acceptable = r
            break
    
    # Print summary table
    print("\n")
    print("=" * 70)
    print("RESULTS SORTED BY SHARPE (DESCENDING)")
    print("=" * 70)
    print(f"{'Z-Score':<10} {'Sharpe':<10} {'Δ Sharpe':<12} {'Max DD':<10} {'Δ DD':<10} {'Trades':<8} {'Accept':<8}")
    print("-" * 70)
    
    # Add baseline row
    print(f"{'BASELINE':<10} {BASELINE_METRICS['avg_sharpe']:<10.4f} {'—':<12} {BASELINE_METRICS['avg_max_drawdown']*100:<10.2f}% {'—':<10} {BASELINE_METRICS['total_trades']:<8} {'REF':<8}")
    print("-" * 70)
    
    for r in all_results:
        accept = "✅" if r["acceptable"] else "❌"
        best_mark = " ⭐ BEST" if r == best_acceptable else ""
        print(f"{r['z_score_threshold']:<10} {r['avg_sharpe']:<10.4f} {r['sharpe_vs_baseline']:+<12.4f} {r['avg_max_drawdown']*100:<10.2f}% {r['maxdd_vs_baseline']*100:+<10.2f}% {r['total_trades']:<8} {accept:<8}{best_mark}")
    
    print("=" * 70)
    
    if best_acceptable:
        print(f"\n🏆 BEST ACCEPTABLE: z_score_threshold = {best_acceptable['z_score_threshold']}")
        print(f"   Sharpe: {best_acceptable['avg_sharpe']:.4f} (Δ{best_acceptable['sharpe_vs_baseline']:+.4f})")
        print(f"   Max DD: {best_acceptable['avg_max_drawdown']*100:.2f}%")
    else:
        print("\n⚠️ NO ACCEPTABLE RESULT - All options either worsen Sharpe or MaxDD")
    
    # Save results
    output = {
        "experiment_id": EXPERIMENT_ID,
        "created": datetime.now().isoformat(),
        "variable_tested": "z_score_threshold",
        "values_tested": ZSCORE_VALUES,
        "baseline_metrics": BASELINE_METRICS,
        "results": all_results,
        "best_acceptable": best_acceptable,
        "recommendation": f"z_score_threshold={best_acceptable['z_score_threshold']}" if best_acceptable else "KEEP BASELINE",
    }
    
    output_path = PROJECT_ROOT / "experiments" / "zscore_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_zscore_sweep()
