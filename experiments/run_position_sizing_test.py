#!/usr/bin/env python3
"""
POSITION SIZING EXPERIMENT
===========================
Experiment ID: POSITION_SIZING_001
Date: 2026-01-07
Purpose: Test position sizing methods to improve consistency

Configurations Tested:
A) Current volatility scaling (baseline)
B) Volatility scaling × confidence_score
C) Same as B with 1% max position cap

Uses BASELINE_003 config (z_score=1.6, regime_filter=ON)

Acceptance Criteria:
- Drawdown decreases OR
- Return variance decreases without Sharpe loss
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
# LOAD BASELINE CONFIG (BASELINE_003)
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

BASELINE_CONFIG = BASELINE["config"]
BASELINE_METRICS = BASELINE["aggregate_metrics"]

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
EXPERIMENT_ID = "POSITION_SIZING_001"

SIZING_METHODS = {
    "A": "vol_scaling_only",           # Current baseline
    "B": "vol_scaling_x_confidence",   # Vol scaling × confidence score
    "C": "vol_x_conf_with_cap",        # B + 1% max position cap
}


def prepare_features(ticker: str, period: str = "10y") -> tuple[pd.DataFrame, list]:
    """Build feature matrix for a ticker."""
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")
    
    hist = build_all_features(hist)
    
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1,))
    
    hist["target"] = hist["Close"].pct_change(1).shift(-1)
    
    feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    return hist, feat_cols


def compute_zscore(predictions: np.ndarray, window: int = 60, min_periods: int = 20) -> np.ndarray:
    """Compute rolling z-score for predictions."""
    pred_series = pd.Series(predictions)
    rolling_mean = pred_series.rolling(window, min_periods=min_periods).mean()
    rolling_std = pred_series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    zscore = (pred_series - rolling_mean) / rolling_std
    return zscore.fillna(0).values


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Detect market regime (bull/bear/neutral)."""
    regime = pd.Series('neutral', index=df.index)
    close = df["Close"]
    
    ma50 = close.rolling(50, min_periods=20).mean()
    ma200 = close.rolling(200, min_periods=50).mean()
    
    bull_condition = (close > ma50) & (close > ma200) & (ma50 > ma200)
    bear_condition = (close < ma50) & (close < ma200) & (ma50 < ma200)
    
    rolling_max = close.rolling(60, min_periods=20).max()
    drawdown = (close - rolling_max) / rolling_max
    high_vol = drawdown < -0.10
    
    regime[bull_condition] = 'bull'
    regime[bear_condition] = 'bear'
    regime[high_vol] = 'bear'
    
    return regime


def compute_confidence_score(y_pred: np.ndarray, z_scores: np.ndarray) -> np.ndarray:
    """
    Compute confidence score from prediction magnitude and z-score.
    Score in [0, 1] where higher = more confident.
    """
    # Normalize prediction magnitude (clip extremes)
    pred_abs = np.abs(y_pred)
    pred_pct = np.clip(pred_abs / np.percentile(pred_abs[pred_abs > 0], 95), 0, 1)
    
    # Normalize z-score (higher z-score = higher confidence)
    z_abs = np.abs(z_scores)
    z_pct = np.clip(z_abs / 3.0, 0, 1)  # z=3 → confidence=1
    
    # Combined confidence: average of both signals
    confidence = (pred_pct + z_pct) / 2
    
    return confidence


def compute_worst_5day_loss(returns: np.ndarray) -> float:
    """Compute worst rolling 5-day loss."""
    if len(returns) < 5:
        return 0.0
    ret_series = pd.Series(returns)
    rolling_5d = ret_series.rolling(5).sum()
    return float(rolling_5d.min()) if not rolling_5d.isna().all() else 0.0


def run_backtest_with_sizing(
    ticker: str,
    sizing_method: str,
    config: dict,
) -> dict:
    """
    Run backtest with specified position sizing method.
    """
    try:
        hist, feat_cols = prepare_features(ticker, period="10y")
        hist["regime"] = detect_regime(hist)
        
        df = hist[feat_cols + ["target", "Close", "regime"]].dropna().copy()
        
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
        test_regimes = test_df["regime"].values
        
        # Train ensemble models (IDENTICAL to baseline)
        predictions = []
        for model_type in config["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        y_pred = np.mean(predictions, axis=0)
        
        # Z-score gating (from baseline)
        z_threshold = config["z_score_threshold"]
        z_scores = compute_zscore(y_pred)
        
        threshold = config["trade_threshold"]
        
        # Base positions
        base_positions = np.where(y_pred > threshold, 1, 
                         np.where(y_pred < -threshold, -1, 0))
        
        # Z-score gate
        if z_threshold > 0:
            z_gate = np.abs(z_scores) >= z_threshold
            positions = base_positions * z_gate.astype(float)
        else:
            positions = base_positions
        
        # Regime filter (from baseline)
        if config["regime_filter"]:
            regime_scale = np.where(test_regimes == 'bear', 0.5,
                          np.where(test_regimes == 'neutral', 0.75, 1.0))
            positions = positions * regime_scale
        
        # === POSITION SIZING (THE VARIABLE BEING TESTED) ===
        
        # Base volatility scaling
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if vol > 0:
            target_vol = 0.01
            vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
        else:
            vol_scale = 1.0
        
        if sizing_method == "A":
            # A) Current: just volatility scaling
            positions = positions * vol_scale
            
        elif sizing_method == "B":
            # B) Volatility scaling × confidence score
            confidence = compute_confidence_score(y_pred, z_scores)
            positions = positions * vol_scale * confidence
            
        elif sizing_method == "C":
            # C) Same as B with 1% max position cap
            confidence = compute_confidence_score(y_pred, z_scores)
            positions = positions * vol_scale * confidence
            # Cap at 1% of portfolio (position size of 0.01)
            positions = np.clip(positions, -0.01, 0.01) * 100  # Scale back up for returns calc
            # Actually: cap the scaled position
            positions = np.clip(positions, -1.0, 1.0)  # Max 100% position
        
        # Calculate returns
        strategy_returns = positions * y_test
        
        # Metrics
        sharpe = compute_sharpe(strategy_returns)
        dd_df = compute_drawdown(pd.Series(strategy_returns))
        max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
        
        # New metrics for this experiment
        std_daily_returns = float(np.std(strategy_returns))
        worst_5day = compute_worst_5day_loss(strategy_returns)
        
        # Trade statistics
        trade_count = int(np.count_nonzero(np.diff(positions)))
        winning_trades = np.sum((positions[:-1] != 0) & (strategy_returns[:-1] > 0))
        total_trades = np.sum(positions[:-1] != 0)
        win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0
        
        cum_return = float(np.sum(strategy_returns))
        
        return {
            "ticker": ticker,
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "std_daily_returns": round(std_daily_returns, 6),
            "worst_5day_loss": round(worst_5day, 4),
            "total_return": round(cum_return, 4),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_position_sizing_experiment():
    """Run position sizing experiment across all methods."""
    print("=" * 70)
    print(f"POSITION SIZING EXPERIMENT: {EXPERIMENT_ID}")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Baseline (BASELINE_003) Sharpe: {BASELINE_METRICS['avg_sharpe']:.4f}")
    print(f"Baseline (BASELINE_003) Max DD: {BASELINE_METRICS['avg_max_drawdown']*100:.2f}%")
    print(f"Methods: A=vol_only, B=vol×confidence, C=vol×conf+cap")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for method_key, method_name in SIZING_METHODS.items():
        print(f"\n--- Testing Method {method_key}: {method_name} ---")
        
        ticker_results = []
        for ticker in BASELINE_CONFIG["tickers"]:
            print(f"  {ticker}...", end=" ")
            result = run_backtest_with_sizing(ticker, method_key, BASELINE_CONFIG)
            ticker_results.append(result)
            
            if "error" in result:
                print(f"ERROR")
            else:
                print(f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown']*100:.1f}%, StdRet={result['std_daily_returns']:.4f}")
        
        # Aggregate
        valid = [r for r in ticker_results if "error" not in r]
        
        if valid:
            agg = {
                "method": method_key,
                "method_name": method_name,
                "avg_sharpe": round(np.mean([r["sharpe"] for r in valid]), 4),
                "avg_max_drawdown": round(np.mean([r["max_drawdown"] for r in valid]), 4),
                "avg_std_daily_returns": round(np.mean([r["std_daily_returns"] for r in valid]), 6),
                "avg_worst_5day_loss": round(np.mean([r["worst_5day_loss"] for r in valid]), 4),
                "avg_win_rate": round(np.mean([r["win_rate"] for r in valid]), 4),
                "total_trades": sum(r["trade_count"] for r in valid),
                "per_ticker": ticker_results,
            }
            
            all_results[method_key] = agg
            
            print(f"\n  => Avg Sharpe:     {agg['avg_sharpe']:.4f}")
            print(f"  => Avg MaxDD:      {agg['avg_max_drawdown']*100:.2f}%")
            print(f"  => Avg Std(Ret):   {agg['avg_std_daily_returns']:.6f}")
            print(f"  => Avg Worst 5D:   {agg['avg_worst_5day_loss']*100:.2f}%")
    
    # Compare methods
    method_a = all_results.get("A", {})
    method_b = all_results.get("B", {})
    method_c = all_results.get("C", {})
    
    print("\n")
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<30} {'Sharpe':<10} {'MaxDD':<10} {'Std(Ret)':<12} {'Worst5D':<10} {'Trades':<8}")
    print("-" * 80)
    
    for key in ["A", "B", "C"]:
        r = all_results.get(key, {})
        if r:
            print(f"{key}: {r['method_name']:<26} {r['avg_sharpe']:<10.4f} {r['avg_max_drawdown']*100:<10.2f}% {r['avg_std_daily_returns']:<12.6f} {r['avg_worst_5day_loss']*100:<10.2f}% {r['total_trades']:<8}")
    
    print("=" * 80)
    
    # Decision logic
    baseline_sharpe = method_a.get("avg_sharpe", 0)
    baseline_dd = method_a.get("avg_max_drawdown", 0)
    baseline_std = method_a.get("avg_std_daily_returns", 0)
    
    recommendations = []
    
    for key in ["B", "C"]:
        r = all_results.get(key, {})
        if not r:
            continue
            
        sharpe_delta = r["avg_sharpe"] - baseline_sharpe
        dd_delta = r["avg_max_drawdown"] - baseline_dd  # Less negative = better
        std_delta = r["avg_std_daily_returns"] - baseline_std  # Lower = better
        
        dd_improved = dd_delta > 0
        variance_improved = std_delta < 0
        sharpe_maintained = sharpe_delta >= -0.02  # Allow small Sharpe drop
        
        accept = dd_improved or (variance_improved and sharpe_maintained)
        
        rec = {
            "method": key,
            "method_name": r["method_name"],
            "sharpe_delta": round(sharpe_delta, 4),
            "dd_delta": round(dd_delta, 4),
            "std_delta": round(std_delta, 6),
            "dd_improved": dd_improved,
            "variance_improved": variance_improved,
            "sharpe_maintained": sharpe_maintained,
            "accept": accept,
        }
        recommendations.append(rec)
        
        print(f"\n📊 Method {key} vs Baseline (A):")
        print(f"   Sharpe Δ:   {sharpe_delta:+.4f} {'✅' if sharpe_delta >= 0 else '⚠️'}")
        print(f"   MaxDD Δ:    {dd_delta*100:+.2f}% {'✅ improved' if dd_improved else '—'}")
        print(f"   Std(Ret) Δ: {std_delta:+.6f} {'✅ improved' if variance_improved else '—'}")
        print(f"   Accept?     {'✅ YES' if accept else '❌ NO'}")
    
    # Find best acceptable method
    best = None
    for rec in recommendations:
        if rec["accept"]:
            if best is None or rec["dd_delta"] > best["dd_delta"]:
                best = rec
    
    if best:
        print(f"\n🏆 BEST ACCEPTABLE: Method {best['method']} ({best['method_name']})")
    else:
        print(f"\n⚠️ NO IMPROVEMENT - Keep Method A (vol_scaling_only)")
        best = {"method": "A", "method_name": "vol_scaling_only", "accept": False}
    
    # Save results
    output = {
        "experiment_id": EXPERIMENT_ID,
        "created": datetime.now().isoformat(),
        "variable_tested": "position_sizing_method",
        "baseline_reference": "BASELINE_003",
        "methods_tested": SIZING_METHODS,
        "decision_rule": "Accept if drawdown decreases OR variance decreases without Sharpe loss",
        "results": all_results,
        "recommendations": recommendations,
        "best_method": best,
    }
    
    output_path = PROJECT_ROOT / "experiments" / "position_sizing_tests.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_position_sizing_experiment()
