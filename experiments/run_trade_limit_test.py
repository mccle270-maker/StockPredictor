#!/usr/bin/env python3
"""
TRADE LIMIT EXPERIMENT
=======================
Experiment ID: TRADE_LIMIT_001
Date: 2026-01-07
Purpose: Test trade frequency constraints to reduce overtrading

Configurations Tested:
A) No limits (baseline)
B) max_trades_per_ticker = 1 per 3 days, max_positions = 10
C) max_trades_per_ticker = 1 per 3 days, max_positions = 8

Uses BASELINE_004 config (z_score=1.6, regime_filter=ON, position_sizing=vol_x_confidence)

Acceptance Criteria:
- Sharpe improves OR
- Drawdown meaningfully decreases
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
# LOAD BASELINE CONFIG (BASELINE_004)
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

BASELINE_CONFIG = BASELINE["config"]
BASELINE_METRICS = BASELINE["aggregate_metrics"]

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
EXPERIMENT_ID = "TRADE_LIMIT_001"

LIMIT_CONFIGS = {
    "A": {"name": "no_limits", "min_days_between_trades": 0, "max_positions": 10},
    "B": {"name": "1_per_3d_max10", "min_days_between_trades": 3, "max_positions": 10},
    "C": {"name": "1_per_3d_max8", "min_days_between_trades": 3, "max_positions": 8},
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
    """Compute confidence score from prediction magnitude and z-score."""
    pred_abs = np.abs(y_pred)
    pred_pct = np.clip(pred_abs / np.percentile(pred_abs[pred_abs > 0], 95), 0, 1)
    z_abs = np.abs(z_scores)
    z_pct = np.clip(z_abs / 3.0, 0, 1)
    confidence = (pred_pct + z_pct) / 2
    return confidence


def apply_trade_limits(
    positions: np.ndarray,
    min_days_between_trades: int,
    max_positions: int,
    predictions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Apply trade frequency constraints.
    
    Returns: (constrained_positions, stats_dict)
    """
    n = len(positions)
    constrained = positions.copy()
    
    # Track last trade day for each position
    last_trade_day = -999  # Days since last trade
    days_in_position = 0
    total_trades = 0
    trade_durations = []
    
    # Track position changes
    prev_pos = 0
    
    for i in range(n):
        current_signal = positions[i]
        days_since_last = i - last_trade_day
        
        # Check if we want to change position
        wants_to_trade = (current_signal != 0 and prev_pos == 0) or \
                        (current_signal == 0 and prev_pos != 0) or \
                        (current_signal * prev_pos < 0)  # Flip direction
        
        if wants_to_trade and min_days_between_trades > 0:
            # Enforce minimum days between trades
            if days_since_last < min_days_between_trades:
                # Can't trade yet, hold previous position
                constrained[i] = prev_pos
            else:
                # Can trade
                if current_signal != 0:
                    last_trade_day = i
                    total_trades += 1
                    if days_in_position > 0:
                        trade_durations.append(days_in_position)
                    days_in_position = 1
                else:
                    # Closing position
                    if days_in_position > 0:
                        trade_durations.append(days_in_position)
                    days_in_position = 0
        else:
            # Either not trading or no limits
            if current_signal != 0:
                days_in_position += 1
            else:
                if days_in_position > 0:
                    trade_durations.append(days_in_position)
                days_in_position = 0
        
        prev_pos = constrained[i]
    
    # Final duration
    if days_in_position > 0:
        trade_durations.append(days_in_position)
    
    stats = {
        "total_trades": total_trades if min_days_between_trades > 0 else int(np.count_nonzero(np.diff(positions))),
        "avg_trade_duration": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "max_trade_duration": int(np.max(trade_durations)) if trade_durations else 0,
    }
    
    return constrained, stats


def run_backtest_with_limits(
    ticker: str,
    limit_config: dict,
    base_config: dict,
) -> dict:
    """
    Run backtest with specified trade limits.
    """
    try:
        hist, feat_cols = prepare_features(ticker, period="10y")
        hist["regime"] = detect_regime(hist)
        
        df = hist[feat_cols + ["target", "Close", "regime"]].dropna().copy()
        
        if len(df) < base_config["min_train_rows"] + 100:
            return {"ticker": ticker, "error": "Insufficient data"}
        
        # Split: IDENTICAL to baseline
        n = len(df)
        split_idx = int(n * (1 - base_config["test_fraction"]))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        test_regimes = test_df["regime"].values
        
        # Train ensemble models (IDENTICAL to baseline)
        predictions = []
        for model_type in base_config["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        y_pred = np.mean(predictions, axis=0)
        
        # Z-score gating (from baseline)
        z_threshold = base_config["z_score_threshold"]
        z_scores = compute_zscore(y_pred)
        
        threshold = base_config["trade_threshold"]
        
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
        if base_config["regime_filter"]:
            regime_scale = np.where(test_regimes == 'bear', 0.5,
                          np.where(test_regimes == 'neutral', 0.75, 1.0))
            positions = positions * regime_scale
        
        # Position sizing: vol × confidence (from baseline)
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if vol > 0:
            target_vol = 0.01
            vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
        else:
            vol_scale = 1.0
        
        confidence = compute_confidence_score(y_pred, z_scores)
        positions = positions * vol_scale * confidence
        
        # === TRADE LIMITS (THE VARIABLE BEING TESTED) ===
        min_days = limit_config["min_days_between_trades"]
        max_pos = limit_config["max_positions"]
        
        constrained_positions, trade_stats = apply_trade_limits(
            positions, min_days, max_pos, y_pred
        )
        
        # Calculate returns
        strategy_returns = constrained_positions * y_test
        
        # Metrics
        sharpe = compute_sharpe(strategy_returns)
        dd_df = compute_drawdown(pd.Series(strategy_returns))
        max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
        
        # Trade statistics
        trade_count = trade_stats["total_trades"]
        avg_duration = trade_stats["avg_trade_duration"]
        
        # Turnover: average absolute position change per day
        pos_changes = np.abs(np.diff(constrained_positions))
        turnover = float(np.mean(pos_changes))
        
        # Win rate
        winning_trades = np.sum((constrained_positions[:-1] != 0) & (strategy_returns[:-1] > 0))
        total_active = np.sum(constrained_positions[:-1] != 0)
        win_rate = float(winning_trades / total_active) if total_active > 0 else 0.0
        
        cum_return = float(np.sum(strategy_returns))
        
        return {
            "ticker": ticker,
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(cum_return, 4),
            "trade_count": trade_count,
            "avg_trade_duration": round(avg_duration, 2),
            "turnover": round(turnover, 6),
            "win_rate": round(win_rate, 4),
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_trade_limit_experiment():
    """Run trade limit experiment across all configurations."""
    print("=" * 70)
    print(f"TRADE LIMIT EXPERIMENT: {EXPERIMENT_ID}")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Baseline (BASELINE_004) Sharpe: {BASELINE_METRICS['avg_sharpe']:.4f}")
    print(f"Baseline (BASELINE_004) Max DD: {BASELINE_METRICS['avg_max_drawdown']*100:.2f}%")
    print(f"Configs: A=no_limits, B=1per3d_max10, C=1per3d_max8")
    print("=" * 70)
    print()
    
    all_results = {}
    
    for config_key, limit_config in LIMIT_CONFIGS.items():
        print(f"\n--- Testing Config {config_key}: {limit_config['name']} ---")
        print(f"    min_days_between_trades={limit_config['min_days_between_trades']}, max_positions={limit_config['max_positions']}")
        
        ticker_results = []
        for ticker in BASELINE_CONFIG["tickers"]:
            print(f"  {ticker}...", end=" ")
            result = run_backtest_with_limits(ticker, limit_config, BASELINE_CONFIG)
            ticker_results.append(result)
            
            if "error" in result:
                print(f"ERROR")
            else:
                print(f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown']*100:.1f}%, AvgDur={result['avg_trade_duration']:.1f}d")
        
        # Aggregate
        valid = [r for r in ticker_results if "error" not in r]
        
        if valid:
            agg = {
                "config": config_key,
                "config_name": limit_config["name"],
                "min_days_between_trades": limit_config["min_days_between_trades"],
                "max_positions": limit_config["max_positions"],
                "avg_sharpe": round(np.mean([r["sharpe"] for r in valid]), 4),
                "avg_max_drawdown": round(np.mean([r["max_drawdown"] for r in valid]), 4),
                "avg_trade_duration": round(np.mean([r["avg_trade_duration"] for r in valid]), 2),
                "avg_turnover": round(np.mean([r["turnover"] for r in valid]), 6),
                "avg_win_rate": round(np.mean([r["win_rate"] for r in valid]), 4),
                "total_trades": sum(r["trade_count"] for r in valid),
                "per_ticker": ticker_results,
            }
            
            all_results[config_key] = agg
            
            print(f"\n  => Avg Sharpe:       {agg['avg_sharpe']:.4f}")
            print(f"  => Avg MaxDD:        {agg['avg_max_drawdown']*100:.2f}%")
            print(f"  => Avg Trade Dur:    {agg['avg_trade_duration']:.1f} days")
            print(f"  => Avg Turnover:     {agg['avg_turnover']:.6f}")
            print(f"  => Total Trades:     {agg['total_trades']}")
    
    # Compare methods
    baseline = all_results.get("A", {})
    
    print("\n")
    print("=" * 85)
    print("COMPARISON TABLE")
    print("=" * 85)
    print(f"{'Config':<25} {'Sharpe':<10} {'MaxDD':<10} {'AvgDur':<10} {'Turnover':<12} {'Trades':<8}")
    print("-" * 85)
    
    for key in ["A", "B", "C"]:
        r = all_results.get(key, {})
        if r:
            print(f"{key}: {r['config_name']:<21} {r['avg_sharpe']:<10.4f} {r['avg_max_drawdown']*100:<10.2f}% {r['avg_trade_duration']:<10.1f}d {r['avg_turnover']:<12.6f} {r['total_trades']:<8}")
    
    print("=" * 85)
    
    # Decision logic
    baseline_sharpe = baseline.get("avg_sharpe", 0)
    baseline_dd = baseline.get("avg_max_drawdown", 0)
    
    recommendations = []
    
    for key in ["B", "C"]:
        r = all_results.get(key, {})
        if not r:
            continue
            
        sharpe_delta = r["avg_sharpe"] - baseline_sharpe
        dd_delta = r["avg_max_drawdown"] - baseline_dd  # Less negative = better
        
        sharpe_improved = sharpe_delta > 0
        dd_meaningfully_improved = dd_delta > 0.01  # At least 1% improvement
        
        accept = sharpe_improved or dd_meaningfully_improved
        
        rec = {
            "config": key,
            "config_name": r["config_name"],
            "sharpe_delta": round(sharpe_delta, 4),
            "dd_delta": round(dd_delta, 4),
            "sharpe_improved": sharpe_improved,
            "dd_meaningfully_improved": dd_meaningfully_improved,
            "accept": accept,
        }
        recommendations.append(rec)
        
        print(f"\n📊 Config {key} vs Baseline (A):")
        print(f"   Sharpe Δ:   {sharpe_delta:+.4f} {'✅ improved' if sharpe_improved else '—'}")
        print(f"   MaxDD Δ:    {dd_delta*100:+.2f}% {'✅ meaningfully improved' if dd_meaningfully_improved else '—'}")
        print(f"   Accept?     {'✅ YES' if accept else '❌ NO'}")
    
    # Find best acceptable config
    best = None
    for rec in recommendations:
        if rec["accept"]:
            # Prefer higher Sharpe, then better drawdown
            if best is None:
                best = rec
            elif rec["sharpe_delta"] > best["sharpe_delta"]:
                best = rec
            elif rec["sharpe_delta"] == best["sharpe_delta"] and rec["dd_delta"] > best["dd_delta"]:
                best = rec
    
    if best:
        print(f"\n🏆 BEST ACCEPTABLE: Config {best['config']} ({best['config_name']})")
    else:
        print(f"\n⚠️ NO IMPROVEMENT - Keep Config A (no_limits)")
        best = {"config": "A", "config_name": "no_limits", "accept": False}
    
    # Save results
    output = {
        "experiment_id": EXPERIMENT_ID,
        "created": datetime.now().isoformat(),
        "variable_tested": "trade_limits",
        "baseline_reference": "BASELINE_004",
        "configs_tested": LIMIT_CONFIGS,
        "decision_rule": "Accept if Sharpe improves OR drawdown meaningfully decreases (>1%)",
        "results": all_results,
        "recommendations": recommendations,
        "best_config": best,
    }
    
    output_path = PROJECT_ROOT / "experiments" / "trade_limit_tests.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_trade_limit_experiment()
