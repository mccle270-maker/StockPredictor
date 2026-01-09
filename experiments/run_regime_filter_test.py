#!/usr/bin/env python3
"""
REGIME FILTER EXPERIMENT
=========================
Experiment ID: REGIME_FILTER_001
Date: 2026-01-07
Purpose: Evaluate regime filter as a RISK CONTROL mechanism

Variable Changed: regime_filter (OFF vs ON)
Decision Rule: Accept ONLY if drawdown improves (Sharpe may slightly decrease)

Uses BASELINE_002 config (z_score_threshold=1.6)
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
# LOAD BASELINE CONFIG (BASELINE_002 with z_score=1.6)
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

BASELINE_CONFIG = BASELINE["config"]
BASELINE_METRICS = BASELINE["aggregate_metrics"]

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================
EXPERIMENT_ID = "REGIME_FILTER_001"


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
    
    return hist, feat_cols


def compute_zscore(predictions: np.ndarray, window: int = 60, min_periods: int = 20) -> np.ndarray:
    """
    Compute rolling z-score for predictions.
    """
    pred_series = pd.Series(predictions)
    rolling_mean = pred_series.rolling(window, min_periods=min_periods).mean()
    rolling_std = pred_series.rolling(window, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    zscore = (pred_series - rolling_mean) / rolling_std
    return zscore.fillna(0).values


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """
    Detect market regime based on:
    - VIX level (if available)
    - 50-day vs 200-day moving average crossover
    - Recent drawdown
    
    Returns: Series with regime labels ('bull', 'bear', 'neutral')
    """
    regime = pd.Series('neutral', index=df.index)
    
    close = df["Close"]
    
    # Moving average crossover
    ma50 = close.rolling(50, min_periods=20).mean()
    ma200 = close.rolling(200, min_periods=50).mean()
    
    # Bull: price above both MAs and MA50 > MA200
    bull_condition = (close > ma50) & (close > ma200) & (ma50 > ma200)
    
    # Bear: price below both MAs and MA50 < MA200
    bear_condition = (close < ma50) & (close < ma200) & (ma50 < ma200)
    
    # Check for high volatility (recent drawdown > 10%)
    rolling_max = close.rolling(60, min_periods=20).max()
    drawdown = (close - rolling_max) / rolling_max
    high_vol = drawdown < -0.10
    
    # Assign regimes
    regime[bull_condition] = 'bull'
    regime[bear_condition] = 'bear'
    regime[high_vol] = 'bear'  # Override to bear if high drawdown
    
    return regime


def run_backtest_with_regime(
    ticker: str,
    regime_filter: bool,
    config: dict,
) -> dict:
    """
    Run backtest with optional regime filter.
    Uses SAME data split and z-score threshold as baseline.
    """
    try:
        hist, feat_cols = prepare_features(ticker, period="10y")
        
        # Add regime detection
        hist["regime"] = detect_regime(hist)
        
        # Prepare final dataframe
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
        
        # Ensemble: average predictions
        y_pred = np.mean(predictions, axis=0)
        
        # Volatility scaling (IDENTICAL to baseline)
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if vol > 0:
            target_vol = 0.01
            vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
        else:
            vol_scale = 1.0
        
        # Z-score gating (from BASELINE_002)
        z_threshold = config["z_score_threshold"]
        z_scores = compute_zscore(y_pred)
        
        threshold = config["trade_threshold"]
        
        # Base positions from predictions
        base_positions = np.where(y_pred > threshold, 1, 
                         np.where(y_pred < -threshold, -1, 0))
        
        # Apply z-score gate
        if z_threshold > 0:
            z_gate = np.abs(z_scores) >= z_threshold
            positions = base_positions * z_gate.astype(float)
        else:
            positions = base_positions
        
        # === REGIME FILTER (THE VARIABLE BEING TESTED) ===
        if regime_filter:
            # In bear regime: reduce position size by 50%
            # In neutral regime: keep full position
            # In bull regime: keep full position
            regime_scale = np.where(test_regimes == 'bear', 0.5,
                          np.where(test_regimes == 'neutral', 0.75, 1.0))
            positions = positions * regime_scale
        
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
        
        # Time in market
        days_in_market = int(np.sum(positions != 0))
        pct_time_in_market = float(days_in_market / len(positions))
        
        # Cumulative return
        cum_return = float(np.sum(strategy_returns))
        
        # Regime breakdown
        bull_days = int(np.sum(test_regimes == 'bull'))
        bear_days = int(np.sum(test_regimes == 'bear'))
        neutral_days = int(np.sum(test_regimes == 'neutral'))
        
        return {
            "ticker": ticker,
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "total_return": round(cum_return, 4),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "days_in_market": days_in_market,
            "pct_time_in_market": round(pct_time_in_market, 4),
            "regime_breakdown": {
                "bull": bull_days,
                "bear": bear_days,
                "neutral": neutral_days
            }
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_regime_filter_experiment():
    """
    Run regime filter experiment: OFF vs ON.
    """
    print("=" * 70)
    print(f"REGIME FILTER EXPERIMENT: {EXPERIMENT_ID}")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Baseline (BASELINE_002) Sharpe: {BASELINE_METRICS['avg_sharpe']:.4f}")
    print(f"Baseline (BASELINE_002) Max DD: {BASELINE_METRICS['avg_max_drawdown']*100:.2f}%")
    print(f"Z-Score Threshold: {BASELINE_CONFIG['z_score_threshold']} (from baseline)")
    print("=" * 70)
    print()
    
    results = {}
    
    for regime_filter in [False, True]:
        label = "ON" if regime_filter else "OFF"
        print(f"\n--- Testing regime_filter = {label} ---")
        
        ticker_results = []
        for ticker in BASELINE_CONFIG["tickers"]:
            print(f"  {ticker}...", end=" ")
            result = run_backtest_with_regime(ticker, regime_filter, BASELINE_CONFIG)
            ticker_results.append(result)
            
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown']*100:.1f}%, InMkt={result['pct_time_in_market']*100:.0f}%")
        
        # Aggregate
        valid = [r for r in ticker_results if "error" not in r]
        
        if valid:
            agg = {
                "regime_filter": label,
                "avg_sharpe": round(np.mean([r["sharpe"] for r in valid]), 4),
                "avg_max_drawdown": round(np.mean([r["max_drawdown"] for r in valid]), 4),
                "avg_win_rate": round(np.mean([r["win_rate"] for r in valid]), 4),
                "avg_pct_time_in_market": round(np.mean([r["pct_time_in_market"] for r in valid]), 4),
                "total_trades": sum(r["trade_count"] for r in valid),
                "per_ticker": ticker_results,
            }
            
            results[label] = agg
            
            print(f"\n  => Avg Sharpe: {agg['avg_sharpe']:.4f}")
            print(f"  => Avg MaxDD:  {agg['avg_max_drawdown']*100:.2f}%")
            print(f"  => Avg Time in Market: {agg['avg_pct_time_in_market']*100:.1f}%")
            print(f"  => Total Trades: {agg['total_trades']}")
    
    # Compare OFF vs ON
    off = results["OFF"]
    on = results["ON"]
    
    sharpe_delta = on["avg_sharpe"] - off["avg_sharpe"]
    dd_delta = on["avg_max_drawdown"] - off["avg_max_drawdown"]  # Less negative = better
    time_delta = on["avg_pct_time_in_market"] - off["avg_pct_time_in_market"]
    trades_delta = on["total_trades"] - off["total_trades"]
    
    # Decision rule: Accept if drawdown improves
    # dd_delta > 0 means ON has less negative drawdown (better)
    drawdown_improved = dd_delta > 0
    sharpe_acceptable = sharpe_delta >= -0.05  # Allow slight Sharpe decrease
    
    accept_regime_filter = drawdown_improved and sharpe_acceptable
    
    # Print comparison table
    print("\n")
    print("=" * 70)
    print("COMPARISON: regime_filter OFF vs ON")
    print("=" * 70)
    print(f"{'Metric':<25} {'OFF':<15} {'ON':<15} {'Delta':<15} {'Better?':<10}")
    print("-" * 70)
    print(f"{'Sharpe':<25} {off['avg_sharpe']:<15.4f} {on['avg_sharpe']:<15.4f} {sharpe_delta:+<15.4f} {'—' if abs(sharpe_delta) < 0.05 else ('ON' if sharpe_delta > 0 else 'OFF'):<10}")
    print(f"{'Max Drawdown':<25} {off['avg_max_drawdown']*100:<15.2f}% {on['avg_max_drawdown']*100:<15.2f}% {dd_delta*100:+<15.2f}% {'ON' if dd_delta > 0 else 'OFF':<10}")
    print(f"{'% Time in Market':<25} {off['avg_pct_time_in_market']*100:<15.1f}% {on['avg_pct_time_in_market']*100:<15.1f}% {time_delta*100:+<15.1f}% {'—':<10}")
    print(f"{'Trade Count':<25} {off['total_trades']:<15} {on['total_trades']:<15} {trades_delta:+<15} {'—':<10}")
    print("=" * 70)
    
    print(f"\n📊 DECISION CRITERIA:")
    print(f"   Drawdown improved? {'✅ YES' if drawdown_improved else '❌ NO'} (Δ{dd_delta*100:+.2f}%)")
    print(f"   Sharpe acceptable? {'✅ YES' if sharpe_acceptable else '❌ NO'} (Δ{sharpe_delta:+.4f}, threshold: -0.05)")
    
    if accept_regime_filter:
        print(f"\n🏆 RECOMMENDATION: ACCEPT regime_filter = ON")
        print(f"   Drawdown improved by {dd_delta*100:.2f}%")
        recommendation = "ACCEPT regime_filter=ON"
    else:
        print(f"\n⚠️ RECOMMENDATION: KEEP regime_filter = OFF")
        if not drawdown_improved:
            print(f"   Reason: Drawdown did not improve")
        if not sharpe_acceptable:
            print(f"   Reason: Sharpe decreased too much ({sharpe_delta:+.4f})")
        recommendation = "KEEP regime_filter=OFF"
    
    # Save results
    output = {
        "experiment_id": EXPERIMENT_ID,
        "created": datetime.now().isoformat(),
        "variable_tested": "regime_filter",
        "baseline_reference": "BASELINE_002",
        "z_score_threshold_used": BASELINE_CONFIG["z_score_threshold"],
        "decision_rule": "Accept if drawdown improves AND Sharpe does not decrease more than 0.05",
        "results": {
            "OFF": off,
            "ON": on
        },
        "comparison": {
            "sharpe_delta": round(sharpe_delta, 4),
            "max_drawdown_delta": round(dd_delta, 4),
            "time_in_market_delta": round(time_delta, 4),
            "trades_delta": trades_delta
        },
        "decision": {
            "drawdown_improved": drawdown_improved,
            "sharpe_acceptable": sharpe_acceptable,
            "accept_regime_filter": accept_regime_filter,
            "recommendation": recommendation
        }
    }
    
    output_path = PROJECT_ROOT / "experiments" / "regime_filter_test.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_regime_filter_experiment()
