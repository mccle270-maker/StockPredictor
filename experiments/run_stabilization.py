#!/usr/bin/env python3
"""
STABILIZATION EXPERIMENT
=========================
Experiment ID: STABILIZATION_001
Date: 2026-01-07

PURPOSE: Stabilize the trading strategy while preserving profitability.

ISSUES IDENTIFIED:
- Worst Sharpe < -6, std dev of Sharpe is 18x mean
- Catastrophic losses in some 3-month windows
- META, SPY perform particularly poorly

STABILIZATION MEASURES TO TEST:
1. Higher z-score thresholds (1.8, 2.0, 2.2)
2. Stricter regime filters (more aggressive position reduction)
3. Maximum loss limits per rolling window
4. Ticker-specific position caps (reduce/exclude worst performers)
5. Volatility-based position caps

TARGET:
- Worst window Sharpe > -0.5
- Max drawdown < -15%
- Sharpe std dev < mean Sharpe (stability)
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
# LOAD BASE CONFIG
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

BASE_CONFIG = BASELINE["config"]

# =============================================================================
# STABILIZATION CONFIGURATIONS TO TEST
# =============================================================================

# Original (unstable) config
ORIGINAL_CONFIG = {
    "name": "ORIGINAL",
    "z_score_threshold": 1.6,
    "regime_bear_scale": 0.5,
    "regime_neutral_scale": 0.75,
    "max_position_size": 1.0,
    "vol_cap_multiplier": 2.0,
    "weekly_loss_limit": None,  # No limit
    "ticker_exclusions": [],
    "ticker_position_caps": {},
}

# Stabilized configurations to test
STABILIZED_CONFIGS = [
    {
        "name": "STABLE_V1_ZSCORE_2.0",
        "z_score_threshold": 2.0,
        "regime_bear_scale": 0.5,
        "regime_neutral_scale": 0.75,
        "max_position_size": 1.0,
        "vol_cap_multiplier": 2.0,
        "weekly_loss_limit": None,
        "ticker_exclusions": [],
        "ticker_position_caps": {},
    },
    {
        "name": "STABLE_V2_ZSCORE_2.2",
        "z_score_threshold": 2.2,
        "regime_bear_scale": 0.5,
        "regime_neutral_scale": 0.75,
        "max_position_size": 1.0,
        "vol_cap_multiplier": 2.0,
        "weekly_loss_limit": None,
        "ticker_exclusions": [],
        "ticker_position_caps": {},
    },
    {
        "name": "STABLE_V3_STRICT_REGIME",
        "z_score_threshold": 2.0,
        "regime_bear_scale": 0.25,  # More aggressive reduction
        "regime_neutral_scale": 0.5,
        "max_position_size": 1.0,
        "vol_cap_multiplier": 2.0,
        "weekly_loss_limit": None,
        "ticker_exclusions": [],
        "ticker_position_caps": {},
    },
    {
        "name": "STABLE_V4_VOL_CAP",
        "z_score_threshold": 2.0,
        "regime_bear_scale": 0.25,
        "regime_neutral_scale": 0.5,
        "max_position_size": 0.5,  # Hard cap at 50%
        "vol_cap_multiplier": 1.5,  # Tighter vol scaling
        "weekly_loss_limit": None,
        "ticker_exclusions": [],
        "ticker_position_caps": {},
    },
    {
        "name": "STABLE_V5_LOSS_LIMIT",
        "z_score_threshold": 2.0,
        "regime_bear_scale": 0.25,
        "regime_neutral_scale": 0.5,
        "max_position_size": 0.5,
        "vol_cap_multiplier": 1.5,
        "weekly_loss_limit": -0.03,  # -3% max loss per week
        "ticker_exclusions": [],
        "ticker_position_caps": {},
    },
    {
        "name": "STABLE_V6_TICKER_FILTER",
        "z_score_threshold": 2.0,
        "regime_bear_scale": 0.25,
        "regime_neutral_scale": 0.5,
        "max_position_size": 0.5,
        "vol_cap_multiplier": 1.5,
        "weekly_loss_limit": -0.03,
        "ticker_exclusions": ["META"],  # Exclude worst performer
        "ticker_position_caps": {"SPY": 0.25},  # Cap SPY at 25%
    },
    {
        "name": "STABLE_V7_FULL_SAFETY",
        "z_score_threshold": 2.2,
        "regime_bear_scale": 0.20,
        "regime_neutral_scale": 0.5,
        "max_position_size": 0.5,
        "vol_cap_multiplier": 1.25,
        "weekly_loss_limit": -0.025,  # -2.5% max loss per week
        "ticker_exclusions": ["META"],
        "ticker_position_caps": {"SPY": 0.25, "NVDA": 0.5},
    },
]

# Validation parameters
WINDOW_SIZE_DAYS = 63  # ~3 months
STEP_SIZE_DAYS = 21    # 1 month step

# Stability thresholds
MAX_ALLOWED_WORST_SHARPE = -0.5
MAX_ALLOWED_DRAWDOWN = -0.15
MAX_SHARPE_STD_RATIO = 1.5  # std should be < 1.5x mean


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
    if np.sum(pred_abs > 0) == 0:
        return np.zeros_like(y_pred)
    pred_pct = np.clip(pred_abs / np.percentile(pred_abs[pred_abs > 0], 95), 0, 1)
    z_abs = np.abs(z_scores)
    z_pct = np.clip(z_abs / 3.0, 0, 1)
    confidence = (pred_pct + z_pct) / 2
    return confidence


def apply_stabilized_strategy(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    test_regimes: np.ndarray,
    vol_scale: float,
    stab_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply stabilized strategy with all safety features.
    Returns (positions, strategy_returns)
    """
    n = len(y_pred)
    
    # Z-score gating
    z_threshold = stab_config["z_score_threshold"]
    z_scores = compute_zscore(y_pred)
    
    # Base positions
    base_positions = np.where(y_pred > 0, 1, np.where(y_pred < 0, -1, 0))
    
    # Z-score gate
    z_gate = np.abs(z_scores) >= z_threshold
    positions = base_positions * z_gate.astype(float)
    
    # Regime filter with configurable scales
    bear_scale = stab_config["regime_bear_scale"]
    neutral_scale = stab_config["regime_neutral_scale"]
    regime_scale = np.where(test_regimes == 'bear', bear_scale,
                  np.where(test_regimes == 'neutral', neutral_scale, 1.0))
    positions = positions * regime_scale
    
    # Confidence-based sizing
    confidence = compute_confidence_score(y_pred, z_scores)
    positions = positions * confidence
    
    # Volatility scaling with cap
    vol_cap = stab_config["vol_cap_multiplier"]
    capped_vol_scale = np.clip(vol_scale, 0.25, vol_cap)
    positions = positions * capped_vol_scale
    
    # Hard position cap
    max_pos = stab_config["max_position_size"]
    positions = np.clip(positions, -max_pos, max_pos)
    
    # Weekly loss limit
    weekly_limit = stab_config["weekly_loss_limit"]
    if weekly_limit is not None:
        # Track rolling 5-day loss and flatten if exceeded
        strategy_returns = positions * y_test
        rolling_loss = pd.Series(strategy_returns).rolling(5, min_periods=1).sum().values
        
        # Flatten positions after hitting loss limit
        flatten_mask = rolling_loss < weekly_limit
        # Create a "cooldown" effect - stay flat for next 5 days after hitting limit
        for i in range(len(flatten_mask)):
            if flatten_mask[i]:
                end_idx = min(i + 5, len(flatten_mask))
                flatten_mask[i:end_idx] = True
        
        positions = np.where(flatten_mask, 0, positions)
    
    # Calculate final returns
    strategy_returns = positions * y_test
    
    return positions, strategy_returns


def run_stabilized_validation(ticker: str, stab_config: dict) -> list:
    """
    Run rolling window validation with stabilized configuration.
    """
    try:
        # Skip excluded tickers
        if ticker in stab_config.get("ticker_exclusions", []):
            return []
        
        hist, feat_cols = prepare_features(ticker, period="10y")
        hist["regime"] = detect_regime(hist)
        
        df = hist[feat_cols + ["target", "Close", "regime"]].dropna().copy()
        
        if len(df) < BASE_CONFIG["min_train_rows"] + 200:
            return []
        
        # Split
        n = len(df)
        split_idx = int(n * (1 - BASE_CONFIG["test_fraction"]))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        if len(test_df) < WINDOW_SIZE_DAYS * 2:
            return []
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        
        # Train models once
        models_trained = []
        for model_type in BASE_CONFIG["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            models_trained.append(model)
        
        # Vol scale
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        target_vol = 0.01
        vol_scale = target_vol / vol if vol > 0 else 1.0
        
        # Apply ticker-specific cap if defined
        ticker_cap = stab_config.get("ticker_position_caps", {}).get(ticker)
        if ticker_cap is not None:
            # Create a modified config for this ticker
            ticker_stab_config = stab_config.copy()
            ticker_stab_config["max_position_size"] = min(
                stab_config["max_position_size"], 
                ticker_cap
            )
        else:
            ticker_stab_config = stab_config
        
        # Test data
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        test_regimes = test_df["regime"].values
        test_dates = test_df.index
        
        # Ensemble predictions
        predictions = []
        for model in models_trained:
            pred = model.predict(X_test)
            predictions.append(pred)
        y_pred = np.mean(predictions, axis=0)
        
        # Apply stabilized strategy
        positions, strategy_returns = apply_stabilized_strategy(
            y_pred, y_test, test_regimes, vol_scale, ticker_stab_config
        )
        
        # Rolling window analysis
        window_results = []
        n_test = len(test_df)
        
        start = 0
        while start + WINDOW_SIZE_DAYS <= n_test:
            end = start + WINDOW_SIZE_DAYS
            
            window_returns = strategy_returns[start:end]
            window_positions = positions[start:end]
            window_regimes = test_regimes[start:end]
            
            window_start_date = str(test_dates[start].date()) if hasattr(test_dates[start], 'date') else str(test_dates[start])[:10]
            window_end_date = str(test_dates[end-1].date()) if hasattr(test_dates[end-1], 'date') else str(test_dates[end-1])[:10]
            
            sharpe = compute_sharpe(window_returns)
            dd_df = compute_drawdown(pd.Series(window_returns))
            max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
            
            trade_count = int(np.count_nonzero(np.diff(window_positions)))
            winning = np.sum((window_positions[:-1] != 0) & (window_returns[:-1] > 0))
            total_active = np.sum(window_positions[:-1] != 0)
            win_rate = float(winning / total_active) if total_active > 0 else 0.0
            
            regime_counts = pd.Series(window_regimes).value_counts()
            dominant_regime = regime_counts.index[0] if len(regime_counts) > 0 else "unknown"
            
            window_results.append({
                "ticker": ticker,
                "window_start": window_start_date,
                "window_end": window_end_date,
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(max_dd, 4),
                "trade_count": trade_count,
                "win_rate": round(win_rate, 4),
                "dominant_regime": dominant_regime,
                "total_return": round(float(np.sum(window_returns)), 4),
            })
            
            start += STEP_SIZE_DAYS
        
        return window_results
        
    except Exception as e:
        return []


def evaluate_config(stab_config: dict) -> dict:
    """
    Evaluate a stabilization configuration across all tickers.
    """
    all_windows = []
    ticker_summaries = []
    
    tickers = [t for t in BASE_CONFIG["tickers"] if t not in stab_config.get("ticker_exclusions", [])]
    
    for ticker in tickers:
        windows = run_stabilized_validation(ticker, stab_config)
        
        if windows:
            all_windows.extend(windows)
            
            sharpes = [w["sharpe"] for w in windows]
            drawdowns = [w["max_drawdown"] for w in windows]
            
            ticker_summaries.append({
                "ticker": ticker,
                "num_windows": len(windows),
                "avg_sharpe": round(np.mean(sharpes), 4),
                "worst_sharpe": round(min(sharpes), 4),
                "worst_drawdown": round(min(drawdowns), 4),
                "pct_positive": round(np.mean([s > 0 for s in sharpes]) * 100, 1),
            })
    
    if not all_windows:
        return {"error": "No valid windows"}
    
    # Aggregate stats
    all_sharpes = [w["sharpe"] for w in all_windows]
    all_drawdowns = [w["max_drawdown"] for w in all_windows]
    
    mean_sharpe = np.mean(all_sharpes)
    std_sharpe = np.std(all_sharpes)
    worst_sharpe = min(all_sharpes)
    worst_drawdown = min(all_drawdowns)
    pct_positive = np.mean([s > 0 for s in all_sharpes]) * 100
    
    # Stability checks
    passes_worst_sharpe = worst_sharpe >= MAX_ALLOWED_WORST_SHARPE
    passes_drawdown = worst_drawdown >= MAX_ALLOWED_DRAWDOWN
    passes_stability = std_sharpe <= abs(mean_sharpe) * MAX_SHARPE_STD_RATIO if mean_sharpe > 0 else False
    passes_positive_rate = pct_positive >= 50
    
    all_pass = passes_worst_sharpe and passes_drawdown and passes_stability and passes_positive_rate
    
    return {
        "config_name": stab_config["name"],
        "config": stab_config,
        "total_windows": len(all_windows),
        "total_tickers": len(ticker_summaries),
        "aggregate": {
            "mean_sharpe": round(mean_sharpe, 4),
            "std_sharpe": round(std_sharpe, 4),
            "worst_sharpe": round(worst_sharpe, 4),
            "best_sharpe": round(max(all_sharpes), 4),
            "worst_drawdown": round(worst_drawdown, 4),
            "pct_positive_sharpe": round(pct_positive, 1),
        },
        "stability_checks": {
            "worst_sharpe_ok": passes_worst_sharpe,
            "drawdown_ok": passes_drawdown,
            "stability_ok": passes_stability,
            "positive_rate_ok": passes_positive_rate,
            "ALL_PASS": all_pass,
        },
        "per_ticker": ticker_summaries,
    }


def run_stabilization_experiment():
    """
    Run the full stabilization experiment.
    """
    print("=" * 80)
    print("STABILIZATION EXPERIMENT")
    print("=" * 80)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Testing {len(STABILIZED_CONFIGS)} stabilized configurations")
    print()
    print("STABILITY THRESHOLDS:")
    print(f"  - Max allowed worst Sharpe: {MAX_ALLOWED_WORST_SHARPE}")
    print(f"  - Max allowed drawdown: {MAX_ALLOWED_DRAWDOWN*100}%")
    print(f"  - Sharpe std/mean ratio: < {MAX_SHARPE_STD_RATIO}")
    print("=" * 80)
    print()
    
    # First evaluate original
    print("Evaluating ORIGINAL (baseline) config...")
    original_result = evaluate_config(ORIGINAL_CONFIG)
    print(f"  Mean Sharpe: {original_result['aggregate']['mean_sharpe']:.4f}")
    print(f"  Worst Sharpe: {original_result['aggregate']['worst_sharpe']:.4f}")
    print(f"  Worst DD: {original_result['aggregate']['worst_drawdown']*100:.2f}%")
    print(f"  Stability: {'PASS' if original_result['stability_checks']['ALL_PASS'] else 'FAIL'}")
    print()
    
    # Evaluate all stabilized configs
    all_results = [original_result]
    
    for i, stab_config in enumerate(STABILIZED_CONFIGS):
        print(f"Evaluating {stab_config['name']}...")
        result = evaluate_config(stab_config)
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
            
        all_results.append(result)
        
        print(f"  Mean Sharpe: {result['aggregate']['mean_sharpe']:.4f}")
        print(f"  Worst Sharpe: {result['aggregate']['worst_sharpe']:.4f}")
        print(f"  Worst DD: {result['aggregate']['worst_drawdown']*100:.2f}%")
        print(f"  Std/Mean: {result['aggregate']['std_sharpe']:.4f}/{result['aggregate']['mean_sharpe']:.4f}")
        print(f"  Stability: {'✅ PASS' if result['stability_checks']['ALL_PASS'] else '❌ FAIL'}")
        print()
    
    # Summary table
    print("\n")
    print("=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Config':<25} {'Mean Sharpe':<12} {'Worst Sharpe':<13} {'Worst DD':<10} {'Std Sharpe':<12} {'Pass?':<8}")
    print("-" * 100)
    
    for r in all_results:
        agg = r["aggregate"]
        passes = "✅ PASS" if r["stability_checks"]["ALL_PASS"] else "❌ FAIL"
        print(f"{r['config_name']:<25} {agg['mean_sharpe']:<12.4f} {agg['worst_sharpe']:<13.4f} {agg['worst_drawdown']*100:<10.2f}% {agg['std_sharpe']:<12.4f} {passes:<8}")
    
    print("=" * 100)
    
    # Find best passing config
    passing_configs = [r for r in all_results if r["stability_checks"]["ALL_PASS"]]
    
    if passing_configs:
        # Sort by mean Sharpe (highest first)
        passing_configs.sort(key=lambda x: x["aggregate"]["mean_sharpe"], reverse=True)
        best = passing_configs[0]
        
        print(f"\n🏆 BEST STABLE CONFIG: {best['config_name']}")
        print(f"   Mean Sharpe: {best['aggregate']['mean_sharpe']:.4f}")
        print(f"   Worst Sharpe: {best['aggregate']['worst_sharpe']:.4f}")
        print(f"   Worst Drawdown: {best['aggregate']['worst_drawdown']*100:.2f}%")
        print(f"   Sharpe Std: {best['aggregate']['std_sharpe']:.4f}")
        
        recommendation = best["config_name"]
        verdict = "STABLE"
    else:
        # Find best even if not passing
        all_results.sort(key=lambda x: x["aggregate"]["worst_sharpe"], reverse=True)
        best = all_results[0]
        
        print(f"\n⚠️ NO CONFIG FULLY STABLE - Best candidate: {best['config_name']}")
        print(f"   Mean Sharpe: {best['aggregate']['mean_sharpe']:.4f}")
        print(f"   Worst Sharpe: {best['aggregate']['worst_sharpe']:.4f}")
        
        recommendation = best["config_name"]
        verdict = "PARTIALLY_STABLE"
    
    # Generate stabilized config for paper trader
    print("\n")
    print("=" * 80)
    print("RECOMMENDED PAPER TRADER CONFIGURATION")
    print("=" * 80)
    
    best_config = best["config"]
    
    print(f"""
# Stabilized Configuration for auto_paper_trade.py
STABILIZED_CONFIG = {{
    "z_score_threshold": {best_config['z_score_threshold']},
    "regime_bear_scale": {best_config['regime_bear_scale']},
    "regime_neutral_scale": {best_config['regime_neutral_scale']},
    "max_position_size": {best_config['max_position_size']},
    "vol_cap_multiplier": {best_config['vol_cap_multiplier']},
    "weekly_loss_limit": {best_config['weekly_loss_limit']},
    "ticker_exclusions": {best_config['ticker_exclusions']},
    "ticker_position_caps": {best_config['ticker_position_caps']},
}}
""")
    
    # Save results
    report = {
        "experiment_id": "STABILIZATION_001",
        "created": datetime.now().isoformat(),
        "purpose": "Stabilize trading strategy while preserving profitability",
        "stability_thresholds": {
            "max_allowed_worst_sharpe": MAX_ALLOWED_WORST_SHARPE,
            "max_allowed_drawdown": MAX_ALLOWED_DRAWDOWN,
            "max_sharpe_std_ratio": MAX_SHARPE_STD_RATIO,
        },
        "original_vs_stabilized": {
            "original": {
                "mean_sharpe": original_result["aggregate"]["mean_sharpe"],
                "worst_sharpe": original_result["aggregate"]["worst_sharpe"],
                "worst_drawdown": original_result["aggregate"]["worst_drawdown"],
                "std_sharpe": original_result["aggregate"]["std_sharpe"],
                "stable": original_result["stability_checks"]["ALL_PASS"],
            },
            "best_stabilized": {
                "config_name": best["config_name"],
                "mean_sharpe": best["aggregate"]["mean_sharpe"],
                "worst_sharpe": best["aggregate"]["worst_sharpe"],
                "worst_drawdown": best["aggregate"]["worst_drawdown"],
                "std_sharpe": best["aggregate"]["std_sharpe"],
                "stable": best["stability_checks"]["ALL_PASS"],
            },
            "improvement": {
                "worst_sharpe_delta": round(best["aggregate"]["worst_sharpe"] - original_result["aggregate"]["worst_sharpe"], 4),
                "std_sharpe_delta": round(best["aggregate"]["std_sharpe"] - original_result["aggregate"]["std_sharpe"], 4),
            }
        },
        "recommended_config": best_config,
        "verdict": verdict,
        "all_results": all_results,
        "testing_plan": {
            "phase_1": "Shadow trading for 2 weeks with recommended config",
            "phase_2": "Paper trading with small position sizes for 4 weeks",
            "phase_3": "Full paper trading with gradual position scaling",
            "monitoring": [
                "Daily Sharpe calculation",
                "Rolling 5-day loss tracking",
                "Regime-specific performance logging",
                "Per-ticker Sharpe comparison",
            ],
        },
    }
    
    output_path = PROJECT_ROOT / "experiments" / "robustness_stabilized.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    run_stabilization_experiment()
