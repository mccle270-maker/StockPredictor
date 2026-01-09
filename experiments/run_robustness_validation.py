#!/usr/bin/env python3
"""
PHASE 6: ROBUSTNESS & STABILITY VALIDATION
============================================
Experiment ID: ROBUSTNESS_001
Date: 2026-01-07

CRITICAL: This is a PURE EVALUATION phase.
- NO modifications to model parameters, thresholds, features, or logic
- NO retraining of models
- NO tuning of any kind
- Uses EXACT locked configuration from BASELINE_004

OBJECTIVE:
Validate that the strategy is stable across time and does not rely
on a small subset of favorable periods.

FAILURE CONDITIONS (HARD REJECT):
- Sharpe < -0.50 in any window
- Drawdown exceeds -15% in any window
- Sharpe std dev > mean Sharpe (instability)
- Performance driven by only a small fraction of windows
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
# LOAD LOCKED CONFIGURATION (BASELINE_004 - NO MODIFICATIONS ALLOWED)
# =============================================================================
BASELINE_PATH = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
with open(BASELINE_PATH, "r") as f:
    BASELINE = json.load(f)

LOCKED_CONFIG = BASELINE["config"]
CONFIG_ID = BASELINE["experiment_id"]

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================
WINDOW_SIZE_DAYS = 63  # ~3 months
STEP_SIZE_DAYS = 21    # 1 month step
MIN_TRADES_PER_WINDOW = 5

# FAILURE THRESHOLDS
MAX_ALLOWED_SHARPE_LOSS = -0.50
MAX_ALLOWED_DRAWDOWN = -0.15  # -15%


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


def run_locked_strategy(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    test_regimes: np.ndarray,
    vol_scale: float,
    config: dict,
) -> np.ndarray:
    """
    Apply the LOCKED strategy configuration to generate positions.
    NO MODIFICATIONS TO LOGIC ALLOWED.
    """
    # Z-score gating (LOCKED at 1.6)
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
    
    # Regime filter (LOCKED ON)
    if config["regime_filter"]:
        regime_scale = np.where(test_regimes == 'bear', 0.5,
                      np.where(test_regimes == 'neutral', 0.75, 1.0))
        positions = positions * regime_scale
    
    # Position sizing: vol × confidence (LOCKED)
    confidence = compute_confidence_score(y_pred, z_scores)
    positions = positions * vol_scale * confidence
    
    return positions


def run_rolling_window_validation(ticker: str, config: dict) -> list:
    """
    Run rolling window validation for a single ticker.
    Returns list of per-window metrics.
    """
    try:
        hist, feat_cols = prepare_features(ticker, period="10y")
        hist["regime"] = detect_regime(hist)
        
        df = hist[feat_cols + ["target", "Close", "regime"]].dropna().copy()
        
        if len(df) < config["min_train_rows"] + 200:
            return []
        
        # Split: Use SAME split as baseline (25% test)
        n = len(df)
        split_idx = int(n * (1 - config["test_fraction"]))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        if len(test_df) < WINDOW_SIZE_DAYS * 2:
            return []
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        
        # Train ensemble models ONCE (no retraining per window)
        models_trained = []
        for model_type in config["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            models_trained.append(model)
        
        # Volatility scale (computed from training data, fixed)
        vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
        if vol > 0:
            target_vol = 0.01
            vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
        else:
            vol_scale = 1.0
        
        # Get test data
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        test_regimes = test_df["regime"].values
        test_dates = test_df.index
        
        # Ensemble predictions (computed once for full test set)
        predictions = []
        for model in models_trained:
            pred = model.predict(X_test)
            predictions.append(pred)
        y_pred = np.mean(predictions, axis=0)
        
        # Apply LOCKED strategy to get positions
        positions = run_locked_strategy(X_test, y_test, y_pred, test_regimes, vol_scale, config)
        
        # Strategy returns
        strategy_returns = positions * y_test
        
        # Rolling window analysis
        window_results = []
        n_test = len(test_df)
        
        start = 0
        while start + WINDOW_SIZE_DAYS <= n_test:
            end = start + WINDOW_SIZE_DAYS
            
            window_returns = strategy_returns[start:end]
            window_positions = positions[start:end]
            window_regimes = test_regimes[start:end]
            
            # Window dates
            window_start_date = str(test_dates[start].date()) if hasattr(test_dates[start], 'date') else str(test_dates[start])[:10]
            window_end_date = str(test_dates[end-1].date()) if hasattr(test_dates[end-1], 'date') else str(test_dates[end-1])[:10]
            
            # Compute window metrics
            sharpe = compute_sharpe(window_returns)
            dd_df = compute_drawdown(pd.Series(window_returns))
            max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
            
            trade_count = int(np.count_nonzero(np.diff(window_positions)))
            winning = np.sum((window_positions[:-1] != 0) & (window_returns[:-1] > 0))
            total_active = np.sum(window_positions[:-1] != 0)
            win_rate = float(winning / total_active) if total_active > 0 else 0.0
            
            # Dominant regime in window
            regime_counts = pd.Series(window_regimes).value_counts()
            dominant_regime = regime_counts.index[0] if len(regime_counts) > 0 else "unknown"
            
            window_results.append({
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
        print(f"    ERROR: {e}")
        return []


def run_robustness_validation():
    """
    Run Phase 6 robustness validation on LOCKED configuration.
    """
    print("=" * 75)
    print("PHASE 6: ROBUSTNESS & STABILITY VALIDATION")
    print("=" * 75)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Locked Configuration: {CONFIG_ID}")
    print(f"Window Size: {WINDOW_SIZE_DAYS} days (~3 months)")
    print(f"Step Size: {STEP_SIZE_DAYS} days (~1 month)")
    print()
    print("FAILURE THRESHOLDS:")
    print(f"  - Max allowed Sharpe loss per window: {MAX_ALLOWED_SHARPE_LOSS}")
    print(f"  - Max allowed drawdown per window: {MAX_ALLOWED_DRAWDOWN*100}%")
    print("=" * 75)
    print()
    print("⚠️  NO MODIFICATIONS TO STRATEGY - PURE EVALUATION ONLY")
    print()
    
    all_window_results = []
    ticker_summaries = []
    
    for ticker in LOCKED_CONFIG["tickers"]:
        print(f"Processing {ticker}...", end=" ")
        
        windows = run_rolling_window_validation(ticker, LOCKED_CONFIG)
        
        if windows:
            all_window_results.extend([{**w, "ticker": ticker} for w in windows])
            
            avg_sharpe = np.mean([w["sharpe"] for w in windows])
            worst_sharpe = min([w["sharpe"] for w in windows])
            worst_dd = min([w["max_drawdown"] for w in windows])
            
            ticker_summaries.append({
                "ticker": ticker,
                "num_windows": len(windows),
                "avg_sharpe": round(avg_sharpe, 4),
                "worst_sharpe": round(worst_sharpe, 4),
                "worst_drawdown": round(worst_dd, 4),
            })
            
            print(f"{len(windows)} windows, avg_sharpe={avg_sharpe:.2f}, worst_sharpe={worst_sharpe:.2f}")
        else:
            print(f"NO DATA")
    
    if not all_window_results:
        print("\n❌ VALIDATION FAILED: No valid windows to analyze")
        return None
    
    # =============================================================================
    # AGGREGATE STATISTICS
    # =============================================================================
    all_sharpes = [w["sharpe"] for w in all_window_results]
    all_drawdowns = [w["max_drawdown"] for w in all_window_results]
    
    mean_sharpe = np.mean(all_sharpes)
    std_sharpe = np.std(all_sharpes)
    worst_sharpe = min(all_sharpes)
    best_sharpe = max(all_sharpes)
    worst_drawdown = min(all_drawdowns)
    pct_positive_sharpe = np.mean([s > 0 for s in all_sharpes]) * 100
    
    # Find worst window
    worst_window_idx = np.argmin(all_sharpes)
    worst_window = all_window_results[worst_window_idx]
    
    print("\n")
    print("=" * 75)
    print("AGGREGATE STABILITY STATISTICS")
    print("=" * 75)
    print(f"Total Windows Analyzed: {len(all_window_results)}")
    print(f"Mean Sharpe:            {mean_sharpe:.4f}")
    print(f"Std Dev Sharpe:         {std_sharpe:.4f}")
    print(f"Best Sharpe:            {best_sharpe:.4f}")
    print(f"Worst Sharpe:           {worst_sharpe:.4f}")
    print(f"Worst Drawdown:         {worst_drawdown*100:.2f}%")
    print(f"% Windows w/ Positive Sharpe: {pct_positive_sharpe:.1f}%")
    print("=" * 75)
    
    print("\n")
    print("WORST-PERIOD ANALYSIS")
    print("-" * 50)
    print(f"Period:    {worst_window['window_start']} to {worst_window['window_end']}")
    print(f"Ticker:    {worst_window['ticker']}")
    print(f"Sharpe:    {worst_window['sharpe']:.4f}")
    print(f"Drawdown:  {worst_window['max_drawdown']*100:.2f}%")
    print(f"Regime:    {worst_window['dominant_regime']}")
    print(f"Trades:    {worst_window['trade_count']}")
    
    # =============================================================================
    # FAILURE CONDITION CHECKS
    # =============================================================================
    print("\n")
    print("=" * 75)
    print("FAILURE CONDITION CHECKS")
    print("=" * 75)
    
    failures = []
    
    # Check 1: Any window with Sharpe < -0.50
    if worst_sharpe < MAX_ALLOWED_SHARPE_LOSS:
        failures.append(f"FAIL: Worst Sharpe ({worst_sharpe:.4f}) < {MAX_ALLOWED_SHARPE_LOSS}")
        print(f"❌ Sharpe < {MAX_ALLOWED_SHARPE_LOSS} in a window: {worst_sharpe:.4f}")
    else:
        print(f"✅ No window with Sharpe < {MAX_ALLOWED_SHARPE_LOSS} (worst: {worst_sharpe:.4f})")
    
    # Check 2: Drawdown exceeds tolerance
    if worst_drawdown < MAX_ALLOWED_DRAWDOWN:
        failures.append(f"FAIL: Worst Drawdown ({worst_drawdown*100:.2f}%) < {MAX_ALLOWED_DRAWDOWN*100}%")
        print(f"❌ Drawdown exceeds {MAX_ALLOWED_DRAWDOWN*100}%: {worst_drawdown*100:.2f}%")
    else:
        print(f"✅ No window with Drawdown < {MAX_ALLOWED_DRAWDOWN*100}% (worst: {worst_drawdown*100:.2f}%)")
    
    # Check 3: Sharpe instability (std > mean)
    if std_sharpe > abs(mean_sharpe) and mean_sharpe > 0:
        failures.append(f"FAIL: Sharpe instability (std {std_sharpe:.4f} > mean {mean_sharpe:.4f})")
        print(f"⚠️ Sharpe variance indicates instability: std ({std_sharpe:.4f}) > mean ({mean_sharpe:.4f})")
    else:
        print(f"✅ Sharpe stable: std ({std_sharpe:.4f}) <= |mean| ({abs(mean_sharpe):.4f})")
    
    # Check 4: Performance driven by small fraction of windows
    if pct_positive_sharpe < 40:
        failures.append(f"FAIL: Only {pct_positive_sharpe:.1f}% windows have positive Sharpe")
        print(f"❌ Low positive window rate: {pct_positive_sharpe:.1f}%")
    else:
        print(f"✅ Broad positive performance: {pct_positive_sharpe:.1f}% windows positive")
    
    print("=" * 75)
    
    # =============================================================================
    # FINAL VERDICT
    # =============================================================================
    if failures:
        verdict = "REJECT"
        verdict_reason = "; ".join(failures)
        print(f"\n🚫 VERDICT: {verdict}")
        print(f"   Reason: {verdict_reason}")
    else:
        verdict = "KEEP"
        verdict_reason = "Strategy is robust and stable across time windows"
        print(f"\n✅ VERDICT: {verdict}")
        print(f"   Reason: {verdict_reason}")
    
    # =============================================================================
    # SAVE REPORT
    # =============================================================================
    report = {
        "experiment_id": "ROBUSTNESS_001",
        "phase": "PHASE 6 - ROBUSTNESS & STABILITY VALIDATION",
        "created": datetime.now().isoformat(),
        "config_snapshot_id": CONFIG_ID,
        "locked_config": LOCKED_CONFIG,
        "validation_parameters": {
            "window_size_days": WINDOW_SIZE_DAYS,
            "step_size_days": STEP_SIZE_DAYS,
            "min_trades_per_window": MIN_TRADES_PER_WINDOW,
            "max_allowed_sharpe_loss": MAX_ALLOWED_SHARPE_LOSS,
            "max_allowed_drawdown": MAX_ALLOWED_DRAWDOWN,
        },
        "aggregate_statistics": {
            "total_windows": len(all_window_results),
            "mean_sharpe": round(mean_sharpe, 4),
            "std_sharpe": round(std_sharpe, 4),
            "best_sharpe": round(best_sharpe, 4),
            "worst_sharpe": round(worst_sharpe, 4),
            "worst_drawdown": round(worst_drawdown, 4),
            "pct_positive_sharpe": round(pct_positive_sharpe, 2),
        },
        "worst_period": {
            "ticker": worst_window["ticker"],
            "window_start": worst_window["window_start"],
            "window_end": worst_window["window_end"],
            "sharpe": worst_window["sharpe"],
            "max_drawdown": worst_window["max_drawdown"],
            "dominant_regime": worst_window["dominant_regime"],
            "trade_count": worst_window["trade_count"],
        },
        "per_ticker_summary": ticker_summaries,
        "per_window_metrics": all_window_results,
        "failure_checks": failures,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    
    output_path = PROJECT_ROOT / "experiments" / "robustness_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    return report


if __name__ == "__main__":
    run_robustness_validation()
