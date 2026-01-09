#!/usr/bin/env python3
"""
Comprehensive Diagnostic Baseline Script
=========================================

Analyzes the current state of the StockPredictor system using the new src/ module structure.
Generates JSON results and Markdown report.

Usage:
    python run_diagnostic_baseline.py

Output:
    experiments/diagnostic_results_{timestamp}.json
    experiments/DIAGNOSTIC_REPORT.md
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable ElasticNet for full feature diagnostic
os.environ["USE_ELASTICNET_SELECT"] = "0"

# Import from new src/ structure
from src.data.market import get_price_history
from src.data.macro import get_macro_df
from src.data.fundamentals import get_fundamental_features
from src.core.features import (
    build_all_features, add_returns, add_volatility, 
    add_all_technicals, add_gbm_features, add_regime_features,
    validate_features, get_feature_quality_summary
)
from src.core.models import make_model
from src.core.metrics import compute_sharpe, compute_drawdown
from src.core.regime_filter import MarketRegime
from src.core.zscore_filter import ZScoreFilter
from src.services.backtest import backtest_one_ticker, track_predictions
from src.config import (
    FEATURE_COLUMNS, MACRO_COLUMNS, REGIME_COLUMNS,
    REGIME_FILTER_CONFIG, POSITION_SIZING_CONFIG, ZSCORE_GATING_CONFIG,
    TICKER_WALKFORWARD_METRICS, TICKER_ELIGIBILITY_THRESHOLDS,
    MODEL_VERSIONS, ACTIVE_MODEL_VERSIONS,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DIAGNOSTIC_TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"]
MODEL_TYPES = ["rf", "xgb"]  # GBRT removed - see GBRT_INVESTIGATION_REPORT.md
DATA_PERIOD = "2y"  # 2 years of data
BACKTEST_MONTHS = 6  # Last 6 months for quick backtest
OUTPUT_DIR = PROJECT_ROOT / "experiments"

# BASELINE_005 recommended config (for comparison)
BASELINE_005_CONFIG = {
    "z_score_threshold": 1.0,
    "regime_filter_enabled": True,
    "regime_spy_dma": 200,
    "regime_vix_high": 25.0,
    "position_sizing_target_vol": 0.01,
    "elasticnet_enabled": False,
    "model_type": "xgb",
    "min_sharpe_eligible": 0.75,
    "min_hitrate_eligible": 0.55,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_mean(arr) -> float:
    """Safely compute mean, handling empty arrays."""
    if arr is None or len(arr) == 0:
        return 0.0
    return float(np.nanmean(arr))


def safe_std(arr) -> float:
    """Safely compute std, handling empty arrays."""
    if arr is None or len(arr) == 0:
        return 0.0
    return float(np.nanstd(arr))


def rolling_sharpe(returns: pd.Series, window: int = 21) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return np.sqrt(252) * roll_mean / roll_std.replace(0, np.nan)


def find_worst_periods(pnl_series: pd.Series, window: int = 21, top_n: int = 5) -> List[Dict]:
    """Find worst performing periods by rolling Sharpe."""
    if len(pnl_series) < window:
        return []
    
    roll_sharpe = rolling_sharpe(pnl_series, window)
    worst_periods = []
    
    # Find local minima
    sorted_idx = roll_sharpe.dropna().sort_values().head(top_n * 2).index
    
    used_ranges = []
    for idx in sorted_idx:
        if len(worst_periods) >= top_n:
            break
        
        # Check overlap with already used ranges
        start = idx - timedelta(days=window)
        end = idx
        
        overlaps = False
        for used_start, used_end in used_ranges:
            if not (end < used_start or start > used_end):
                overlaps = True
                break
        
        if not overlaps:
            sharpe_val = float(roll_sharpe.loc[idx])
            worst_periods.append({
                "end_date": str(idx.date()) if hasattr(idx, 'date') else str(idx),
                "start_date": str(start.date()) if hasattr(start, 'date') else str(start),
                "sharpe": round(sharpe_val, 3),
                "window_days": window,
            })
            used_ranges.append((start, end))
    
    return worst_periods


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_progress(msg: str):
    """Print progress message."""
    print(f"  → {msg}")


# ============================================================================
# PART 1: DATA QUALITY CHECK
# ============================================================================

def check_data_quality(tickers: List[str], period: str = "2y") -> Dict[str, Any]:
    """Check data quality for all tickers."""
    print_section("PART 1: DATA QUALITY CHECK")
    
    results = {
        "per_ticker": {},
        "feature_health": {},
        "validation_summary": {},  # NEW: After validate_features() cleanup
        "summary": {
            "total_tickers": len(tickers),
            "passed": 0,
            "failed": 0,
            "validated_pass": 0,  # NEW: Count after validation
            "issues": [],
        }
    }
    
    all_feature_nan_rates = {}
    all_validated_results = {}
    
    for ticker in tickers:
        print_progress(f"Checking {ticker}...")
        ticker_result = {
            "status": "PASS",
            "issues": [],
            "total_rows": 0,
            "date_range": {},
            "gaps": [],
            "missing_features": [],
        }
        
        try:
            # Fetch data
            hist = get_price_history(ticker, period=period, interval="1d")
            
            if hist is None or hist.empty:
                ticker_result["status"] = "FAIL"
                ticker_result["issues"].append("No data returned")
                results["per_ticker"][ticker] = ticker_result
                results["summary"]["failed"] += 1
                continue
            
            ticker_result["total_rows"] = len(hist)
            ticker_result["date_range"] = {
                "start": str(hist.index[0].date()),
                "end": str(hist.index[-1].date()),
            }
            
            # Check for gaps > 3 days (excluding weekends)
            date_diff = hist.index.to_series().diff()
            # Filter out weekends (Sat=5, Sun=6)
            gaps = date_diff[date_diff > pd.Timedelta(days=3)]
            if len(gaps) > 0:
                ticker_result["gaps"] = [
                    {"date": str(g.date()) if hasattr(g, 'date') else str(g), 
                     "days": int(date_diff.loc[g].days)}
                    for g in gaps.index[:5]  # Top 5 gaps
                ]
                if len(gaps) > 2:
                    ticker_result["issues"].append(f"{len(gaps)} gaps > 3 days found")
            
            # Build features and check for NaNs
            try:
                hist_with_features = build_all_features(hist.copy())
                
                # Check each feature (RAW - before validation)
                for col in FEATURE_COLUMNS:
                    if col not in hist_with_features.columns:
                        ticker_result["missing_features"].append(col)
                    else:
                        nan_rate = hist_with_features[col].isna().mean()
                        if col not in all_feature_nan_rates:
                            all_feature_nan_rates[col] = []
                        all_feature_nan_rates[col].append(nan_rate)
                
                if len(ticker_result["missing_features"]) > 10:
                    ticker_result["issues"].append(
                        f"{len(ticker_result['missing_features'])} features missing"
                    )
                
                # NEW: Also check AFTER validate_features() cleanup
                try:
                    validated_df, quality_report = validate_features(
                        hist_with_features.copy(),
                        required_features=FEATURE_COLUMNS,
                        max_row_nan_pct=0.20,
                        max_feature_nan_pct=0.10,
                        drop_warmup=True,
                        min_rows_after_clean=60
                    )
                    ticker_result["validated"] = {
                        "status": "PASS",
                        "rows_before": quality_report.get("original_rows", len(hist_with_features)),
                        "rows_after": quality_report.get("final_rows", len(validated_df)),
                        "rows_dropped": quality_report.get("rows_dropped", 0),
                        "features_filled": quality_report.get("features_with_nans_filled", []),
                    }
                    all_validated_results[ticker] = True
                except Exception as val_err:
                    ticker_result["validated"] = {
                        "status": "FAIL",
                        "error": str(val_err)[:100],
                    }
                    all_validated_results[ticker] = False
                    
            except Exception as e:
                ticker_result["issues"].append(f"Feature calculation failed: {str(e)[:50]}")
            
            # Check missing values in OHLCV
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in hist.columns:
                    missing = hist[col].isna().sum()
                    if missing > 0:
                        ticker_result["issues"].append(f"{col}: {missing} missing values")
            
            # Set status
            if len(ticker_result["issues"]) > 0:
                ticker_result["status"] = "WARN"
            
            results["per_ticker"][ticker] = ticker_result
            
            if ticker_result["status"] == "PASS":
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
                
        except Exception as e:
            ticker_result["status"] = "FAIL"
            ticker_result["issues"].append(f"Error: {str(e)[:100]}")
            results["per_ticker"][ticker] = ticker_result
            results["summary"]["failed"] += 1
    
    # Feature health summary
    print_progress("Calculating feature health metrics...")
    feature_health = {}
    problem_features = []
    
    for feat, nan_rates in all_feature_nan_rates.items():
        avg_nan_rate = safe_mean(nan_rates)
        feature_health[feat] = {
            "avg_nan_rate": round(avg_nan_rate, 4),
            "max_nan_rate": round(max(nan_rates) if nan_rates else 0, 4),
            "status": "OK" if avg_nan_rate < 0.05 else "HIGH_NAN"
        }
        if avg_nan_rate >= 0.05:
            problem_features.append(feat)
    
    results["feature_health"] = {
        "total_features": len(all_feature_nan_rates),
        "problem_features": problem_features,
        "problem_count": len(problem_features),
        "per_feature": feature_health,
    }
    
    if problem_features:
        results["summary"]["issues"].append(
            f"{len(problem_features)} features with >5% NaN rate (raw)"
        )
    
    # NEW: Count validated passes
    validated_pass_count = sum(1 for v in all_validated_results.values() if v)
    results["summary"]["validated_pass"] = validated_pass_count
    results["validation_summary"] = {
        "passed": validated_pass_count,
        "failed": len(all_validated_results) - validated_pass_count,
        "per_ticker": all_validated_results,
    }
    
    print(f"  ✓ Data quality check complete:")
    print(f"      Raw: {results['summary']['passed']}/{len(tickers)} passed")
    print(f"      After validation: {validated_pass_count}/{len(tickers)} passed")
    
    return results


# ============================================================================
# PART 2: MODEL PERFORMANCE SNAPSHOT
# ============================================================================

def run_performance_snapshot(
    tickers: List[str], 
    model_types: List[str] = MODEL_TYPES,
    test_months: int = 6
) -> Dict[str, Any]:
    """Run quick backtest for each ticker and model."""
    print_section("PART 2: MODEL PERFORMANCE SNAPSHOT")
    
    results = {
        "per_ticker": {},
        "per_model": {m: {"sharpes": [], "accuracies": []} for m in model_types},
        "model_agreement": {},
        "summary": {},
    }
    
    all_predictions = []
    
    for ticker in tickers:
        print_progress(f"Backtesting {ticker}...")
        ticker_result = {
            "models": {},
            "best_model": None,
            "worst_model": None,
        }
        
        model_predictions = {}
        
        for model_type in model_types:
            try:
                # Use 1 year data, test on last N months
                test_years = test_months / 12
                bt_result = backtest_one_ticker(
                    ticker=ticker,
                    period="2y",
                    test_years=test_years,
                    threshold=0.002,
                    model_type=model_type,
                    horizon=1,
                )
                
                if "error" in bt_result:
                    ticker_result["models"][model_type] = {"error": bt_result["error"]}
                    continue
                
                ticker_result["models"][model_type] = {
                    "sharpe": round(bt_result.get("sharpe", 0), 3),
                    "accuracy": round(bt_result.get("accuracy", 0), 3),
                    "max_drawdown": round(bt_result.get("max_drawdown", 0), 4),
                    "num_trades": bt_result.get("num_trades", 0),
                    "total_return": round(bt_result.get("total_return", 0), 4),
                }
                
                results["per_model"][model_type]["sharpes"].append(bt_result.get("sharpe", 0))
                results["per_model"][model_type]["accuracies"].append(bt_result.get("accuracy", 0))
                
                # Track predictions for agreement analysis
                try:
                    track_df, acc = track_predictions(ticker, period="1y", model_type=model_type)
                    if not track_df.empty:
                        model_predictions[model_type] = track_df
                except:
                    pass
                    
            except Exception as e:
                ticker_result["models"][model_type] = {"error": str(e)[:50]}
        
        # Find best/worst model
        sharpes = {m: d.get("sharpe", -999) for m, d in ticker_result["models"].items() 
                   if "sharpe" in d}
        if sharpes:
            ticker_result["best_model"] = max(sharpes, key=sharpes.get)
            ticker_result["worst_model"] = min(sharpes, key=sharpes.get)
        
        results["per_ticker"][ticker] = ticker_result
        
        # Compute model agreement for this ticker
        if len(model_predictions) == 3:
            try:
                # Align predictions by date
                aligned = pd.DataFrame({m: df["predicted_return"] for m, df in model_predictions.items()})
                aligned = aligned.dropna()
                
                if len(aligned) > 0:
                    # Check if all models agree on direction
                    signs = np.sign(aligned)
                    all_agree = (signs.nunique(axis=1) == 1).mean()
                    
                    all_predictions.append({
                        "ticker": ticker,
                        "agreement_rate": all_agree,
                        "sample_size": len(aligned),
                    })
            except:
                pass
    
    # Model agreement summary
    if all_predictions:
        results["model_agreement"] = {
            "overall_agreement_rate": round(safe_mean([p["agreement_rate"] for p in all_predictions]), 3),
            "per_ticker": all_predictions,
        }
    
    # Summary statistics
    for model_type in model_types:
        results["per_model"][model_type]["avg_sharpe"] = round(
            safe_mean(results["per_model"][model_type]["sharpes"]), 3
        )
        results["per_model"][model_type]["avg_accuracy"] = round(
            safe_mean(results["per_model"][model_type]["accuracies"]), 3
        )
    
    best_model = max(model_types, key=lambda m: results["per_model"][m]["avg_sharpe"])
    results["summary"] = {
        "best_model_overall": best_model,
        "best_avg_sharpe": results["per_model"][best_model]["avg_sharpe"],
        "tickers_tested": len(tickers),
    }
    
    print(f"  ✓ Performance snapshot complete. Best model: {best_model}")
    
    return results


# ============================================================================
# PART 3: REGIME ANALYSIS
# ============================================================================

def analyze_regime(period: str = "2y") -> Dict[str, Any]:
    """Analyze market regime detection and performance."""
    print_section("PART 3: REGIME ANALYSIS")
    
    results = {
        "current_regime": {},
        "regime_accuracy": {},
        "regime_conditional_performance": {},
    }
    
    print_progress("Fetching SPY and VIX data...")
    
    try:
        # Get SPY data
        spy = get_price_history("SPY", period=period, interval="1d")
        if spy is None or spy.empty:
            return {"error": "Could not fetch SPY data"}
        
        # Calculate regime features
        spy = add_returns(spy)
        spy = add_all_technicals(spy)
        
        # 200-day moving average
        spy["ma_200"] = spy["Close"].rolling(200).mean()
        spy["above_200dma"] = (spy["Close"] > spy["ma_200"]).astype(int)
        
        # Simple regime classification
        spy["regime"] = "neutral"
        spy.loc[spy["above_200dma"] == 1, "regime"] = "bull"
        spy.loc[spy["above_200dma"] == 0, "regime"] = "bear"
        
        # Get VIX
        try:
            vix = get_price_history("^VIX", period=period, interval="1d")
            if vix is not None and not vix.empty:
                spy = spy.join(vix["Close"].rename("vix"), how="left")
                spy["vix"] = spy["vix"].ffill()
                
                # High VIX regime
                spy.loc[(spy["vix"] > 25) & (spy["regime"] == "bear"), "regime"] = "strong_bear"
        except:
            pass
        
        # Current regime
        current = spy.iloc[-1]
        results["current_regime"] = {
            "regime": current["regime"],
            "spy_price": round(float(current["Close"]), 2),
            "spy_200dma": round(float(current["ma_200"]), 2) if pd.notna(current["ma_200"]) else None,
            "above_200dma": bool(current["above_200dma"]),
            "vix": round(float(current["vix"]), 2) if "vix" in current and pd.notna(current["vix"]) else None,
            "rsi": round(float(current["rsi14"]), 2) if pd.notna(current["rsi14"]) else None,
            "date": str(current.name.date()) if hasattr(current.name, 'date') else str(current.name),
        }
        
        print_progress(f"Current regime: {results['current_regime']['regime'].upper()}")
        
        # Regime accuracy: when regime=bull, what % of days were actually up?
        spy["actual_up"] = (spy["ret_1d"] > 0).astype(int)
        
        regime_accuracy = {}
        for regime in ["bull", "bear", "neutral", "strong_bear"]:
            mask = spy["regime"] == regime
            if mask.sum() > 10:
                up_pct = spy.loc[mask, "actual_up"].mean()
                regime_accuracy[regime] = {
                    "days": int(mask.sum()),
                    "pct_up": round(float(up_pct), 3),
                    "expected_direction": "up" if regime in ["bull"] else "down" if regime in ["bear", "strong_bear"] else "mixed",
                    "accuracy": round(float(up_pct) if regime == "bull" else 1 - float(up_pct), 3),
                }
        
        results["regime_accuracy"] = regime_accuracy
        
        # Regime-conditional Sharpe (hypothetical)
        print_progress("Calculating regime-conditional performance...")
        
        regime_performance = {}
        for regime in ["bull", "bear", "neutral"]:
            mask = spy["regime"] == regime
            if mask.sum() > 20:
                regime_rets = spy.loc[mask, "ret_1d"].dropna()
                sharpe = np.sqrt(252) * regime_rets.mean() / regime_rets.std() if regime_rets.std() > 0 else 0
                regime_performance[regime] = {
                    "sharpe": round(float(sharpe), 3),
                    "avg_return": round(float(regime_rets.mean()), 5),
                    "volatility": round(float(regime_rets.std()), 4),
                    "days": int(mask.sum()),
                }
        
        results["regime_conditional_performance"] = regime_performance
        
        # Identify worst regime
        sharpes = {r: d["sharpe"] for r, d in regime_performance.items()}
        if sharpes:
            results["worst_regime"] = min(sharpes, key=sharpes.get)
            results["best_regime"] = max(sharpes, key=sharpes.get)
        
    except Exception as e:
        results["error"] = str(e)
    
    print(f"  ✓ Regime analysis complete")
    
    return results


# ============================================================================
# PART 4: WORST CASE ANALYSIS
# ============================================================================

def analyze_worst_periods(tickers: List[str]) -> Dict[str, Any]:
    """Find worst performing periods for each ticker."""
    print_section("PART 4: WORST CASE ANALYSIS")
    
    results = {
        "worst_1month_periods": [],
        "worst_3month_periods": [],
        "ticker_rankings": {},
    }
    
    ticker_metrics = {}
    
    for ticker in tickers:
        print_progress(f"Analyzing worst periods for {ticker}...")
        
        try:
            # Run backtest and get PnL series
            hist = get_price_history(ticker, period="2y", interval="1d")
            if hist is None or hist.empty:
                continue
            
            hist = add_returns(hist)
            
            # Simulate simple strategy returns (using actual returns as proxy)
            pnl = hist["ret_1d"].dropna()
            
            if len(pnl) < 63:
                continue
            
            # 6-month Sharpe
            recent_pnl = pnl.iloc[-126:]  # ~6 months
            sharpe_6m = np.sqrt(252) * recent_pnl.mean() / recent_pnl.std() if recent_pnl.std() > 0 else 0
            
            # Find worst 1-month periods
            worst_1m = find_worst_periods(pnl, window=21, top_n=2)
            for period in worst_1m:
                period["ticker"] = ticker
                results["worst_1month_periods"].append(period)
            
            # Find worst 3-month periods
            worst_3m = find_worst_periods(pnl, window=63, top_n=2)
            for period in worst_3m:
                period["ticker"] = ticker
                results["worst_3month_periods"].append(period)
            
            # Worst 1-month Sharpe for this ticker
            worst_1m_sharpe = min([p["sharpe"] for p in worst_1m]) if worst_1m else 0
            
            ticker_metrics[ticker] = {
                "sharpe_6m": round(float(sharpe_6m), 3),
                "worst_1m_sharpe": round(float(worst_1m_sharpe), 3),
                "volatility": round(float(pnl.std() * np.sqrt(252)), 4),
            }
            
        except Exception as e:
            print(f"    Warning: Error for {ticker}: {e}")
    
    # Sort worst periods
    results["worst_1month_periods"] = sorted(
        results["worst_1month_periods"], 
        key=lambda x: x["sharpe"]
    )[:5]
    
    results["worst_3month_periods"] = sorted(
        results["worst_3month_periods"], 
        key=lambda x: x["sharpe"]
    )[:5]
    
    # Ticker rankings
    if ticker_metrics:
        # Rank by 6-month Sharpe
        ranked_6m = sorted(ticker_metrics.items(), key=lambda x: x[1]["sharpe_6m"], reverse=True)
        results["ticker_rankings"]["by_6m_sharpe"] = [
            {"rank": i+1, "ticker": t, "sharpe": m["sharpe_6m"]} 
            for i, (t, m) in enumerate(ranked_6m)
        ]
        
        # Rank by worst 1-month Sharpe (higher is better)
        ranked_worst = sorted(ticker_metrics.items(), key=lambda x: x[1]["worst_1m_sharpe"], reverse=True)
        results["ticker_rankings"]["by_worst_1m_sharpe"] = [
            {"rank": i+1, "ticker": t, "worst_sharpe": m["worst_1m_sharpe"]} 
            for i, (t, m) in enumerate(ranked_worst)
        ]
        
        # Identify stable vs unstable tickers
        stable = []
        unstable = []
        for ticker, metrics in ticker_metrics.items():
            if metrics["sharpe_6m"] > 0.5 and metrics["worst_1m_sharpe"] > -2.0:
                stable.append(ticker)
            elif metrics["sharpe_6m"] < -0.5 or metrics["worst_1m_sharpe"] < -3.0:
                unstable.append(ticker)
        
        results["ticker_rankings"]["stable_tickers"] = stable
        results["ticker_rankings"]["unstable_tickers"] = unstable
    
    print(f"  ✓ Worst case analysis complete")
    
    return results


# ============================================================================
# PART 5: CONFIGURATION AUDIT
# ============================================================================

def audit_configuration() -> Dict[str, Any]:
    """Audit current configuration vs recommended baseline."""
    print_section("PART 5: CONFIGURATION AUDIT")
    
    results = {
        "current_config": {},
        "baseline_005_config": BASELINE_005_CONFIG,
        "differences": [],
        "recommendations": [],
    }
    
    # Get current config
    current = {
        "z_score_threshold": ZSCORE_GATING_CONFIG.get("min_zscore", 1.0),
        "regime_filter_enabled": REGIME_FILTER_CONFIG.get("enabled", True),
        "regime_spy_dma": REGIME_FILTER_CONFIG.get("spy_dma_period", 200),
        "regime_vix_high": REGIME_FILTER_CONFIG.get("vix_high_threshold", 25.0),
        "position_sizing_target_vol": POSITION_SIZING_CONFIG.get("target_daily_vol", 0.01),
        "elasticnet_enabled": os.environ.get("USE_ELASTICNET_SELECT", "0") == "1",
        "min_sharpe_eligible": TICKER_ELIGIBILITY_THRESHOLDS.get("min_sharpe", 0.75),
        "min_hitrate_eligible": TICKER_ELIGIBILITY_THRESHOLDS.get("min_hitrate", 0.55),
    }
    
    # Add model version info
    for model_type, version_key in ACTIVE_MODEL_VERSIONS.items():
        if version_key in MODEL_VERSIONS:
            info = MODEL_VERSIONS[version_key]
            current[f"model_{model_type}_version"] = version_key
            current[f"model_{model_type}_status"] = info.get("status", "unknown")
    
    results["current_config"] = current
    
    print_progress("Comparing to BASELINE_005 recommended config...")
    
    # Compare to baseline
    for key, baseline_val in BASELINE_005_CONFIG.items():
        if key in current:
            current_val = current[key]
            if current_val != baseline_val:
                diff = {
                    "parameter": key,
                    "current": current_val,
                    "recommended": baseline_val,
                }
                
                # Estimate impact
                if key == "z_score_threshold" and current_val < baseline_val:
                    diff["impact"] = "May include weak signals - higher false positive rate"
                elif key == "regime_filter_enabled" and not current_val:
                    diff["impact"] = "No protection during bear markets - higher drawdowns"
                elif key == "elasticnet_enabled" and current_val:
                    diff["impact"] = "Feature selection active - may reduce robustness"
                else:
                    diff["impact"] = "Minor difference"
                
                results["differences"].append(diff)
    
    # Generate recommendations
    if not results["differences"]:
        results["recommendations"].append("Configuration matches BASELINE_005 - no changes needed")
    else:
        for diff in results["differences"]:
            if diff["parameter"] == "regime_filter_enabled" and not diff["current"]:
                results["recommendations"].append(
                    "Enable regime filter to reduce drawdowns during bear markets"
                )
            elif diff["parameter"] == "z_score_threshold":
                results["recommendations"].append(
                    f"Consider z_score_threshold={diff['recommended']} for higher conviction signals"
                )
    
    print(f"  ✓ Configuration audit complete. {len(results['differences'])} differences found.")
    
    return results


# ============================================================================
# GENERATE REPORTS
# ============================================================================

def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate human-readable Markdown report."""
    print_progress("Generating Markdown report...")
    
    lines = [
        "# Stock Predictor Diagnostic Report",
        f"",
        f"**Generated:** {results['timestamp']}",
        f"",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Executive summary bullets
    summary_bullets = []
    
    # What's working
    if results.get("model_performance", {}).get("summary", {}).get("best_avg_sharpe", 0) > 0.5:
        best_model = results["model_performance"]["summary"]["best_model_overall"]
        sharpe = results["model_performance"]["summary"]["best_avg_sharpe"]
        summary_bullets.append(f"✅ **Working:** {best_model.upper()} model achieving {sharpe:.2f} average Sharpe")
    
    # What's broken
    data_issues = results.get("data_quality", {}).get("summary", {}).get("failed", 0)
    if data_issues > 0:
        summary_bullets.append(f"❌ **Issue:** {data_issues} tickers failing data quality checks")
    
    problem_features = results.get("data_quality", {}).get("feature_health", {}).get("problem_count", 0)
    if problem_features > 5:
        summary_bullets.append(f"⚠️ **Warning:** {problem_features} features with >5% NaN rate")
    
    # Top priority
    if results.get("worst_periods", {}).get("ticker_rankings", {}).get("unstable_tickers"):
        unstable = results["worst_periods"]["ticker_rankings"]["unstable_tickers"]
        summary_bullets.append(f"🎯 **Priority:** Review unstable tickers: {', '.join(unstable[:3])}")
    
    for bullet in summary_bullets:
        lines.append(f"- {bullet}")
    
    lines.extend(["", "---", "", "## Data Quality", ""])
    
    # Data quality table
    dq = results.get("data_quality", {})
    raw_passed = dq.get("summary", {}).get("passed", 0)
    validated_passed = dq.get("summary", {}).get("validated_pass", 0)
    total = dq.get("summary", {}).get("total_tickers", 0)
    
    # Use validated status (after cleanup) as the primary status
    status = "✅ PASS" if validated_passed == total else "⚠️ PARTIAL" if validated_passed > 0 else "❌ FAIL"
    
    lines.append(f"**Status:** {status}")
    lines.append(f"- Raw data: {raw_passed}/{total} tickers passed (before validation)")
    lines.append(f"- **After validation:** {validated_passed}/{total} tickers passed")
    lines.append("")
    
    if dq.get("per_ticker"):
        lines.append("| Ticker | Rows | After Validation | Dropped | Status |")
        lines.append("|--------|------|------------------|---------|--------|")
        
        for ticker, info in dq["per_ticker"].items():
            rows = info.get("total_rows", "N/A")
            validated = info.get("validated", {})
            if validated.get("status") == "PASS":
                final_rows = validated.get("rows_after", "?")
                dropped = validated.get("rows_dropped", 0)
                val_status = "✅ PASS"
            else:
                final_rows = "N/A"
                dropped = "-"
                val_status = "❌ FAIL"
            lines.append(f"| {ticker} | {rows} | {final_rows} | {dropped} | {val_status} |")
    
    lines.extend(["", "---", "", "## Model Performance (6-Month Backtest)", ""])
    
    # Performance table
    if results.get("model_performance", {}).get("per_ticker"):
        lines.append("| Ticker | RF Sharpe | XGB Sharpe | GBRT Sharpe | Best Model |")
        lines.append("|--------|-----------|------------|-------------|------------|")
        
        for ticker, info in results["model_performance"]["per_ticker"].items():
            rf = info.get("models", {}).get("rf", {}).get("sharpe", "N/A")
            xgb = info.get("models", {}).get("xgb", {}).get("sharpe", "N/A")
            gbrt = info.get("models", {}).get("gbrt", {}).get("sharpe", "N/A")
            best = info.get("best_model", "?")
            lines.append(f"| {ticker} | {rf} | {xgb} | {gbrt} | {best} |")
    
    lines.extend(["", "---", "", "## Regime Analysis", ""])
    
    if results.get("regime_analysis", {}).get("current_regime"):
        cr = results["regime_analysis"]["current_regime"]
        lines.append(f"**Current Regime:** {cr.get('regime', '?').upper()}")
        lines.append(f"- SPY: ${cr.get('spy_price', '?')}")
        lines.append(f"- 200 DMA: ${cr.get('spy_200dma', '?')}")
        lines.append(f"- VIX: {cr.get('vix', '?')}")
        lines.append(f"- RSI: {cr.get('rsi', '?')}")
    
    lines.extend(["", "---", "", "## Worst Periods", ""])
    
    if results.get("worst_periods", {}).get("worst_1month_periods"):
        lines.append("### Worst 1-Month Periods")
        lines.append("| Ticker | Period | Sharpe |")
        lines.append("|--------|--------|--------|")
        
        for period in results["worst_periods"]["worst_1month_periods"][:5]:
            lines.append(f"| {period['ticker']} | {period['start_date']} to {period['end_date']} | {period['sharpe']} |")
    
    lines.extend(["", "---", "", "## Top 3 Recommendations", ""])
    
    recs = results.get("config_audit", {}).get("recommendations", [])
    if not recs:
        recs = ["No specific recommendations - system appears well configured"]
    
    for i, rec in enumerate(recs[:3], 1):
        lines.append(f"{i}. {rec}")
    
    lines.extend(["", "---", "", "*Report generated by run_diagnostic_baseline.py*"])
    
    # Write file
    output_path.write_text("\n".join(lines))
    print(f"  ✓ Report saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run full diagnostic and generate reports."""
    print("\n" + "="*60)
    print("  STOCK PREDICTOR DIAGNOSTIC BASELINE")
    print("="*60)
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Tickers: {', '.join(DIAGNOSTIC_TICKERS)}")
    print(f"  Models: {', '.join(MODEL_TYPES)}")
    print("="*60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all results
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
        "tickers_analyzed": DIAGNOSTIC_TICKERS,
        "models_tested": MODEL_TYPES,
    }
    
    start_time = datetime.now()
    
    # Part 1: Data Quality
    results["data_quality"] = check_data_quality(DIAGNOSTIC_TICKERS, period=DATA_PERIOD)
    
    # Part 2: Model Performance
    results["model_performance"] = run_performance_snapshot(
        DIAGNOSTIC_TICKERS, MODEL_TYPES, test_months=BACKTEST_MONTHS
    )
    
    # Part 3: Regime Analysis
    results["regime_analysis"] = analyze_regime(period=DATA_PERIOD)
    
    # Part 4: Worst Case Analysis
    results["worst_periods"] = analyze_worst_periods(DIAGNOSTIC_TICKERS)
    
    # Part 5: Configuration Audit
    results["config_audit"] = audit_configuration()
    
    # Timing
    elapsed = (datetime.now() - start_time).total_seconds()
    results["runtime_seconds"] = round(elapsed, 1)
    
    # Generate outputs
    print_section("GENERATING OUTPUTS")
    
    # JSON output
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"diagnostic_results_{timestamp_str}.json"
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ JSON saved to: {json_path}")
    
    # Markdown report
    md_path = OUTPUT_DIR / "DIAGNOSTIC_REPORT.md"
    generate_markdown_report(results, md_path)
    
    # Summary
    print_section("DIAGNOSTIC COMPLETE")
    print(f"  Runtime: {elapsed:.1f} seconds")
    print(f"  JSON: {json_path}")
    print(f"  Report: {md_path}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
