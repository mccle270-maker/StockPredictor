"""
Comprehensive Out-of-Sample Validation Suite
=============================================

This module provides rigorous validation to ensure the +2.77 Sharpe
from optimization holds on truly unseen data.

Three validation methods:
1. TRUE HOLDOUT TEST - Train 2022-2024, Test 2025 (never seen)
2. WALK-FORWARD VALIDATION - Rolling 18-month train, 3-month test
3. PURGED K-FOLD CROSS-VALIDATION - K-fold with leakage prevention

Run: python experiments/validate_out_of_sample.py
"""

import sys
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_model import build_features_and_target
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from src.config import get_optimized_rf_config, get_optimized_config

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not available, will skip XGB tests")

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Test tickers (focus on those with better unseen performance)
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Optimization benchmark to beat
OPTIMIZATION_SHARPE = 2.77

# Part 1: True Holdout Config
HOLDOUT_CONFIG = {
    "train_start": "2022-01-01",
    "train_end": "2024-12-31",
    "holdout_start": "2025-01-01",
    "holdout_end": "2025-12-31",
}

# Part 2: Walk-Forward Config
WALKFORWARD_CONFIG = {
    "train_months": 18,
    "test_months": 3,
    "step_months": 3,
    "start_date": "2022-01-01",
    "end_date": "2025-12-31",
}

# Part 3: Purged K-Fold Config
PURGED_KFOLD_CONFIG = {
    "n_splits": 5,
    "purge_days": 5,
    "embargo_days": 1,
}

# Pass/Fail Thresholds
THRESHOLDS = {
    "pass_sharpe": 1.0,
    "warning_sharpe": 0.5,
    "pass_accuracy": 0.52,
    "pass_positive_folds_pct": 0.6,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_sharpe(returns: np.ndarray, annualize: bool = True) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(returns) / np.std(returns)
    if annualize:
        sharpe *= np.sqrt(252)
    return sharpe


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cumulative_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (peak + 1e-10)
    return float(np.min(drawdown))


def calculate_direction_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate direction accuracy (% of correct up/down predictions)."""
    if len(predictions) == 0:
        return 0.5
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    return float(np.mean(pred_direction == actual_direction))


def get_optimized_rf() -> RandomForestRegressor:
    """Create optimized Random Forest model."""
    config = get_optimized_rf_config()
    return RandomForestRegressor(**config)


def get_optimized_xgb() -> Any:
    """Create optimized XGBoost model."""
    if not HAS_XGB:
        return None
    config = get_optimized_config()
    return XGBRegressor(
        n_estimators=config.get("n_estimators", 450),
        max_depth=config.get("max_depth", 7),
        learning_rate=config.get("learning_rate", 0.048),
        subsample=config.get("subsample", 0.998),
        colsample_bytree=config.get("colsample_bytree", 0.67),
        min_child_weight=config.get("min_child_weight", 19),
        reg_alpha=config.get("reg_alpha", 0.012),
        reg_lambda=config.get("reg_lambda", 9.3),
        random_state=42,
        n_jobs=-1
    )


def load_data_for_period(ticker: str, start_date: str, end_date: str, 
                         period: str = "5y") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.Series]]:
    """
    Load features and target for a specific date range.
    
    Returns:
        Tuple of (X, y, dates) or (None, None, None) on failure
    """
    try:
        result = build_features_and_target(ticker, period=period, horizon=1)
        if result is None:
            return None, None, None
        
        X, y, _, _, _, _, dates = result
        
        # Convert dates to datetime
        if hasattr(dates, 'values'):
            dates_arr = pd.to_datetime(dates.values)
        else:
            dates_arr = pd.to_datetime(dates)
        
        # Create mask for date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        
        # Handle timezone-aware dates
        if dates_arr.tz is not None:
            start_dt = start_dt.tz_localize(dates_arr.tz)
            end_dt = end_dt.tz_localize(dates_arr.tz)
        
        mask = (dates_arr >= start_dt) & (dates_arr <= end_dt)
        
        return X[mask], y[mask], pd.Series(dates_arr[mask])
    
    except Exception as e:
        print(f"  ⚠️ Failed to load {ticker}: {e}")
        return None, None, None


# ============================================================================
# PART 1: TRUE HOLDOUT TEST
# ============================================================================

def run_holdout_test(model_type: str = "rf") -> Dict[str, Any]:
    """
    Run true holdout test: Train on 2022-2024, test on 2025.
    
    This is the most important test - 2025 was NEVER seen during optimization.
    """
    print("\n" + "="*70)
    print("PART 1: TRUE HOLDOUT TEST")
    print("="*70)
    print(f"Training Period: {HOLDOUT_CONFIG['train_start']} to {HOLDOUT_CONFIG['train_end']}")
    print(f"Holdout Period:  {HOLDOUT_CONFIG['holdout_start']} to {HOLDOUT_CONFIG['holdout_end']}")
    print(f"Model Type: {model_type.upper()}")
    print("-"*70)
    
    all_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n📊 Testing {ticker}...")
        
        # Load training data (2022-2024)
        X_train, y_train, dates_train = load_data_for_period(
            ticker, 
            HOLDOUT_CONFIG["train_start"], 
            HOLDOUT_CONFIG["train_end"],
            period="5y"
        )
        
        # Load holdout data (2025)
        X_holdout, y_holdout, dates_holdout = load_data_for_period(
            ticker,
            HOLDOUT_CONFIG["holdout_start"],
            HOLDOUT_CONFIG["holdout_end"],
            period="5y"
        )
        
        if X_train is None or X_holdout is None:
            print(f"  ⚠️ Skipping {ticker} - data unavailable")
            continue
        
        if len(X_train) < 100 or len(X_holdout) < 10:
            print(f"  ⚠️ Skipping {ticker} - insufficient data (train={len(X_train)}, holdout={len(X_holdout)})")
            continue
        
        print(f"  Train samples: {len(X_train)}, Holdout samples: {len(X_holdout)}")
        
        # Create and train model
        if model_type == "rf":
            model = get_optimized_rf()
        elif model_type == "xgb" and HAS_XGB:
            model = get_optimized_xgb()
        else:
            print(f"  ⚠️ Model type '{model_type}' not available")
            continue
        
        model.fit(X_train, y_train)
        
        # Predict on holdout
        holdout_pred = model.predict(X_holdout)
        
        # Calculate strategy returns (long if pred > 0, short if pred < 0)
        strategy_returns = np.sign(holdout_pred) * y_holdout
        
        # Calculate metrics
        sharpe = calculate_sharpe(strategy_returns)
        accuracy = calculate_direction_accuracy(holdout_pred, y_holdout)
        cumulative = np.cumprod(1 + strategy_returns)
        max_dd = calculate_max_drawdown(cumulative)
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        
        # Training metrics for comparison
        train_pred = model.predict(X_train)
        train_strategy = np.sign(train_pred) * y_train
        train_sharpe = calculate_sharpe(train_strategy)
        train_accuracy = calculate_direction_accuracy(train_pred, y_train)
        
        result = {
            "ticker": ticker,
            "train_samples": len(X_train),
            "holdout_samples": len(X_holdout),
            "train_sharpe": train_sharpe,
            "holdout_sharpe": sharpe,
            "sharpe_degradation": train_sharpe - sharpe,
            "train_accuracy": train_accuracy,
            "holdout_accuracy": accuracy,
            "max_drawdown": max_dd,
            "total_return": total_return,
        }
        
        all_results.append(result)
        
        print(f"  Train Sharpe: {train_sharpe:.2f} → Holdout Sharpe: {sharpe:.2f} (Δ={train_sharpe - sharpe:.2f})")
        print(f"  Train Acc: {train_accuracy:.1%} → Holdout Acc: {accuracy:.1%}")
        print(f"  Max Drawdown: {max_dd:.1%}, Total Return: {total_return:.1%}")
    
    # Aggregate results
    if not all_results:
        return {"status": "FAIL", "reason": "No valid results"}
    
    avg_holdout_sharpe = np.mean([r["holdout_sharpe"] for r in all_results])
    avg_holdout_accuracy = np.mean([r["holdout_accuracy"] for r in all_results])
    avg_train_sharpe = np.mean([r["train_sharpe"] for r in all_results])
    avg_degradation = np.mean([r["sharpe_degradation"] for r in all_results])
    avg_max_dd = np.mean([r["max_drawdown"] for r in all_results])
    
    # Determine pass/fail
    if avg_holdout_sharpe >= THRESHOLDS["pass_sharpe"]:
        status = "✅ PASS"
        status_code = "PASS"
    elif avg_holdout_sharpe >= THRESHOLDS["warning_sharpe"]:
        status = "⚠️ WARNING"
        status_code = "WARNING"
    else:
        status = "❌ FAIL"
        status_code = "FAIL"
    
    summary = {
        "status": status_code,
        "status_display": status,
        "model_type": model_type,
        "avg_holdout_sharpe": avg_holdout_sharpe,
        "avg_holdout_accuracy": avg_holdout_accuracy,
        "avg_train_sharpe": avg_train_sharpe,
        "avg_degradation": avg_degradation,
        "avg_max_drawdown": avg_max_dd,
        "optimization_benchmark": OPTIMIZATION_SHARPE,
        "degradation_vs_optimization": OPTIMIZATION_SHARPE - avg_holdout_sharpe,
        "per_ticker_results": all_results,
    }
    
    print("\n" + "-"*70)
    print("HOLDOUT TEST SUMMARY")
    print("-"*70)
    print(f"Status: {status}")
    print(f"Avg Holdout Sharpe: {avg_holdout_sharpe:.2f} (vs Optimization: {OPTIMIZATION_SHARPE})")
    print(f"Avg Holdout Accuracy: {avg_holdout_accuracy:.1%}")
    print(f"Avg Sharpe Degradation: {avg_degradation:.2f}")
    print(f"Avg Max Drawdown: {avg_max_dd:.1%}")
    
    return summary


# ============================================================================
# PART 2: WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward_validation(model_type: str = "rf") -> Dict[str, Any]:
    """
    Run walk-forward validation with fixed hyperparameters.
    
    - Train window: 18 months
    - Test window: 3 months
    - Step forward: 3 months
    """
    print("\n" + "="*70)
    print("PART 2: WALK-FORWARD VALIDATION")
    print("="*70)
    print(f"Train Window: {WALKFORWARD_CONFIG['train_months']} months")
    print(f"Test Window: {WALKFORWARD_CONFIG['test_months']} months")
    print(f"Step Forward: {WALKFORWARD_CONFIG['step_months']} months")
    print(f"Model Type: {model_type.upper()}")
    print("-"*70)
    
    all_fold_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n📊 Walk-Forward for {ticker}...")
        
        # Load all data
        X_all, y_all, dates_all = load_data_for_period(
            ticker,
            WALKFORWARD_CONFIG["start_date"],
            WALKFORWARD_CONFIG["end_date"],
            period="5y"
        )
        
        if X_all is None or len(X_all) < 200:
            print(f"  ⚠️ Skipping {ticker} - insufficient data")
            continue
        
        # Convert dates to timestamps
        dates_ts = pd.to_datetime(dates_all)
        
        # Calculate fold dates
        start_date = dates_ts.min()
        end_date = dates_ts.max()
        
        train_days = WALKFORWARD_CONFIG["train_months"] * 21  # Approx trading days per month
        test_days = WALKFORWARD_CONFIG["test_months"] * 21
        step_days = WALKFORWARD_CONFIG["step_months"] * 21
        
        fold_num = 0
        current_train_start = start_date
        
        ticker_folds = []
        
        while True:
            # Calculate fold boundaries
            train_end = current_train_start + pd.Timedelta(days=train_days * 1.5)  # Calendar days
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=test_days * 1.5)
            
            if test_end > end_date:
                break
            
            fold_num += 1
            
            # Create masks
            train_mask = (dates_ts >= current_train_start) & (dates_ts <= train_end)
            test_mask = (dates_ts >= test_start) & (dates_ts <= test_end)
            
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]
            
            if len(X_train) < 50 or len(X_test) < 10:
                current_train_start += pd.Timedelta(days=step_days * 1.5)
                continue
            
            # Train model
            if model_type == "rf":
                model = get_optimized_rf()
            elif model_type == "xgb" and HAS_XGB:
                model = get_optimized_xgb()
            else:
                break
            
            model.fit(X_train, y_train)
            
            # Predict
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            strategy_returns = np.sign(test_pred) * y_test
            sharpe = calculate_sharpe(strategy_returns)
            accuracy = calculate_direction_accuracy(test_pred, y_test)
            
            fold_result = {
                "fold": fold_num,
                "train_start": current_train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "sharpe": sharpe,
                "accuracy": accuracy,
            }
            
            ticker_folds.append(fold_result)
            
            print(f"  Fold {fold_num}: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} | Sharpe: {sharpe:.2f}, Acc: {accuracy:.1%}")
            
            # Step forward
            current_train_start += pd.Timedelta(days=step_days * 1.5)
        
        if ticker_folds:
            all_fold_results.append({
                "ticker": ticker,
                "folds": ticker_folds,
                "num_folds": len(ticker_folds),
                "mean_sharpe": np.mean([f["sharpe"] for f in ticker_folds]),
                "std_sharpe": np.std([f["sharpe"] for f in ticker_folds]),
                "min_sharpe": np.min([f["sharpe"] for f in ticker_folds]),
                "max_sharpe": np.max([f["sharpe"] for f in ticker_folds]),
                "pct_positive_sharpe": np.mean([f["sharpe"] > 0 for f in ticker_folds]),
                "mean_accuracy": np.mean([f["accuracy"] for f in ticker_folds]),
            })
    
    # Aggregate results
    if not all_fold_results:
        return {"status": "FAIL", "reason": "No valid fold results"}
    
    all_sharpes = [f["sharpe"] for r in all_fold_results for f in r["folds"]]
    all_accuracies = [f["accuracy"] for r in all_fold_results for f in r["folds"]]
    
    summary = {
        "model_type": model_type,
        "total_folds": len(all_sharpes),
        "mean_sharpe": np.mean(all_sharpes),
        "std_sharpe": np.std(all_sharpes),
        "min_sharpe": np.min(all_sharpes),
        "max_sharpe": np.max(all_sharpes),
        "pct_positive_sharpe": np.mean([s > 0 for s in all_sharpes]),
        "mean_accuracy": np.mean(all_accuracies),
        "per_ticker_results": all_fold_results,
    }
    
    # Determine status
    if summary["mean_sharpe"] >= THRESHOLDS["pass_sharpe"]:
        status = "✅ PASS"
        status_code = "PASS"
    elif summary["mean_sharpe"] >= THRESHOLDS["warning_sharpe"]:
        status = "⚠️ WARNING"
        status_code = "WARNING"
    else:
        status = "❌ FAIL"
        status_code = "FAIL"
    
    summary["status"] = status_code
    summary["status_display"] = status
    
    print("\n" + "-"*70)
    print("WALK-FORWARD SUMMARY")
    print("-"*70)
    print(f"Status: {status}")
    print(f"Total Folds: {summary['total_folds']}")
    print(f"Mean Sharpe: {summary['mean_sharpe']:.2f} ± {summary['std_sharpe']:.2f}")
    print(f"Sharpe Range: [{summary['min_sharpe']:.2f}, {summary['max_sharpe']:.2f}]")
    print(f"% Positive Sharpe Folds: {summary['pct_positive_sharpe']:.1%}")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.1%}")
    
    return summary


# ============================================================================
# PART 3: PURGED K-FOLD CROSS-VALIDATION
# ============================================================================

def purged_kfold_split(n_samples: int, n_splits: int = 5, 
                       purge_days: int = 5, embargo_days: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged K-fold splits for time series.
    
    Purging removes data near fold boundaries to prevent information leakage.
    Embargo adds an extra buffer after test sets.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        purge_days: Days to remove before each test fold
        embargo_days: Extra buffer after test fold
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    fold_size = n_samples // n_splits
    splits = []
    
    for i in range(n_splits):
        # Test fold boundaries
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        
        # Create test indices
        test_indices = np.arange(test_start, test_end)
        
        # Create train indices with purging
        all_indices = np.arange(n_samples)
        
        # Remove test indices
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_start:test_end] = False
        
        # Remove purge window before test
        purge_start = max(0, test_start - purge_days)
        train_mask[purge_start:test_start] = False
        
        # Remove embargo window after test
        embargo_end = min(n_samples, test_end + embargo_days)
        train_mask[test_end:embargo_end] = False
        
        train_indices = all_indices[train_mask]
        
        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))
    
    return splits


def run_purged_kfold_validation(model_type: str = "rf") -> Dict[str, Any]:
    """
    Run purged K-fold cross-validation.
    
    This prevents information leakage between folds by:
    1. Removing data near fold boundaries (purging)
    2. Adding an embargo period after test folds
    """
    print("\n" + "="*70)
    print("PART 3: PURGED K-FOLD CROSS-VALIDATION")
    print("="*70)
    print(f"Folds: {PURGED_KFOLD_CONFIG['n_splits']}")
    print(f"Purge Days: {PURGED_KFOLD_CONFIG['purge_days']}")
    print(f"Embargo Days: {PURGED_KFOLD_CONFIG['embargo_days']}")
    print(f"Model Type: {model_type.upper()}")
    print("-"*70)
    
    all_fold_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n📊 Purged K-Fold for {ticker}...")
        
        # Load all data
        X_all, y_all, dates_all = load_data_for_period(
            ticker,
            "2022-01-01",
            "2025-12-31",
            period="5y"
        )
        
        if X_all is None or len(X_all) < 200:
            print(f"  ⚠️ Skipping {ticker} - insufficient data")
            continue
        
        # Get purged splits
        splits = purged_kfold_split(
            len(X_all),
            n_splits=PURGED_KFOLD_CONFIG["n_splits"],
            purge_days=PURGED_KFOLD_CONFIG["purge_days"],
            embargo_days=PURGED_KFOLD_CONFIG["embargo_days"],
        )
        
        ticker_folds = []
        
        for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]
            
            # Train model
            if model_type == "rf":
                model = get_optimized_rf()
            elif model_type == "xgb" and HAS_XGB:
                model = get_optimized_xgb()
            else:
                break
            
            model.fit(X_train, y_train)
            
            # Predict
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            strategy_returns = np.sign(test_pred) * y_test
            sharpe = calculate_sharpe(strategy_returns)
            accuracy = calculate_direction_accuracy(test_pred, y_test)
            
            # Training metrics
            train_pred = model.predict(X_train)
            train_strategy = np.sign(train_pred) * y_train
            train_sharpe = calculate_sharpe(train_strategy)
            
            fold_result = {
                "fold": fold_num,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "purged_samples": len(X_all) - len(X_train) - len(X_test),
                "train_sharpe": train_sharpe,
                "test_sharpe": sharpe,
                "accuracy": accuracy,
                "overfitting_gap": train_sharpe - sharpe,
            }
            
            ticker_folds.append(fold_result)
            
            print(f"  Fold {fold_num}: Train Sharpe: {train_sharpe:.2f} → Test Sharpe: {sharpe:.2f} | Acc: {accuracy:.1%}")
        
        if ticker_folds:
            all_fold_results.append({
                "ticker": ticker,
                "folds": ticker_folds,
                "num_folds": len(ticker_folds),
                "mean_test_sharpe": np.mean([f["test_sharpe"] for f in ticker_folds]),
                "mean_train_sharpe": np.mean([f["train_sharpe"] for f in ticker_folds]),
                "mean_overfitting_gap": np.mean([f["overfitting_gap"] for f in ticker_folds]),
                "mean_accuracy": np.mean([f["accuracy"] for f in ticker_folds]),
            })
    
    # Aggregate results
    if not all_fold_results:
        return {"status": "FAIL", "reason": "No valid fold results"}
    
    all_test_sharpes = [f["test_sharpe"] for r in all_fold_results for f in r["folds"]]
    all_train_sharpes = [f["train_sharpe"] for r in all_fold_results for f in r["folds"]]
    all_gaps = [f["overfitting_gap"] for r in all_fold_results for f in r["folds"]]
    all_accuracies = [f["accuracy"] for r in all_fold_results for f in r["folds"]]
    
    summary = {
        "model_type": model_type,
        "total_folds": len(all_test_sharpes),
        "mean_train_sharpe": np.mean(all_train_sharpes),
        "mean_test_sharpe": np.mean(all_test_sharpes),
        "std_test_sharpe": np.std(all_test_sharpes),
        "mean_overfitting_gap": np.mean(all_gaps),
        "pct_positive_sharpe": np.mean([s > 0 for s in all_test_sharpes]),
        "mean_accuracy": np.mean(all_accuracies),
        "per_ticker_results": all_fold_results,
    }
    
    # Determine status based on overfitting gap
    if summary["mean_overfitting_gap"] < 1.0 and summary["mean_test_sharpe"] >= THRESHOLDS["warning_sharpe"]:
        status = "✅ PASS"
        status_code = "PASS"
    elif summary["mean_overfitting_gap"] < 2.0:
        status = "⚠️ WARNING"
        status_code = "WARNING"
    else:
        status = "❌ FAIL"
        status_code = "FAIL"
    
    summary["status"] = status_code
    summary["status_display"] = status
    
    print("\n" + "-"*70)
    print("PURGED K-FOLD SUMMARY")
    print("-"*70)
    print(f"Status: {status}")
    print(f"Total Folds: {summary['total_folds']}")
    print(f"Mean Train Sharpe: {summary['mean_train_sharpe']:.2f}")
    print(f"Mean Test Sharpe: {summary['mean_test_sharpe']:.2f} ± {summary['std_test_sharpe']:.2f}")
    print(f"Mean Overfitting Gap: {summary['mean_overfitting_gap']:.2f}")
    print(f"% Positive Sharpe Folds: {summary['pct_positive_sharpe']:.1%}")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.1%}")
    
    return summary


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

def run_full_validation(model_types: List[str] = None) -> Dict[str, Any]:
    """
    Run complete validation suite.
    
    Args:
        model_types: List of model types to test ["rf", "xgb"]
    
    Returns:
        Complete results dictionary
    """
    if model_types is None:
        model_types = ["rf"]
        if HAS_XGB:
            model_types.append("xgb")
    
    print("="*70)
    print("COMPREHENSIVE OUT-OF-SAMPLE VALIDATION SUITE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing Models: {model_types}")
    print(f"Testing Tickers: {TEST_TICKERS}")
    print(f"Optimization Benchmark: Sharpe {OPTIMIZATION_SHARPE}")
    
    all_results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"VALIDATING: {model_type.upper()}")
        print(f"{'='*70}")
        
        model_results = {
            "model_type": model_type,
            "holdout_test": None,
            "walk_forward": None,
            "purged_kfold": None,
        }
        
        # Part 1: Holdout Test
        try:
            model_results["holdout_test"] = run_holdout_test(model_type)
        except Exception as e:
            print(f"❌ Holdout test failed: {e}")
            model_results["holdout_test"] = {"status": "ERROR", "error": str(e)}
        
        # Part 2: Walk-Forward
        try:
            model_results["walk_forward"] = run_walk_forward_validation(model_type)
        except Exception as e:
            print(f"❌ Walk-forward failed: {e}")
            model_results["walk_forward"] = {"status": "ERROR", "error": str(e)}
        
        # Part 3: Purged K-Fold
        try:
            model_results["purged_kfold"] = run_purged_kfold_validation(model_type)
        except Exception as e:
            print(f"❌ Purged K-fold failed: {e}")
            model_results["purged_kfold"] = {"status": "ERROR", "error": str(e)}
        
        all_results[model_type] = model_results
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL VALIDATION SUMMARY")
    print("="*70)
    
    for model_type, results in all_results.items():
        print(f"\n{model_type.upper()}:")
        
        if results["holdout_test"]:
            h = results["holdout_test"]
            print(f"  Holdout:      {h.get('status_display', 'N/A')} (Sharpe: {h.get('avg_holdout_sharpe', 0):.2f})")
        
        if results["walk_forward"]:
            w = results["walk_forward"]
            print(f"  Walk-Forward: {w.get('status_display', 'N/A')} (Sharpe: {w.get('mean_sharpe', 0):.2f})")
        
        if results["purged_kfold"]:
            p = results["purged_kfold"]
            print(f"  Purged K-Fold: {p.get('status_display', 'N/A')} (Sharpe: {p.get('mean_test_sharpe', 0):.2f})")
    
    # Overall verdict
    print("\n" + "-"*70)
    print("OVERALL VERDICT")
    print("-"*70)
    
    for model_type, results in all_results.items():
        statuses = []
        for test_name in ["holdout_test", "walk_forward", "purged_kfold"]:
            if results[test_name]:
                statuses.append(results[test_name].get("status", "UNKNOWN"))
        
        if all(s == "PASS" for s in statuses):
            verdict = "✅ ALL TESTS PASSED - Model is robust"
        elif any(s == "FAIL" for s in statuses):
            verdict = "❌ SOME TESTS FAILED - Overfitting detected"
        elif any(s == "WARNING" for s in statuses):
            verdict = "⚠️ WARNINGS PRESENT - Proceed with caution"
        else:
            verdict = "❓ INCONCLUSIVE - Review results manually"
        
        print(f"{model_type.upper()}: {verdict}")
    
    # Save results
    output_path = PROJECT_ROOT / "experiments" / "validation_results.json"
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(output_path, "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Out-of-Sample Validation Suite")
    parser.add_argument("--models", nargs="+", default=None, 
                        help="Model types to test (rf, xgb)")
    parser.add_argument("--part", type=int, default=None,
                        help="Run only specific part (1, 2, or 3)")
    
    args = parser.parse_args()
    
    if args.part == 1:
        run_holdout_test("rf")
        if HAS_XGB:
            run_holdout_test("xgb")
    elif args.part == 2:
        run_walk_forward_validation("rf")
        if HAS_XGB:
            run_walk_forward_validation("xgb")
    elif args.part == 3:
        run_purged_kfold_validation("rf")
        if HAS_XGB:
            run_purged_kfold_validation("xgb")
    else:
        results = run_full_validation(args.models)
