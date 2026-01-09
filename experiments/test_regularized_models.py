"""
Test Regularized Models
=======================

This script tests the new regularized configurations against
the original overfitted models to see if we've reduced the
train-test gap.

Run: python experiments/test_regularized_models.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_model import build_features_and_target
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Import original configs for comparison
from src.config import get_optimized_rf_config, get_optimized_config

# Import new regularized configs
from experiments.anti_overfitting_config import (
    REGULARIZED_RF_CONFIG,
    REGULARIZED_XGB_CONFIG,
    MINIMAL_FEATURES,
    CONSERVATIVE_FEATURES,
    validate_before_deploy,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# Test config
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
HOLDOUT_START = "2025-01-01"


def calculate_sharpe(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def calculate_direction_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Calculate direction prediction accuracy."""
    return float(np.mean(np.sign(pred) == np.sign(actual)))


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-10)
    return float(np.min(drawdown))


def filter_features(X: np.ndarray, feature_names: list, keep_features: list) -> np.ndarray:
    """Filter feature matrix to keep only specified features."""
    indices = []
    for i, name in enumerate(feature_names):
        if name in keep_features:
            indices.append(i)
    
    if len(indices) == 0:
        print(f"  ⚠️ No matching features found! Using all features.")
        return X
    
    return X[:, indices]


def test_model_config(ticker: str, model_type: str, config: dict, 
                      config_name: str, use_minimal_features: bool = False) -> dict:
    """Test a specific model configuration."""
    
    # Load data
    result = build_features_and_target(ticker, period="5y", horizon=1)
    if result is None:
        return None
    
    X, y, _, _, _, _, dates = result
    
    # Get feature names (approximate - we don't have exact names)
    # For now, we'll test with full features vs reduced count
    
    # Split data
    dates_arr = pd.to_datetime(dates.values if hasattr(dates, 'values') else dates)
    holdout_start = pd.Timestamp(HOLDOUT_START)
    if dates_arr.tz is not None:
        holdout_start = holdout_start.tz_localize(dates_arr.tz)
    
    train_mask = dates_arr < holdout_start
    test_mask = dates_arr >= holdout_start
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 100 or len(X_test) < 20:
        return None
    
    # Optionally reduce features by random selection (simulating minimal features)
    if use_minimal_features:
        n_features = min(10, X_train.shape[1])
        np.random.seed(42)
        feature_idx = np.random.choice(X_train.shape[1], n_features, replace=False)
        X_train = X_train[:, feature_idx]
        X_test = X_test[:, feature_idx]
    
    # Create model
    if model_type == "rf":
        model = RandomForestRegressor(**config)
    elif model_type == "xgb" and HAS_XGB:
        model = XGBRegressor(**config)
    else:
        return None
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_strategy = np.sign(train_pred) * y_train
    test_strategy = np.sign(test_pred) * y_test
    
    train_sharpe = calculate_sharpe(train_strategy)
    test_sharpe = calculate_sharpe(test_strategy)
    train_accuracy = calculate_direction_accuracy(train_pred, y_train)
    test_accuracy = calculate_direction_accuracy(test_pred, y_test)
    test_drawdown = calculate_max_drawdown(test_strategy)
    
    return {
        "ticker": ticker,
        "model_type": model_type,
        "config_name": config_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "overfitting_gap": train_sharpe - test_sharpe,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "test_drawdown": test_drawdown,
    }


def main():
    """Run comparison tests."""
    print("="*70)
    print("REGULARIZED MODEL COMPARISON TEST")
    print("="*70)
    print(f"Testing: Original vs Regularized configurations")
    print(f"Holdout Period: {HOLDOUT_START} onwards")
    print("-"*70)
    
    all_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n📊 Testing {ticker}...")
        
        # Test RF configurations
        # 1. Original RF config
        orig_rf_config = get_optimized_rf_config()
        result = test_model_config(ticker, "rf", orig_rf_config, "Original RF")
        if result:
            all_results.append(result)
            print(f"  Original RF:     Train Sharpe {result['train_sharpe']:6.2f} → Test Sharpe {result['test_sharpe']:6.2f} (Gap: {result['overfitting_gap']:5.2f})")
        
        # 2. Regularized RF config
        result = test_model_config(ticker, "rf", REGULARIZED_RF_CONFIG, "Regularized RF")
        if result:
            all_results.append(result)
            print(f"  Regularized RF:  Train Sharpe {result['train_sharpe']:6.2f} → Test Sharpe {result['test_sharpe']:6.2f} (Gap: {result['overfitting_gap']:5.2f})")
        
        # 3. Regularized RF with fewer features
        result = test_model_config(ticker, "rf", REGULARIZED_RF_CONFIG, "Regularized RF + Min Features", use_minimal_features=True)
        if result:
            all_results.append(result)
            print(f"  Reg RF + MinFeat: Train Sharpe {result['train_sharpe']:6.2f} → Test Sharpe {result['test_sharpe']:6.2f} (Gap: {result['overfitting_gap']:5.2f})")
        
        # Test XGB configurations
        if HAS_XGB:
            # 1. Original XGB config
            orig_xgb_config = get_optimized_config()
            orig_xgb_config["random_state"] = 42
            orig_xgb_config["n_jobs"] = -1
            result = test_model_config(ticker, "xgb", orig_xgb_config, "Original XGB")
            if result:
                all_results.append(result)
                print(f"  Original XGB:    Train Sharpe {result['train_sharpe']:6.2f} → Test Sharpe {result['test_sharpe']:6.2f} (Gap: {result['overfitting_gap']:5.2f})")
            
            # 2. Regularized XGB config
            result = test_model_config(ticker, "xgb", REGULARIZED_XGB_CONFIG, "Regularized XGB")
            if result:
                all_results.append(result)
                print(f"  Regularized XGB: Train Sharpe {result['train_sharpe']:6.2f} → Test Sharpe {result['test_sharpe']:6.2f} (Gap: {result['overfitting_gap']:5.2f})")
    
    # Aggregate results
    print("\n" + "="*70)
    print("SUMMARY BY CONFIGURATION")
    print("="*70)
    
    configs = set(r["config_name"] for r in all_results)
    
    summary_data = []
    for config_name in sorted(configs):
        config_results = [r for r in all_results if r["config_name"] == config_name]
        
        avg_train_sharpe = np.mean([r["train_sharpe"] for r in config_results])
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in config_results])
        avg_gap = np.mean([r["overfitting_gap"] for r in config_results])
        avg_train_acc = np.mean([r["train_accuracy"] for r in config_results])
        avg_test_acc = np.mean([r["test_accuracy"] for r in config_results])
        
        summary_data.append({
            "config": config_name,
            "train_sharpe": avg_train_sharpe,
            "test_sharpe": avg_test_sharpe,
            "gap": avg_gap,
            "train_acc": avg_train_acc,
            "test_acc": avg_test_acc,
        })
    
    # Sort by test Sharpe (what we care about)
    summary_data.sort(key=lambda x: x["test_sharpe"], reverse=True)
    
    print(f"\n{'Configuration':<25} {'Train Sharpe':>12} {'Test Sharpe':>12} {'Gap':>8} {'Test Acc':>10}")
    print("-"*70)
    
    for s in summary_data:
        gap_indicator = "✅" if s["gap"] < 5 else "⚠️" if s["gap"] < 10 else "❌"
        print(f"{s['config']:<25} {s['train_sharpe']:>12.2f} {s['test_sharpe']:>12.2f} {s['gap']:>6.2f} {gap_indicator} {s['test_acc']*100:>8.1f}%")
    
    # Determine winner
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    best_config = summary_data[0]
    print(f"\n🏆 Best Test Sharpe: {best_config['config']} ({best_config['test_sharpe']:.2f})")
    
    # Find lowest overfitting gap
    lowest_gap = min(summary_data, key=lambda x: x["gap"])
    print(f"🎯 Lowest Overfitting Gap: {lowest_gap['config']} ({lowest_gap['gap']:.2f})")
    
    # Check if regularization helped
    orig_rf = next((s for s in summary_data if s["config"] == "Original RF"), None)
    reg_rf = next((s for s in summary_data if s["config"] == "Regularized RF"), None)
    
    if orig_rf and reg_rf:
        gap_reduction = orig_rf["gap"] - reg_rf["gap"]
        test_improvement = reg_rf["test_sharpe"] - orig_rf["test_sharpe"]
        print(f"\n📊 RF Regularization Impact:")
        print(f"   Overfitting gap reduced by: {gap_reduction:.2f}")
        print(f"   Test Sharpe changed by: {test_improvement:+.2f}")
        
        if gap_reduction > 0:
            print(f"   ✅ Regularization reduced overfitting!")
        else:
            print(f"   ⚠️ Regularization did not reduce overfitting gap")
    
    if HAS_XGB:
        orig_xgb = next((s for s in summary_data if s["config"] == "Original XGB"), None)
        reg_xgb = next((s for s in summary_data if s["config"] == "Regularized XGB"), None)
        
        if orig_xgb and reg_xgb:
            gap_reduction = orig_xgb["gap"] - reg_xgb["gap"]
            test_improvement = reg_xgb["test_sharpe"] - orig_xgb["test_sharpe"]
            print(f"\n📊 XGB Regularization Impact:")
            print(f"   Overfitting gap reduced by: {gap_reduction:.2f}")
            print(f"   Test Sharpe changed by: {test_improvement:+.2f}")
            
            if gap_reduction > 0:
                print(f"   ✅ Regularization reduced overfitting!")
            else:
                print(f"   ⚠️ Regularization did not reduce overfitting gap")
    
    # Deployment recommendations
    print("\n" + "="*70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*70)
    
    for s in summary_data:
        deployment = validate_before_deploy(
            holdout_sharpe=s["test_sharpe"],
            train_sharpe=s["train_sharpe"],
            holdout_accuracy=s["test_acc"],
            holdout_drawdown=-0.25  # Placeholder
        )
        
        status = "✅ APPROVED" if deployment["deploy_approved"] else "❌ NOT APPROVED"
        print(f"\n{s['config']}: {status}")
        for check, passed in deployment.items():
            if check != "deploy_approved":
                icon = "✓" if passed else "✗"
                print(f"   {icon} {check}")
    
    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
