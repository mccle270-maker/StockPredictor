"""
Test Optimized Model
====================
Validates the optimized configuration from Model Improvement Pipeline.

Tests:
1. Optimized XGBoost hyperparameters (xgb_optimized_v2)
2. Temperature Scaling calibration
3. OptimizedPredictor class
4. Comparison with baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TESTING OPTIMIZED MODEL CONFIGURATION")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Test 1: Verify Optimized Hyperparameters in Config
# ============================================================================
print("\n" + "-" * 70)
print("TEST 1: Verify Optimized Hyperparameters")
print("-" * 70)

from src.config import (
    MODEL_VERSIONS, 
    ACTIVE_MODEL_VERSIONS, 
    get_model_config,
    log_model_version
)

# Check that xgb_optimized_v2 exists
assert "xgb_optimized_v2" in MODEL_VERSIONS, "xgb_optimized_v2 not found in MODEL_VERSIONS"
print("✓ xgb_optimized_v2 exists in MODEL_VERSIONS")

# Check that it's the active version
assert ACTIVE_MODEL_VERSIONS["xgb"] == "xgb_optimized_v2", "xgb_optimized_v2 not active"
print("✓ xgb_optimized_v2 is the active XGB version")

# Verify key parameters
config = get_model_config("xgb")
expected = {
    "n_estimators": 450,
    "max_depth": 7,
    "learning_rate": 0.048,
    "reg_lambda": 9.3,
}
for key, expected_val in expected.items():
    actual_val = config.get(key)
    assert abs(actual_val - expected_val) < 0.01, f"{key}: expected {expected_val}, got {actual_val}"
    print(f"  ✓ {key} = {actual_val}")

print(f"\n📋 Active version: {log_model_version('xgb')}")

# ============================================================================
# Test 2: Temperature Scaling
# ============================================================================
print("\n" + "-" * 70)
print("TEST 2: Temperature Scaling Calibration")
print("-" * 70)

from model_improvements import TemperatureScaler

scaler = TemperatureScaler(temperature=2.9)
print(f"✓ TemperatureScaler initialized with T={scaler.temperature}")

# Test calibration
test_probs = np.array([0.3, 0.5, 0.7, 0.9])
calibrated = scaler.calibrate(test_probs)

print(f"  Raw probabilities: {test_probs}")
print(f"  Calibrated:        {calibrated.round(3)}")

# Temperature scaling should soften extreme probabilities
assert calibrated[0] > test_probs[0], "T>1 should increase low probabilities"
assert calibrated[3] < test_probs[3], "T>1 should decrease high probabilities"
print("✓ Temperature scaling works correctly (softens extremes)")

# ============================================================================
# Test 3: OptimizedPredictor Class
# ============================================================================
print("\n" + "-" * 70)
print("TEST 3: OptimizedPredictor Class")
print("-" * 70)

from model_improvements import OptimizedPredictor

predictor = OptimizedPredictor(
    use_calibration=True,
    use_volatility_weighting=True,
    confidence_threshold=0.55
)
print("✓ OptimizedPredictor initialized")

# Check optimized params
assert predictor.OPTIMIZED_PARAMS["n_estimators"] == 450
assert predictor.OPTIMIZED_PARAMS["max_depth"] == 7
assert predictor.OPTIMAL_TEMPERATURE == 2.9
print("✓ Optimized parameters verified")

# ============================================================================
# Test 4: Full Backtest Comparison
# ============================================================================
print("\n" + "-" * 70)
print("TEST 4: Full Backtest Comparison")
print("-" * 70)

from prediction_model import build_features_and_target

TICKERS = ["AAPL", "MSFT", "AMZN"]
PERIOD = "2y"

print(f"Loading data for: {TICKERS}")

all_X, all_y = [], []
for ticker in TICKERS:
    try:
        X, y, _, _, _, _, dates = build_features_and_target(
            ticker=ticker,
            period=PERIOD,
            horizon=1
        )
        if X is not None and len(X) >= 100:
            all_X.append(X)
            all_y.append(y)
            print(f"  ✓ {ticker}: {len(X)} samples")
    except Exception as e:
        print(f"  ⚠ {ticker}: {e}")

X = np.vstack(all_X)
y = np.concatenate(all_y)
print(f"\nTotal: {X.shape[0]} samples, {X.shape[1]} features")

# Split data
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ---- Baseline: Old XGBoost ----
print("\n📊 Training Baseline (old params)...")
from xgboost import XGBClassifier, XGBRegressor

baseline_params = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_child_weight": 100,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "random_state": 42,
}

baseline_model = XGBClassifier(
    **baseline_params,
    objective='binary:logistic',
    verbosity=0
)
y_train_binary = (y_train > 0).astype(int)
y_test_binary = (y_test > 0).astype(int)
baseline_model.fit(X_train, y_train_binary)

baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
baseline_preds = (baseline_probs > 0.5).astype(int)
baseline_positions = np.where(baseline_probs > 0.5, 1, 0)
baseline_returns = baseline_positions * y_test

baseline_sharpe = (baseline_returns.mean() / baseline_returns.std()) * np.sqrt(252) if baseline_returns.std() > 0 else 0
baseline_accuracy = (baseline_preds == y_test_binary).mean()

print(f"  Baseline Sharpe: {baseline_sharpe:.4f}")
print(f"  Baseline Accuracy: {baseline_accuracy:.1%}")

# ---- Optimized: New Config ----
print("\n📊 Training Optimized (new params + calibration)...")

predictor = OptimizedPredictor(
    use_calibration=True,
    use_volatility_weighting=False,  # No vol data in this test
    confidence_threshold=0.55
)
predictor.fit(X_train, y_train)

# Get backtest metrics
metrics = predictor.backtest(X_test, y_test)

print(f"  Optimized Sharpe: {metrics['sharpe']:.4f}")
print(f"  Optimized Accuracy: {metrics['accuracy']:.1%}")
print(f"  Total Return: {metrics['total_return']:.1%}")
print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("OPTIMIZATION RESULTS SUMMARY")
print("=" * 70)

sharpe_improvement = ((metrics['sharpe'] / baseline_sharpe) - 1) * 100 if baseline_sharpe > 0 else 0
accuracy_improvement = (metrics['accuracy'] - baseline_accuracy) * 100

print(f"\n{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
print("-" * 56)
print(f"{'Sharpe Ratio':<20} {baseline_sharpe:>12.4f} {metrics['sharpe']:>12.4f} {sharpe_improvement:>+11.1f}%")
print(f"{'Accuracy':<20} {baseline_accuracy:>11.1%} {metrics['accuracy']:>11.1%} {accuracy_improvement:>+11.1f}pp")
print("-" * 56)

if metrics['sharpe'] > baseline_sharpe:
    print("\n✅ OPTIMIZATION SUCCESSFUL!")
    print(f"   Sharpe improved by {sharpe_improvement:.1f}%")
else:
    print("\n⚠️ WARNING: Optimized model underperformed baseline")
    print("   This may be due to market regime changes or data differences")

print("\n📁 Files updated:")
print("   - src/config.py (xgb_optimized_v2 added and activated)")
print("   - model_improvements.py (TemperatureScaler, OptimizedPredictor added)")

print("\n🔧 To use the optimized predictor:")
print("   from model_improvements import OptimizedPredictor")
print("   predictor = OptimizedPredictor()")
print("   predictor.fit(X_train, y_train)")
print("   positions, confidence = predictor.predict_positions(X_test)")

print("\n" + "=" * 70)
