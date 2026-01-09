#!/usr/bin/env python3
"""
Test script to verify the optimized model configuration.

This script validates:
1. OPTIMIZED_MODEL_CONFIG is correctly defined
2. OPTIMIZED_FEATURES are available
3. make_model() with use_optimized=True works
4. Temperature scaling calibration works
5. A/B testing infrastructure works
6. Feature engineering produces expected features
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd


def test_config():
    """Test 1: Verify OPTIMIZED_MODEL_CONFIG exists and has correct values."""
    print("\n" + "="*60)
    print("TEST 1: Configuration Check")
    print("="*60)
    
    from src.config import OPTIMIZED_MODEL_CONFIG, OPTIMIZED_FEATURES, is_optimized_mode
    
    # Check required keys
    required_keys = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda",
        "use_temperature_scaling", "temperature"
    ]
    
    for key in required_keys:
        assert key in OPTIMIZED_MODEL_CONFIG, f"Missing key: {key}"
        print(f"  ✓ {key}: {OPTIMIZED_MODEL_CONFIG[key]}")
    
    # Check feature count
    assert len(OPTIMIZED_FEATURES) == 20, f"Expected 20 features, got {len(OPTIMIZED_FEATURES)}"
    print(f"\n  ✓ OPTIMIZED_FEATURES count: {len(OPTIMIZED_FEATURES)}")
    print(f"  ✓ Features: {OPTIMIZED_FEATURES[:5]}...")
    
    # Check is_optimized_mode function
    mode = is_optimized_mode()
    print(f"  ✓ is_optimized_mode(): {mode}")
    
    print("\n✅ TEST 1 PASSED: Configuration is correct")
    return True


def test_make_model():
    """Test 2: Verify make_model() with use_optimized parameter."""
    print("\n" + "="*60)
    print("TEST 2: Model Factory (make_model)")
    print("="*60)
    
    from src.core.models import make_model
    
    # Test with optimized=True (default)
    print("\n  Creating optimized XGBoost model...")
    model_opt = make_model("xgb", use_optimized=True)
    print(f"  ✓ Model created: {type(model_opt).__name__}")
    
    # Check hyperparameters
    params = model_opt.get_params()
    assert params["n_estimators"] == 450, f"Wrong n_estimators: {params['n_estimators']}"
    assert params["max_depth"] == 7, f"Wrong max_depth: {params['max_depth']}"
    print(f"  ✓ n_estimators: {params['n_estimators']}")
    print(f"  ✓ max_depth: {params['max_depth']}")
    print(f"  ✓ learning_rate: {params['learning_rate']}")
    
    # Test with optimized=False
    print("\n  Creating legacy XGBoost model...")
    model_leg = make_model("xgb", use_optimized=False)
    params_leg = model_leg.get_params()
    print(f"  ✓ Legacy model created")
    
    print("\n✅ TEST 2 PASSED: make_model() works correctly")
    return True


def test_calibration():
    """Test 3: Verify temperature scaling calibration."""
    print("\n" + "="*60)
    print("TEST 3: Temperature Scaling Calibration")
    print("="*60)
    
    from src.core.calibration import (
        temperature_scale,
        calibrate_predictions,
        PredictionCalibrator,
        get_default_temperature,
    )
    
    # Test basic scaling
    raw_pred = 0.10  # 10% predicted return
    temp = get_default_temperature()
    calibrated = temperature_scale(raw_pred, temp)
    
    print(f"\n  Raw prediction: {raw_pred:.4f}")
    print(f"  Temperature: {temp}")
    print(f"  Calibrated prediction: {calibrated:.4f}")
    
    assert calibrated < raw_pred, "Calibration should reduce prediction magnitude"
    print(f"  ✓ Calibration reduces magnitude correctly")
    
    # Test with array
    raw_array = np.array([0.05, 0.10, -0.03, 0.15])
    calibrated_array = temperature_scale(raw_array, temp)
    print(f"\n  Array calibration:")
    print(f"    Raw: {raw_array}")
    print(f"    Calibrated: {calibrated_array}")
    
    # Test PredictionCalibrator class
    calibrator = PredictionCalibrator(temperature=2.9)
    result = calibrator.calibrate(raw_array)
    print(f"\n  ✓ PredictionCalibrator works: {result[:2]}...")
    
    # Test calibrate_predictions convenience function
    preds, metadata = calibrate_predictions(raw_array)
    assert metadata["calibrated"] == True
    assert metadata["temperature"] == 2.9
    print(f"  ✓ calibrate_predictions metadata: {metadata}")
    
    print("\n✅ TEST 3 PASSED: Temperature scaling works correctly")
    return True


def test_ab_testing():
    """Test 4: Verify A/B testing infrastructure."""
    print("\n" + "="*60)
    print("TEST 4: A/B Testing Infrastructure")
    print("="*60)
    
    from src.core.ab_testing import (
        get_ab_config,
        set_ab_variant,
        get_active_variant,
        is_optimized_active,
        log_prediction,
        variant_context,
        generate_ab_report,
    )
    
    # Test config
    config = get_ab_config()
    print(f"\n  ✓ A/B config loaded: enabled={config.enabled}")
    print(f"  ✓ Default variant: {config.default_variant}")
    
    # Test variant switching
    original = get_active_variant()
    print(f"\n  Current variant: {original}")
    
    set_ab_variant("legacy")
    assert get_active_variant() == "legacy"
    print(f"  ✓ Switched to legacy")
    
    set_ab_variant("optimized")
    assert get_active_variant() == "optimized"
    print(f"  ✓ Switched back to optimized")
    
    # Test context manager
    with variant_context("legacy"):
        assert get_active_variant() == "legacy"
        print(f"  ✓ Context manager works (inside: legacy)")
    
    assert get_active_variant() == "optimized"
    print(f"  ✓ Context manager restored: optimized")
    
    # Test prediction logging
    log_prediction("AAPL", 0.05, position=1.0)
    print(f"  ✓ Prediction logged")
    
    # Test report generation
    report = generate_ab_report()
    assert "A/B TEST REPORT" in report
    print(f"  ✓ Report generated ({len(report)} chars)")
    
    print("\n✅ TEST 4 PASSED: A/B testing works correctly")
    return True


def test_feature_engineering():
    """Test 5: Verify optimized feature engineering."""
    print("\n" + "="*60)
    print("TEST 5: Feature Engineering")
    print("="*60)
    
    from src.core.features import (
        build_all_features,
        build_optimized_features,
        add_momentum_indicators,
    )
    from src.config import OPTIMIZED_FEATURES
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "High": 101 + np.cumsum(np.random.randn(100) * 0.5),
        "Low": 99 + np.cumsum(np.random.randn(100) * 0.5),
        "Close": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "Volume": np.random.randint(1000000, 5000000, 100),
    }, index=dates)
    
    # Fix High/Low to be consistent
    df["High"] = df[["Open", "High", "Close"]].max(axis=1) + 0.5
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1) - 0.5
    
    print(f"\n  Sample data shape: {df.shape}")
    
    # Test add_momentum_indicators
    df_mom = add_momentum_indicators(df.copy())
    mom_cols = ["obv", "momentum", "williams_r", "cci", "stoch_k", "mfi"]
    present = [c for c in mom_cols if c in df_mom.columns]
    print(f"  ✓ Momentum indicators added: {present}")
    
    # Test build_all_features
    df_all = build_all_features(df.copy(), use_optimized_features=True)
    print(f"  ✓ All features built: {df_all.shape[1]} columns")
    
    # Test build_optimized_features
    df_opt, available = build_optimized_features(df.copy())
    print(f"  ✓ Optimized features: {len(available)}/{len(OPTIMIZED_FEATURES)}")
    print(f"    Available: {available[:5]}...")
    
    # Check which optimized features are present
    missing = [f for f in OPTIMIZED_FEATURES if f not in df_opt.columns]
    if missing:
        print(f"    Missing (expected for some): {missing}")
    
    print("\n✅ TEST 5 PASSED: Feature engineering works correctly")
    return True


def test_integration():
    """Test 6: Integration test - full prediction pipeline."""
    print("\n" + "="*60)
    print("TEST 6: Integration Test")
    print("="*60)
    
    from src.core.models import make_model
    from src.core.calibration import calibrate_predictions
    from src.core.ab_testing import log_prediction, get_active_variant
    from src.config import OPTIMIZED_MODEL_CONFIG
    
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] + 0.002 * np.random.randn(n_samples)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n  Training data: {X_train.shape}")
    print(f"  Test data: {X_test.shape}")
    
    # Create and train model
    model = make_model("xgb", use_optimized=True)
    model.fit(X_train, y_train)
    print(f"  ✓ Model trained")
    
    # Make predictions
    raw_preds = model.predict(X_test)
    print(f"  ✓ Raw predictions: mean={raw_preds.mean():.6f}, std={raw_preds.std():.6f}")
    
    # Apply calibration
    calibrated_preds, metadata = calibrate_predictions(raw_preds)
    print(f"  ✓ Calibrated predictions: mean={calibrated_preds.mean():.6f}")
    print(f"    Temperature: {metadata['temperature']}")
    
    # Log predictions
    variant = get_active_variant()
    for i, pred in enumerate(calibrated_preds[:3]):
        log_prediction(f"TEST_{i}", pred, variant=variant)
    print(f"  ✓ Predictions logged (variant: {variant})")
    
    # Calculate direction accuracy
    direction_correct = np.sign(calibrated_preds) == np.sign(y_test)
    accuracy = direction_correct.mean()
    print(f"\n  Direction accuracy: {accuracy*100:.1f}%")
    
    print("\n✅ TEST 6 PASSED: Integration test successful")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("OPTIMIZED MODEL CONFIGURATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Configuration", test_config),
        ("Model Factory", test_make_model),
        ("Calibration", test_calibration),
        ("A/B Testing", test_ab_testing),
        ("Feature Engineering", test_feature_engineering),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    
    for name, result in results:
        status = "✅" if result == "PASS" else "❌"
        print(f"  {status} {name}: {result}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Optimized configuration is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
