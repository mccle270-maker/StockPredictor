"""
True Unseen Data Test
=====================

This script tests models on TRULY UNSEEN data by:
1. Training on data up to a cutoff date (e.g., 2025-10-01)
2. Testing on data AFTER that date (that the model has NEVER seen)

This is the only valid way to assess overfitting risk.

Run: python experiments/test_unseen_data.py
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_model import build_features_and_target
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from src.config import get_optimized_rf_config, get_optimized_config

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# Test configuration
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
TRAINING_PERIOD = "3y"  # Use 3 years of data
HOLDOUT_DAYS = 60       # Last 60 days are UNSEEN data (never in training)
HORIZON = 1             # Predict next-day returns


def get_optimized_rf():
    """Create optimized Random Forest model."""
    config = get_optimized_rf_config()
    return RandomForestRegressor(**config)


def get_optimized_xgb():
    """Create optimized XGBoost model."""
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


def evaluate_on_unseen_data(ticker: str) -> dict:
    """
    Evaluate models on truly unseen data.
    
    Returns:
        Dict with metrics for RF, XGB, and Ensemble
    """
    print(f"\n{'='*60}")
    print(f"Testing {ticker}")
    print(f"{'='*60}")
    
    # Load data
    try:
        result = build_features_and_target(ticker, period=TRAINING_PERIOD, horizon=HORIZON)
        if result is None:
            print(f"  ❌ Failed to load data for {ticker}")
            return None
        X, y, _, _, _, _, dates = result
    except Exception as e:
        print(f"  ❌ Error loading {ticker}: {e}")
        return None
    
    # Convert dates to datetime if needed
    if hasattr(dates, 'values'):
        dates = dates.values
    
    # Find cutoff date (HOLDOUT_DAYS before last date)
    last_date = pd.Timestamp(dates[-1])
    cutoff_date = last_date - pd.Timedelta(days=HOLDOUT_DAYS)
    
    print(f"  Data range: {dates[0]} to {dates[-1]}")
    print(f"  Cutoff date: {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Training samples: data before {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Holdout samples: last {HOLDOUT_DAYS} days (UNSEEN)")
    
    # Split data chronologically
    dates_series = pd.Series(dates)
    train_mask = dates_series < cutoff_date
    test_mask = dates_series >= cutoff_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    if len(X_test) < 10:
        print(f"  ⚠️ Insufficient holdout data ({len(X_test)} samples)")
        return None
    
    results = {
        "ticker": ticker,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cutoff_date": cutoff_date.strftime('%Y-%m-%d'),
    }
    
    # =========================================================================
    # Test 1: Optimized Random Forest
    # =========================================================================
    print("\n  📊 Testing OPTIMIZED Random Forest...")
    rf = get_optimized_rf()
    rf.fit(X_train, y_train)
    
    # Training metrics
    rf_train_pred = rf.predict(X_train)
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    
    # Test (UNSEEN) metrics
    rf_test_pred = rf.predict(X_test)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    
    # Direction accuracy
    rf_direction_train = np.mean(np.sign(rf_train_pred) == np.sign(y_train))
    rf_direction_test = np.mean(np.sign(rf_test_pred) == np.sign(y_test))
    
    # Calculate Sharpe-like metric on test returns
    rf_strategy_returns = np.sign(rf_test_pred) * y_test
    rf_sharpe = np.mean(rf_strategy_returns) / (np.std(rf_strategy_returns) + 1e-8) * np.sqrt(252)
    
    results["rf"] = {
        "train_r2": rf_train_r2,
        "test_r2": rf_test_r2,
        "overfitting_gap": rf_train_r2 - rf_test_r2,
        "train_direction_acc": rf_direction_train,
        "test_direction_acc": rf_direction_test,
        "test_sharpe": rf_sharpe,
        "test_rmse": rf_test_rmse,
    }
    
    print(f"     Train R²: {rf_train_r2:.4f} | Test R²: {rf_test_r2:.4f} | Gap: {rf_train_r2 - rf_test_r2:.4f}")
    print(f"     Train Dir: {rf_direction_train:.1%} | Test Dir: {rf_direction_test:.1%}")
    print(f"     Test Sharpe: {rf_sharpe:.2f}")
    
    # =========================================================================
    # Test 2: Optimized XGBoost
    # =========================================================================
    if HAS_XGB:
        print("\n  📊 Testing OPTIMIZED XGBoost...")
        xgb = get_optimized_xgb()
        xgb.fit(X_train, y_train)
        
        # Training metrics
        xgb_train_pred = xgb.predict(X_train)
        xgb_train_r2 = r2_score(y_train, xgb_train_pred)
        
        # Test (UNSEEN) metrics
        xgb_test_pred = xgb.predict(X_test)
        xgb_test_r2 = r2_score(y_test, xgb_test_pred)
        xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
        
        # Direction accuracy
        xgb_direction_train = np.mean(np.sign(xgb_train_pred) == np.sign(y_train))
        xgb_direction_test = np.mean(np.sign(xgb_test_pred) == np.sign(y_test))
        
        # Sharpe
        xgb_strategy_returns = np.sign(xgb_test_pred) * y_test
        xgb_sharpe = np.mean(xgb_strategy_returns) / (np.std(xgb_strategy_returns) + 1e-8) * np.sqrt(252)
        
        results["xgb"] = {
            "train_r2": xgb_train_r2,
            "test_r2": xgb_test_r2,
            "overfitting_gap": xgb_train_r2 - xgb_test_r2,
            "train_direction_acc": xgb_direction_train,
            "test_direction_acc": xgb_direction_test,
            "test_sharpe": xgb_sharpe,
            "test_rmse": xgb_test_rmse,
        }
        
        print(f"     Train R²: {xgb_train_r2:.4f} | Test R²: {xgb_test_r2:.4f} | Gap: {xgb_train_r2 - xgb_test_r2:.4f}")
        print(f"     Train Dir: {xgb_direction_train:.1%} | Test Dir: {xgb_direction_test:.1%}")
        print(f"     Test Sharpe: {xgb_sharpe:.2f}")
    
    # =========================================================================
    # Test 3: Ensemble (RF + XGB, equal weights)
    # =========================================================================
    if HAS_XGB:
        print("\n  📊 Testing ENSEMBLE (RF + XGB, 50/50)...")
        
        # Average predictions
        ensemble_test_pred = (rf_test_pred + xgb_test_pred) / 2
        ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
        ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
        
        # Direction accuracy
        ensemble_direction_test = np.mean(np.sign(ensemble_test_pred) == np.sign(y_test))
        
        # Sharpe
        ensemble_strategy_returns = np.sign(ensemble_test_pred) * y_test
        ensemble_sharpe = np.mean(ensemble_strategy_returns) / (np.std(ensemble_strategy_returns) + 1e-8) * np.sqrt(252)
        
        # Disagreement rate
        disagreement = np.mean(np.sign(rf_test_pred) != np.sign(xgb_test_pred))
        
        results["ensemble"] = {
            "test_r2": ensemble_test_r2,
            "test_direction_acc": ensemble_direction_test,
            "test_sharpe": ensemble_sharpe,
            "test_rmse": ensemble_test_rmse,
            "disagreement_rate": disagreement,
        }
        
        print(f"     Test R²: {ensemble_test_r2:.4f}")
        print(f"     Test Dir: {ensemble_direction_test:.1%}")
        print(f"     Test Sharpe: {ensemble_sharpe:.2f}")
        print(f"     RF vs XGB disagreement: {disagreement:.1%}")
    
    return results


def main():
    """Run comprehensive unseen data test."""
    print("="*70)
    print("TRUE UNSEEN DATA TEST")
    print("="*70)
    print(f"\nThis test evaluates models on data they have NEVER seen:")
    print(f"  - Training: All data BEFORE the last {HOLDOUT_DAYS} days")
    print(f"  - Testing:  Last {HOLDOUT_DAYS} days (completely unseen)")
    print(f"  - Tickers:  {', '.join(TEST_TICKERS)}")
    print(f"  - Horizon:  {HORIZON} day(s)")
    
    all_results = []
    
    for ticker in TEST_TICKERS:
        result = evaluate_on_unseen_data(ticker)
        if result:
            all_results.append(result)
    
    # =========================================================================
    # Aggregate Results
    # =========================================================================
    print("\n" + "="*70)
    print("AGGREGATE RESULTS (UNSEEN DATA)")
    print("="*70)
    
    if not all_results:
        print("❌ No results to aggregate")
        return
    
    # RF Results
    rf_results = [r["rf"] for r in all_results if "rf" in r]
    if rf_results:
        avg_rf_train_r2 = np.mean([r["train_r2"] for r in rf_results])
        avg_rf_test_r2 = np.mean([r["test_r2"] for r in rf_results])
        avg_rf_gap = np.mean([r["overfitting_gap"] for r in rf_results])
        avg_rf_train_dir = np.mean([r["train_direction_acc"] for r in rf_results])
        avg_rf_test_dir = np.mean([r["test_direction_acc"] for r in rf_results])
        avg_rf_sharpe = np.mean([r["test_sharpe"] for r in rf_results])
        
        print(f"\n📊 OPTIMIZED RANDOM FOREST:")
        print(f"   Avg Train R²:        {avg_rf_train_r2:.4f}")
        print(f"   Avg Test R²:         {avg_rf_test_r2:.4f}")
        print(f"   Avg Overfitting Gap: {avg_rf_gap:.4f}")
        print(f"   Avg Train Direction: {avg_rf_train_dir:.1%}")
        print(f"   Avg Test Direction:  {avg_rf_test_dir:.1%}")
        print(f"   Avg Test Sharpe:     {avg_rf_sharpe:.2f}")
        
        # Overfitting assessment
        if avg_rf_gap > 0.3:
            print(f"   ⚠️  SEVERE OVERFITTING (gap > 0.3)")
        elif avg_rf_gap > 0.15:
            print(f"   ⚠️  MODERATE OVERFITTING (gap > 0.15)")
        elif avg_rf_gap > 0.05:
            print(f"   🟡 MILD OVERFITTING (gap > 0.05)")
        else:
            print(f"   ✅ GOOD GENERALIZATION (gap <= 0.05)")
    
    # XGB Results
    xgb_results = [r["xgb"] for r in all_results if "xgb" in r]
    if xgb_results:
        avg_xgb_train_r2 = np.mean([r["train_r2"] for r in xgb_results])
        avg_xgb_test_r2 = np.mean([r["test_r2"] for r in xgb_results])
        avg_xgb_gap = np.mean([r["overfitting_gap"] for r in xgb_results])
        avg_xgb_train_dir = np.mean([r["train_direction_acc"] for r in xgb_results])
        avg_xgb_test_dir = np.mean([r["test_direction_acc"] for r in xgb_results])
        avg_xgb_sharpe = np.mean([r["test_sharpe"] for r in xgb_results])
        
        print(f"\n📊 OPTIMIZED XGBOOST:")
        print(f"   Avg Train R²:        {avg_xgb_train_r2:.4f}")
        print(f"   Avg Test R²:         {avg_xgb_test_r2:.4f}")
        print(f"   Avg Overfitting Gap: {avg_xgb_gap:.4f}")
        print(f"   Avg Train Direction: {avg_xgb_train_dir:.1%}")
        print(f"   Avg Test Direction:  {avg_xgb_test_dir:.1%}")
        print(f"   Avg Test Sharpe:     {avg_xgb_sharpe:.2f}")
        
        if avg_xgb_gap > 0.3:
            print(f"   ⚠️  SEVERE OVERFITTING (gap > 0.3)")
        elif avg_xgb_gap > 0.15:
            print(f"   ⚠️  MODERATE OVERFITTING (gap > 0.15)")
        elif avg_xgb_gap > 0.05:
            print(f"   🟡 MILD OVERFITTING (gap > 0.05)")
        else:
            print(f"   ✅ GOOD GENERALIZATION (gap <= 0.05)")
    
    # Ensemble Results
    ensemble_results = [r["ensemble"] for r in all_results if "ensemble" in r]
    if ensemble_results:
        avg_ens_test_r2 = np.mean([r["test_r2"] for r in ensemble_results])
        avg_ens_test_dir = np.mean([r["test_direction_acc"] for r in ensemble_results])
        avg_ens_sharpe = np.mean([r["test_sharpe"] for r in ensemble_results])
        avg_disagree = np.mean([r["disagreement_rate"] for r in ensemble_results])
        
        print(f"\n📊 ENSEMBLE (RF + XGB, 50/50):")
        print(f"   Avg Test R²:         {avg_ens_test_r2:.4f}")
        print(f"   Avg Test Direction:  {avg_ens_test_dir:.1%}")
        print(f"   Avg Test Sharpe:     {avg_ens_sharpe:.2f}")
        print(f"   Avg Disagreement:    {avg_disagree:.1%}")
    
    # =========================================================================
    # Comparison Table
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON TABLE (UNSEEN DATA ONLY)")
    print("="*70)
    print(f"\n{'Model':<20} {'Test R²':<12} {'Test Dir':<12} {'Test Sharpe':<12} {'Overfitting':<15}")
    print("-"*70)
    
    if rf_results:
        print(f"{'Optimized RF':<20} {avg_rf_test_r2:<12.4f} {avg_rf_test_dir*100:<12.1f}% {avg_rf_sharpe:<12.2f} {avg_rf_gap:.4f}")
    if xgb_results:
        print(f"{'Optimized XGB':<20} {avg_xgb_test_r2:<12.4f} {avg_xgb_test_dir*100:<12.1f}% {avg_xgb_sharpe:<12.2f} {avg_xgb_gap:.4f}")
    if ensemble_results:
        print(f"{'Ensemble (50/50)':<20} {avg_ens_test_r2:<12.4f} {avg_ens_test_dir*100:<12.1f}% {avg_ens_sharpe:<12.2f} {'N/A':<15}")
    
    # =========================================================================
    # Final Assessment
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL OVERFITTING ASSESSMENT")
    print("="*70)
    
    # Reality check on those 98%+ accuracy numbers
    if rf_results and xgb_results:
        # During optimization, accuracy was 98%+ 
        # Let's see what it really is on unseen data
        print(f"\n⚠️  CRITICAL CHECK:")
        print(f"   Optimization report claimed ~98% accuracy")
        print(f"   Actual unseen data accuracy:")
        print(f"     - RF:       {avg_rf_test_dir:.1%}")
        print(f"     - XGB:      {avg_xgb_test_dir:.1%}")
        print(f"     - Ensemble: {avg_ens_test_dir:.1%}")
        
        if max(avg_rf_test_dir, avg_xgb_test_dir) < 0.55:
            print(f"\n⚠️  WARNING: Direction accuracy is near random (50%)")
            print(f"   This suggests the optimization may have overfit to training data.")
            print(f"   Consider: fewer features, more regularization, longer holdout period.")
        elif max(avg_rf_test_dir, avg_xgb_test_dir) < 0.52:
            print(f"\n❌ SEVERE OVERFITTING DETECTED")
            print(f"   Models are essentially random on unseen data.")
        else:
            print(f"\n✅ Models show some predictive power on unseen data")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == "__main__":
    results = main()
