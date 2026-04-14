#!/usr/bin/env python3
"""
LSTM Model Comparison Test

Tests the new LSTM model against existing models (RF, XGB, Ensemble)
and compares accuracy and Sharpe ratios.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("🧠 LSTM MODEL COMPARISON TEST")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Test TensorFlow import
print("📦 Checking dependencies...")
try:
    import tensorflow as tf
    print(f"  ✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"  ❌ TensorFlow not available: {e}")
    sys.exit(1)

try:
    from src.core.lstm_model import LSTMPredictor, LSTMWrapper, create_lstm_model, HAS_TF
    print(f"  ✅ LSTM model module loaded (HAS_TF={HAS_TF})")
except ImportError as e:
    print(f"  ❌ LSTM model import failed: {e}")
    sys.exit(1)

try:
    from src.core.models import make_model
    print("  ✅ make_model imported")
except ImportError as e:
    print(f"  ❌ make_model import failed: {e}")
    sys.exit(1)

try:
    from prediction_model import build_features_and_target
    print("  ✅ prediction_model imported")
except ImportError as e:
    print(f"  ❌ prediction_model import failed: {e}")
    sys.exit(1)

print()

# Configuration
TICKERS = ["AAPL", "MSFT", "NVDA"]
PERIOD = "2y"
HORIZON = 5
TEST_DAYS = 60  # Last 60 days for testing

# Model configurations to test
MODEL_CONFIGS = {
    "rf": {"model_type": "rf", "use_optimized": True},
    "xgb": {"model_type": "xgb", "use_optimized": True},
    "lstm_small": {"model_type": "lstm", "lookback": 30, "lstm_units": 32, "epochs": 50, "verbose": 0},
    "lstm_xgb": {"model_type": "lstm_xgb", "lstm_weight": 0.5, "xgb_weight": 0.5, 
                 "lstm_config": {"lookback": 30, "lstm_units": 32, "epochs": 50, "verbose": 0}},
}


def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """Compute Sharpe ratio from returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(252)
    return sr


def compute_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute directional accuracy."""
    correct = ((actual > 0) == (predicted > 0)).sum()
    return correct / len(actual) * 100 if len(actual) > 0 else 0.0


def test_model(ticker: str, model_name: str, config: Dict) -> Dict[str, Any]:
    """Test a single model configuration on a ticker."""
    result = {
        "ticker": ticker,
        "model": model_name,
        "accuracy": 0.0,
        "sharpe": 0.0,
        "status": "failed",
        "error": None,
        "train_samples": 0,
        "test_samples": 0,
    }
    
    try:
        # Build features and target
        feat_result = build_features_and_target(
            ticker, 
            period=PERIOD, 
            horizon=HORIZON,
        )
        
        if feat_result is None:
            result["error"] = "No data returned"
            return result
        
        # build_features_and_target returns 7 values:
        # X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates
        X_arr, y_arr, _, _, _, _, dates = feat_result
        
        if X_arr is None or y_arr is None or len(X_arr) < 100:
            result["error"] = f"Insufficient data: {len(X_arr) if X_arr is not None else 0} samples"
            return result
        
        # Convert to DataFrames
        X = pd.DataFrame(X_arr, index=dates)
        y = pd.Series(y_arr, index=dates)
        
        # Split: train on all but last TEST_DAYS, test on last TEST_DAYS
        train_end = len(X) - TEST_DAYS
        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]
        
        result["train_samples"] = len(X_train)
        result["test_samples"] = len(X_test)
        
        # Create and train model
        model = make_model(**config)
        
        if config["model_type"] in ("lstm", "lstm_xgb"):
            # LSTM and LSTM+XGB need DataFrame
            model.fit(X_train, y_train)
            
            # Predict - LSTM predicts one at a time with lookback
            if config["model_type"] == "lstm":
                lookback = config.get("lookback", 60)
            else:
                lookback = config.get("lstm_config", {}).get("lookback", 30)
            predictions = []
            
            for i in range(len(X_test)):
                # Use lookback window ending at current position
                start_idx = max(0, train_end + i - lookback)
                end_idx = train_end + i
                window = X.iloc[start_idx:end_idx]
                
                pred = model.predict(window)
                predictions.append(pred[0] if len(pred) > 0 else 0.0)
            
            y_pred = np.array(predictions)
        else:
            # Traditional sklearn models
            model.fit(X_train.values, y_train.values)
            y_pred = model.predict(X_test.values)
        
        # Compute metrics
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        
        # Accuracy (directional)
        result["accuracy"] = compute_accuracy(y_test, y_pred_series)
        
        # Strategy returns (sign of prediction * actual return)
        strategy_returns = np.sign(y_pred) * y_test.values
        result["sharpe"] = compute_sharpe(pd.Series(strategy_returns))
        
        result["status"] = "success"
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


def main():
    """Run all tests and display results."""
    print("="*70)
    print("📊 RUNNING MODEL COMPARISONS")
    print("="*70)
    print(f"Tickers: {TICKERS}")
    print(f"Period: {PERIOD}, Horizon: {HORIZON}D, Test Days: {TEST_DAYS}")
    print()
    
    all_results = []
    
    for ticker in TICKERS:
        print(f"\n🎯 Testing {ticker}")
        print("-" * 50)
        
        for model_name, config in MODEL_CONFIGS.items():
            print(f"  Testing {model_name}...", end=" ", flush=True)
            
            result = test_model(ticker, model_name, config)
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"✅ Acc: {result['accuracy']:.1f}% | Sharpe: {result['sharpe']:+.3f}")
            else:
                print(f"❌ {result['error']}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    # Success rate
    success_rate = (df["status"] == "success").mean() * 100
    print(f"\nSuccess Rate: {success_rate:.0f}%")
    
    # Filter successful results
    success_df = df[df["status"] == "success"].copy()
    
    if not success_df.empty:
        # Average by model
        print("\n📈 AVERAGE PERFORMANCE BY MODEL:")
        print("-" * 50)
        model_summary = success_df.groupby("model").agg({
            "accuracy": "mean",
            "sharpe": "mean",
            "train_samples": "mean",
            "test_samples": "mean",
        }).round(2)
        
        model_summary = model_summary.sort_values("sharpe", ascending=False)
        
        print(f"{'Model':<15} {'Accuracy':>10} {'Sharpe':>10} {'Train':>8} {'Test':>6}")
        print("-" * 50)
        for model_name, row in model_summary.iterrows():
            print(f"{model_name:<15} {row['accuracy']:>9.1f}% {row['sharpe']:>+9.3f} {row['train_samples']:>8.0f} {row['test_samples']:>6.0f}")
        
        # Best model
        best_model = model_summary["sharpe"].idxmax()
        best_sharpe = model_summary.loc[best_model, "sharpe"]
        best_acc = model_summary.loc[best_model, "accuracy"]
        
        print("\n" + "="*70)
        print(f"🏆 BEST MODEL: {best_model.upper()}")
        print(f"   Sharpe: {best_sharpe:+.3f} | Accuracy: {best_acc:.1f}%")
        print("="*70)
        
        # Detailed results table
        print("\n📋 DETAILED RESULTS:")
        print("-" * 70)
        print(f"{'Ticker':<8} {'Model':<15} {'Status':<10} {'Accuracy':>10} {'Sharpe':>10}")
        print("-" * 70)
        for _, row in df.iterrows():
            status_icon = "✅" if row["status"] == "success" else "❌"
            acc = f"{row['accuracy']:.1f}%" if row["status"] == "success" else "—"
            sharpe = f"{row['sharpe']:+.3f}" if row["status"] == "success" else "—"
            print(f"{row['ticker']:<8} {row['model']:<15} {status_icon:<10} {acc:>10} {sharpe:>10}")
    
    print(f"\n✅ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df


if __name__ == "__main__":
    results = main()
