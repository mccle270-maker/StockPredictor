#!/usr/bin/env python3
"""
Comprehensive LSTM+XGB Ensemble Test

Tests different weight combinations across:
- Multiple tickers (10+)
- Longer periods (3-5 years)
- Different market regimes (bullish 2021, bearish 2022, recovery 2023-2024)
- Multiple weight configurations
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

print("="*80)
print("🔬 COMPREHENSIVE LSTM+XGB ENSEMBLE TEST")
print("="*80)
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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Diverse ticker universe (tech, finance, healthcare, consumer, energy)
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",  # Tech
    "JPM", "BAC", "GS",                        # Finance
    "JNJ", "PFE",                              # Healthcare
    "WMT", "COST",                             # Consumer
    "XOM", "CVX",                              # Energy
]

PERIOD = "5y"  # 5 years of data for robust testing
HORIZON = 5    # 5-day prediction horizon

# Test periods for market regime analysis
# We'll split the data into different periods
MARKET_REGIMES = {
    "bull_2021": ("2021-01-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2022-12-31"),
    "recovery_2023": ("2023-01-01", "2023-12-31"),
    "current_2024": ("2024-01-01", "2024-12-31"),
}

# Weight configurations to test
WEIGHT_CONFIGS = {
    "lstm_only": {"lstm_weight": 1.0, "xgb_weight": 0.0},
    "lstm_heavy_70_30": {"lstm_weight": 0.7, "xgb_weight": 0.3},
    "equal_50_50": {"lstm_weight": 0.5, "xgb_weight": 0.5},
    "xgb_heavy_30_70": {"lstm_weight": 0.3, "xgb_weight": 0.7},
    "xgb_only": {"lstm_weight": 0.0, "xgb_weight": 1.0},
}

# Model configurations
MODEL_CONFIGS = {
    "rf": {"model_type": "rf", "use_optimized": True},
    "xgb": {"model_type": "xgb", "use_optimized": True},
    "lstm_small": {"model_type": "lstm", "lookback": 30, "lstm_units": 32, "epochs": 50, "verbose": 0},
}

# Add ensemble configs dynamically
for name, weights in WEIGHT_CONFIGS.items():
    if weights["lstm_weight"] > 0 and weights["xgb_weight"] > 0:
        MODEL_CONFIGS[f"ensemble_{name}"] = {
            "model_type": "lstm_xgb",
            "lstm_weight": weights["lstm_weight"],
            "xgb_weight": weights["xgb_weight"],
            "lstm_config": {"lookback": 30, "lstm_units": 32, "epochs": 50, "verbose": 0}
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


def compute_max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from returns."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min() * 100  # Return as percentage


def compute_win_rate(returns: pd.Series) -> float:
    """Compute win rate (% of positive returns)."""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns) * 100


def test_model_on_period(
    ticker: str, 
    model_name: str, 
    config: Dict,
    start_date: str = None,
    end_date: str = None,
    test_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Test a model on a specific time period.
    
    Args:
        ticker: Stock ticker
        model_name: Name of model config
        config: Model configuration dict
        start_date: Optional start date filter
        end_date: Optional end date filter  
        test_ratio: Fraction of data to use for testing
    """
    result = {
        "ticker": ticker,
        "model": model_name,
        "accuracy": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "status": "failed",
        "error": None,
        "train_samples": 0,
        "test_samples": 0,
        "period": f"{start_date} to {end_date}" if start_date else "full",
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
        
        # Unpack results
        X_arr, y_arr, _, _, _, _, dates = feat_result
        
        if X_arr is None or y_arr is None or len(X_arr) < 100:
            result["error"] = f"Insufficient data: {len(X_arr) if X_arr is not None else 0} samples"
            return result
        
        # Convert to DataFrames
        X = pd.DataFrame(X_arr, index=dates)
        y = pd.Series(y_arr, index=dates)
        
        # Filter by date range if specified
        if start_date:
            mask = X.index >= pd.Timestamp(start_date)
            if end_date:
                mask &= X.index <= pd.Timestamp(end_date)
            X = X[mask]
            y = y[mask]
        
        if len(X) < 50:
            result["error"] = f"Insufficient data after date filter: {len(X)}"
            return result
        
        # Split into train/test
        test_size = int(len(X) * test_ratio)
        train_end = len(X) - test_size
        
        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]
        
        result["train_samples"] = len(X_train)
        result["test_samples"] = len(X_test)
        
        if len(X_train) < 30 or len(X_test) < 10:
            result["error"] = f"Not enough samples: train={len(X_train)}, test={len(X_test)}"
            return result
        
        # Create and train model
        model = make_model(**config)
        
        model_type = config.get("model_type", "")
        
        if model_type in ("lstm", "lstm_xgb"):
            # LSTM-based models need DataFrame
            model.fit(X_train, y_train)
            
            # Get lookback
            if model_type == "lstm":
                lookback = config.get("lookback", 60)
            else:
                lookback = config.get("lstm_config", {}).get("lookback", 30)
            
            predictions = []
            for i in range(len(X_test)):
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
        strategy_returns = pd.Series(np.sign(y_pred) * y_test.values, index=y_test.index)
        
        result["sharpe"] = compute_sharpe(strategy_returns)
        result["max_drawdown"] = compute_max_drawdown(strategy_returns)
        result["win_rate"] = compute_win_rate(strategy_returns)
        
        result["status"] = "success"
        
    except Exception as e:
        result["error"] = str(e)[:100]
    
    return result


def run_comprehensive_test():
    """Run the full comprehensive test suite."""
    
    all_results = []
    
    # ========================================================================
    # TEST 1: Full Period Test (All Models, All Tickers)
    # ========================================================================
    print("\n" + "="*80)
    print("📊 TEST 1: FULL PERIOD COMPARISON (5 Years)")
    print("="*80)
    print(f"Tickers: {len(TICKERS)} | Models: {len(MODEL_CONFIGS)}")
    print("-"*80)
    
    total_tests = len(TICKERS) * len(MODEL_CONFIGS)
    test_count = 0
    
    for ticker in TICKERS:
        print(f"\n🎯 {ticker}")
        for model_name, config in MODEL_CONFIGS.items():
            test_count += 1
            print(f"  [{test_count}/{total_tests}] Testing {model_name}...", end=" ", flush=True)
            
            result = test_model_on_period(ticker, model_name, config, test_ratio=0.15)
            result["test_type"] = "full_period"
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"✅ Acc: {result['accuracy']:.1f}% | Sharpe: {result['sharpe']:+.3f}")
            else:
                print(f"❌ {result['error'][:40]}")
    
    # ========================================================================
    # TEST 2: Market Regime Tests
    # ========================================================================
    print("\n" + "="*80)
    print("📈📉 TEST 2: MARKET REGIME ANALYSIS")
    print("="*80)
    
    # Use subset of tickers for regime testing
    regime_tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    
    # Focus on key ensemble configurations
    regime_models = {
        "xgb": MODEL_CONFIGS["xgb"],
        "lstm_small": MODEL_CONFIGS["lstm_small"],
        "ensemble_equal_50_50": MODEL_CONFIGS.get("ensemble_equal_50_50"),
        "ensemble_lstm_heavy_70_30": MODEL_CONFIGS.get("ensemble_lstm_heavy_70_30"),
        "ensemble_xgb_heavy_30_70": MODEL_CONFIGS.get("ensemble_xgb_heavy_30_70"),
    }
    regime_models = {k: v for k, v in regime_models.items() if v is not None}
    
    for regime_name, (start, end) in MARKET_REGIMES.items():
        print(f"\n📅 {regime_name.upper()} ({start} to {end})")
        print("-"*60)
        
        for ticker in regime_tickers:
            print(f"  {ticker}: ", end="")
            best_model = None
            best_sharpe = -999
            
            for model_name, config in regime_models.items():
                result = test_model_on_period(
                    ticker, model_name, config,
                    start_date=start, end_date=end,
                    test_ratio=0.25  # 25% test for shorter periods
                )
                result["test_type"] = f"regime_{regime_name}"
                all_results.append(result)
                
                if result["status"] == "success" and result["sharpe"] > best_sharpe:
                    best_sharpe = result["sharpe"]
                    best_model = model_name
            
            if best_model:
                print(f"Best: {best_model} (Sharpe: {best_sharpe:+.3f})")
            else:
                print("No valid results")
    
    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    df_success = df[df["status"] == "success"].copy()
    
    if df_success.empty:
        print("❌ No successful tests!")
        return all_results
    
    # Overall model comparison
    print("\n📈 OVERALL MODEL PERFORMANCE (All Tests)")
    print("-"*70)
    
    model_summary = df_success.groupby("model").agg({
        "accuracy": ["mean", "std"],
        "sharpe": ["mean", "std", "min", "max"],
        "max_drawdown": "mean",
        "win_rate": "mean",
    }).round(3)
    
    # Flatten column names
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns.values]
    model_summary = model_summary.sort_values("sharpe_mean", ascending=False)
    
    print(f"{'Model':<30} {'Acc%':>8} {'Sharpe':>10} {'Sharpe Std':>12} {'MaxDD%':>10} {'WinRate%':>10}")
    print("-"*80)
    for idx, row in model_summary.iterrows():
        print(f"{idx:<30} {row['accuracy_mean']:>7.1f}% {row['sharpe_mean']:>+10.3f} {row['sharpe_std']:>12.3f} {row['max_drawdown_mean']:>9.1f}% {row['win_rate_mean']:>9.1f}%")
    
    # Best model overall
    best_model = model_summary.index[0]
    best_sharpe = model_summary.iloc[0]["sharpe_mean"]
    print(f"\n🏆 BEST OVERALL MODEL: {best_model} (Avg Sharpe: {best_sharpe:+.3f})")
    
    # Full period only comparison
    print("\n" + "-"*70)
    print("📊 FULL PERIOD TEST RESULTS ONLY")
    print("-"*70)
    
    df_full = df_success[df_success["test_type"] == "full_period"]
    if not df_full.empty:
        full_summary = df_full.groupby("model").agg({
            "accuracy": "mean",
            "sharpe": ["mean", "std"],
            "max_drawdown": "mean",
            "win_rate": "mean",
        }).round(3)
        full_summary.columns = ['_'.join(col).strip() for col in full_summary.columns.values]
        full_summary = full_summary.sort_values("sharpe_mean", ascending=False)
        
        print(f"{'Model':<30} {'Acc%':>8} {'Sharpe':>10} {'Sharpe Std':>12} {'MaxDD%':>10} {'WinRate%':>10}")
        print("-"*80)
        for idx, row in full_summary.iterrows():
            print(f"{idx:<30} {row['accuracy_mean']:>7.1f}% {row['sharpe_mean']:>+10.3f} {row['sharpe_std']:>12.3f} {row['max_drawdown_mean']:>9.1f}% {row['win_rate_mean']:>9.1f}%")
    
    # Market regime analysis
    print("\n" + "-"*70)
    print("📈📉 PERFORMANCE BY MARKET REGIME")
    print("-"*70)
    
    for regime_name in MARKET_REGIMES.keys():
        df_regime = df_success[df_success["test_type"] == f"regime_{regime_name}"]
        if not df_regime.empty:
            regime_avg = df_regime.groupby("model")["sharpe"].mean().sort_values(ascending=False)
            top_model = regime_avg.index[0]
            top_sharpe = regime_avg.iloc[0]
            print(f"  {regime_name.upper():<20} Best: {top_model:<30} Sharpe: {top_sharpe:+.3f}")
    
    # Per-ticker analysis
    print("\n" + "-"*70)
    print("🎯 BEST MODEL PER TICKER (Full Period)")
    print("-"*70)
    
    if not df_full.empty:
        for ticker in TICKERS:
            ticker_df = df_full[df_full["ticker"] == ticker]
            if not ticker_df.empty:
                best_idx = ticker_df["sharpe"].idxmax()
                best = ticker_df.loc[best_idx]
                print(f"  {ticker:<8} → {best['model']:<30} Sharpe: {best['sharpe']:+.3f} | Acc: {best['accuracy']:.1f}%")
    
    # Weight configuration analysis (for ensembles)
    print("\n" + "-"*70)
    print("⚖️ ENSEMBLE WEIGHT ANALYSIS")
    print("-"*70)
    
    ensemble_models = [m for m in df_success["model"].unique() if "ensemble" in m]
    if ensemble_models:
        ensemble_df = df_success[df_success["model"].isin(ensemble_models)]
        ensemble_avg = ensemble_df.groupby("model").agg({
            "sharpe": ["mean", "std"],
            "accuracy": "mean",
            "max_drawdown": "mean",
        }).round(3)
        ensemble_avg.columns = ['_'.join(col).strip() for col in ensemble_avg.columns.values]
        ensemble_avg = ensemble_avg.sort_values("sharpe_mean", ascending=False)
        
        print(f"{'Weight Config':<35} {'Sharpe':>10} {'Std':>10} {'Acc%':>8} {'MaxDD%':>10}")
        print("-"*75)
        for idx, row in ensemble_avg.iterrows():
            print(f"{idx:<35} {row['sharpe_mean']:>+10.3f} {row['sharpe_std']:>10.3f} {row['accuracy_mean']:>7.1f}% {row['max_drawdown_mean']:>9.1f}%")
        
        best_ensemble = ensemble_avg.index[0]
        print(f"\n🏆 BEST ENSEMBLE CONFIG: {best_ensemble}")
    
    # Stability analysis
    print("\n" + "-"*70)
    print("📉 STABILITY ANALYSIS (Lower Std = More Stable)")
    print("-"*70)
    
    stability = df_success.groupby("model")["sharpe"].std().sort_values()
    for model, std in stability.items():
        print(f"  {model:<35} Sharpe Std: {std:.3f}")
    
    # Save results
    results_file = f"lstm_xgb_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(results_file, index=False)
    print(f"\n📁 Results saved to: {results_file}")
    
    print("\n" + "="*80)
    print(f"✅ COMPREHENSIVE TEST COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    run_comprehensive_test()
