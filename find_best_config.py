"""
Find Best Model Configuration
Tests all combinations of models, filters, and settings to find optimal research parameters
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction_model import backtest_one_ticker, build_features_and_target

print("=" * 70)
print("🔬 FINDING BEST MODEL CONFIGURATION")
print("=" * 70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# Test Configuration
# =============================================================================

# Tickers to test (reliable ones based on your BASELINE notes)
TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

# Parameters to test
MODELS = ["xgb", "rf", "ensemble"]  # GBRT excluded per your notes (overfitting)
PERIODS = ["2y", "5y"]
HORIZONS = [1, 3, 5]

# Trading modes for adaptive model
TRADING_MODES = ["conservative", "balanced", "aggressive"]

# =============================================================================
# Test Functions
# =============================================================================

def test_model_config(ticker, model_type, period, horizon):
    """Test a single model configuration and return metrics."""
    try:
        result = backtest_one_ticker(
            ticker,
            model_type=model_type,
            period=period,
            horizon=horizon,
        )
        
        if result is None:
            return None
            
        # Extract key metrics
        return {
            "ticker": ticker,
            "model": model_type,
            "period": period,
            "horizon": horizon,
            "sharpe": result.get("sharpe_ratio", 0),
            "accuracy": result.get("accuracy", 0),
            "total_return": result.get("total_return", 0),
            "max_drawdown": result.get("max_drawdown", 0),
            "win_rate": result.get("win_rate", 0),
            "num_trades": result.get("num_trades", 0),
        }
    except Exception as e:
        print(f"  ⚠️ Error testing {ticker}/{model_type}/{period}/{horizon}d: {str(e)[:50]}")
        return None


def test_adaptive_model(ticker, mode):
    """Test adaptive model with different trading modes."""
    try:
        from src.core.production_predictor import quick_predict
        result = quick_predict(ticker, mode=mode)
        
        return {
            "ticker": ticker,
            "mode": mode,
            "signal": result.signal,
            "confidence": result.confidence,
            "predicted_return": result.predicted_return,
            "position_size": result.position_size,
        }
    except Exception as e:
        print(f"  ⚠️ Adaptive model error for {ticker}/{mode}: {str(e)[:50]}")
        return None


# =============================================================================
# Run Tests
# =============================================================================

print("📊 PHASE 1: Testing Model Types, Periods, and Horizons")
print("-" * 70)

all_results = []
total_tests = len(TEST_TICKERS) * len(MODELS) * len(PERIODS) * len(HORIZONS)
current_test = 0

for ticker in TEST_TICKERS:
    print(f"\n🔍 Testing {ticker}...")
    
    for model in MODELS:
        for period in PERIODS:
            for horizon in HORIZONS:
                current_test += 1
                print(f"  [{current_test}/{total_tests}] {model.upper()} | {period} | {horizon}d", end="")
                
                result = test_model_config(ticker, model, period, horizon)
                
                if result:
                    sharpe = result["sharpe"]
                    acc = result["accuracy"] * 100
                    print(f" → Sharpe: {sharpe:+.3f} | Acc: {acc:.1f}%")
                    all_results.append(result)
                else:
                    print(" → FAILED")

# Create results DataFrame
results_df = pd.DataFrame(all_results)

if results_df.empty:
    print("\n❌ No results collected. Check for errors above.")
    sys.exit(1)

print("\n" + "=" * 70)
print("📈 PHASE 1 RESULTS: Best Configurations by Sharpe Ratio")
print("=" * 70)

# Aggregate by configuration (average across tickers)
config_stats = results_df.groupby(["model", "period", "horizon"]).agg({
    "sharpe": ["mean", "std", "min", "max"],
    "accuracy": ["mean"],
    "win_rate": ["mean"],
    "total_return": ["mean"],
    "max_drawdown": ["mean"],
}).round(4)

config_stats.columns = ["_".join(col) for col in config_stats.columns]
config_stats = config_stats.sort_values("sharpe_mean", ascending=False)

print("\n🏆 TOP 10 CONFIGURATIONS (by Mean Sharpe):")
print("-" * 70)
print(f"{'Rank':<5} {'Model':<10} {'Period':<8} {'Horizon':<8} {'Sharpe':<12} {'Accuracy':<10} {'Win Rate':<10}")
print("-" * 70)

for i, (idx, row) in enumerate(config_stats.head(10).iterrows(), 1):
    model, period, horizon = idx
    print(f"{i:<5} {model.upper():<10} {period:<8} {horizon}d{'':<5} {row['sharpe_mean']:+.4f}±{row['sharpe_std']:.3f} {row['accuracy_mean']*100:.1f}%{'':<5} {row['win_rate_mean']*100:.1f}%")

print("\n" + "=" * 70)
print("📊 BEST CONFIG BY MODEL TYPE")
print("=" * 70)

for model in MODELS:
    model_df = results_df[results_df["model"] == model]
    if not model_df.empty:
        best = model_df.loc[model_df["sharpe"].idxmax()]
        avg_sharpe = model_df["sharpe"].mean()
        avg_acc = model_df["accuracy"].mean() * 100
        print(f"\n{model.upper()}:")
        print(f"  Best Config: {best['period']} period, {best['horizon']}d horizon")
        print(f"  Best Sharpe: {best['sharpe']:+.4f} (on {best['ticker']})")
        print(f"  Avg Sharpe:  {avg_sharpe:+.4f}")
        print(f"  Avg Accuracy: {avg_acc:.1f}%")

print("\n" + "=" * 70)
print("📊 BEST CONFIG BY TICKER")
print("=" * 70)

for ticker in TEST_TICKERS:
    ticker_df = results_df[results_df["ticker"] == ticker]
    if not ticker_df.empty:
        best = ticker_df.loc[ticker_df["sharpe"].idxmax()]
        print(f"\n{ticker}:")
        print(f"  Best: {best['model'].upper()} | {best['period']} | {best['horizon']}d")
        print(f"  Sharpe: {best['sharpe']:+.4f} | Accuracy: {best['accuracy']*100:.1f}% | Win Rate: {best['win_rate']*100:.1f}%")

# =============================================================================
# Phase 2: Test Adaptive Model Modes
# =============================================================================

print("\n" + "=" * 70)
print("🧠 PHASE 2: Testing Adaptive Model Trading Modes")
print("=" * 70)

adaptive_results = []

for ticker in TEST_TICKERS:
    print(f"\n🔍 {ticker}:", end=" ")
    for mode in TRADING_MODES:
        result = test_adaptive_model(ticker, mode)
        if result:
            adaptive_results.append(result)
            sig = result["signal"]
            conf = result["confidence"] * 100
            ret = result["predicted_return"] * 100
            print(f"{mode[:4]}={sig}({conf:.0f}%,{ret:+.1f}%)", end=" ")
    print()

adaptive_df = pd.DataFrame(adaptive_results)

if not adaptive_df.empty:
    print("\n📊 ADAPTIVE MODEL SUMMARY BY MODE:")
    print("-" * 70)
    
    for mode in TRADING_MODES:
        mode_df = adaptive_df[adaptive_df["mode"] == mode]
        if not mode_df.empty:
            buys = (mode_df["signal"] == "BUY").sum()
            sells = (mode_df["signal"] == "SELL").sum()
            holds = (mode_df["signal"] == "HOLD").sum()
            avg_conf = mode_df["confidence"].mean() * 100
            avg_ret = mode_df["predicted_return"].mean() * 100
            
            print(f"\n{mode.upper()}:")
            print(f"  Signals: {buys} BUY | {sells} SELL | {holds} HOLD")
            print(f"  Avg Confidence: {avg_conf:.1f}%")
            print(f"  Avg Predicted Return: {avg_ret:+.2f}%")

# =============================================================================
# Final Recommendations
# =============================================================================

print("\n" + "=" * 70)
print("🏆 FINAL RECOMMENDATIONS")
print("=" * 70)

# Get overall best config
if not results_df.empty:
    # Best by mean Sharpe across all tickers
    best_config = config_stats["sharpe_mean"].idxmax()
    best_sharpe = config_stats.loc[best_config, "sharpe_mean"]
    best_acc = config_stats.loc[best_config, "accuracy_mean"] * 100
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTIMAL CONFIGURATION FOR RESEARCH                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  MODEL:      {best_config[0].upper():<54} ║
║  PERIOD:     {best_config[1]:<54} ║
║  HORIZON:    {best_config[2]}d{'':<53} ║
║                                                                      ║
║  Mean Sharpe:   {best_sharpe:+.4f}{'':<49} ║
║  Mean Accuracy: {best_acc:.1f}%{'':<51} ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Additional recommendations
    print("\n📝 ADDITIONAL INSIGHTS:")
    print("-" * 70)
    
    # Best model overall
    model_means = results_df.groupby("model")["sharpe"].mean().sort_values(ascending=False)
    print(f"\n1. BEST MODEL OVERALL: {model_means.index[0].upper()}")
    print(f"   Mean Sharpe across all tests: {model_means.iloc[0]:+.4f}")
    
    # Best period
    period_means = results_df.groupby("period")["sharpe"].mean().sort_values(ascending=False)
    print(f"\n2. BEST PERIOD: {period_means.index[0]}")
    print(f"   Mean Sharpe: {period_means.iloc[0]:+.4f}")
    
    # Best horizon
    horizon_means = results_df.groupby("horizon")["sharpe"].mean().sort_values(ascending=False)
    print(f"\n3. BEST HORIZON: {horizon_means.index[0]}d")
    print(f"   Mean Sharpe: {horizon_means.iloc[0]:+.4f}")
    
    # Stability analysis
    print("\n4. STABILITY ANALYSIS:")
    stable_configs = config_stats[config_stats["sharpe_std"] < 0.5].sort_values("sharpe_mean", ascending=False)
    if not stable_configs.empty:
        best_stable = stable_configs.index[0]
        print(f"   Most Stable High-Performer: {best_stable[0].upper()} | {best_stable[1]} | {best_stable[2]}d")
        print(f"   Sharpe: {stable_configs.iloc[0]['sharpe_mean']:+.4f} ± {stable_configs.iloc[0]['sharpe_std']:.3f}")
    
    # Best for each ticker
    print("\n5. TICKER-SPECIFIC RECOMMENDATIONS:")
    for ticker in TEST_TICKERS:
        ticker_df = results_df[results_df["ticker"] == ticker]
        if not ticker_df.empty:
            best = ticker_df.loc[ticker_df["sharpe"].idxmax()]
            print(f"   {ticker}: {best['model'].upper()} | {best['period']} | {best['horizon']}d (Sharpe: {best['sharpe']:+.3f})")

# Save results
results_df.to_csv("model_optimization_results.csv", index=False)
print(f"\n✅ Detailed results saved to: model_optimization_results.csv")

print("\n" + "=" * 70)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
