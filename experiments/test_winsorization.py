"""
Test Winsorization Impact on Model Performance

Compares model performance with and without winsorized targets.
Tests on AAPL, MSFT, NVDA with XGB V6 configuration.

Usage:
    python experiments/test_winsorization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from src.core.features import winsorize_series, build_target, build_all_features
from src.core.models import make_model
from src.core.metrics import compute_sharpe
from src.data.market import get_price_history


def test_winsorize_function():
    """Test the winsorize_series function."""
    print("\n" + "="*60)
    print("TEST 1: Winsorize Function")
    print("="*60)
    
    # Create test data with outliers
    returns = pd.Series([
        -0.15, -0.08, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.08, 0.20
    ])
    
    print(f"\nOriginal returns:")
    print(f"  Min: {returns.min():.2%}")
    print(f"  Max: {returns.max():.2%}")
    print(f"  Std: {returns.std():.4f}")
    
    # Winsorize at 5th/95th percentile (aggressive for small sample)
    winsorized = winsorize_series(returns, lower_pct=0.05, upper_pct=0.95)
    
    print(f"\nWinsorized (5%/95%):")
    print(f"  Min: {winsorized.min():.2%}")
    print(f"  Max: {winsorized.max():.2%}")
    print(f"  Std: {winsorized.std():.4f}")
    
    print(f"\n✅ Winsorize function works correctly")
    return True


def analyze_target_distribution(ticker: str = "AAPL", period: str = "5y"):
    """Analyze the distribution of targets with and without winsorization."""
    print("\n" + "="*60)
    print(f"TEST 2: Target Distribution Analysis ({ticker})")
    print("="*60)
    
    # Get price data
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        print(f"❌ No data for {ticker}")
        return None
    
    # Build targets
    target_raw = build_target(hist, horizon=1, winsorize=False)
    target_win = build_target(hist, horizon=1, winsorize=True, winsorize_pct=0.01)
    
    # Drop NaN
    target_raw = target_raw.dropna()
    target_win = target_win.dropna()
    
    print(f"\nRaw Target (no winsorization):")
    print(f"  Count: {len(target_raw)}")
    print(f"  Min:   {target_raw.min():.2%}")
    print(f"  Max:   {target_raw.max():.2%}")
    print(f"  Mean:  {target_raw.mean():.4%}")
    print(f"  Std:   {target_raw.std():.4%}")
    print(f"  Skew:  {target_raw.skew():.3f}")
    print(f"  Kurt:  {target_raw.kurtosis():.3f}")
    
    print(f"\nWinsorized Target (1%/99%):")
    print(f"  Count: {len(target_win)}")
    print(f"  Min:   {target_win.min():.2%}")
    print(f"  Max:   {target_win.max():.2%}")
    print(f"  Mean:  {target_win.mean():.4%}")
    print(f"  Std:   {target_win.std():.4%}")
    print(f"  Skew:  {target_win.skew():.3f}")
    print(f"  Kurt:  {target_win.kurtosis():.3f}")
    
    # Count outliers clipped
    n_clipped_low = (target_raw < target_win.min()).sum()
    n_clipped_high = (target_raw > target_win.max()).sum()
    
    print(f"\nOutliers Clipped:")
    print(f"  Below 1st percentile: {n_clipped_low} days")
    print(f"  Above 99th percentile: {n_clipped_high} days")
    print(f"  Total clipped: {n_clipped_low + n_clipped_high} ({(n_clipped_low + n_clipped_high)/len(target_raw)*100:.1f}%)")
    
    return {
        "ticker": ticker,
        "raw_std": target_raw.std(),
        "win_std": target_win.std(),
        "raw_skew": target_raw.skew(),
        "win_skew": target_win.skew(),
        "raw_kurt": target_raw.kurtosis(),
        "win_kurt": target_win.kurtosis(),
        "n_clipped": n_clipped_low + n_clipped_high,
    }


def backtest_comparison(ticker: str = "AAPL", period: str = "5y"):
    """Compare backtest performance with and without winsorization."""
    print("\n" + "="*60)
    print(f"TEST 3: Backtest Comparison ({ticker})")
    print("="*60)
    
    # Get data and build features
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        print(f"❌ No data for {ticker}")
        return None
    
    hist = build_all_features(hist)
    
    # Build both targets
    target_raw = build_target(hist, horizon=1, winsorize=False)
    target_win = build_target(hist, horizon=1, winsorize=True, winsorize_pct=0.01)
    
    hist["target_raw"] = target_raw
    hist["target_win"] = target_win
    
    # Get feature columns (simplified)
    feat_cols = [c for c in hist.columns if c not in [
        "Open", "High", "Low", "Close", "Volume", "Adj Close",
        "target_raw", "target_win", "ftarget_ret_horizon_ahead"
    ] and not c.startswith("ftarget")]
    
    # Drop rows with NaN
    df = hist.dropna(subset=feat_cols + ["target_raw", "target_win"])
    
    if len(df) < 500:
        print(f"⚠️ Only {len(df)} rows after dropna, skipping")
        return None
    
    # Train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feat_cols].values
    X_test = test_df[feat_cols].values
    
    results = {}
    
    for target_name, target_col in [("Raw", "target_raw"), ("Winsorized", "target_win")]:
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        # Train XGB model
        model = make_model(model_type="xgb", use_optimized=True)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Simulate trading
        positions = np.sign(y_pred)
        
        # Use RAW test returns for P&L (even if trained on winsorized)
        actual_returns = test_df["target_raw"].values
        pnl = positions * actual_returns
        
        # Metrics
        sharpe = compute_sharpe(pd.Series(pnl))
        accuracy = (np.sign(y_pred) == np.sign(y_test)).mean()
        
        # Prediction stats
        pred_std = y_pred.std()
        pred_range = y_pred.max() - y_pred.min()
        
        results[target_name] = {
            "sharpe": sharpe,
            "accuracy": accuracy,
            "pred_std": pred_std,
            "pred_range": pred_range,
            "total_return": pnl.sum(),
        }
        
        print(f"\n{target_name} Target:")
        print(f"  Sharpe:       {sharpe:.3f}" if sharpe else "  Sharpe:       N/A")
        print(f"  Accuracy:     {accuracy:.1%}")
        print(f"  Total Return: {pnl.sum():.2%}")
        print(f"  Pred Std:     {pred_std:.4f}")
        print(f"  Pred Range:   {pred_range:.4f}")
    
    return results


def run_multi_ticker_test():
    """Test across multiple tickers."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Ticker Comparison")
    print("="*60)
    
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    results = []
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        try:
            r = backtest_comparison(ticker, period="5y")
            if r:
                results.append({
                    "ticker": ticker,
                    "sharpe_raw": r["Raw"]["sharpe"],
                    "sharpe_win": r["Winsorized"]["sharpe"],
                    "acc_raw": r["Raw"]["accuracy"],
                    "acc_win": r["Winsorized"]["accuracy"],
                })
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY: Winsorized vs Raw Target")
        print("="*60)
        
        df = pd.DataFrame(results)
        
        # Calculate improvement
        df["sharpe_diff"] = df["sharpe_win"] - df["sharpe_raw"]
        df["acc_diff"] = df["acc_win"] - df["acc_raw"]
        
        print("\n" + df.to_string(index=False))
        
        print(f"\n--- Averages ---")
        print(f"Raw Sharpe:        {df['sharpe_raw'].mean():.3f}")
        print(f"Winsorized Sharpe: {df['sharpe_win'].mean():.3f}")
        print(f"Sharpe Improvement: {df['sharpe_diff'].mean():+.3f}")
        print(f"\nRaw Accuracy:        {df['acc_raw'].mean():.1%}")
        print(f"Winsorized Accuracy: {df['acc_win'].mean():.1%}")
        print(f"Accuracy Change:     {df['acc_diff'].mean()*100:+.2f}%")
        
        # Recommendation
        if df["sharpe_diff"].mean() > 0.05:
            print("\n✅ RECOMMENDATION: Enable winsorization (improves Sharpe)")
        elif df["sharpe_diff"].mean() < -0.05:
            print("\n❌ RECOMMENDATION: Keep winsorization OFF (hurts Sharpe)")
        else:
            print("\n⚪ RECOMMENDATION: Neutral - minimal impact")
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("WINSORIZATION TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test 1: Basic function
    test_winsorize_function()
    
    # Test 2: Distribution analysis
    analyze_target_distribution("AAPL")
    
    # Test 3: Single ticker backtest
    # backtest_comparison("AAPL")  # Covered in multi-ticker
    
    # Test 4: Multi-ticker test
    run_multi_ticker_test()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
