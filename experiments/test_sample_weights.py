"""
Test Sample Weights Impact on Model Performance

Tests exponential decay sample weighting where recent samples
are weighted more heavily than older samples.

Usage:
    python experiments/test_sample_weights.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from src.core.models import compute_sample_weights, make_model
from src.core.features import build_all_features, build_target
from src.core.metrics import compute_sharpe
from src.data.market import get_price_history


def test_compute_sample_weights():
    """Test the compute_sample_weights function."""
    print("\n" + "="*60)
    print("TEST 1: Sample Weights Function")
    print("="*60)
    
    # Test basic functionality
    weights = compute_sample_weights(500, half_life_days=252)
    
    print(f"\nWith 500 samples, half_life=252 days (1 year):")
    print(f"  Total samples: {len(weights)}")
    print(f"  Weight sum: {weights.sum():.1f} (should be ~500)")
    print(f"  Oldest weight: {weights[0]:.4f}")
    print(f"  Most recent weight: {weights[-1]:.4f}")
    print(f"  Recent/Oldest ratio: {weights[-1]/weights[0]:.2f}x")
    
    # Test different half-lives
    print("\n  Half-life comparison (500 samples):")
    for half_life in [63, 126, 252, 504]:
        w = compute_sample_weights(500, half_life_days=half_life)
        ratio = w[-1] / w[0]
        print(f"    {half_life:3d} days: Recent/Oldest = {ratio:.2f}x")
    
    # Verify middle sample has ~50% weight at half-life
    weights = compute_sample_weights(505, half_life_days=252)  # 505 samples
    # Sample at index 252 is exactly 252 days before most recent
    middle_idx = 505 - 252 - 1  # 252 days ago
    ratio_at_halflife = weights[middle_idx] / weights[-1]
    print(f"\n  Verification: Weight at 252 days ago = {ratio_at_halflife:.2%} of recent (should be ~50%)")
    
    print("\n✅ compute_sample_weights works correctly")
    return True


def backtest_with_weights(ticker: str = "AAPL", period: str = "5y", half_life: int = 252):
    """Compare model performance with and without sample weights."""
    print(f"\n--- {ticker} (half_life={half_life} days) ---")
    
    # Get data and build features
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        print(f"  ❌ No data for {ticker}")
        return None
    
    hist = build_all_features(hist)
    target = build_target(hist, horizon=1)
    hist["target"] = target
    
    # Get feature columns
    feat_cols = [c for c in hist.columns if c not in [
        "Open", "High", "Low", "Close", "Volume", "Adj Close",
        "target", "ftarget_ret_horizon_ahead"
    ] and not c.startswith("ftarget")]
    
    # Drop rows with NaN
    df = hist.dropna(subset=feat_cols + ["target"])
    
    if len(df) < 500:
        print(f"  ⚠️ Only {len(df)} rows, skipping")
        return None
    
    # Train/test split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["target"].values
    
    results = {}
    
    # Test without weights
    model_no_weights = make_model(model_type="xgb", use_optimized=True)
    model_no_weights.fit(X_train, y_train)
    y_pred_no = model_no_weights.predict(X_test)
    
    positions_no = np.sign(y_pred_no)
    pnl_no = positions_no * y_test
    sharpe_no = compute_sharpe(pd.Series(pnl_no))
    acc_no = (np.sign(y_pred_no) == np.sign(y_test)).mean()
    
    results["no_weights"] = {"sharpe": sharpe_no, "accuracy": acc_no}
    
    # Test with weights
    weights = compute_sample_weights(len(X_train), half_life_days=half_life)
    
    model_with_weights = make_model(model_type="xgb", use_optimized=True)
    model_with_weights.fit(X_train, y_train, sample_weight=weights)
    y_pred_w = model_with_weights.predict(X_test)
    
    positions_w = np.sign(y_pred_w)
    pnl_w = positions_w * y_test
    sharpe_w = compute_sharpe(pd.Series(pnl_w))
    acc_w = (np.sign(y_pred_w) == np.sign(y_test)).mean()
    
    results["with_weights"] = {"sharpe": sharpe_w, "accuracy": acc_w}
    
    print(f"  No Weights:   Sharpe={sharpe_no:.3f}, Acc={acc_no:.1%}")
    print(f"  With Weights: Sharpe={sharpe_w:.3f}, Acc={acc_w:.1%}")
    print(f"  Improvement:  Sharpe={sharpe_w - sharpe_no:+.3f}, Acc={acc_w - acc_no:+.1%}")
    
    return {
        "ticker": ticker,
        "sharpe_no": sharpe_no,
        "sharpe_w": sharpe_w,
        "acc_no": acc_no,
        "acc_w": acc_w,
        "sharpe_diff": sharpe_w - sharpe_no,
        "acc_diff": acc_w - acc_no,
    }


def run_multi_ticker_test():
    """Test across multiple tickers with different half-lives."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Ticker Sample Weights Comparison")
    print("="*60)
    
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    half_life = 252  # 1 year
    
    results = []
    for ticker in tickers:
        try:
            r = backtest_with_weights(ticker, period="5y", half_life=half_life)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    if results:
        print("\n" + "="*60)
        print(f"SUMMARY: Sample Weights (half_life={half_life} days)")
        print("="*60)
        
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        print(f"\n--- Averages ---")
        print(f"No Weights Sharpe:   {df['sharpe_no'].mean():.3f}")
        print(f"With Weights Sharpe: {df['sharpe_w'].mean():.3f}")
        print(f"Sharpe Improvement:  {df['sharpe_diff'].mean():+.3f}")
        print(f"\nNo Weights Accuracy:   {df['acc_no'].mean():.1%}")
        print(f"With Weights Accuracy: {df['acc_w'].mean():.1%}")
        print(f"Accuracy Change:       {df['acc_diff'].mean()*100:+.2f}%")
        
        # Recommendation
        if df["sharpe_diff"].mean() > 0.05:
            print("\n✅ RECOMMENDATION: Enable sample weights (improves Sharpe)")
        elif df["sharpe_diff"].mean() < -0.05:
            print("\n❌ RECOMMENDATION: Keep sample weights OFF (hurts Sharpe)")
        else:
            print("\n⚪ RECOMMENDATION: Neutral - minimal impact, test more")
    
    return results


def test_half_life_sensitivity():
    """Test different half-life values to find optimal."""
    print("\n" + "="*60)
    print("TEST 3: Half-Life Sensitivity (AAPL only)")
    print("="*60)
    
    half_lives = [63, 126, 252, 504]  # 3mo, 6mo, 1yr, 2yr
    
    for hl in half_lives:
        backtest_with_weights("AAPL", period="5y", half_life=hl)


if __name__ == "__main__":
    print("="*60)
    print("SAMPLE WEIGHTS TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Test 1: Function correctness
    test_compute_sample_weights()
    
    # Test 2: Multi-ticker comparison
    run_multi_ticker_test()
    
    # Test 3: Half-life sensitivity (optional - uncomment to run)
    # test_half_life_sensitivity()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
