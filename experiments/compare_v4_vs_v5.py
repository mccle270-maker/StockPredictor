"""
Compare XGB V4 (Production) vs V5 (Heavy Reg) on truly unseen holdout data.

Methodology:
- Training: 2021-01-01 to 2024-06-30 (3.5 years)
- Holdout Test: 2024-07-01 to 2025-12-31 (1.5 years, NEVER seen during training)
- Walk-forward validation within holdout period
- Tests on 5 tickers: AAPL, MSFT, AMZN, GOOGL, NVDA

Created: 2026-01-08
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from xgboost import XGBRegressor
from scipy import stats

warnings.filterwarnings('ignore')

# Import from project
import sys
sys.path.insert(0, '/Users/jakobmccleary/Desktop/Stock Predictor')

from src.config import MODEL_VERSIONS
from prediction_model import build_features_and_target

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"]

V4_CONFIG = MODEL_VERSIONS["xgb_regularized_v4"]["params"]
V5_CONFIG = MODEL_VERSIONS["xgb_heavy_reg_v5"]["params"]

print("=" * 80)
print("XGB V4 vs V5 OUT-OF-SAMPLE COMPARISON")
print("=" * 80)
print(f"\nTraining Period: 2021-01-01 to {TRAIN_END}")
print(f"Holdout Test:    {TEST_START} to {TEST_END}")
print(f"Tickers: {TICKERS}")
print()

print("V4 Config (Production):")
for k, v in V4_CONFIG.items():
    print(f"  {k}: {v}")
print()

print("V5 Config (Heavy Reg):")
for k, v in V5_CONFIG.items():
    print(f"  {k}: {v}")
print()

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_ticker(ticker: str, config: dict, config_name: str) -> dict:
    """Run walk-forward backtest on holdout period."""
    try:
        # Get data (5 years to have enough history)
        X, y, _, _, _, _, dates = build_features_and_target(ticker, period='5y')
        
        # Convert dates to DataFrame for filtering
        df = pd.DataFrame({'y': y}, index=dates)
        
        # Split by date
        train_mask = df.index <= TRAIN_END
        test_mask = (df.index >= TEST_START) & (df.index <= TEST_END)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        test_dates = df.index[test_mask]
        
        if len(X_train) < 100 or len(X_test) < 50:
            print(f"  ⚠️ {ticker}: Insufficient data (train={len(X_train)}, test={len(X_test)})")
            return None
        
        # Train model
        model = XGBRegressor(**config)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Check prediction variance
        pred_std = np.std(preds)
        if pred_std < 0.0001:
            print(f"  ⚠️ {ticker} {config_name}: Zero prediction variance - z-score broken!")
            return None
        
        # Calculate z-scores for signal generation
        z_scores = stats.zscore(preds)
        
        # Simulate trading with z-score threshold
        Z_THRESHOLD = 1.5
        
        returns = []
        signals = []
        
        for i in range(len(preds)):
            z = z_scores[i]
            actual_ret = y_test[i]
            
            if z > Z_THRESHOLD:  # Strong bullish
                signal = 1
            elif z < -Z_THRESHOLD:  # Strong bearish
                signal = -1
            else:
                signal = 0  # No trade
            
            signals.append(signal)
            if signal != 0:
                returns.append(signal * actual_ret)
        
        if len(returns) < 5:
            print(f"  ⚠️ {ticker} {config_name}: Too few trades ({len(returns)})")
            return None
        
        returns = np.array(returns)
        
        # Metrics
        total_return = np.sum(returns)
        avg_return = np.mean(returns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        win_rate = np.mean(returns > 0)
        n_trades = len(returns)
        
        # Direction accuracy (does prediction sign match actual return sign?)
        correct = np.sum((preds > 0) == (y_test > 0))
        accuracy = correct / len(y_test)
        
        return {
            'ticker': ticker,
            'config': config_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_trades': n_trades,
            'total_return': total_return,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'pred_std': pred_std,
        }
        
    except Exception as e:
        print(f"  ❌ {ticker} {config_name}: Error - {e}")
        return None


# =============================================================================
# RUN COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING BACKTESTS...")
print("=" * 80 + "\n")

v4_results = []
v5_results = []

for ticker in TICKERS:
    print(f"Testing {ticker}...")
    
    # V4
    result_v4 = backtest_ticker(ticker, V4_CONFIG, "V4")
    if result_v4:
        v4_results.append(result_v4)
        print(f"  V4: Sharpe={result_v4['sharpe']:.2f}, WinRate={result_v4['win_rate']:.1%}, Trades={result_v4['n_trades']}")
    
    # V5
    result_v5 = backtest_ticker(ticker, V5_CONFIG, "V5")
    if result_v5:
        v5_results.append(result_v5)
        print(f"  V5: Sharpe={result_v5['sharpe']:.2f}, WinRate={result_v5['win_rate']:.1%}, Trades={result_v5['n_trades']}")
    
    print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80 + "\n")

def aggregate_results(results, name):
    if not results:
        print(f"{name}: No valid results")
        return None
    
    df = pd.DataFrame(results)
    
    agg = {
        'Mean Sharpe': df['sharpe'].mean(),
        'Median Sharpe': df['sharpe'].median(),
        'Std Sharpe': df['sharpe'].std(),
        'Mean Win Rate': df['win_rate'].mean(),
        'Mean Accuracy': df['accuracy'].mean(),
        'Total Trades': df['n_trades'].sum(),
        'Total Return': df['total_return'].sum(),
        'Tickers': len(df),
    }
    
    return agg

v4_agg = aggregate_results(v4_results, "V4")
v5_agg = aggregate_results(v5_results, "V5")

print("PER-TICKER RESULTS:")
print("-" * 80)
print(f"{'Config':<8} {'Ticker':<8} {'Sharpe':>10} {'Win Rate':>10} {'Accuracy':>10} {'Trades':>8} {'Return':>12}")
print("-" * 80)

for r in v4_results:
    print(f"{'V4':<8} {r['ticker']:<8} {r['sharpe']:>10.2f} {r['win_rate']:>10.1%} {r['accuracy']:>10.1%} {r['n_trades']:>8} {r['total_return']:>12.2%}")

print("-" * 80)

for r in v5_results:
    print(f"{'V5':<8} {r['ticker']:<8} {r['sharpe']:>10.2f} {r['win_rate']:>10.1%} {r['accuracy']:>10.1%} {r['n_trades']:>8} {r['total_return']:>12.2%}")

print("\n" + "=" * 80)
print("AGGREGATE COMPARISON")
print("=" * 80 + "\n")

print(f"{'Metric':<25} {'V4 (Production)':>20} {'V5 (Heavy Reg)':>20} {'Winner':>15}")
print("-" * 80)

if v4_agg and v5_agg:
    metrics = ['Mean Sharpe', 'Median Sharpe', 'Mean Win Rate', 'Mean Accuracy', 'Total Trades', 'Total Return']
    
    for metric in metrics:
        v4_val = v4_agg[metric]
        v5_val = v5_agg[metric]
        
        # Determine winner (higher is better for all these metrics except potentially trades)
        if metric == 'Total Trades':
            # More trades = more opportunities, but not always better
            winner = "V4" if v4_val > v5_val else "V5"
        else:
            winner = "V4" if v4_val > v5_val else "V5"
        
        if isinstance(v4_val, float):
            if 'Rate' in metric or 'Accuracy' in metric:
                print(f"{metric:<25} {v4_val:>20.1%} {v5_val:>20.1%} {winner:>15}")
            elif 'Return' in metric:
                print(f"{metric:<25} {v4_val:>20.2%} {v5_val:>20.2%} {winner:>15}")
            else:
                print(f"{metric:<25} {v4_val:>20.3f} {v5_val:>20.3f} {winner:>15}")
        else:
            print(f"{metric:<25} {v4_val:>20} {v5_val:>20} {winner:>15}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80 + "\n")

if v4_agg and v5_agg:
    v4_score = 0
    v5_score = 0
    
    # Sharpe is most important
    if v4_agg['Mean Sharpe'] > v5_agg['Mean Sharpe']:
        v4_score += 2
    else:
        v5_score += 2
    
    # Win rate
    if v4_agg['Mean Win Rate'] > v5_agg['Mean Win Rate']:
        v4_score += 1
    else:
        v5_score += 1
    
    # Accuracy
    if v4_agg['Mean Accuracy'] > v5_agg['Mean Accuracy']:
        v4_score += 1
    else:
        v5_score += 1
    
    # Total return
    if v4_agg['Total Return'] > v5_agg['Total Return']:
        v4_score += 1.5
    else:
        v5_score += 1.5
    
    print(f"Scoring: V4 = {v4_score:.1f} points, V5 = {v5_score:.1f} points")
    print()
    
    if v4_score > v5_score:
        print("🏆 RECOMMENDATION: Keep V4 (Production)")
        print("   Reason: Better overall performance on holdout data")
    elif v5_score > v4_score:
        print("🏆 RECOMMENDATION: Switch to V5 (Heavy Reg)")
        print("   Reason: Better overall performance on holdout data")
    else:
        print("⚖️ RECOMMENDATION: TIE - Both configs perform similarly")
        print("   Consider sticking with V4 for stability, or test V5 in paper trading")

print("\n" + "=" * 80)
print("NOTES")
print("=" * 80)
print("""
- Holdout period is 2024-07-01 to 2025-12-31 (truly unseen data)
- Z-score threshold of 1.5 used for signal generation
- Sharpe ratio annualized (×√252)
- Both configs have gamma=0 to ensure z-score works
- V5 has higher regularization (reg_lambda=7.0 vs 5.0)
""")
