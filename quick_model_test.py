#!/usr/bin/env python3
"""
Quick Model Combination Test - Faster version
Tests key configurations across ETF tickers with 5-year walk-forward backtest.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_quick_test():
    """Run a quick but realistic backtest comparison."""
    
    # Import prediction system
    from prediction_model import backtest_one_ticker
    
    # Tickers to test (mix of equity, bonds, commodities, low-vol)
    tickers = ["SPY", "QQQ", "DIA", "TLT", "GLD", "XLP", "XLU"]
    
    # Model configurations
    configs = [
        ("rf", "RF (Random Forest)"),
        ("xgb", "XGB (XGBoost)"),
    ]
    
    print("=" * 70)
    print("QUICK MODEL COMBINATION BACKTEST")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Period: 5 years | Walk-forward validation")
    print("=" * 70)
    
    all_results = []
    
    for model_type, model_name in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print(f"{'='*50}")
        
        for ticker in tickers:
            print(f"  {ticker}...", end=" ", flush=True)
            
            try:
                result = backtest_one_ticker(
                    ticker=ticker,
                    model_type=model_type,
                    period="5y",
                    horizon=1
                )
                
                if result and "error" not in result:
                    sharpe = result.get("sharpe", 0)
                    accuracy = result.get("accuracy", 0) * 100
                    max_dd = result.get("max_drawdown", 0) * 100
                    total_ret = result.get("total_return", 0) * 100
                    
                    status = "✅" if sharpe > 0 else "❌"
                    print(f"{status} Sharpe: {sharpe:+.2f}, Acc: {accuracy:.0f}%, DD: {max_dd:.1f}%")
                    
                    all_results.append({
                        "model": model_name,
                        "model_type": model_type,
                        "ticker": ticker,
                        "sharpe": sharpe,
                        "accuracy": accuracy,
                        "max_drawdown": max_dd,
                        "total_return": total_ret,
                        "num_trades": result.get("num_trades", 0)
                    })
                else:
                    print(f"⚠️ Error: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ Failed: {str(e)[:40]}")
    
    if not all_results:
        print("\n❌ No results collected")
        return
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Summary by model
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY BY MODEL")
    print("=" * 70)
    
    model_summary = results_df.groupby('model').agg({
        'sharpe': ['mean', 'std', 'min', 'max'],
        'accuracy': 'mean',
        'max_drawdown': 'mean',
        'total_return': 'mean',
        'num_trades': 'sum'
    }).round(2)
    
    print(model_summary.to_string())
    
    # Rank models by average Sharpe
    print("\n" + "-" * 70)
    print("MODEL RANKING (by Average Sharpe)")
    print("-" * 70)
    
    model_ranks = results_df.groupby('model').agg({
        'sharpe': 'mean',
        'accuracy': 'mean',
        'max_drawdown': 'mean'
    }).sort_values('sharpe', ascending=False)
    
    for i, (model, row) in enumerate(model_ranks.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal} {model}")
        print(f"    Avg Sharpe: {row['sharpe']:+.3f}")
        print(f"    Avg Accuracy: {row['accuracy']:.1f}%")
        print(f"    Avg Max DD: {row['max_drawdown']:.1f}%")
    
    # Best per ticker
    print("\n" + "-" * 70)
    print("BEST MODEL PER TICKER")
    print("-" * 70)
    
    for ticker in tickers:
        ticker_data = results_df[results_df['ticker'] == ticker]
        if not ticker_data.empty:
            best_idx = ticker_data['sharpe'].idxmax()
            best = ticker_data.loc[best_idx]
            print(f"  {ticker}: {best['model']} (Sharpe: {best['sharpe']:+.2f}, Acc: {best['accuracy']:.0f}%)")
    
    # Win rate by asset class
    print("\n" + "-" * 70)
    print("PERFORMANCE BY ASSET CLASS")
    print("-" * 70)
    
    asset_classes = {
        "Equity (SPY, QQQ, DIA)": ["SPY", "QQQ", "DIA"],
        "Bonds (TLT)": ["TLT"],
        "Commodities (GLD)": ["GLD"],
        "Defensive (XLP, XLU)": ["XLP", "XLU"]
    }
    
    for asset_class, tks in asset_classes.items():
        class_data = results_df[results_df['ticker'].isin(tks)]
        if not class_data.empty:
            avg_sharpe = class_data['sharpe'].mean()
            avg_acc = class_data['accuracy'].mean()
            status = "✅" if avg_sharpe > 0 else "⚠️" if avg_sharpe > -0.2 else "❌"
            print(f"  {status} {asset_class}: Sharpe {avg_sharpe:+.2f}, Acc {avg_acc:.0f}%")
    
    # Overall recommendation
    print("\n" + "=" * 70)
    print("📊 RECOMMENDATION")
    print("=" * 70)
    
    best_model = model_ranks.index[0]
    best_sharpe = model_ranks.iloc[0]['sharpe']
    
    # Which tickers work best
    positive_sharpe = results_df[results_df['sharpe'] > 0]
    good_tickers = positive_sharpe.groupby('ticker')['sharpe'].mean().sort_values(ascending=False)
    
    print(f"\n  Best Overall Model: {best_model} (Avg Sharpe: {best_sharpe:+.3f})")
    print(f"\n  Tickers with Positive Edge:")
    for tk, sh in good_tickers.head(5).items():
        print(f"    ✅ {tk}: Sharpe {sh:+.2f}")
    
    bad_tickers = results_df.groupby('ticker')['sharpe'].mean()
    bad_tickers = bad_tickers[bad_tickers < 0].sort_values()
    if not bad_tickers.empty:
        print(f"\n  Tickers to AVOID:")
        for tk, sh in bad_tickers.items():
            print(f"    ❌ {tk}: Sharpe {sh:+.2f}")
    
    # Save results
    results_df.to_csv("quick_model_test_results.csv", index=False)
    print(f"\n✅ Results saved to quick_model_test_results.csv")
    
    return results_df


if __name__ == "__main__":
    results = run_quick_test()
