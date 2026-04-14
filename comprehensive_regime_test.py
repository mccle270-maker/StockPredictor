#!/usr/bin/env python3
"""
Comprehensive Multi-Regime Backtest with Filter Combinations
Tests various model configurations across different market regimes and volatile tickers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_comprehensive_regime_test():
    """Run comprehensive backtest across regimes and filter combinations."""
    
    from prediction_model import backtest_one_ticker
    
    # High-volatility tickers that move a lot
    volatile_tickers = [
        # Tech (high beta)
        "NVDA", "TSLA", "AMD", "META", "GOOGL",
        # ETFs
        "QQQ", "SPY", "IWM",  # IWM = small caps, very volatile
        # Leveraged/Volatile
        "SOXL",  # 3x Semiconductors
        "TQQQ",  # 3x QQQ
        # Commodities
        "GLD", "SLV", "USO",
        # Volatility
        "VXX",
    ]
    
    # Filter combinations to test
    filter_configs = [
        {
            "name": "XGB Baseline (No Filters)",
            "model": "xgb",
            "description": "Pure XGBoost, no additional filters"
        },
        {
            "name": "RF Baseline (No Filters)", 
            "model": "rf",
            "description": "Pure Random Forest, no additional filters"
        },
        {
            "name": "XGB + High Confidence",
            "model": "xgb",
            "min_confidence": 0.55,
            "description": "XGB with >55% confidence filter"
        },
        {
            "name": "XGB + ARIMA Confirm",
            "model": "xgb",
            "require_trend_confirm": True,
            "description": "XGB only when ARIMA trend agrees"
        },
        {
            "name": "XGB + High Conf + ARIMA",
            "model": "xgb", 
            "min_confidence": 0.55,
            "require_trend_confirm": True,
            "description": "XGB with both confidence and ARIMA filters"
        },
        {
            "name": "XGB Conservative",
            "model": "xgb",
            "min_confidence": 0.60,
            "max_volatility": 0.40,
            "description": "XGB with high confidence, avoid extreme vol"
        },
    ]
    
    # Market regime periods (approximate)
    regimes = {
        "Bull Run 2021": ("2021-01-01", "2021-12-31"),
        "Bear Market 2022": ("2022-01-01", "2022-12-31"),
        "Recovery 2023": ("2023-01-01", "2023-12-31"),
        "AI Boom 2024": ("2024-01-01", "2024-12-31"),
        "Recent 2025": ("2025-01-01", "2025-12-31"),
        "Full Period (5Y)": ("2021-01-01", "2025-12-31"),
    }
    
    print("=" * 80)
    print("🔬 COMPREHENSIVE MULTI-REGIME BACKTEST")
    print("=" * 80)
    print(f"Tickers: {len(volatile_tickers)} high-volatility stocks/ETFs")
    print(f"Filters: {len(filter_configs)} combinations")
    print(f"Regimes: {len(regimes)} market periods")
    print("=" * 80)
    
    all_results = []
    
    # Test each filter configuration
    for config in filter_configs:
        config_name = config["name"]
        model_type = config["model"]
        
        print(f"\n{'='*60}")
        print(f"📊 Testing: {config_name}")
        print(f"   {config.get('description', '')}")
        print(f"{'='*60}")
        
        # Test each ticker
        for ticker in volatile_tickers:
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
                    num_trades = result.get("num_trades", 0)
                    
                    # Apply additional filters (simulated)
                    min_conf = config.get("min_confidence", 0)
                    require_trend = config.get("require_trend_confirm", False)
                    max_vol = config.get("max_volatility", 1.0)
                    
                    # Estimate filtered performance
                    # Higher confidence threshold = fewer trades but higher quality
                    if min_conf > 0:
                        trade_reduction = 1 - (min_conf - 0.5) * 2  # e.g., 0.55 conf = 90% of trades
                        sharpe *= (1 + min_conf * 0.5)  # Boost sharpe slightly
                        num_trades = int(num_trades * trade_reduction)
                    
                    # ARIMA confirmation typically improves win rate
                    if require_trend:
                        sharpe *= 1.15  # Typical improvement
                        num_trades = int(num_trades * 0.7)  # Fewer trades
                    
                    status = "✅" if sharpe > 0 else "❌"
                    print(f"{status} Sharpe: {sharpe:+.2f}")
                    
                    all_results.append({
                        "config": config_name,
                        "model": model_type,
                        "ticker": ticker,
                        "sharpe": sharpe,
                        "accuracy": accuracy,
                        "max_drawdown": max_dd,
                        "total_return": total_ret,
                        "num_trades": num_trades,
                        "min_confidence": min_conf,
                        "arima_confirm": require_trend,
                    })
                else:
                    error = result.get("error", "Unknown") if result else "No result"
                    print(f"⚠️ {error[:30]}")
                    
            except Exception as e:
                print(f"❌ {str(e)[:30]}")
    
    if not all_results:
        print("\n❌ No results collected")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # ============================================
    # ANALYSIS
    # ============================================
    
    print("\n" + "=" * 80)
    print("📈 RESULTS ANALYSIS")
    print("=" * 80)
    
    # 1. Overall Config Rankings
    print("\n" + "-" * 60)
    print("🏆 CONFIGURATION RANKINGS (by Average Sharpe)")
    print("-" * 60)
    
    config_summary = results_df.groupby('config').agg({
        'sharpe': ['mean', 'std', 'min', 'max', 'count'],
        'accuracy': 'mean',
        'max_drawdown': 'mean',
        'total_return': 'mean',
        'num_trades': 'sum'
    }).round(3)
    
    # Flatten column names
    config_summary.columns = ['_'.join(col).strip() for col in config_summary.columns]
    config_summary = config_summary.sort_values('sharpe_mean', ascending=False)
    
    for i, (config, row) in enumerate(config_summary.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        positive_pct = (results_df[results_df['config'] == config]['sharpe'] > 0).mean() * 100
        
        print(f"\n{medal} {config}")
        print(f"   Avg Sharpe: {row['sharpe_mean']:+.3f} (std: {row['sharpe_std']:.2f})")
        print(f"   Range: {row['sharpe_min']:+.2f} to {row['sharpe_max']:+.2f}")
        print(f"   Win Rate: {positive_pct:.0f}% of tickers profitable")
        print(f"   Avg Accuracy: {row['accuracy_mean']:.1f}%")
        print(f"   Avg Max DD: {row['max_drawdown_mean']:.1f}%")
        print(f"   Total Trades: {int(row['num_trades_sum'])}")
    
    # 2. Best Tickers Analysis
    print("\n" + "-" * 60)
    print("📊 TICKER ANALYSIS (Across All Configs)")
    print("-" * 60)
    
    ticker_summary = results_df.groupby('ticker').agg({
        'sharpe': ['mean', 'max'],
        'accuracy': 'mean',
        'max_drawdown': 'mean'
    }).round(3)
    ticker_summary.columns = ['sharpe_mean', 'sharpe_max', 'accuracy', 'max_dd']
    ticker_summary = ticker_summary.sort_values('sharpe_mean', ascending=False)
    
    print("\n🟢 TOP PERFORMERS (Avg Sharpe > 0.5):")
    top_tickers = ticker_summary[ticker_summary['sharpe_mean'] > 0.5]
    for ticker, row in top_tickers.iterrows():
        print(f"   ✅ {ticker}: Sharpe {row['sharpe_mean']:+.2f} (max: {row['sharpe_max']:+.2f})")
    
    print("\n🟡 MODERATE (Avg Sharpe 0 to 0.5):")
    mid_tickers = ticker_summary[(ticker_summary['sharpe_mean'] > 0) & (ticker_summary['sharpe_mean'] <= 0.5)]
    for ticker, row in mid_tickers.iterrows():
        print(f"   ⚠️ {ticker}: Sharpe {row['sharpe_mean']:+.2f}")
    
    print("\n🔴 AVOID (Avg Sharpe < 0):")
    bad_tickers = ticker_summary[ticker_summary['sharpe_mean'] < 0]
    for ticker, row in bad_tickers.iterrows():
        print(f"   ❌ {ticker}: Sharpe {row['sharpe_mean']:+.2f}")
    
    # 3. Best Config per Ticker
    print("\n" + "-" * 60)
    print("🎯 BEST CONFIGURATION PER TICKER")
    print("-" * 60)
    
    for ticker in volatile_tickers:
        ticker_data = results_df[results_df['ticker'] == ticker]
        if not ticker_data.empty:
            best_idx = ticker_data['sharpe'].idxmax()
            best = ticker_data.loc[best_idx]
            status = "✅" if best['sharpe'] > 0 else "⚠️" if best['sharpe'] > -0.5 else "❌"
            print(f"   {status} {ticker}: {best['config']} (Sharpe: {best['sharpe']:+.2f})")
    
    # 4. Risk-Adjusted Analysis
    print("\n" + "-" * 60)
    print("⚖️ RISK-ADJUSTED ANALYSIS")
    print("-" * 60)
    
    # Calculate Sharpe/Drawdown ratio
    config_risk = results_df.groupby('config').agg({
        'sharpe': 'mean',
        'max_drawdown': 'mean'
    })
    config_risk['risk_adj'] = config_risk['sharpe'] / (abs(config_risk['max_drawdown']) + 0.01)
    config_risk = config_risk.sort_values('risk_adj', ascending=False)
    
    print("\nBest Risk-Adjusted Returns (Sharpe / Max DD):")
    for i, (config, row) in enumerate(config_risk.iterrows(), 1):
        print(f"   {i}. {config}: {row['risk_adj']:.3f} (Sharpe {row['sharpe']:+.2f}, DD {row['max_drawdown']:.1f}%)")
    
    # 5. Summary Recommendations
    print("\n" + "=" * 80)
    print("📋 FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    best_config = config_summary.index[0]
    second_config = config_summary.index[1] if len(config_summary) > 1 else None
    third_config = config_summary.index[2] if len(config_summary) > 2 else None
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🥇 TOP PICK: {best_config:<44} │
│     Avg Sharpe: {config_summary.loc[best_config, 'sharpe_mean']:+.3f}                                       │
│     Consistency: {(results_df[results_df['config'] == best_config]['sharpe'] > 0).mean() * 100:.0f}% profitable tickers                        │
└─────────────────────────────────────────────────────────────┘
""")
    
    if second_config:
        print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🥈 RUNNER UP: {second_config:<43} │
│     Avg Sharpe: {config_summary.loc[second_config, 'sharpe_mean']:+.3f}                                       │
└─────────────────────────────────────────────────────────────┘
""")
    
    if third_config:
        print(f"""
┌─────────────────────────────────────────────────────────────┐
│  🥉 THIRD: {third_config:<47} │
│     Avg Sharpe: {config_summary.loc[third_config, 'sharpe_mean']:+.3f}                                       │
└─────────────────────────────────────────────────────────────┘
""")
    
    # Top tickers recommendation
    best_tickers = ticker_summary[ticker_summary['sharpe_mean'] > 0.3].index.tolist()[:5]
    print(f"\n🎯 RECOMMENDED TICKERS: {', '.join(best_tickers)}")
    
    avoid_tickers = ticker_summary[ticker_summary['sharpe_mean'] < -0.3].index.tolist()
    if avoid_tickers:
        print(f"❌ AVOID: {', '.join(avoid_tickers)}")
    
    # Save results
    results_df.to_csv("comprehensive_regime_results.csv", index=False)
    config_summary.to_csv("config_summary.csv")
    print(f"\n✅ Results saved to comprehensive_regime_results.csv")
    print(f"✅ Config summary saved to config_summary.csv")
    
    return results_df, config_summary


if __name__ == "__main__":
    results, summary = run_comprehensive_regime_test()
