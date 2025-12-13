from prediction_model import backtest_compare_one_ticker
import pandas as pd

# ==========================================
# COMPREHENSIVE STOCK BACKTEST
# Tests 20+ stocks across multiple sectors
# ==========================================

# Define stock universe
test_stocks = {
    # High-Beta Tech (Your Sweet Spot)
    "High-Beta Tech": ["NVDA", "TSLA", "AMD", "SMCI", "PLTR", "COIN"],
    
    # Mega-Cap Tech (Stable)
    "Mega-Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    
    # Semiconductors
    "Semiconductors": ["AVGO", "TSM", "INTC", "MU", "QCOM"],
    
    # Volatile Growth
    "Volatile Growth": ["SHOP", "SQ", "ROKU", "ZM", "ABNB"],
    
    # Indices
    "Indices": ["SPY", "QQQ", "IWM", "XLK"],
    
    # Traditional Sectors (Should Fail)
    "Defensive": ["JNJ", "PG", "KO", "WMT"],
    
    # Financials
    "Financials": ["JPM", "BAC", "GS", "MS"],
}

print("=" * 80)
print("COMPREHENSIVE MULTI-STOCK BACKTEST")
print("=" * 80)
print("\nTesting 5-day predictions with 2-year out-of-sample window")
print("Auto-optimization: ON (prunes weak features per stock)\n")

# Storage for results
all_results = []

# Test each category
for category, tickers in test_stocks.items():
    print(f"\n{'='*80}")
    print(f"CATEGORY: {category}")
    print(f"{'='*80}\n")
    
    for ticker in tickers:
        print(f"Testing {ticker}...")
        
        try:
            # Run backtest with auto-optimization
            results = backtest_compare_one_ticker(
                ticker=ticker,
                period="5y",
                test_years=2,
                horizon=5,
                auto_optimize=True,
            )
            
            # Extract RF results (our best model)
            rf_res = results['rf']
            xgb_res = results['xgb']
            
            # Store results
            all_results.append({
                'Category': category,
                'Ticker': ticker,
                'RF_Sharpe': rf_res['sharpe'],
                'RF_Return': rf_res['total_return'] * 100,
                'RF_HitRate': rf_res['hit_rate'] * 100,
                'RF_Features': rf_res['num_features_used'],
                'XGB_Sharpe': xgb_res['sharpe'],
                'XGB_Features': xgb_res['num_features_used'],
                'Test_Days': rf_res['test_days'],
            })
            
            print(f"  ✅ {ticker}: RF Sharpe={rf_res['sharpe']:.3f} "
                  f"({rf_res['num_features_used']} features), "
                  f"Return={rf_res['total_return']*100:.1f}%")
            
        except Exception as e:
            print(f"  ❌ {ticker}: FAILED - {str(e)[:50]}")
            all_results.append({
                'Category': category,
                'Ticker': ticker,
                'RF_Sharpe': None,
                'RF_Return': None,
                'RF_HitRate': None,
                'RF_Features': None,
                'XGB_Sharpe': None,
                'XGB_Features': None,
                'Test_Days': None,
            })

# Create summary DataFrame
results_df = pd.DataFrame(all_results)

# ==========================================
# ANALYSIS & SUMMARY
# ==========================================

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Overall statistics
valid_results = results_df.dropna(subset=['RF_Sharpe'])
print(f"\nSuccessful Tests: {len(valid_results)}/{len(results_df)}")
print(f"Average RF Sharpe: {valid_results['RF_Sharpe'].mean():.3f}")
print(f"Median RF Sharpe: {valid_results['RF_Sharpe'].median():.3f}")
print(f"Average Features Used: {valid_results['RF_Features'].mean():.1f}/60")

# Top performers
print("\n" + "=" * 80)
print("TOP 10 PERFORMERS (by RF Sharpe)")
print("=" * 80)
top_10 = valid_results.nlargest(10, 'RF_Sharpe')[
    ['Ticker', 'Category', 'RF_Sharpe', 'RF_Return', 'RF_HitRate', 'RF_Features']
]
print(top_10.to_string(index=False))

# Bottom performers
print("\n" + "=" * 80)
print("BOTTOM 10 PERFORMERS (by RF Sharpe)")
print("=" * 80)
bottom_10 = valid_results.nsmallest(10, 'RF_Sharpe')[
    ['Ticker', 'Category', 'RF_Sharpe', 'RF_Return', 'RF_HitRate', 'RF_Features']
]
print(bottom_10.to_string(index=False))

# Category analysis
print("\n" + "=" * 80)
print("PERFORMANCE BY CATEGORY")
print("=" * 80)
category_stats = valid_results.groupby('Category').agg({
    'RF_Sharpe': ['mean', 'median', 'count'],
    'RF_Features': 'mean',
}).round(3)
category_stats.columns = ['Avg_Sharpe', 'Med_Sharpe', 'Count', 'Avg_Features']
category_stats = category_stats.sort_values('Avg_Sharpe', ascending=False)
print(category_stats)

# Model comparison (RF vs XGB)
print("\n" + "=" * 80)
print("MODEL COMPARISON (RF vs XGB)")
print("=" * 80)
rf_better = (valid_results['RF_Sharpe'] > valid_results['XGB_Sharpe']).sum()
xgb_better = (valid_results['XGB_Sharpe'] > valid_results['RF_Sharpe']).sum()
print(f"RF wins: {rf_better} stocks")
print(f"XGB wins: {xgb_better} stocks")
print(f"RF average: {valid_results['RF_Sharpe'].mean():.3f}")
print(f"XGB average: {valid_results['XGB_Sharpe'].mean():.3f}")

# Sharpe distribution
print("\n" + "=" * 80)
print("SHARPE RATIO DISTRIBUTION")
print("=" * 80)
sharpe_bins = [
    ("Elite (>1.5)", (valid_results['RF_Sharpe'] > 1.5).sum()),
    ("Very Good (1.0-1.5)", ((valid_results['RF_Sharpe'] >= 1.0) & (valid_results['RF_Sharpe'] <= 1.5)).sum()),
    ("Good (0.5-1.0)", ((valid_results['RF_Sharpe'] >= 0.5) & (valid_results['RF_Sharpe'] < 1.0)).sum()),
    ("Marginal (0.0-0.5)", ((valid_results['RF_Sharpe'] >= 0.0) & (valid_results['RF_Sharpe'] < 0.5)).sum()),
    ("Negative (<0.0)", (valid_results['RF_Sharpe'] < 0.0).sum()),
]
for label, count in sharpe_bins:
    pct = count / len(valid_results) * 100
    print(f"{label}: {count} stocks ({pct:.1f}%)")

# Save results to CSV
results_df.to_csv("backtest_results_comprehensive.csv", index=False)
print(f"\n✅ Full results saved to: backtest_results_comprehensive.csv")

# ==========================================
# TRADEABLE UNIVERSE (Sharpe > 1.0)
# ==========================================

print("\n" + "=" * 80)
print("RECOMMENDED TRADEABLE UNIVERSE (Sharpe > 1.0)")
print("=" * 80)
tradeable = valid_results[valid_results['RF_Sharpe'] > 1.0].sort_values('RF_Sharpe', ascending=False)
if not tradeable.empty:
    print(f"\nFound {len(tradeable)} stocks with Sharpe > 1.0:\n")
    print(tradeable[['Ticker', 'Category', 'RF_Sharpe', 'RF_Return', 'RF_HitRate', 'RF_Features']].to_string(index=False))
    print(f"\nTradeable tickers: {', '.join(tradeable['Ticker'].tolist())}")
else:
    print("No stocks with Sharpe > 1.0 found.")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
