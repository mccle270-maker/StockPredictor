#!/usr/bin/env python3
"""
Test fix for missing macro data issue
"""
import os
os.environ['FRED_API_KEY'] = '357745ca92b751bf20b6131ca8bd8646'

from prediction_model import predict_next_for_ticker

print("\n" + "="*80)
print("Testing prediction with missing macro data")
print("="*80)

tickers = ['AAPL', 'NVDA']

for ticker in tickers:
    print(f"\n🔍 Testing {ticker}...")
    try:
        pred = predict_next_for_ticker(ticker, period='1y', model_type='rf', horizon=1)
        
        if isinstance(pred, dict):
            print(f"   ✅ SUCCESS!")
            print(f"      • Next-day return: {pred.get('pred_next_ret', 'N/A'):.2%}")
            print(f"      • Prediction price: ${pred.get('pred_next_price', 'N/A'):.2f}")
            print(f"      • Probability up: {pred.get('prob_up', 'N/A'):.1%}")
            print(f"      • Num features: {pred.get('num_features', 'N/A')}")
        else:
            print(f"   ❌ Unexpected result: {type(pred)}")
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)[:200]}")

print("\n" + "="*80)
print("Test complete!")
print("="*80 + "\n")
