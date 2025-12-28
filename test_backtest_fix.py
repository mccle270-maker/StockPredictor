#!/usr/bin/env python3
"""
Test script to verify backtest functions work with the fix
"""

import sys
import json
from prediction_model import (
    backtest_one_ticker,
    backtest_one_ticker_auto_optimized,
    walk_forward_backtest,
)

def test_backtest_one_ticker():
    """Test basic backtest on one ticker"""
    print("\n" + "=" * 80)
    print("🧪 TEST 1: backtest_one_ticker (AAPL, 2y history, 1y test)")
    print("=" * 80)
    
    try:
        result = backtest_one_ticker(
            ticker="AAPL",
            period="2y",
            test_years=1,
            threshold=0.002,
            model_type="rf",
            horizon=1
        )
        
        if result is None or (isinstance(result, dict) and not result):
            print("❌ FAILED: Empty result")
            return False
        
        print(f"✅ SUCCESS!")
        if isinstance(result, dict):
            print(f"   Sharpe Ratio: {result.get('sharpe', 'N/A')}")
            print(f"   Hit Rate: {result.get('hit_rate', 'N/A')}")
            print(f"   Num Features: {result.get('num_features', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_one_ticker_auto_optimized():
    """Test auto-optimized backtest"""
    print("\n" + "=" * 80)
    print("🧪 TEST 2: backtest_one_ticker_auto_optimized (NVDA, 2y history)")
    print("=" * 80)
    
    try:
        result = backtest_one_ticker_auto_optimized(
            ticker="NVDA",
            period="2y",
            test_years=1,
            threshold=0.002,
            model_type="rf",
            horizon=1
        )
        
        if result is None or (isinstance(result, dict) and not result):
            print("❌ FAILED: Empty result")
            return False
        
        print(f"✅ SUCCESS!")
        if isinstance(result, dict):
            print(f"   Keys: {list(result.keys())}")
        else:
            print(f"   Type: {type(result)}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False


def test_walk_forward_backtest():
    """Test walk-forward backtest"""
    print("\n" + "=" * 80)
    print("🧪 TEST 3: walk_forward_backtest (MSFT, 2y history, 2 folds)")
    print("=" * 80)
    
    try:
        results = walk_forward_backtest(
            ticker="MSFT",
            period="2y",
            horizon=1,
            model_type="rf",
            train_years=1,
            test_years=0.5,
            threshold=0.002,
            step_days=None
        )
        
        if not results or (isinstance(results, list) and len(results) == 0):
            print("❌ FAILED: Empty results list")
            return False
        
        print(f"✅ SUCCESS!")
        print(f"   Num Folds: {len(results)}")
        if isinstance(results, list) and len(results) > 0:
            print(f"   First fold keys: {list(results[0].keys())}")
            print(f"   First fold Sharpe: {results[0].get('sharpe', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "BACKTEST FUNCTIONS FIX VERIFICATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = {
        "test_backtest_one_ticker": test_backtest_one_ticker(),
        "test_backtest_one_ticker_auto_optimized": test_backtest_one_ticker_auto_optimized(),
        "test_walk_forward_backtest": test_walk_forward_backtest(),
    }
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All backtest functions working! Fix is successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
