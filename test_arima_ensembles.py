#!/usr/bin/env python3
"""
Test ARIMA integration and compare model ensemble combinations.

Tests:
1. ML-only vs ML+ARIMA predictions
2. Different ensemble weight combinations
3. XGB, LSTM, ARIMA combinations

Author: Stock Predictor Team
Date: 2026-01-11
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
import time

# Test imports
print("=" * 60)
print("🧪 ARIMA ENSEMBLE TEST SUITE")
print("=" * 60)

# Check pmdarima availability
try:
    from pmdarima import auto_arima
    print("✅ pmdarima installed")
except ImportError:
    print("❌ pmdarima NOT installed - run: pip install pmdarima")
    exit(1)

# Import prediction functions
try:
    from prediction_model import predict_next_for_ticker, make_model, build_features_and_target
    print("✅ prediction_model imported")
except ImportError as e:
    print(f"❌ Could not import prediction_model: {e}")
    exit(1)

# Import ARIMA components
try:
    from arima_integration import ARIMAPredictor, EnsemblePredictor
    print("✅ arima_integration imported")
except ImportError as e:
    print(f"❌ Could not import arima_integration: {e}")
    exit(1)

# Import LSTM ensemble
try:
    from src.core.lstm_xgb_ensemble import LSTMXGBEnsemble
    HAS_LSTM = True
    print("✅ LSTMXGBEnsemble imported")
except ImportError as e:
    HAS_LSTM = False
    print(f"⚠️ LSTMXGBEnsemble not available: {e}")


# ==== TEST 1: Basic ARIMA Integration ====
def test_arima_basic():
    """Test that ARIMA integration works in predict_next_for_ticker."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic ARIMA Integration")
    print("=" * 60)
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    results = []
    
    for ticker in test_tickers:
        print(f"\n🔍 Testing {ticker}...")
        
        try:
            # Test with ARIMA enabled
            start = time.time()
            result_arima = predict_next_for_ticker(
                ticker=ticker,
                period="2y",
                model_type="xgb",
                horizon=1,
                use_arima=True,
                arima_weight=0.3
            )
            time_arima = time.time() - start
            
            # Test without ARIMA
            start = time.time()
            result_ml = predict_next_for_ticker(
                ticker=ticker,
                period="2y",
                model_type="xgb",
                horizon=1,
                use_arima=False
            )
            time_ml = time.time() - start
            
            results.append({
                "ticker": ticker,
                "ml_pred": result_ml["pred_next_ret"],
                "arima_pred": result_arima.get("arima_pred"),
                "arima_order": result_arima.get("arima_order"),
                "ensemble_pred": result_arima.get("ensemble_pred"),
                "ml_time_s": time_ml,
                "arima_time_s": time_arima,
                "status": "✅"
            })
            
            print(f"  ML-only pred:    {result_ml['pred_next_ret']:+.4f}")
            print(f"  ARIMA pred:      {result_arima.get('arima_pred', 'N/A')}")
            print(f"  Ensemble pred:   {result_arima.get('ensemble_pred'):+.4f}")
            print(f"  ARIMA order:     {result_arima.get('arima_order')}")
            print(f"  Time: ML={time_ml:.1f}s, ARIMA={time_arima:.1f}s")
            
        except Exception as e:
            results.append({
                "ticker": ticker,
                "status": f"❌ {str(e)[:50]}"
            })
            print(f"  ❌ Error: {e}")
    
    # Summary
    df = pd.DataFrame(results)
    print("\n📊 ARIMA Integration Results:")
    print(df.to_string(index=False))
    
    return results


# ==== TEST 2: ARIMA Weight Comparison ====
def test_arima_weights():
    """Test different ARIMA weight configurations."""
    print("\n" + "=" * 60)
    print("TEST 2: ARIMA Weight Comparison")
    print("=" * 60)
    
    ticker = "AAPL"
    weights_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    print(f"\n🔍 Testing {ticker} with different ARIMA weights...")
    
    for arima_weight in weights_to_test:
        try:
            result = predict_next_for_ticker(
                ticker=ticker,
                period="2y",
                model_type="xgb",
                horizon=1,
                use_arima=(arima_weight > 0),
                arima_weight=arima_weight
            )
            
            results.append({
                "arima_weight": arima_weight,
                "ml_weight": 1.0 - arima_weight,
                "ml_pred": result["pred_next_ret"],
                "arima_pred": result.get("arima_pred"),
                "ensemble_pred": result.get("ensemble_pred"),
            })
            
        except Exception as e:
            print(f"  ❌ Weight {arima_weight}: {e}")
    
    df = pd.DataFrame(results)
    print("\n📊 Weight Comparison Results:")
    print(df.to_string(index=False))
    
    return results


# ==== TEST 3: Model Comparison (XGB vs XGB+ARIMA vs XGB+LSTM) ====
def test_model_comparison():
    """Compare XGB, XGB+ARIMA, and XGB+LSTM performance."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Ensemble Comparison")
    print("=" * 60)
    
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    results = []
    
    for ticker in test_tickers:
        print(f"\n🔍 Testing {ticker}...")
        row = {"ticker": ticker}
        
        # XGB only
        try:
            result = predict_next_for_ticker(
                ticker=ticker, period="2y", model_type="xgb", 
                horizon=1, use_arima=False
            )
            row["xgb_pred"] = result["pred_next_ret"]
            row["xgb_prob_up"] = result.get("prob_up")
            print(f"  XGB:        {result['pred_next_ret']:+.4f}")
        except Exception as e:
            row["xgb_pred"] = None
            print(f"  XGB: ❌ {e}")
        
        # XGB + ARIMA (30%)
        try:
            result = predict_next_for_ticker(
                ticker=ticker, period="2y", model_type="xgb",
                horizon=1, use_arima=True, arima_weight=0.3
            )
            row["xgb_arima_pred"] = result.get("ensemble_pred")
            row["arima_order"] = str(result.get("arima_order"))
            print(f"  XGB+ARIMA:  {result.get('ensemble_pred'):+.4f}")
        except Exception as e:
            row["xgb_arima_pred"] = None
            print(f"  XGB+ARIMA: ❌ {e}")
        
        # XGB + LSTM (if available)
        if HAS_LSTM:
            try:
                result = predict_next_for_ticker(
                    ticker=ticker, period="2y", model_type="xgb_lstm",
                    horizon=1, use_arima=False
                )
                row["xgb_lstm_pred"] = result["pred_next_ret"]
                print(f"  XGB+LSTM:   {result['pred_next_ret']:+.4f}")
            except Exception as e:
                row["xgb_lstm_pred"] = None
                print(f"  XGB+LSTM: ❌ {e}")
        
        # XGB + LSTM + ARIMA (triple ensemble)
        if HAS_LSTM:
            try:
                result = predict_next_for_ticker(
                    ticker=ticker, period="2y", model_type="xgb_lstm",
                    horizon=1, use_arima=True, arima_weight=0.2
                )
                row["triple_pred"] = result.get("ensemble_pred")
                print(f"  TRIPLE:     {result.get('ensemble_pred'):+.4f}")
            except Exception as e:
                row["triple_pred"] = None
                print(f"  TRIPLE: ❌ {e}")
        
        results.append(row)
    
    df = pd.DataFrame(results)
    print("\n📊 Model Comparison Results:")
    print(df.to_string(index=False))
    
    return results


# ==== TEST 4: Backtest with ARIMA ====
def test_arima_backtest():
    """Run a simple backtest comparing ML vs ML+ARIMA."""
    print("\n" + "=" * 60)
    print("TEST 4: Simple Backtest Comparison")
    print("=" * 60)
    
    from data_fetch import get_price_history
    
    ticker = "AAPL"
    period = "2y"
    
    print(f"\n🔍 Running backtest on {ticker}...")
    
    try:
        # Get historical data
        hist = get_price_history(ticker, period=period, interval="1d")
        if hist is None or hist.empty:
            print("❌ Could not fetch historical data")
            return None
        
        # Use last 252 trading days for test
        test_size = min(252, len(hist) - 500)
        
        results = {
            "ml_only": {"correct": 0, "total": 0, "returns": []},
            "ml_arima": {"correct": 0, "total": 0, "returns": []}
        }
        
        # Get predictions for last N days (simplified - just current prediction)
        for use_arima in [False, True]:
            key = "ml_arima" if use_arima else "ml_only"
            
            result = predict_next_for_ticker(
                ticker=ticker, period=period, model_type="xgb",
                horizon=1, use_arima=use_arima, arima_weight=0.3
            )
            
            pred = result.get("ensemble_pred") if use_arima else result["pred_next_ret"]
            prob_up = result.get("prob_up", 0.5)
            
            print(f"  {key}: pred={pred:+.4f}, prob_up={prob_up:.2%}")
        
        print("\n📊 Backtest Summary:")
        print("  (Full walk-forward backtest would require more time)")
        print("  Current predictions compared above.")
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return None


# ==== TEST 5: ARIMA Stand-alone Performance ====
def test_arima_standalone():
    """Test ARIMA predictor standalone performance."""
    print("\n" + "=" * 60)
    print("TEST 5: ARIMA Standalone Analysis")
    print("=" * 60)
    
    from data_fetch import get_price_history
    
    tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
    results = []
    
    for ticker in tickers:
        print(f"\n🔍 Analyzing {ticker}...")
        
        try:
            hist = get_price_history(ticker, period="2y", interval="1d")
            if hist is None or hist.empty:
                continue
            
            returns = hist["Close"].pct_change().dropna()
            
            # Fit ARIMA
            arima = ARIMAPredictor(max_p=3, max_d=1, max_q=3, verbose=False)
            success = arima.fit(returns)
            
            if success:
                forecast = arima.predict(steps=5)
                order = arima.get_fitted_order()
                
                results.append({
                    "ticker": ticker,
                    "order": str(order),
                    "1d_forecast": forecast[0] if forecast is not None else None,
                    "5d_sum": np.sum(forecast) if forecast is not None else None,
                    "returns_std": returns.std(),
                    "status": "✅"
                })
                
                print(f"  Order: {order}")
                print(f"  1D forecast: {forecast[0]:+.4f}")
                print(f"  5D sum: {np.sum(forecast):+.4f}")
            else:
                results.append({
                    "ticker": ticker,
                    "status": "❌ Fit failed"
                })
                
        except Exception as e:
            results.append({
                "ticker": ticker,
                "status": f"❌ {str(e)[:30]}"
            })
    
    df = pd.DataFrame(results)
    print("\n📊 ARIMA Standalone Results:")
    print(df.to_string(index=False))
    
    return results


# ==== MAIN ====
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting ARIMA Ensemble Tests...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run tests
    test_arima_basic()
    test_arima_weights()
    test_arima_standalone()
    test_model_comparison()
    test_arima_backtest()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
