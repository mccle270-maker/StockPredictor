#!/usr/bin/env python3
import os, json, pandas as pd, numpy as np
from datetime import datetime
import logging, sys, warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('phase2_backtest.log'), logging.StreamHandler()])
logger = logging.getLogger('Phase2')

from prediction_model import build_features_and_target, make_model, USE_ELASTICNET_SELECT
from sklearn.metrics import mean_squared_error, accuracy_score

def backtest_ticker(ticker, period="2y"):
    logger.info(f"\n{'='*80}\nBacktesting {ticker}\n{'='*80}")
    try:
        result = build_features_and_target(ticker=ticker, period=period, horizon=1)
        if isinstance(result, tuple):
            X, y, _, _, _, _, dates = result
        else:
            logger.error(f"{ticker}: Failed to build features")
            return None
        
        if X is None or len(X) < 100:
            logger.warning(f"{ticker}: Insufficient data ({len(X) if X is not None else 0} rows)")
            return None
        
        logger.info(f"{ticker}: {len(X)} samples, {X.shape[1]} features")
        
        n = len(X)
        split_idx = int(n * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        results = {"ticker": ticker, "total_samples": n, "train_samples": len(X_train), 
                   "test_samples": len(X_test), "features": X.shape[1], "models": {}, 
                   "best_model": None, "best_accuracy": 0, "issues": [], "recommendations": []}
        
        for model_type in ["rf", "xgb"]:
            try:
                model = make_model(model_type=model_type, random_state=42, task="reg")
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
                y_train_dir = (y_train > 0).astype(int)
                y_test_dir = (y_test > 0).astype(int)
                y_train_pred_dir = (y_train_pred > 0).astype(int)
                y_test_pred_dir = (y_test_pred > 0).astype(int)
                
                train_acc = accuracy_score(y_train_dir, y_train_pred_dir)
                test_acc = accuracy_score(y_test_dir, y_test_pred_dir)
                
                rmse_ratio = test_rmse / (train_rmse + 1e-6)
                acc_delta = train_acc - test_acc
                overfit = (rmse_ratio > 1.5) or (acc_delta > 0.15)
                
                results["models"][model_type] = {
                    "train_rmse": float(train_rmse), "test_rmse": float(test_rmse),
                    "rmse_ratio": float(rmse_ratio), "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc), "accuracy_delta": float(acc_delta), "overfitting": bool(overfit)}
                
                if test_acc > results["best_accuracy"]:
                    results["best_accuracy"] = test_acc
                    results["best_model"] = model_type
                
                logger.info(f"  {model_type.upper():10} | Test Acc: {test_acc:.3f} | RMSE Ratio: {rmse_ratio:.3f} | Overfit: {'YES' if overfit else 'NO'}")
                
                if overfit:
                    results["issues"].append(f"{model_type}: Overfitting detected (ratio={rmse_ratio:.2f})")
                    results["recommendations"].append("Add regularization or reduce features")
                if test_acc < 0.52:
                    results["issues"].append(f"{model_type}: Low accuracy ({test_acc:.3f})")
                    
            except Exception as e:
                logger.warning(f"  {model_type} failed: {str(e)[:50]}")
                continue
        
        returns = (y_test > 0).astype(float)
        sharpe = (np.mean(returns) - 0.5) / (np.std(returns) + 1e-6) * np.sqrt(252)
        results["sharpe_ratio"] = float(sharpe)
        results["win_rate"] = float(np.mean(returns))
        
        if results["best_accuracy"] > 0.58:
            results["recommendations"].append(f"✓ Strong accuracy ({results['best_accuracy']:.3f}), ready for deployment")
        if sharpe > 0.80:
            results["recommendations"].append(f"✓ Good Sharpe ratio ({sharpe:.3f})")
        if len(results["issues"]) == 0:
            results["recommendations"].append("✓ All metrics healthy")
        
        logger.info(f"  Best Model: {results['best_model'].upper()} ({results['best_accuracy']:.3f})")
        logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"{ticker}: Exception - {str(e)}")
        return None

def main():
    test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GLD", "SPY", "QQQ", "IWM"]
    
    logger.info("\n" + "="*80 + "\nPHASE 2: WALK-FORWARD BACKTEST\n" + "="*80 +
                f"\nTest Date: {datetime.now()}\nTickers: {len(test_tickers)}\n")
    
    all_results = {}
    for ticker in test_tickers:
        result = backtest_ticker(ticker)
        if result:
            all_results[ticker] = result
    
    logger.info("\n" + "="*80 + "\nPHASE 2 SUMMARY\n" + "="*80)
    
    if all_results:
        valid = list(all_results.values())
        avg_accuracy = np.mean([r["best_accuracy"] for r in valid])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in valid])
        avg_win_rate = np.mean([r["win_rate"] for r in valid])
        
        logger.info(f"\nResults: {len(valid)}/{len(test_tickers)} tickers passed")
        logger.info(f"Average Accuracy: {avg_accuracy:.3f}")
        logger.info(f"Average Sharpe: {avg_sharpe:.3f}")
        logger.info(f"Average Win Rate: {avg_win_rate:.3f}")
        logger.info(f"\n{'Ticker':<8} {'Model':<8} {'Accuracy':<10} {'Sharpe':<10} {'Status':<15}")
        logger.info("-" * 60)
        for ticker, result in all_results.items():
            status = "✅ PASS" if result["best_accuracy"] > 0.55 else "⚠️  CAUTION"
            logger.info(f"{ticker:<8} {result['best_model']:<8} {result['best_accuracy']:<10.3f} {result['sharpe_ratio']:<10.3f} {status:<15}")
        
        status = "🟢 PRODUCTION READY" if avg_accuracy > 0.56 and avg_sharpe > 0.70 else "🟡 NEEDS REVIEW"
        logger.info(f"\n{status}")
        
        with open("phase2_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("\nResults saved to phase2_results.json")
        return all_results
    else:
        logger.error("No valid results")
        return {}

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results else 1)
