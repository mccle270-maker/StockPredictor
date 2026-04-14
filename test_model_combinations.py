#!/usr/bin/env python3
"""
Comprehensive Model Combination Backtest
Tests different model configurations across multiple tickers over 10 years.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our prediction systems
from prediction_model import predict_next_for_ticker, build_features_and_target
from src.core.production_predictor import ProductionPredictor, PredictionResult

def get_historical_prices(ticker: str, years: int = 10) -> pd.DataFrame:
    """Fetch historical price data."""
    import yfinance as yf
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return None
    
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def simulate_strategy(
    ticker: str,
    model_type: str = "adaptive",
    trading_mode: str = "balanced",
    require_arima_confirm: bool = False,
    min_confidence: float = 0.0,
    years: int = 10
) -> dict:
    """
    Simulate a trading strategy over historical data.
    
    Uses walk-forward approach: retrain periodically, predict out-of-sample.
    """
    print(f"  Testing {ticker} with {model_type}/{trading_mode}, ARIMA confirm={require_arima_confirm}, min_conf={min_confidence:.0%}...")
    
    try:
        # Get price data
        df = get_historical_prices(ticker, years)
        if df is None or len(df) < 252:  # Need at least 1 year
            return {"error": f"Insufficient data for {ticker}"}
        
        # Calculate daily returns for simulation
        df['returns'] = df['Close'].pct_change()
        df['fwd_ret_1d'] = df['returns'].shift(-1)  # Next day return
        
        # We'll simulate by checking what prediction we would have made
        # and seeing if the next day moved in our favor
        
        # For a realistic backtest, we need to do walk-forward
        # But for speed, we'll use a simplified simulation based on
        # running the current model on historical feature patterns
        
        # Get features for the full period
        result = build_features_and_target(ticker, period=f"{years}y", horizon=1)
        if result is None:
            return {"error": f"Could not build features for {ticker}"}
        
        features_df, target, feature_cols = result
        
        if len(features_df) < 500:
            return {"error": f"Insufficient feature data for {ticker}"}
        
        # Walk-forward simulation
        train_size = 252 * 2  # 2 years training
        step_size = 63  # Retrain every quarter
        
        trades = []
        equity = [100000.0]  # Start with $100k
        
        from sklearn.ensemble import RandomForestRegressor
        import xgboost as xgb
        
        # Align features and target
        common_idx = features_df.index.intersection(target.index)
        X = features_df.loc[common_idx]
        y = target.loc[common_idx]
        
        # Merge with price data for forward returns
        price_df = df.set_index('Date') if 'Date' in df.columns else df
        
        for start_idx in range(train_size, len(X) - 1, step_size):
            end_idx = min(start_idx + step_size, len(X) - 1)
            
            # Training data
            X_train = X.iloc[:start_idx]
            y_train = y.iloc[:start_idx]
            
            # Test data
            X_test = X.iloc[start_idx:end_idx]
            y_test = y.iloc[start_idx:end_idx]
            
            if len(X_train) < 100 or len(X_test) < 1:
                continue
            
            # Train model based on type
            if model_type == "xgb":
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
            else:  # rf or adaptive
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_leaf=4,
                    max_features=0.7,
                    random_state=42,
                    n_jobs=-1
                )
            
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            except Exception as e:
                continue
            
            # Generate signals
            for i, (idx, pred) in enumerate(zip(X_test.index, predictions)):
                if i >= len(y_test):
                    break
                
                actual_ret = y_test.iloc[i]
                
                # Calculate probability (simplified)
                # Use prediction magnitude relative to historical std
                pred_std = np.std(y_train)
                if pred_std > 0:
                    z_score = pred / pred_std
                    prob_up = 1 / (1 + np.exp(-z_score * 2))  # Sigmoid
                else:
                    prob_up = 0.5
                
                confidence = abs(prob_up - 0.5) * 2
                
                # Apply trading mode thresholds
                if trading_mode == "conservative":
                    long_thresh, short_thresh = 0.45, 0.70
                elif trading_mode == "aggressive":
                    long_thresh, short_thresh = 0.38, 0.45
                else:  # balanced
                    long_thresh, short_thresh = 0.42, 0.55
                
                # Determine signal
                if prob_up > 0.5 and confidence >= long_thresh - 0.5:
                    signal = "LONG"
                elif prob_up < 0.5 and confidence >= short_thresh - 0.5:
                    signal = "SHORT"
                else:
                    signal = "HOLD"
                
                # Apply min confidence filter
                if confidence < min_confidence:
                    signal = "HOLD"
                
                # ARIMA confirmation (simplified - use trend of last 5 predictions)
                if require_arima_confirm and i >= 5:
                    recent_preds = predictions[max(0, i-5):i]
                    trend_up = np.mean(recent_preds) > 0
                    pred_up = pred > 0
                    if trend_up != pred_up:
                        signal = "HOLD"  # No confirmation
                
                # Record trade
                if signal != "HOLD":
                    pnl_pct = actual_ret if signal == "LONG" else -actual_ret
                    trades.append({
                        "date": idx,
                        "signal": signal,
                        "pred": pred,
                        "actual": actual_ret,
                        "prob_up": prob_up,
                        "confidence": confidence,
                        "pnl_pct": pnl_pct,
                        "win": pnl_pct > 0
                    })
                    
                    # Update equity
                    position_size = min(confidence, 0.5)  # Max 50% per trade
                    equity_change = equity[-1] * position_size * pnl_pct
                    equity.append(equity[-1] + equity_change)
                else:
                    equity.append(equity[-1])  # No change
        
        if len(trades) == 0:
            return {"error": f"No trades generated for {ticker}"}
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        
        returns = trades_df['pnl_pct'].values
        wins = trades_df['win'].sum()
        total = len(trades_df)
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        equity_series = pd.Series(equity)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Win rate
        win_rate = wins / total if total > 0 else 0
        
        # Average win/loss
        winning_trades = trades_df[trades_df['win']]['pnl_pct']
        losing_trades = trades_df[~trades_df['win']]['pnl_pct']
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "ticker": ticker,
            "model": model_type,
            "mode": trading_mode,
            "arima_confirm": require_arima_confirm,
            "min_confidence": min_confidence,
            "trades": total,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "total_return": total_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "final_equity": equity[-1]
        }
        
    except Exception as e:
        return {"error": f"{ticker}: {str(e)}"}


def run_comprehensive_test():
    """Run tests across all combinations."""
    
    # Tickers to test
    tickers = ["SPY", "QQQ", "DIA", "TLT", "GLD", "VXX", "XLP", "XLU"]
    
    # Model configurations to test
    configs = [
        # (model_type, trading_mode, arima_confirm, min_confidence, description)
        ("rf", "balanced", False, 0.0, "RF Balanced - No Filters"),
        ("rf", "balanced", True, 0.0, "RF Balanced - ARIMA Confirm"),
        ("rf", "balanced", False, 0.3, "RF Balanced - Min 30% Conf"),
        ("rf", "balanced", True, 0.3, "RF Balanced - ARIMA + 30% Conf"),
        ("xgb", "balanced", False, 0.0, "XGB Balanced - No Filters"),
        ("xgb", "balanced", True, 0.0, "XGB Balanced - ARIMA Confirm"),
        ("xgb", "balanced", False, 0.3, "XGB Balanced - Min 30% Conf"),
        ("xgb", "balanced", True, 0.3, "XGB Balanced - ARIMA + 30% Conf"),
        ("rf", "conservative", False, 0.0, "RF Conservative - No Filters"),
        ("rf", "aggressive", False, 0.0, "RF Aggressive - No Filters"),
        ("xgb", "conservative", False, 0.0, "XGB Conservative - No Filters"),
        ("xgb", "aggressive", False, 0.0, "XGB Aggressive - No Filters"),
    ]
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMBINATION BACKTEST")
    print(f"Testing {len(configs)} configurations across {len(tickers)} tickers")
    print(f"Period: 10 years of historical data")
    print("=" * 80)
    print()
    
    all_results = []
    
    for config in configs:
        model_type, trading_mode, arima_confirm, min_conf, desc = config
        print(f"\n{'='*60}")
        print(f"Configuration: {desc}")
        print(f"{'='*60}")
        
        config_results = []
        
        for ticker in tickers:
            result = simulate_strategy(
                ticker=ticker,
                model_type=model_type,
                trading_mode=trading_mode,
                require_arima_confirm=arima_confirm,
                min_confidence=min_conf,
                years=10
            )
            
            if "error" not in result:
                config_results.append(result)
                all_results.append({**result, "config": desc})
            else:
                print(f"    ⚠️  {result['error']}")
        
        # Summarize config results
        if config_results:
            avg_sharpe = np.mean([r['sharpe'] for r in config_results])
            avg_win_rate = np.mean([r['win_rate'] for r in config_results])
            avg_return = np.mean([r['total_return'] for r in config_results])
            avg_dd = np.mean([r['max_dd'] for r in config_results])
            
            print(f"\n  Config Summary:")
            print(f"    Avg Sharpe: {avg_sharpe:.2f}")
            print(f"    Avg Win Rate: {avg_win_rate:.1%}")
            print(f"    Avg Return: {avg_return:.1%}")
            print(f"    Avg Max DD: {avg_dd:.1%}")
    
    # Overall summary
    if all_results:
        print("\n" + "=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        
        results_df = pd.DataFrame(all_results)
        
        # Aggregate by config
        config_summary = results_df.groupby('config').agg({
            'sharpe': 'mean',
            'win_rate': 'mean',
            'total_return': 'mean',
            'max_dd': 'mean',
            'profit_factor': 'mean',
            'trades': 'sum'
        }).round(3)
        
        # Sort by Sharpe
        config_summary = config_summary.sort_values('sharpe', ascending=False)
        
        print("\nRanked by Average Sharpe Ratio:")
        print("-" * 80)
        
        for i, (config, row) in enumerate(config_summary.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{medal} {i}. {config}")
            print(f"      Sharpe: {row['sharpe']:.2f} | Win: {row['win_rate']:.1%} | "
                  f"Return: {row['total_return']:.1%} | MaxDD: {row['max_dd']:.1%} | "
                  f"PF: {row['profit_factor']:.2f} | Trades: {int(row['trades'])}")
        
        # Best per ticker
        print("\n" + "-" * 80)
        print("Best Configuration Per Ticker:")
        print("-" * 80)
        
        for ticker in tickers:
            ticker_results = results_df[results_df['ticker'] == ticker]
            if len(ticker_results) > 0:
                best = ticker_results.loc[ticker_results['sharpe'].idxmax()]
                print(f"  {ticker}: {best['config']} (Sharpe: {best['sharpe']:.2f}, "
                      f"Win: {best['win_rate']:.1%}, Return: {best['total_return']:.1%})")
        
        # Save results
        results_df.to_csv("model_combination_results.csv", index=False)
        print(f"\n✅ Detailed results saved to model_combination_results.csv")
        
        return results_df, config_summary
    
    return None, None


if __name__ == "__main__":
    results_df, config_summary = run_comprehensive_test()
