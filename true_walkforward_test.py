#!/usr/bin/env python3
"""
TRUE WALK-FORWARD BACKTEST
Tests all model configurations with proper train/test splits on unseen data.
This is the gold standard - train on past, test on future you've never seen.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def run_true_walkforward():
    """
    Proper walk-forward validation:
    - Train on 2 years of data
    - Test on next 6 months (unseen)
    - Roll forward and repeat
    """
    
    from prediction_model import build_features_and_target
    from src.core.models import make_model
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    
    # High-performing tickers from previous test
    test_tickers = ["GOOGL", "SPY", "QQQ", "NVDA", "SLV", "AMD"]
    
    # Model configurations to test
    configs = {
        "XGB_Baseline": {
            "model_type": "xgb",
            "use_elastic_net": False,
            "use_scaler": False,
            "adaptive_lr": False,
        },
        "XGB + StandardScaler": {
            "model_type": "xgb", 
            "use_elastic_net": False,
            "use_scaler": True,
            "adaptive_lr": False,
        },
        "XGB + ElasticNet Prefilter": {
            "model_type": "xgb",
            "use_elastic_net": True,
            "use_scaler": True,
            "adaptive_lr": False,
        },
        "XGB + Adaptive LR": {
            "model_type": "xgb",
            "use_elastic_net": False,
            "use_scaler": True,
            "adaptive_lr": True,
        },
        "XGB + All Enhancements": {
            "model_type": "xgb",
            "use_elastic_net": True,
            "use_scaler": True,
            "adaptive_lr": True,
        },
        "RF_Baseline": {
            "model_type": "rf",
            "use_elastic_net": False,
            "use_scaler": False,
            "adaptive_lr": False,
        },
        "RF + ElasticNet + Scaler": {
            "model_type": "rf",
            "use_elastic_net": True,
            "use_scaler": True,
            "adaptive_lr": False,
        },
    }
    
    # Walk-forward periods (train 2 years, test 6 months)
    # We'll use index-based splits after getting data
    train_days = 504  # ~2 years of trading days
    test_days = 126   # ~6 months
    
    print("=" * 80)
    print("🔬 TRUE WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Training window: {train_days} days (~2 years)")
    print(f"Test window: {test_days} days (~6 months)")
    print(f"Tickers: {', '.join(test_tickers)}")
    print(f"Configurations: {len(configs)}")
    print("=" * 80)
    
    all_results = []
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"📊 Testing {ticker}")
        print(f"{'='*60}")
        
        try:
            # Build features for full period
            # Returns: X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates
            result = build_features_and_target(ticker, period="7y", horizon=1)
            if result is None:
                print(f"  ❌ Could not build features for {ticker}")
                continue
            
            X_arr, y_arr, last_row_feats, last_close, last_vol, prob_gaf, dates = result
            
            # Convert to DataFrames for easier manipulation
            n_features = X_arr.shape[1]
            feature_cols = [f"feat_{i}" for i in range(n_features)]
            
            X = pd.DataFrame(X_arr, index=dates, columns=feature_cols)
            y = pd.Series(y_arr, index=dates)
            
            print(f"  Data: {len(X)} samples, {n_features} features")
            
            if len(X) < train_days + test_days:
                print(f"  ❌ Insufficient data (need {train_days + test_days})")
                continue
            
            # Walk-forward folds
            n_folds = (len(X) - train_days) // test_days
            print(f"  Walk-forward folds: {n_folds}")
            
            for config_name, config in configs.items():
                fold_results = []
                
                for fold in range(min(n_folds, 5)):  # Max 5 folds for speed
                    train_start = fold * test_days
                    train_end = train_start + train_days
                    test_start = train_end
                    test_end = test_start + test_days
                    
                    if test_end > len(X):
                        break
                    
                    X_train = X.iloc[train_start:train_end]
                    y_train = y.iloc[train_start:train_end]
                    X_test = X.iloc[test_start:test_end]
                    y_test = y.iloc[test_start:test_end]
                    
                    # Get dates for reporting
                    train_start_date = X_train.index[0].strftime("%Y-%m") if hasattr(X_train.index[0], 'strftime') else str(X_train.index[0])[:7]
                    test_end_date = X_test.index[-1].strftime("%Y-%m") if hasattr(X_test.index[-1], 'strftime') else str(X_test.index[-1])[:7]
                    
                    try:
                        # Apply preprocessing based on config
                        X_train_proc = X_train.copy()
                        X_test_proc = X_test.copy()
                        
                        # 1. Scaling
                        if config["use_scaler"]:
                            scaler = StandardScaler()
                            X_train_proc = pd.DataFrame(
                                scaler.fit_transform(X_train_proc),
                                index=X_train_proc.index,
                                columns=X_train_proc.columns
                            )
                            X_test_proc = pd.DataFrame(
                                scaler.transform(X_test_proc),
                                index=X_test_proc.index,
                                columns=X_test_proc.columns
                            )
                        
                        # 2. ElasticNet feature selection
                        selected_features = list(X_train_proc.columns)
                        if config["use_elastic_net"]:
                            en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42)
                            en.fit(X_train_proc, y_train)
                            
                            # Select features with non-zero coefficients
                            feature_importance = np.abs(en.coef_)
                            threshold = np.percentile(feature_importance, 25)  # Top 75%
                            selected_mask = feature_importance > threshold
                            selected_features = [f for f, m in zip(X_train_proc.columns, selected_mask) if m]
                            
                            if len(selected_features) < 10:
                                selected_features = list(X_train_proc.columns)  # Fallback
                            
                            X_train_proc = X_train_proc[selected_features]
                            X_test_proc = X_test_proc[selected_features]
                        
                        # 3. Build model
                        if config["model_type"] == "xgb":
                            learning_rate = 0.05
                            if config["adaptive_lr"]:
                                # Adaptive: lower LR for more volatile periods
                                train_vol = y_train.std()
                                if train_vol > 0.02:  # High vol
                                    learning_rate = 0.03
                                elif train_vol < 0.01:  # Low vol
                                    learning_rate = 0.08
                            
                            model = xgb.XGBRegressor(
                                n_estimators=100,
                                max_depth=5,
                                learning_rate=learning_rate,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=0.1,
                                reg_lambda=1.0,
                                random_state=42,
                                verbosity=0
                            )
                        else:  # RF
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(
                                n_estimators=100,
                                max_depth=None,
                                min_samples_leaf=4,
                                max_features=0.7,
                                random_state=42,
                                n_jobs=-1
                            )
                        
                        # Train
                        model.fit(X_train_proc, y_train)
                        
                        # Predict
                        predictions = model.predict(X_test_proc)
                        
                        # Calculate metrics on UNSEEN test data
                        # Direction accuracy
                        correct_direction = ((predictions > 0) == (y_test > 0)).mean()
                        
                        # Trading returns (go long when pred > 0, short when pred < 0)
                        signals = np.sign(predictions)
                        trade_returns = signals * y_test.values
                        
                        # Sharpe ratio (annualized)
                        if trade_returns.std() > 0:
                            sharpe = trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                        else:
                            sharpe = 0
                        
                        # Total return
                        total_ret = (1 + trade_returns).prod() - 1
                        
                        # Max drawdown
                        cumulative = (1 + trade_returns).cumprod()
                        rolling_max = np.maximum.accumulate(cumulative)
                        drawdowns = (cumulative - rolling_max) / rolling_max
                        max_dd = drawdowns.min()
                        
                        fold_results.append({
                            "fold": fold + 1,
                            "train_period": f"{train_start_date}",
                            "test_period": f"{test_end_date}",
                            "accuracy": correct_direction,
                            "sharpe": sharpe,
                            "total_return": total_ret,
                            "max_drawdown": max_dd,
                            "n_features": len(selected_features),
                            "n_trades": len(y_test),
                        })
                        
                    except Exception as e:
                        continue
                
                # Aggregate fold results
                if fold_results:
                    avg_sharpe = np.mean([r["sharpe"] for r in fold_results])
                    avg_acc = np.mean([r["accuracy"] for r in fold_results])
                    avg_ret = np.mean([r["total_return"] for r in fold_results])
                    avg_dd = np.mean([r["max_drawdown"] for r in fold_results])
                    std_sharpe = np.std([r["sharpe"] for r in fold_results])
                    positive_folds = sum(1 for r in fold_results if r["sharpe"] > 0)
                    
                    status = "✅" if avg_sharpe > 0.5 else "⚠️" if avg_sharpe > 0 else "❌"
                    print(f"  {status} {config_name}: Sharpe {avg_sharpe:+.2f}±{std_sharpe:.2f}, Acc {avg_acc:.1%}, {positive_folds}/{len(fold_results)} positive folds")
                    
                    all_results.append({
                        "ticker": ticker,
                        "config": config_name,
                        "avg_sharpe": avg_sharpe,
                        "std_sharpe": std_sharpe,
                        "avg_accuracy": avg_acc,
                        "avg_return": avg_ret,
                        "avg_max_dd": avg_dd,
                        "positive_folds": positive_folds,
                        "total_folds": len(fold_results),
                        "consistency": positive_folds / len(fold_results),
                    })
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}")
            continue
    
    if not all_results:
        print("\n❌ No results collected")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # ============================================
    # ANALYSIS
    # ============================================
    
    print("\n" + "=" * 80)
    print("📈 WALK-FORWARD RESULTS ANALYSIS")
    print("=" * 80)
    
    # 1. Configuration Rankings (across all tickers)
    print("\n" + "-" * 60)
    print("🏆 CONFIGURATION RANKINGS (TRUE OUT-OF-SAMPLE)")
    print("-" * 60)
    
    config_summary = results_df.groupby('config').agg({
        'avg_sharpe': ['mean', 'std'],
        'avg_accuracy': 'mean',
        'avg_return': 'mean',
        'avg_max_dd': 'mean',
        'consistency': 'mean',
        'positive_folds': 'sum',
        'total_folds': 'sum',
    }).round(3)
    
    config_summary.columns = ['sharpe_mean', 'sharpe_std', 'accuracy', 'return', 'max_dd', 'consistency', 'pos_folds', 'total_folds']
    config_summary['overall_consistency'] = config_summary['pos_folds'] / config_summary['total_folds']
    config_summary = config_summary.sort_values('sharpe_mean', ascending=False)
    
    for i, (config, row) in enumerate(config_summary.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        
        print(f"\n{medal} {config}")
        print(f"   Avg Sharpe: {row['sharpe_mean']:+.3f} (±{row['sharpe_std']:.2f})")
        print(f"   Direction Accuracy: {row['accuracy']:.1%}")
        print(f"   Avg Return per fold: {row['return']*100:+.1f}%")
        print(f"   Avg Max DD: {row['max_dd']*100:.1f}%")
        print(f"   Consistency: {row['overall_consistency']:.0%} profitable folds ({int(row['pos_folds'])}/{int(row['total_folds'])})")
    
    # 2. Compare enhanced vs baseline
    print("\n" + "-" * 60)
    print("📊 ENHANCEMENT ANALYSIS: Does it help?")
    print("-" * 60)
    
    xgb_baseline = config_summary.loc["XGB_Baseline", "sharpe_mean"] if "XGB_Baseline" in config_summary.index else 0
    
    enhancements = []
    for config in config_summary.index:
        if config != "XGB_Baseline" and "XGB" in config:
            improvement = config_summary.loc[config, "sharpe_mean"] - xgb_baseline
            enhancements.append((config, improvement, config_summary.loc[config, "sharpe_mean"]))
    
    enhancements.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBaseline XGB Sharpe: {xgb_baseline:+.3f}")
    print("\nImprovements over baseline:")
    for config, improvement, sharpe in enhancements:
        symbol = "📈" if improvement > 0.1 else "➖" if improvement > -0.1 else "📉"
        print(f"  {symbol} {config}: {improvement:+.3f} (total: {sharpe:+.3f})")
    
    # 3. Best config per ticker
    print("\n" + "-" * 60)
    print("🎯 BEST CONFIGURATION PER TICKER (Walk-Forward)")
    print("-" * 60)
    
    for ticker in test_tickers:
        ticker_data = results_df[results_df['ticker'] == ticker]
        if not ticker_data.empty:
            best_idx = ticker_data['avg_sharpe'].idxmax()
            best = ticker_data.loc[best_idx]
            status = "✅" if best['avg_sharpe'] > 0.5 else "⚠️" if best['avg_sharpe'] > 0 else "❌"
            print(f"  {status} {ticker}: {best['config']}")
            print(f"      Sharpe: {best['avg_sharpe']:+.2f}, Acc: {best['avg_accuracy']:.1%}, Consistency: {best['consistency']:.0%}")
    
    # 4. Statistical significance
    print("\n" + "-" * 60)
    print("📏 STATISTICAL SIGNIFICANCE")
    print("-" * 60)
    
    best_config = config_summary.index[0]
    best_sharpe = config_summary.loc[best_config, "sharpe_mean"]
    best_std = config_summary.loc[best_config, "sharpe_std"]
    n_samples = config_summary.loc[best_config, "total_folds"]
    
    # T-statistic for Sharpe > 0
    if best_std > 0 and n_samples > 1:
        t_stat = best_sharpe / (best_std / np.sqrt(n_samples))
        print(f"\nBest config: {best_config}")
        print(f"  Sharpe: {best_sharpe:+.3f} ± {best_std:.3f}")
        print(f"  T-statistic (Sharpe > 0): {t_stat:.2f}")
        print(f"  Significant at 95%: {'Yes ✅' if t_stat > 1.96 else 'No ❌'}")
        print(f"  Significant at 99%: {'Yes ✅' if t_stat > 2.58 else 'No ❌'}")
    
    # 5. Final Recommendations
    print("\n" + "=" * 80)
    print("📋 FINAL RECOMMENDATIONS (Based on True Walk-Forward)")
    print("=" * 80)
    
    # Find truly best config considering both return and stability
    config_summary['score'] = config_summary['sharpe_mean'] * config_summary['overall_consistency']
    best_overall = config_summary['score'].idxmax()
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  🥇 BEST OVERALL: {best_overall:<49} │
│     Walk-Forward Sharpe: {config_summary.loc[best_overall, 'sharpe_mean']:+.3f}                                  │
│     Direction Accuracy: {config_summary.loc[best_overall, 'accuracy']:.1%}                                    │
│     Consistency: {config_summary.loc[best_overall, 'overall_consistency']:.0%} profitable folds                                │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Check if enhancements actually help
    xgb_enhanced = config_summary.loc[config_summary.index.str.contains("All Enhancements"), "sharpe_mean"].values[0] if any(config_summary.index.str.contains("All Enhancements")) else 0
    
    print("\n🔍 ENHANCEMENT VERDICT:")
    if xgb_enhanced > xgb_baseline + 0.1:
        print("  ✅ ElasticNet + Scaler + Adaptive LR IMPROVES performance")
        print(f"     Improvement: {xgb_enhanced - xgb_baseline:+.3f} Sharpe")
    elif xgb_enhanced > xgb_baseline - 0.1:
        print("  ➖ Enhancements have MINIMAL effect")
        print("     Stick with simpler baseline for robustness")
    else:
        print("  ❌ Enhancements HURT performance")
        print("     Use baseline XGB model")
    
    # Save results
    results_df.to_csv("walkforward_results.csv", index=False)
    config_summary.to_csv("walkforward_config_summary.csv")
    print(f"\n✅ Results saved to walkforward_results.csv")
    print(f"✅ Summary saved to walkforward_config_summary.csv")
    
    return results_df, config_summary


if __name__ == "__main__":
    results, summary = run_true_walkforward()
