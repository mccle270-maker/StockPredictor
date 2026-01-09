"""
Out-of-Sample Validation: Final Comparison
===========================================

Tests 4 models on truly out-of-sample holdout data (2024-07-01 to 2025-12-31):

1. Current Production XGB v4 (what you're using now)
2. Heavy Regularized XGB (Config C - the new candidate)
3. Random Forest (default sklearn params)
4. Random Forest Regularized (with constraints)

Training: 2022-01-01 to 2024-06-30
Holdout:  2024-07-01 to 2025-12-31 (NEVER seen during training)
"""

import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prediction_model import build_features_and_target
from src.config import get_model_config
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

print('='*90)
print('FINAL OUT-OF-SAMPLE MODEL COMPARISON')
print('='*90)
print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print()

# ============================================================================
# MODEL CONFIGS
# ============================================================================

# Model 1: Current Production XGB v4 (what you're using)
prod_params = get_model_config('xgb')
MODEL_PROD_XGB = {
    'name': 'Prod XGB v4',
    'model_class': XGBRegressor,
    'params': prod_params,
    'features': None,  # All features
}

# Model 2: Heavy Regularized XGB (Config C - the winner from previous test)
MODEL_HEAVY_XGB = {
    'name': 'Heavy Reg XGB',
    'model_class': XGBRegressor,
    'params': {
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'min_child_weight': 50,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
        'gamma': 0.5,
        'random_state': 42,
        'n_jobs': -1,
    },
    'features': [
        'ret_5d', 'vol_20d', 'rsi14', 'macd', 'atr_14',
        'momentum', 'ret_1d', 'vol_10d', 'bb_width', 'adx_14'
    ],
}

# Model 3: Random Forest (default params - likely overfit)
MODEL_RF_DEFAULT = {
    'name': 'RF Default',
    'model_class': RandomForestRegressor,
    'params': {
        'n_estimators': 100,
        'max_depth': None,  # No limit - will overfit
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1,
    },
    'features': None,  # All features
}

# Model 4: Random Forest Regularized
MODEL_RF_REG = {
    'name': 'RF Regularized',
    'model_class': RandomForestRegressor,
    'params': {
        'n_estimators': 50,
        'max_depth': 5,           # Limited depth
        'min_samples_split': 50,  # Need many samples to split
        'min_samples_leaf': 25,   # Need many samples in leaf
        'max_features': 0.3,      # Only use 30% of features per tree
        'random_state': 42,
        'n_jobs': -1,
    },
    'features': [
        'ret_5d', 'vol_20d', 'rsi14', 'macd', 'atr_14',
        'momentum', 'ret_1d', 'vol_10d', 'bb_width', 'adx_14'
    ],
}

# Test setup
TICKERS = ['AAPL', 'MSFT', 'AMZN']
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-12-31'

ALL_MODELS = [MODEL_PROD_XGB, MODEL_HEAVY_XGB, MODEL_RF_DEFAULT, MODEL_RF_REG]


def calculate_sharpe(returns, annualize=True):
    """Calculate Sharpe ratio from returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(returns) / np.std(returns)
    if annualize:
        sharpe *= np.sqrt(252)
    return sharpe


def calculate_max_drawdown(cum_returns):
    """Calculate maximum drawdown."""
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    return float(np.min(drawdown)) if len(drawdown) > 0 else 0.0


def run_validation(model_config, tickers, train_start, train_end, test_start, test_end, verbose=False):
    """Run validation for a model config."""
    all_train_returns = []
    all_test_returns = []
    all_train_correct = 0
    all_train_total = 0
    all_test_correct = 0
    all_test_total = 0
    
    for ticker in tickers:
        try:
            # Get full data
            result = build_features_and_target(ticker, period='5y', horizon=5)
            X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates = result
            
            if X is None or len(X) == 0:
                if verbose:
                    print(f'  {ticker}: Skipped (no data)')
                continue
            
            # Get feature columns
            all_possible_cols = [
                'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'vol_10d', 'vol_20d', 'vol_60d',
                'ma_50', 'ma_200', 'rsi14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_lower', 'bb_width', 'atr_14', 'adx_14', 'obv', 'vwap',
                'cci', 'williams_r', 'stoch_k', 'stoch_d', 'momentum', 'roc', 'mfi',
                'gbm_mu_60d', 'gbm_sig_60d', 'gbm_prob_up_1d', 'gbm_exp_ret_1d',
                'gbm_p05_ret_1d', 'gbm_p95_ret_1d', 'gbm_prob_up_5d', 'gbm_exp_ret_5d',
                'gbm_p05_ret_5d', 'gbm_p95_ret_5d',
            ]
            
            # Create DataFrame with features
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            feature_cols = all_possible_cols[:n_features] if n_features <= len(all_possible_cols) else [f'feat_{i}' for i in range(n_features)]
            
            df = pd.DataFrame(X, index=dates, columns=feature_cols)
            df['target'] = y
            
            # Filter by date
            df.index = pd.to_datetime(df.index)
            
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            test_mask = (df.index >= test_start) & (df.index <= test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(train_df) < 50 or len(test_df) < 20:
                if verbose:
                    print(f'  {ticker}: Skipped (insufficient data)')
                continue
            
            # Select features
            if model_config['features']:
                use_features = [f for f in model_config['features'] if f in feature_cols]
                if len(use_features) < 3:
                    use_features = feature_cols
            else:
                use_features = feature_cols
            
            X_train = train_df[use_features].values
            y_train = train_df['target'].values
            X_test = test_df[use_features].values
            y_test = test_df['target'].values
            
            # Handle NaNs
            train_valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
            test_valid = ~(np.isnan(X_test).any(axis=1) | np.isnan(y_test))
            
            X_train = X_train[train_valid]
            y_train = y_train[train_valid]
            X_test = X_test[test_valid]
            y_test = y_test[test_valid]
            
            if len(X_train) < 50 or len(X_test) < 20:
                continue
            
            # Train model
            model = model_config['model_class'](**model_config['params'])
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Strategy returns
            train_returns = np.sign(train_pred) * y_train
            test_returns = np.sign(test_pred) * y_test
            
            all_train_returns.extend(train_returns)
            all_test_returns.extend(test_returns)
            
            # Accuracy
            train_correct = np.sum((train_pred > 0) == (y_train > 0))
            test_correct = np.sum((test_pred > 0) == (y_test > 0))
            
            all_train_correct += train_correct
            all_train_total += len(y_train)
            all_test_correct += test_correct
            all_test_total += len(y_test)
            
            if verbose:
                print(f'  {ticker}: train={len(X_train)}, test={len(X_test)}, features={len(use_features)}')
            
        except Exception as e:
            if verbose:
                print(f'  {ticker}: Error - {str(e)[:50]}')
            continue
    
    if len(all_train_returns) == 0 or len(all_test_returns) == 0:
        return None
    
    train_returns = np.array(all_train_returns)
    test_returns = np.array(all_test_returns)
    
    # Calculate metrics
    train_sharpe = calculate_sharpe(train_returns)
    test_sharpe = calculate_sharpe(test_returns)
    
    train_acc = all_train_correct / all_train_total * 100 if all_train_total > 0 else 0
    test_acc = all_test_correct / all_test_total * 100 if all_test_total > 0 else 0
    
    train_cum = np.cumprod(1 + train_returns)
    test_cum = np.cumprod(1 + test_returns)
    
    train_dd = calculate_max_drawdown(train_cum) * 100
    test_dd = calculate_max_drawdown(test_cum) * 100
    
    train_winrate = np.mean(train_returns > 0) * 100
    test_winrate = np.mean(test_returns > 0) * 100
    
    # Total return
    train_total_ret = (train_cum[-1] - 1) * 100 if len(train_cum) > 0 else 0
    test_total_ret = (test_cum[-1] - 1) * 100 if len(test_cum) > 0 else 0
    
    return {
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_dd': train_dd,
        'test_dd': test_dd,
        'train_winrate': train_winrate,
        'test_winrate': test_winrate,
        'train_samples': len(train_returns),
        'test_samples': len(test_returns),
        'sharpe_gap': train_sharpe - test_sharpe,
        'acc_gap': train_acc - test_acc,
        'train_total_ret': train_total_ret,
        'test_total_ret': test_total_ret,
    }


if __name__ == "__main__":
    print(f'Tickers: {TICKERS}')
    print(f'Training: {TRAIN_START} to {TRAIN_END} (2.5 years)')
    print(f'Holdout:  {TEST_START} to {TEST_END} (1.5 years - UNSEEN)')
    print()
    
    # Run all models
    results = {}
    for model_config in ALL_MODELS:
        print(f'Testing: {model_config["name"]}...')
        results[model_config['name']] = run_validation(
            model_config, TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END, verbose=True
        )
        print()
    
    # Print comparison table
    print()
    print('='*100)
    print('COMPARISON TABLE')
    print('='*100)
    print()
    
    # Header
    header = f'{"Metric":<20}'
    for model in ALL_MODELS:
        header += f'{model["name"]:<20}'
    print(header)
    print('-'*100)
    
    # Metrics
    metrics = [
        ('Train Sharpe', 'train_sharpe', '{:.2f}'),
        ('TEST SHARPE', 'test_sharpe', '{:.2f}'),
        ('Sharpe Gap', 'sharpe_gap', '{:.2f}'),
        ('Train Accuracy', 'train_acc', '{:.1f}%'),
        ('TEST ACCURACY', 'test_acc', '{:.1f}%'),
        ('Accuracy Gap', 'acc_gap', '{:.1f}%'),
        ('Test Max DD', 'test_dd', '{:.1f}%'),
        ('Test Win Rate', 'test_winrate', '{:.1f}%'),
        ('Test Total Return', 'test_total_ret', '{:.1f}%'),
    ]
    
    for name, key, fmt in metrics:
        row = f'{name:<20}'
        for model in ALL_MODELS:
            r = results.get(model['name'])
            if r:
                row += f'{fmt.format(r[key]):<20}'
            else:
                row += f'{"N/A":<20}'
        print(row)
    
    # Highlight key metrics
    print()
    print('='*100)
    print('KEY METRICS SUMMARY (TEST DATA ONLY)')
    print('='*100)
    print()
    
    # Sort by test Sharpe
    sorted_models = sorted(
        [(m['name'], results.get(m['name'])) for m in ALL_MODELS if results.get(m['name'])],
        key=lambda x: x[1]['test_sharpe'],
        reverse=True
    )
    
    print(f'{"Rank":<6} {"Model":<20} {"Test Sharpe":<15} {"Test Acc":<12} {"Sharpe Gap":<12} {"Verdict":<20}')
    print('-'*85)
    
    for i, (name, r) in enumerate(sorted_models, 1):
        ts = r['test_sharpe']
        gap = r['sharpe_gap']
        
        if ts > 1.0 and gap < 2:
            verdict = '✅ EXCELLENT'
        elif ts > 0.5 and gap < 5:
            verdict = '✅ GOOD'
        elif ts > 0 and gap < 10:
            verdict = '⚠️ MODERATE'
        elif ts > 0:
            verdict = '⚠️ WEAK (overfit)'
        else:
            verdict = '❌ POOR'
        
        print(f'{i:<6} {name:<20} {ts:<15.2f} {r["test_acc"]:<12.1f}% {gap:<12.2f} {verdict:<20}')
    
    print()
    print('='*100)
    print('RECOMMENDATION')
    print('='*100)
    print()
    
    # Find best
    if sorted_models:
        best_name, best_r = sorted_models[0]
        print(f'🏆 BEST MODEL: {best_name}')
        print(f'   Test Sharpe: {best_r["test_sharpe"]:.2f}')
        print(f'   Test Accuracy: {best_r["test_acc"]:.1f}%')
        print(f'   Sharpe Gap: {best_r["sharpe_gap"]:.2f}')
        print(f'   Test Total Return: {best_r["test_total_ret"]:.1f}%')
        print()
        
        # Compare to production
        prod_r = results.get('Prod XGB v4')
        if prod_r and best_name != 'Prod XGB v4':
            improvement = best_r['test_sharpe'] - prod_r['test_sharpe']
            print(f'📈 Improvement over Production: +{improvement:.2f} Sharpe')
            print()
            
            if improvement > 0.5:
                print('   ✅ STRONG IMPROVEMENT - Consider switching to this model')
            elif improvement > 0:
                print('   ✅ MODERATE IMPROVEMENT - Worth considering')
            else:
                print('   ⚠️ No improvement - Keep current production model')
        
        print()
        print('='*100)
        print('NEXT STEPS')
        print('='*100)
        
        if best_r['test_sharpe'] > 0.5 and best_r['sharpe_gap'] < 5:
            print(f'The {best_name} model is ready for production.')
            print('To implement, I can add it as a new config option in src/config.py')
            print('Your current production model will remain unchanged until you switch.')
        else:
            print('All models need improvement. Consider:')
            print('  1. More data')
            print('  2. Different feature sets')
            print('  3. Walk-forward retraining')
            print('  4. Ensemble methods')
