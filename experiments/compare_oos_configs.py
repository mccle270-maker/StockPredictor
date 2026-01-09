"""
Out-of-Sample Validation: Compare Optuna Optimized vs Production v4
====================================================================

Compares two XGBoost configurations on truly out-of-sample data:
- Config A: Optuna optimized (450 trees, 20 features, T=2.9)
- Config B: Production regularized v4 (150 trees, all features)
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

print('='*70)
print('OUT-OF-SAMPLE VALIDATION COMPARISON')
print('='*70)
print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print()

# ============================================================================
# CONFIG A: Optuna optimized config
# ============================================================================
CONFIG_A = {
    'name': 'Optuna Optimized (450 trees, 20 features)',
    'features': [
        'gbm_exp_ret_5d', 'gbm_prob_up_5d', 'ret_5d', 'gbm_exp_ret_1d',
        'vol_20d', 'rsi14', 'gbm_prob_up_1d', 'macd', 'atr_14', 'adx_14',
        'ret_1d', 'ret_10d', 'vol_10d', 'obv', 'momentum', 'williams_r',
        'cci', 'stoch_k', 'bb_width', 'mfi'
    ],
    'params': {
        'n_estimators': 450,
        'max_depth': 7,
        'learning_rate': 0.048,
        'subsample': 0.998,
        'colsample_bytree': 0.674,
        'min_child_weight': 19,
        'reg_alpha': 0.012,
        'reg_lambda': 9.296,
        'random_state': 42,
        'n_jobs': -1,
    },
    'temp_scale': 2.9,
}

# ============================================================================
# CONFIG B: Current production config (regularized v4)
# ============================================================================
prod_params = get_model_config('xgb')
CONFIG_B = {
    'name': 'Production Regularized v4 (150 trees, all features)',
    'features': None,  # Use all features
    'params': prod_params,
    'temp_scale': 1.0,
}

# Test setup
TICKERS = ['AAPL', 'MSFT', 'AMZN']
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-12-31'


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


def run_validation(config, tickers, train_start, train_end, test_start, test_end):
    """Run validation for a config."""
    all_train_returns = []
    all_test_returns = []
    all_train_correct = 0
    all_train_total = 0
    all_test_correct = 0
    all_test_total = 0
    
    for ticker in tickers:
        try:
            # Get full data - returns 7 values: X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates
            result = build_features_and_target(ticker, period='5y', horizon=5)
            X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates = result
            
            if X is None or len(X) == 0:
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
                print(f'  {ticker}: Skipped (insufficient data - train={len(train_df)}, test={len(test_df)})')
                continue
            
            # Select features
            if config['features']:
                use_features = [f for f in config['features'] if f in feature_cols]
                if len(use_features) < 5:
                    # Fall back to all available if not enough requested features exist
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
            model = XGBRegressor(**config['params'])
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Apply temperature scaling
            if config['temp_scale'] != 1.0:
                test_pred = test_pred / config['temp_scale']
            
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
            
            print(f'  {ticker}: train={len(X_train)}, test={len(X_test)}, features={len(use_features)}')
            
        except Exception as e:
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
    }


if __name__ == "__main__":
    # Run both configs
    print(f'Tickers: {TICKERS}')
    print(f'Training: {TRAIN_START} to {TRAIN_END}')
    print(f'Holdout:  {TEST_START} to {TEST_END}')
    print()

    print('='*70)
    print(f'CONFIG A: {CONFIG_A["name"]}')
    print('='*70)
    results_a = run_validation(CONFIG_A, TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END)

    print()
    print('='*70)
    print(f'CONFIG B: {CONFIG_B["name"]}')
    print('='*70)
    results_b = run_validation(CONFIG_B, TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END)

    # Print comparison table
    print()
    print('='*70)
    print('COMPARISON TABLE')
    print('='*70)
    print()
    print(f'{"Metric":<20} {"Config A (Optuna)":<25} {"Config B (Prod v4)":<25} {"Winner":<10}')
    print('-'*80)

    if results_a and results_b:
        metrics = [
            ('Train Sharpe', 'train_sharpe', '{:.2f}', True),
            ('Test Sharpe', 'test_sharpe', '{:.2f}', True),
            ('Train Accuracy', 'train_acc', '{:.1f}%', True),
            ('Test Accuracy', 'test_acc', '{:.1f}%', True),
            ('Train Max DD', 'train_dd', '{:.1f}%', False),
            ('Test Max DD', 'test_dd', '{:.1f}%', False),
            ('Train Win Rate', 'train_winrate', '{:.1f}%', True),
            ('Test Win Rate', 'test_winrate', '{:.1f}%', True),
        ]
        
        for name, key, fmt, higher_better in metrics:
            val_a = results_a[key]
            val_b = results_b[key]
            
            if higher_better:
                winner = 'A' if val_a > val_b else 'B' if val_b > val_a else 'TIE'
            else:
                # For DD, less negative is better
                winner = 'A' if val_a > val_b else 'B' if val_b > val_a else 'TIE'
            
            str_a = fmt.format(val_a)
            str_b = fmt.format(val_b)
            
            print(f'{name:<20} {str_a:<25} {str_b:<25} {winner:<10}')
        
        print()
        print('='*70)
        print('OVERFITTING ANALYSIS')
        print('='*70)
        
        gap_a = results_a['train_sharpe'] - results_a['test_sharpe']
        gap_b = results_b['train_sharpe'] - results_b['test_sharpe']
        
        print(f'Config A Sharpe Gap (train - test): {gap_a:.2f}')
        print(f'Config B Sharpe Gap (train - test): {gap_b:.2f}')
        print()
        print(f'Config A Accuracy Gap: {results_a["train_acc"] - results_a["test_acc"]:.1f}%')
        print(f'Config B Accuracy Gap: {results_b["train_acc"] - results_b["test_acc"]:.1f}%')
        
        print()
        print('='*70)
        print('VERDICT')
        print('='*70)
        
        # Determine winner based on TEST performance (what matters!)
        test_sharpe_a = results_a['test_sharpe']
        test_sharpe_b = results_b['test_sharpe']
        
        if test_sharpe_a > test_sharpe_b:
            better = 'CONFIG A (Optuna Optimized)'
            better_sharpe = test_sharpe_a
        else:
            better = 'CONFIG B (Production v4)'
            better_sharpe = test_sharpe_b
        
        print(f'🏆 WINNER: {better}')
        print(f'   Test Sharpe: {better_sharpe:.2f}')
        print()
        
        if better_sharpe > 1.5:
            print('✅ EXCELLENT - Deploy immediately')
        elif better_sharpe > 1.0:
            print('✅ GOOD - Deploy with monitoring')
        elif better_sharpe > 0.5:
            print('⚠️ CAUTION - Some overfitting, tighten risk limits')
        else:
            print('❌ WARNING - Significant overfitting detected')
        
        print()
        print('='*70)
        print('RECOMMENDATION')
        print('='*70)
        
        # Compare overfitting
        if gap_a < gap_b:
            less_overfit = 'A'
        else:
            less_overfit = 'B'
        
        if test_sharpe_a > test_sharpe_b and gap_a < gap_b:
            print('Config A wins on BOTH test performance AND less overfitting.')
            print('→ Recommend switching to Optuna optimized config.')
        elif test_sharpe_b > test_sharpe_a and gap_b < gap_a:
            print('Config B wins on BOTH test performance AND less overfitting.')
            print('→ Recommend keeping Production v4 config.')
        elif test_sharpe_a > test_sharpe_b:
            print('Config A has better test performance but more overfitting.')
            print('→ Consider using Config A with additional regularization.')
        else:
            print('Config B has better test performance but more overfitting.')
            print('→ Keep Production v4 but monitor for degradation.')

    else:
        print('ERROR: One or both configs failed to produce results')
