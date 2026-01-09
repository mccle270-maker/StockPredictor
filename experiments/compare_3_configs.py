"""
Out-of-Sample Validation: Compare 3 Configs
============================================

Config A: Optuna Optimized (450 trees, max_depth=7) - Your requested config
Config B: Production Regularized v4 (current production)
Config C: Heavy Regularization (new experimental config)

Tests all three on truly out-of-sample holdout data (2024-07-01 to 2025-12-31)
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

print('='*80)
print('OUT-OF-SAMPLE VALIDATION: 3 CONFIG COMPARISON')
print('='*80)
print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print()

# ============================================================================
# CONFIG A: Optuna optimized config (your requested params)
# ============================================================================
CONFIG_A = {
    'name': 'A: Optuna (450 trees, depth=7)',
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
    'name': 'B: Prod v4 (150 trees, depth=4)',
    'features': None,  # Use all features
    'params': prod_params,
    'temp_scale': 1.0,
}

# ============================================================================
# CONFIG C: HEAVILY REGULARIZED (new experimental)
# ============================================================================
CONFIG_C = {
    'name': 'C: Heavy Reg (50 trees, depth=3)',
    'features': [
        # Only 10 most robust features
        'ret_5d', 'vol_20d', 'rsi14', 'macd', 'atr_14',
        'momentum', 'ret_1d', 'vol_10d', 'bb_width', 'adx_14'
    ],
    'params': {
        'n_estimators': 50,           # Very few trees
        'max_depth': 3,               # Very shallow
        'learning_rate': 0.01,        # Very slow learning
        'subsample': 0.5,             # Heavy row sampling
        'colsample_bytree': 0.5,      # Heavy column sampling
        'min_child_weight': 50,       # Require many samples per leaf
        'reg_alpha': 1.0,             # Strong L1 regularization
        'reg_lambda': 10.0,           # Strong L2 regularization
        'gamma': 0.5,                 # Min loss reduction for split
        'random_state': 42,
        'n_jobs': -1,
    },
    'temp_scale': 1.0,
}

# ============================================================================
# CONFIG D: ULTRA CONSERVATIVE (barely above baseline)
# ============================================================================
CONFIG_D = {
    'name': 'D: Ultra Conservative (25 trees, depth=2)',
    'features': [
        # Only 5 most fundamental features
        'ret_5d', 'vol_20d', 'momentum', 'rsi14', 'macd'
    ],
    'params': {
        'n_estimators': 25,           # Minimal trees
        'max_depth': 2,               # Very shallow (like decision stump)
        'learning_rate': 0.005,       # Extremely slow learning
        'subsample': 0.4,             # Very heavy row sampling
        'colsample_bytree': 0.4,      # Very heavy column sampling
        'min_child_weight': 100,      # Require LOTS of samples per leaf
        'reg_alpha': 2.0,             # Very strong L1
        'reg_lambda': 20.0,           # Very strong L2
        'gamma': 1.0,                 # High min loss reduction
        'random_state': 42,
        'n_jobs': -1,
    },
    'temp_scale': 1.0,
}

# Test setup
TICKERS = ['AAPL', 'MSFT', 'AMZN']
TRAIN_START = '2022-01-01'
TRAIN_END = '2024-06-30'
TEST_START = '2024-07-01'
TEST_END = '2025-12-31'

ALL_CONFIGS = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D]


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


def run_validation(config, tickers, train_start, train_end, test_start, test_end, verbose=False):
    """Run validation for a config."""
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
            if config['features']:
                use_features = [f for f in config['features'] if f in feature_cols]
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
    }


if __name__ == "__main__":
    print(f'Tickers: {TICKERS}')
    print(f'Training: {TRAIN_START} to {TRAIN_END}')
    print(f'Holdout:  {TEST_START} to {TEST_END}')
    print()
    
    # Run all configs
    results = {}
    for config in ALL_CONFIGS:
        print(f'Testing: {config["name"]}...')
        results[config['name']] = run_validation(
            config, TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END, verbose=True
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
    for config in ALL_CONFIGS:
        header += f'{config["name"][:25]:<27}'
    print(header)
    print('-'*100)
    
    # Metrics
    metrics = [
        ('Train Sharpe', 'train_sharpe', '{:.2f}'),
        ('Test Sharpe', 'test_sharpe', '{:.2f}'),
        ('Sharpe Gap', 'sharpe_gap', '{:.2f}'),
        ('Train Accuracy', 'train_acc', '{:.1f}%'),
        ('Test Accuracy', 'test_acc', '{:.1f}%'),
        ('Accuracy Gap', 'acc_gap', '{:.1f}%'),
        ('Test Max DD', 'test_dd', '{:.1f}%'),
        ('Test Win Rate', 'test_winrate', '{:.1f}%'),
    ]
    
    for name, key, fmt in metrics:
        row = f'{name:<20}'
        for config in ALL_CONFIGS:
            r = results.get(config['name'])
            if r:
                row += f'{fmt.format(r[key]):<27}'
            else:
                row += f'{"N/A":<27}'
        print(row)
    
    print()
    print('='*100)
    print('ANALYSIS')
    print('='*100)
    
    # Find best by test Sharpe
    best_test_sharpe = None
    best_config = None
    least_overfit = None
    least_overfit_gap = float('inf')
    
    for config in ALL_CONFIGS:
        r = results.get(config['name'])
        if r:
            if best_test_sharpe is None or r['test_sharpe'] > best_test_sharpe:
                best_test_sharpe = r['test_sharpe']
                best_config = config['name']
            if r['sharpe_gap'] < least_overfit_gap:
                least_overfit_gap = r['sharpe_gap']
                least_overfit = config['name']
    
    print(f'📈 Best Test Sharpe: {best_config} ({best_test_sharpe:.2f})')
    print(f'🎯 Least Overfitting: {least_overfit} (gap: {least_overfit_gap:.2f})')
    print()
    
    # Verdicts
    print('VERDICTS:')
    for config in ALL_CONFIGS:
        r = results.get(config['name'])
        if r:
            ts = r['test_sharpe']
            gap = r['sharpe_gap']
            
            if ts > 1.5:
                verdict = '✅ EXCELLENT'
            elif ts > 1.0:
                verdict = '✅ GOOD'
            elif ts > 0.5:
                verdict = '⚠️ MODERATE'
            elif ts > 0:
                verdict = '⚠️ WEAK'
            else:
                verdict = '❌ POOR'
            
            if gap > 10:
                overfit = 'SEVERE OVERFIT'
            elif gap > 5:
                overfit = 'MODERATE OVERFIT'
            elif gap > 2:
                overfit = 'MILD OVERFIT'
            else:
                overfit = 'STABLE'
            
            print(f'  {config["name"]}: {verdict} | {overfit}')
    
    print()
    print('='*100)
    print('RECOMMENDATION')
    print('='*100)
    
    # Find best balance of performance and stability
    best_balance = None
    best_score = -float('inf')
    
    for config in ALL_CONFIGS:
        r = results.get(config['name'])
        if r:
            # Score = test_sharpe - penalty for overfitting
            # Lower gap is better, so subtract gap/10
            score = r['test_sharpe'] - (r['sharpe_gap'] / 10)
            if score > best_score:
                best_score = score
                best_balance = config['name']
    
    print(f'🏆 Best Balance (Test Sharpe - Overfit Penalty): {best_balance}')
    print()
    
    if best_balance:
        r = results.get(best_balance)
        print(f'   Test Sharpe: {r["test_sharpe"]:.2f}')
        print(f'   Test Accuracy: {r["test_acc"]:.1f}%')
        print(f'   Sharpe Gap: {r["sharpe_gap"]:.2f}')
        print()
        
        if r['test_sharpe'] > 0.5 and r['sharpe_gap'] < 5:
            print('   ✅ This config is ready for production use')
        elif r['test_sharpe'] > 0 and r['sharpe_gap'] < 10:
            print('   ⚠️ This config needs more testing before production')
        else:
            print('   ❌ All configs need improvement - consider:')
            print('      1. More regularization')
            print('      2. Fewer features')
            print('      3. Shorter training window')
            print('      4. Walk-forward retraining')
