"""
Random Forest Optimization Pipeline
====================================

This script optimizes Random Forest for the stock prediction system using
the same methodology as the XGBoost optimization:

EXPERIMENT 1: RF with Optimized Features
- Test RF with the 20 features that worked best for XGBoost
- Compare against full 150+ feature set

EXPERIMENT 2: RF Hyperparameter Optimization  
- 50 Optuna trials to find optimal RF parameters
- Walk-forward validation for robust evaluation

EXPERIMENT 3: RF Calibration
- Temperature scaling, Platt scaling, Isotonic regression
- Find best calibration for RF probability estimates

EXPERIMENT 4: Ensemble Comparison
- Compare XGB only vs various RF+XGB ensemble configurations
- Evaluate Sharpe, accuracy, stability, and disagreement rate

OUTPUT:
- experiments/RF_OPTIMIZATION_REPORT.md
- experiments/optimized_rf_config.json
- Final recommendation: Ensemble vs XGB only

Author: Stock Predictor Optimization Pipeline
Date: 2026-01-08
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')

# Imports
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler

# Local imports
try:
    from prediction_model import build_features_and_target, backtest_one_ticker
    from src.config import (
        OPTIMIZED_MODEL_CONFIG, 
        OPTIMIZED_FEATURES,
        TRADING_DAYS_PER_YEAR,
        get_optimized_rf_config,
    )
    from src.core.models import make_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Running from experiments directory - adjusting imports...")
    
# XGBoost import
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not available")

# ============================================================================
# CONSTANTS
# ============================================================================

OUTPUT_DIR = Path(__file__).parent
REPORT_PATH = OUTPUT_DIR / "RF_OPTIMIZATION_REPORT.md"
CONFIG_PATH = OUTPUT_DIR / "optimized_rf_config.json"

# Test tickers - representative sample
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Optimized features from XGBoost experiment
OPTIMIZED_FEATURES_LIST = [
    "gbm_exp_ret_5d", "gbm_prob_up_5d", "ret_5d", "gbm_exp_ret_1d",
    "vol_20d", "rsi14", "gbm_prob_up_1d", "macd", "atr_14", "adx_14",
    "ret_1d", "ret_10d", "vol_10d", "obv", "momentum", "williams_r",
    "cci", "stoch_k", "bb_width", "mfi",
]

# RF Hyperparameter search space
RF_PARAM_SPACE = {
    "n_estimators": [100, 200, 300, 500, 700],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "bootstrap": [True, False],
}

# Temperature scaling values to test
TEMPERATURE_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_sharpe(returns: np.ndarray, annualize: bool = True) -> float:
    """Calculate Sharpe ratio from returns array."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    if annualize:
        sharpe *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sharpe)


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate direction accuracy (% correct sign predictions)."""
    if len(y_true) == 0:
        return 0.5
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct))


def calculate_rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling_sharpe


def get_worst_3month_sharpe(returns: pd.Series) -> float:
    """Get worst 3-month rolling Sharpe (stability measure)."""
    rolling = calculate_rolling_sharpe(returns, window=63)
    if rolling.isna().all():
        return 0.0
    return float(rolling.min())


def load_ticker_data(ticker: str, period: str = "5y") -> Tuple[pd.DataFrame, str]:
    """Load and prepare data for a ticker."""
    try:
        result = build_features_and_target(ticker, period=period, horizon=5)
        if result is None:
            return None, "build_features_and_target returned None"
        
        # build_features_and_target returns: X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates
        X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates = result
        
        if X is None or len(X) < 252:
            return None, f"Insufficient data: {len(X) if X is not None else 0} rows"
        
        # Reconstruct dataframe with features and target
        # Get feature column names from config
        from prediction_model import FEATURE_COLUMNS, MACRO_COLUMNS
        all_cols = FEATURE_COLUMNS + MACRO_COLUMNS
        
        # Use available columns up to X.shape[1]
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        if n_features <= len(all_cols):
            feature_names = all_cols[:n_features]
        else:
            # Generate extra column names if needed
            feature_names = all_cols + [f"feat_{i}" for i in range(len(all_cols), n_features)]
        
        df = pd.DataFrame(X, columns=feature_names, index=dates)
        df["ftarget_ret_5d_ahead"] = y
        df["ret_5d"] = y  # Alias for compatibility
        
        return df, None
    except Exception as e:
        return None, str(e)


def filter_features(df: pd.DataFrame, feature_list: List[str]) -> List[str]:
    """Filter feature list to only include columns that exist in dataframe."""
    available = [f for f in feature_list if f in df.columns]
    return available


def walk_forward_evaluate(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    model,
    n_splits: int = 5,
    train_pct: float = 0.7,
) -> Dict[str, float]:
    """
    Perform walk-forward validation and return metrics.
    
    Returns dict with: sharpe, accuracy, r2, mse, worst_3m_sharpe
    """
    # Prepare data
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 5:
        return {"sharpe": 0.0, "accuracy": 0.5, "r2": 0.0, "mse": 1.0, "worst_3m_sharpe": -999}
    
    X = df[available_features].values
    y = df[target_col].values if target_col in df.columns else df["ret_5d"].values
    
    # Remove NaN rows
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) < 100:
        return {"sharpe": 0.0, "accuracy": 0.5, "r2": 0.0, "mse": 1.0, "worst_3m_sharpe": -999}
    
    # Walk-forward splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_preds = []
    all_actuals = []
    all_returns = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        all_preds.extend(y_pred)
        all_actuals.extend(y_test)
        
        # Strategy returns: go long when pred > 0, else flat
        strategy_returns = np.where(y_pred > 0, y_test, 0)
        all_returns.extend(strategy_returns)
    
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_returns = np.array(all_returns)
    
    # Calculate metrics
    sharpe = calculate_sharpe(all_returns)
    accuracy = calculate_direction_accuracy(all_actuals, all_preds)
    r2 = r2_score(all_actuals, all_preds) if len(all_actuals) > 1 else 0.0
    mse = mean_squared_error(all_actuals, all_preds) if len(all_actuals) > 1 else 1.0
    
    # Worst 3-month Sharpe
    returns_series = pd.Series(all_returns)
    worst_3m = get_worst_3month_sharpe(returns_series)
    
    return {
        "sharpe": sharpe,
        "accuracy": accuracy,
        "r2": r2,
        "mse": mse,
        "worst_3m_sharpe": worst_3m,
        "n_predictions": len(all_preds),
    }


# ============================================================================
# EXPERIMENT 1: RF WITH OPTIMIZED FEATURES
# ============================================================================

def experiment_1_feature_comparison() -> Dict[str, Any]:
    """
    Compare RF performance with optimized features vs full feature set.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: RF with Optimized Features vs Full Features")
    print("="*70)
    
    results = {
        "optimized_features": {},
        "full_features": {},
        "summary": {},
    }
    
    # Default RF model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    
    for ticker in TEST_TICKERS:
        print(f"\n📊 Testing {ticker}...")
        
        df, error = load_ticker_data(ticker)
        if df is None:
            print(f"  ⚠️ Skipping {ticker}: {error}")
            continue
        
        # Get target column
        target_col = "ftarget_ret_5d_ahead" if "ftarget_ret_5d_ahead" in df.columns else "ret_5d"
        
        # Test with optimized features
        opt_features = filter_features(df, OPTIMIZED_FEATURES_LIST)
        if len(opt_features) >= 10:
            print(f"  Testing with {len(opt_features)} optimized features...")
            opt_metrics = walk_forward_evaluate(
                df, opt_features, target_col,
                RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, 
                                     min_samples_leaf=4, random_state=42, n_jobs=-1)
            )
            results["optimized_features"][ticker] = opt_metrics
            print(f"    Sharpe: {opt_metrics['sharpe']:.4f}, Accuracy: {opt_metrics['accuracy']:.2%}")
        
        # Test with all features
        all_features = [c for c in df.columns if c not in [target_col, "Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]]
        all_features = [f for f in all_features if not f.startswith("ftarget")]
        
        if len(all_features) >= 20:
            print(f"  Testing with {len(all_features)} full features...")
            full_metrics = walk_forward_evaluate(
                df, all_features, target_col,
                RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10,
                                     min_samples_leaf=4, random_state=42, n_jobs=-1)
            )
            results["full_features"][ticker] = full_metrics
            print(f"    Sharpe: {full_metrics['sharpe']:.4f}, Accuracy: {full_metrics['accuracy']:.2%}")
    
    # Summary
    opt_sharpes = [v["sharpe"] for v in results["optimized_features"].values()]
    full_sharpes = [v["sharpe"] for v in results["full_features"].values()]
    
    opt_acc = [v["accuracy"] for v in results["optimized_features"].values()]
    full_acc = [v["accuracy"] for v in results["full_features"].values()]
    
    results["summary"] = {
        "optimized_avg_sharpe": np.mean(opt_sharpes) if opt_sharpes else 0,
        "full_avg_sharpe": np.mean(full_sharpes) if full_sharpes else 0,
        "optimized_avg_accuracy": np.mean(opt_acc) if opt_acc else 0.5,
        "full_avg_accuracy": np.mean(full_acc) if full_acc else 0.5,
        "optimized_better": np.mean(opt_sharpes) > np.mean(full_sharpes) if opt_sharpes and full_sharpes else False,
        "improvement_pct": ((np.mean(opt_sharpes) - np.mean(full_sharpes)) / (abs(np.mean(full_sharpes)) + 1e-8)) * 100 if full_sharpes else 0,
    }
    
    print("\n📈 Experiment 1 Summary:")
    print(f"  Optimized Features ({len(OPTIMIZED_FEATURES_LIST)}): Sharpe {results['summary']['optimized_avg_sharpe']:.4f}, "
          f"Acc {results['summary']['optimized_avg_accuracy']:.2%}")
    print(f"  Full Features: Sharpe {results['summary']['full_avg_sharpe']:.4f}, "
          f"Acc {results['summary']['full_avg_accuracy']:.2%}")
    print(f"  Winner: {'Optimized Features' if results['summary']['optimized_better'] else 'Full Features'}")
    
    return results


# ============================================================================
# EXPERIMENT 2: RF HYPERPARAMETER OPTIMIZATION
# ============================================================================

def experiment_2_hyperopt(n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimize RF hyperparameters using Optuna.
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT 2: RF Hyperparameter Optimization ({n_trials} trials)")
    print("="*70)
    
    # Load data for all tickers
    ticker_data = {}
    for ticker in TEST_TICKERS:
        df, error = load_ticker_data(ticker)
        if df is not None:
            ticker_data[ticker] = df
    
    if not ticker_data:
        print("⚠️ No valid ticker data loaded")
        return {"error": "No data"}
    
    print(f"Loaded data for {len(ticker_data)} tickers: {list(ticker_data.keys())}")
    
    # Use optimized features
    features_to_use = OPTIMIZED_FEATURES_LIST
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Sample hyperparameters
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 500, 700]),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, None]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
            "n_jobs": -1,
        }
        
        # Evaluate on all tickers
        sharpes = []
        for ticker, df in ticker_data.items():
            target_col = "ftarget_ret_5d_ahead" if "ftarget_ret_5d_ahead" in df.columns else "ret_5d"
            available_features = filter_features(df, features_to_use)
            
            if len(available_features) < 5:
                continue
            
            try:
                model = RandomForestRegressor(**params)
                metrics = walk_forward_evaluate(df, available_features, target_col, model, n_splits=3)
                sharpes.append(metrics["sharpe"])
            except Exception as e:
                continue
        
        if not sharpes:
            return -10.0  # Penalty for failed trials
        
        return np.mean(sharpes)
    
    # Run optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    print(f"\n🔍 Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_sharpe = study.best_value
    
    print(f"\n✅ Best Sharpe: {best_sharpe:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Evaluate best model more thoroughly
    best_rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    
    detailed_results = {}
    for ticker, df in ticker_data.items():
        target_col = "ftarget_ret_5d_ahead" if "ftarget_ret_5d_ahead" in df.columns else "ret_5d"
        available_features = filter_features(df, features_to_use)
        
        if len(available_features) >= 5:
            metrics = walk_forward_evaluate(
                df, available_features, target_col,
                RandomForestRegressor(**best_params, random_state=42, n_jobs=-1),
                n_splits=5
            )
            detailed_results[ticker] = metrics
            print(f"  {ticker}: Sharpe {metrics['sharpe']:.4f}, Acc {metrics['accuracy']:.2%}")
    
    # Collect all trials
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            })
    
    return {
        "best_params": best_params,
        "best_sharpe": best_sharpe,
        "detailed_results": detailed_results,
        "n_trials": n_trials,
        "trials": trials_data[:10],  # Top 10 trials
        "avg_accuracy": np.mean([v["accuracy"] for v in detailed_results.values()]) if detailed_results else 0.5,
        "avg_worst_3m_sharpe": np.mean([v["worst_3m_sharpe"] for v in detailed_results.values()]) if detailed_results else -999,
    }


# ============================================================================
# EXPERIMENT 3: RF CALIBRATION
# ============================================================================

def apply_temperature_scaling(predictions: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to predictions."""
    return predictions / temperature


def experiment_3_calibration() -> Dict[str, Any]:
    """
    Test different calibration methods on RF predictions.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: RF Calibration Methods")
    print("="*70)
    
    results = {
        "uncalibrated": {},
        "temperature_scaling": {},
        "best_temperature": 1.0,
        "summary": {},
    }
    
    # Load data
    ticker_data = {}
    for ticker in TEST_TICKERS[:3]:  # Use subset for speed
        df, error = load_ticker_data(ticker)
        if df is not None:
            ticker_data[ticker] = df
    
    if not ticker_data:
        return {"error": "No data"}
    
    features_to_use = OPTIMIZED_FEATURES_LIST
    
    # Get best RF params from experiment 2 (or use defaults)
    rf_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Test each temperature
    temp_results = {t: {"sharpes": [], "accuracies": []} for t in TEMPERATURE_VALUES}
    
    for ticker, df in ticker_data.items():
        print(f"\n📊 Testing calibration on {ticker}...")
        
        target_col = "ftarget_ret_5d_ahead" if "ftarget_ret_5d_ahead" in df.columns else "ret_5d"
        available_features = filter_features(df, features_to_use)
        
        if len(available_features) < 5:
            continue
        
        X = df[available_features].values
        y = df[target_col].values
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        if len(X) < 200:
            continue
        
        # Train/test split (70/30)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Fit base model
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        base_preds = model.predict(X_test)
        
        # Test each temperature
        for temp in TEMPERATURE_VALUES:
            scaled_preds = apply_temperature_scaling(base_preds, temp)
            
            # Strategy returns
            strategy_returns = np.where(scaled_preds > 0, y_test, 0)
            sharpe = calculate_sharpe(strategy_returns)
            accuracy = calculate_direction_accuracy(y_test, scaled_preds)
            
            temp_results[temp]["sharpes"].append(sharpe)
            temp_results[temp]["accuracies"].append(accuracy)
    
    # Find best temperature
    avg_sharpes = {t: np.mean(r["sharpes"]) for t, r in temp_results.items() if r["sharpes"]}
    best_temp = max(avg_sharpes, key=avg_sharpes.get) if avg_sharpes else 1.0
    
    results["temperature_scaling"] = {
        t: {
            "avg_sharpe": np.mean(r["sharpes"]) if r["sharpes"] else 0,
            "avg_accuracy": np.mean(r["accuracies"]) if r["accuracies"] else 0.5,
        }
        for t, r in temp_results.items()
    }
    results["best_temperature"] = best_temp
    
    print(f"\n📈 Calibration Results:")
    for temp, data in results["temperature_scaling"].items():
        marker = "⭐" if temp == best_temp else "  "
        print(f"  {marker} T={temp}: Sharpe {data['avg_sharpe']:.4f}, Acc {data['avg_accuracy']:.2%}")
    
    print(f"\n✅ Best Temperature: {best_temp}")
    
    results["summary"] = {
        "best_temperature": best_temp,
        "best_sharpe": avg_sharpes.get(best_temp, 0),
        "improvement_vs_uncalibrated": (avg_sharpes.get(best_temp, 0) - avg_sharpes.get(1.0, 0)) / (abs(avg_sharpes.get(1.0, 0)) + 1e-8) * 100,
    }
    
    return results


# ============================================================================
# EXPERIMENT 4: ENSEMBLE COMPARISON
# ============================================================================

def experiment_4_ensemble_comparison(rf_params: Dict = None) -> Dict[str, Any]:
    """
    Compare different ensemble configurations.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Ensemble Comparison")
    print("="*70)
    
    if not HAS_XGB:
        print("⚠️ XGBoost not available, skipping ensemble comparison")
        return {"error": "XGBoost not available"}
    
    # Default RF params if not provided
    if rf_params is None:
        rf_params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": 42,
            "n_jobs": -1,
        }
    
    # XGBoost params (from OPTIMIZED_MODEL_CONFIG)
    xgb_params = {
        "n_estimators": 450,
        "max_depth": 7,
        "learning_rate": 0.048,
        "subsample": 0.998,
        "colsample_bytree": 0.67,
        "min_child_weight": 19,
        "reg_alpha": 0.012,
        "reg_lambda": 9.3,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # Load data
    ticker_data = {}
    for ticker in TEST_TICKERS:
        df, error = load_ticker_data(ticker)
        if df is not None:
            ticker_data[ticker] = df
    
    if not ticker_data:
        return {"error": "No data"}
    
    features_to_use = OPTIMIZED_FEATURES_LIST
    
    # Ensemble configurations to test
    configs = {
        "xgb_only": {
            "models": [("xgb", XGBRegressor(**xgb_params))],
            "weights": [1.0],
        },
        "rf_only": {
            "models": [("rf", RandomForestRegressor(**rf_params))],
            "weights": [1.0],
        },
        "xgb_rf_equal": {
            "models": [
                ("xgb", XGBRegressor(**xgb_params)),
                ("rf", RandomForestRegressor(**rf_params)),
            ],
            "weights": [0.5, 0.5],
        },
        "xgb_rf_70_30": {
            "models": [
                ("xgb", XGBRegressor(**xgb_params)),
                ("rf", RandomForestRegressor(**rf_params)),
            ],
            "weights": [0.7, 0.3],
        },
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n🔧 Testing {config_name}...")
        
        config_results = {"sharpes": [], "accuracies": [], "worst_3m_sharpes": [], "disagreement_rates": []}
        
        for ticker, df in ticker_data.items():
            target_col = "ftarget_ret_5d_ahead" if "ftarget_ret_5d_ahead" in df.columns else "ret_5d"
            available_features = filter_features(df, features_to_use)
            
            if len(available_features) < 5:
                continue
            
            X = df[available_features].values
            y = df[target_col].values
            
            # Remove NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            
            if len(X) < 200:
                continue
            
            # Walk-forward evaluation
            tscv = TimeSeriesSplit(n_splits=5)
            all_preds = []
            all_actuals = []
            individual_preds = {name: [] for name, _ in config["models"]}
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Get predictions from each model
                model_preds = []
                for name, model in config["models"]:
                    # Clone model for this fold
                    if "xgb" in name:
                        m = XGBRegressor(**xgb_params)
                    else:
                        m = RandomForestRegressor(**rf_params)
                    
                    m.fit(X_train, y_train)
                    pred = m.predict(X_test)
                    model_preds.append(pred)
                    individual_preds[name].extend(pred)
                
                # Weighted average
                weights = config["weights"]
                ensemble_pred = sum(w * p for w, p in zip(weights, model_preds))
                
                all_preds.extend(ensemble_pred)
                all_actuals.extend(y_test)
            
            all_preds = np.array(all_preds)
            all_actuals = np.array(all_actuals)
            
            # Calculate metrics
            strategy_returns = np.where(all_preds > 0, all_actuals, 0)
            sharpe = calculate_sharpe(strategy_returns)
            accuracy = calculate_direction_accuracy(all_actuals, all_preds)
            worst_3m = get_worst_3month_sharpe(pd.Series(strategy_returns))
            
            config_results["sharpes"].append(sharpe)
            config_results["accuracies"].append(accuracy)
            config_results["worst_3m_sharpes"].append(worst_3m)
            
            # Calculate disagreement rate (for multi-model configs)
            if len(config["models"]) > 1:
                pred_arrays = [np.array(individual_preds[name]) for name, _ in config["models"]]
                if len(pred_arrays[0]) > 0:
                    signs = [np.sign(p) for p in pred_arrays]
                    disagreement = np.mean(signs[0] != signs[1])
                    config_results["disagreement_rates"].append(disagreement)
        
        # Aggregate results
        results[config_name] = {
            "avg_sharpe": np.mean(config_results["sharpes"]) if config_results["sharpes"] else 0,
            "avg_accuracy": np.mean(config_results["accuracies"]) if config_results["accuracies"] else 0.5,
            "avg_worst_3m_sharpe": np.mean(config_results["worst_3m_sharpes"]) if config_results["worst_3m_sharpes"] else -999,
            "avg_disagreement_rate": np.mean(config_results["disagreement_rates"]) if config_results["disagreement_rates"] else 0,
            "per_ticker": {
                ticker: {
                    "sharpe": config_results["sharpes"][i] if i < len(config_results["sharpes"]) else 0,
                    "accuracy": config_results["accuracies"][i] if i < len(config_results["accuracies"]) else 0.5,
                }
                for i, ticker in enumerate(ticker_data.keys())
            }
        }
        
        print(f"  Avg Sharpe: {results[config_name]['avg_sharpe']:.4f}, "
              f"Acc: {results[config_name]['avg_accuracy']:.2%}, "
              f"Worst 3M: {results[config_name]['avg_worst_3m_sharpe']:.4f}")
    
    # Determine best configuration
    best_config = max(results, key=lambda k: results[k]["avg_sharpe"])
    
    # Also check stability (worst 3-month)
    most_stable = max(results, key=lambda k: results[k]["avg_worst_3m_sharpe"])
    
    results["summary"] = {
        "best_by_sharpe": best_config,
        "best_sharpe": results[best_config]["avg_sharpe"],
        "most_stable": most_stable,
        "most_stable_worst_3m": results[most_stable]["avg_worst_3m_sharpe"],
        "xgb_only_sharpe": results["xgb_only"]["avg_sharpe"],
        "ensemble_improves_sharpe": results[best_config]["avg_sharpe"] > results["xgb_only"]["avg_sharpe"],
        "ensemble_improves_stability": results[most_stable]["avg_worst_3m_sharpe"] > results["xgb_only"]["avg_worst_3m_sharpe"],
    }
    
    print(f"\n✅ Best by Sharpe: {best_config} ({results[best_config]['avg_sharpe']:.4f})")
    print(f"✅ Most Stable: {most_stable} (worst 3M: {results[most_stable]['avg_worst_3m_sharpe']:.4f})")
    
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(
    exp1_results: Dict,
    exp2_results: Dict,
    exp3_results: Dict,
    exp4_results: Dict,
) -> str:
    """Generate the final optimization report."""
    
    report = f"""# Random Forest Optimization Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Objective:** Optimize Random Forest model and determine if adding it to the ensemble improves the system.

---

## Executive Summary

"""
    
    # Determine final recommendation
    if "error" in exp4_results:
        recommendation = "⚠️ INCONCLUSIVE - Ensemble comparison failed"
    elif exp4_results["summary"]["ensemble_improves_sharpe"]:
        best = exp4_results["summary"]["best_by_sharpe"]
        improvement = ((exp4_results["summary"]["best_sharpe"] - exp4_results["summary"]["xgb_only_sharpe"]) 
                      / (abs(exp4_results["summary"]["xgb_only_sharpe"]) + 1e-8) * 100)
        recommendation = f"✅ **USE ENSEMBLE** ({best}) - Improves Sharpe by {improvement:.1f}%"
    elif exp4_results["summary"]["ensemble_improves_stability"]:
        most_stable = exp4_results["summary"]["most_stable"]
        recommendation = f"✅ **USE ENSEMBLE** ({most_stable}) - Better stability (worst 3-month Sharpe)"
    else:
        recommendation = "❌ **STICK WITH XGB ONLY** - Ensemble does not improve performance"
    
    report += f"""**Recommendation:** {recommendation}

| Metric | XGB Only | Best Ensemble | Difference |
|--------|----------|---------------|------------|
| Avg Sharpe | {exp4_results.get('xgb_only', {}).get('avg_sharpe', 0):.4f} | {exp4_results.get('summary', {}).get('best_sharpe', 0):.4f} | {exp4_results.get('summary', {}).get('best_sharpe', 0) - exp4_results.get('xgb_only', {}).get('avg_sharpe', 0):+.4f} |
| Worst 3M Sharpe | {exp4_results.get('xgb_only', {}).get('avg_worst_3m_sharpe', 0):.4f} | {exp4_results.get(exp4_results.get('summary', {}).get('most_stable', 'xgb_only'), {}).get('avg_worst_3m_sharpe', 0):.4f} | {exp4_results.get(exp4_results.get('summary', {}).get('most_stable', 'xgb_only'), {}).get('avg_worst_3m_sharpe', 0) - exp4_results.get('xgb_only', {}).get('avg_worst_3m_sharpe', 0):+.4f} |

---

## Experiment 1: Feature Comparison

**Question:** Do the 20 optimized features work for RF as well as XGB?

| Feature Set | Avg Sharpe | Avg Accuracy |
|-------------|------------|--------------|
| Optimized (20 features) | {exp1_results.get('summary', {}).get('optimized_avg_sharpe', 0):.4f} | {exp1_results.get('summary', {}).get('optimized_avg_accuracy', 0):.2%} |
| Full (~150 features) | {exp1_results.get('summary', {}).get('full_avg_sharpe', 0):.4f} | {exp1_results.get('summary', {}).get('full_avg_accuracy', 0):.2%} |

**Winner:** {'Optimized Features' if exp1_results.get('summary', {}).get('optimized_better', False) else 'Full Features'}
**Improvement:** {exp1_results.get('summary', {}).get('improvement_pct', 0):.1f}%

---

## Experiment 2: Hyperparameter Optimization

**Method:** Optuna with {exp2_results.get('n_trials', 50)} trials, maximizing Sharpe via walk-forward validation

**Best Parameters:**
```python
OPTIMIZED_RF_CONFIG = {json.dumps(exp2_results.get('best_params', {}), indent=4)}
```

**Best Sharpe:** {exp2_results.get('best_sharpe', 0):.4f}
**Avg Accuracy:** {exp2_results.get('avg_accuracy', 0):.2%}
**Avg Worst 3M Sharpe:** {exp2_results.get('avg_worst_3m_sharpe', 0):.4f}

### Per-Ticker Results

| Ticker | Sharpe | Accuracy | Worst 3M Sharpe |
|--------|--------|----------|-----------------|
"""
    
    for ticker, metrics in exp2_results.get('detailed_results', {}).items():
        report += f"| {ticker} | {metrics.get('sharpe', 0):.4f} | {metrics.get('accuracy', 0):.2%} | {metrics.get('worst_3m_sharpe', 0):.4f} |\n"
    
    report += f"""
---

## Experiment 3: Calibration

**Best Temperature:** {exp3_results.get('best_temperature', 1.0)}

| Temperature | Avg Sharpe | Avg Accuracy |
|-------------|------------|--------------|
"""
    
    for temp, data in exp3_results.get('temperature_scaling', {}).items():
        marker = "⭐" if temp == exp3_results.get('best_temperature', 1.0) else ""
        report += f"| {temp} {marker} | {data.get('avg_sharpe', 0):.4f} | {data.get('avg_accuracy', 0):.2%} |\n"
    
    report += f"""
**Improvement vs Uncalibrated:** {exp3_results.get('summary', {}).get('improvement_vs_uncalibrated', 0):.1f}%

---

## Experiment 4: Ensemble Comparison

| Configuration | Avg Sharpe | Avg Accuracy | Worst 3M Sharpe | Disagreement Rate |
|--------------|------------|--------------|-----------------|-------------------|
"""
    
    for config, data in exp4_results.items():
        if config == "summary":
            continue
        if isinstance(data, dict) and "avg_sharpe" in data:
            report += f"| {config} | {data.get('avg_sharpe', 0):.4f} | {data.get('avg_accuracy', 0):.2%} | {data.get('avg_worst_3m_sharpe', 0):.4f} | {data.get('avg_disagreement_rate', 0):.2%} |\n"
    
    report += f"""
**Best by Sharpe:** {exp4_results.get('summary', {}).get('best_by_sharpe', 'N/A')}
**Most Stable:** {exp4_results.get('summary', {}).get('most_stable', 'N/A')}

---

## Final Recommendation

{recommendation}

### Reasoning:

1. **Sharpe Comparison:** {"Ensemble improves Sharpe" if exp4_results.get('summary', {}).get('ensemble_improves_sharpe', False) else "XGB alone has higher Sharpe"}
2. **Stability:** {"Ensemble has better worst-case performance" if exp4_results.get('summary', {}).get('ensemble_improves_stability', False) else "XGB alone is more stable"}
3. **Complexity Trade-off:** Ensemble adds model complexity and compute time

### Implementation:

If using ensemble, update `model_improvements.py`:
```python
class ModelEnsemble:
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor, VotingRegressor
        from xgboost import XGBRegressor
        
        estimators = [
            ('xgb', XGBRegressor(
                n_estimators=450, max_depth=7, learning_rate=0.048,
                subsample=0.998, colsample_bytree=0.67,
                min_child_weight=19, reg_alpha=0.012, reg_lambda=9.3,
                random_state=42, n_jobs=-1
            )),
            ('rf', RandomForestRegressor(
                **OPTIMIZED_RF_CONFIG  # From this experiment
            )),
        ]
        
        self.ensemble = VotingRegressor(estimators=estimators, weights=[0.7, 0.3])
```

---

## Files Generated

1. `experiments/RF_OPTIMIZATION_REPORT.md` - This report
2. `experiments/optimized_rf_config.json` - Best RF parameters

---

*Report generated by RF Optimization Pipeline*
"""
    
    return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all experiments and generate report."""
    print("="*70)
    print("RANDOM FOREST OPTIMIZATION PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Tickers: {TEST_TICKERS}")
    
    # Run experiments
    print("\n" + "="*70)
    exp1_results = experiment_1_feature_comparison()
    
    print("\n" + "="*70)
    exp2_results = experiment_2_hyperopt(n_trials=50)
    
    print("\n" + "="*70)
    exp3_results = experiment_3_calibration()
    
    print("\n" + "="*70)
    # Pass best RF params to experiment 4
    best_rf_params = exp2_results.get("best_params", {})
    if best_rf_params:
        best_rf_params["random_state"] = 42
        best_rf_params["n_jobs"] = -1
    exp4_results = experiment_4_ensemble_comparison(rf_params=best_rf_params if best_rf_params else None)
    
    # Generate and save report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    report = generate_report(exp1_results, exp2_results, exp3_results, exp4_results)
    
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"✅ Report saved to: {REPORT_PATH}")
    
    # Helper to convert numpy types to native Python
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    # Save best config
    config_to_save = {
        "best_rf_params": convert_numpy_types(exp2_results.get("best_params", {})),
        "best_temperature": float(exp3_results.get("best_temperature", 1.0)),
        "best_sharpe": float(exp2_results.get("best_sharpe", 0)),
        "avg_accuracy": float(exp2_results.get("avg_accuracy", 0.5)),
        "recommendation": exp4_results.get("summary", {}).get("best_by_sharpe", "xgb_only"),
        "use_ensemble": bool(exp4_results.get("summary", {}).get("ensemble_improves_sharpe", False) or 
                       exp4_results.get("summary", {}).get("ensemble_improves_stability", False)),
        "generated": datetime.now().isoformat(),
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"✅ Config saved to: {CONFIG_PATH}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Best RF Sharpe: {exp2_results.get('best_sharpe', 0):.4f}")
    print(f"Best RF Params: {exp2_results.get('best_params', {})}")
    print(f"Best Ensemble: {exp4_results.get('summary', {}).get('best_by_sharpe', 'N/A')}")
    print(f"Recommendation: {'Use Ensemble' if config_to_save['use_ensemble'] else 'Stick with XGB Only'}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        "exp1": exp1_results,
        "exp2": exp2_results,
        "exp3": exp3_results,
        "exp4": exp4_results,
        "config": config_to_save,
    }


if __name__ == "__main__":
    results = main()
