#!/usr/bin/env python3
"""
Test XGB+LSTM and XGB+LSTM+RF ensembles across market regimes.

Tests:
1. XGB Only (baseline)
2. XGB + LSTM (0.7/0.3 weights - XGB dominant since lower overfitting)
3. XGB + LSTM + RF (0.6/0.25/0.15 weights)

Across:
- 5 years of data (2020-2025)
- Bull, bear, and neutral markets
- 6 tickers with good data quality
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction_model import build_features_and_target
from src.core.models import make_model, get_feature_importance
from src.config import get_model_config


# ==============================================================================
# LSTM MODEL
# ==============================================================================

def create_lstm_model(input_shape: Tuple[int, int]) -> "Sequential":
    """Create a small LSTM model to prevent overfitting."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l2
    
    model = Sequential([
        LSTM(16, input_shape=input_shape, return_sequences=False,
             kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def prepare_lstm_data(X: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Convert 2D features to 3D sequences for LSTM."""
    n_samples, n_features = X.shape
    X_seq = np.zeros((n_samples - lookback + 1, lookback, n_features))
    
    for i in range(lookback - 1, n_samples):
        X_seq[i - lookback + 1] = X[i - lookback + 1:i + 1]
    
    return X_seq


# ==============================================================================
# ENSEMBLE MODELS
# ==============================================================================

class XGBOnlyModel:
    """XGBoost only baseline."""
    
    def __init__(self):
        self.xgb = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit XGBoost model."""
        params = get_model_config("xgb", "xgb_balanced_v6")
        self.xgb = make_model("xgb", "regression", random_state=42)
        self.xgb.set_params(**params)
        self.xgb.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using XGBoost."""
        return self.xgb.predict(X)
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.Series:
        """Get feature importance."""
        return get_feature_importance(self.xgb, feature_names)


class XGBLSTMEnsemble:
    """XGBoost + LSTM ensemble with configurable weights."""
    
    def __init__(self, xgb_weight: float = 0.7, lookback: int = 20):
        self.xgb_weight = xgb_weight
        self.lstm_weight = 1.0 - xgb_weight
        self.lookback = lookback
        self.xgb = None
        self.lstm = None
        self.n_features = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, verbose: int = 0):
        """Fit both models."""
        import tensorflow as tf
        
        # XGBoost
        params = get_model_config("xgb", "xgb_balanced_v6")
        self.xgb = make_model("xgb", "regression", random_state=42)
        self.xgb.set_params(**params)
        self.xgb.fit(X, y)
        
        # LSTM
        self.n_features = X.shape[1]
        X_seq = prepare_lstm_data(X, self.lookback)
        y_lstm = y[self.lookback - 1:]
        
        self.lstm = create_lstm_model((self.lookback, self.n_features))
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        self.lstm.fit(
            X_seq, y_lstm,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        xgb_pred = self.xgb.predict(X)
        
        if len(X) < self.lookback:
            return xgb_pred
        
        X_seq = prepare_lstm_data(X, self.lookback)
        lstm_pred_short = self.lstm.predict(X_seq, verbose=0).flatten()
        
        # Align predictions
        lstm_pred = np.full(len(X), np.nan)
        lstm_pred[self.lookback - 1:] = lstm_pred_short
        
        # Weighted average (use XGB where LSTM unavailable)
        ensemble_pred = np.where(
            np.isnan(lstm_pred),
            xgb_pred,
            self.xgb_weight * xgb_pred + self.lstm_weight * lstm_pred
        )
        
        return ensemble_pred


class XGBLSTMRFEnsemble:
    """XGBoost + LSTM + RandomForest ensemble."""
    
    def __init__(self, xgb_weight: float = 0.6, lstm_weight: float = 0.25, 
                 rf_weight: float = 0.15, lookback: int = 20):
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.rf_weight = rf_weight
        self.lookback = lookback
        self.xgb = None
        self.lstm = None
        self.rf = None
        self.n_features = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, verbose: int = 0):
        """Fit all three models."""
        import tensorflow as tf
        
        # XGBoost
        params = get_model_config("xgb", "xgb_balanced_v6")
        self.xgb = make_model("xgb", "regression", random_state=42)
        self.xgb.set_params(**params)
        self.xgb.fit(X, y)
        
        # RandomForest (with regularization)
        self.rf = make_model("rf", "regression", random_state=42)
        self.rf.set_params(
            n_estimators=100,
            max_depth=4,  # Limited depth to prevent overfitting
            min_samples_leaf=20,
            max_features=0.5
        )
        self.rf.fit(X, y)
        
        # LSTM
        self.n_features = X.shape[1]
        X_seq = prepare_lstm_data(X, self.lookback)
        y_lstm = y[self.lookback - 1:]
        
        self.lstm = create_lstm_model((self.lookback, self.n_features))
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        self.lstm.fit(
            X_seq, y_lstm,
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction."""
        xgb_pred = self.xgb.predict(X)
        rf_pred = self.rf.predict(X)
        
        if len(X) < self.lookback:
            # Fallback to XGB + RF only
            return self.xgb_weight * xgb_pred + self.rf_weight * rf_pred
        
        X_seq = prepare_lstm_data(X, self.lookback)
        lstm_pred_short = self.lstm.predict(X_seq, verbose=0).flatten()
        
        # Align predictions
        lstm_pred = np.full(len(X), np.nan)
        lstm_pred[self.lookback - 1:] = lstm_pred_short
        
        # Weighted average
        ensemble_pred = np.where(
            np.isnan(lstm_pred),
            # Where LSTM unavailable, rescale XGB + RF weights
            (self.xgb_weight / (self.xgb_weight + self.rf_weight)) * xgb_pred +
            (self.rf_weight / (self.xgb_weight + self.rf_weight)) * rf_pred,
            # Full ensemble
            self.xgb_weight * xgb_pred + self.lstm_weight * lstm_pred + self.rf_weight * rf_pred
        )
        
        return ensemble_pred


# ==============================================================================
# BACKTESTING
# ==============================================================================

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
    """Calculate trading metrics."""
    # Classification
    pred_dir = np.sign(predictions)
    actual_dir = np.sign(actuals)
    correct = (pred_dir == actual_dir)
    accuracy = correct.mean() if len(correct) > 0 else 0.0
    
    # Returns (long/short strategy)
    returns = pred_dir * actuals
    
    # Sharpe (annualized)
    if len(returns) > 0 and returns.std() > 1e-10:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Cumulative return
    cum_return = (1 + returns).prod() - 1
    
    # Max drawdown
    cumsum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = cumsum - running_max
    max_dd = drawdowns.min() if len(drawdowns) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "sharpe": sharpe,
        "cum_return": cum_return,
        "max_drawdown": max_dd,
        "n_trades": len(returns),
    }


def backtest_model(
    model,
    ticker: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Dict:
    """Walk-forward backtest for a model."""
    
    # Build full dataset using period
    try:
        result = build_features_and_target(
            ticker,
            period="5y",  # Get 5 years of data
            horizon=1
        )
        
        # Result is a tuple: (X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates)
        if result is None:
            return {"error": "No data"}
        
        X, y, _, _, _, _, dates = result
        
        if X is None or len(X) < 100:
            return {"error": f"Insufficient data: {len(X) if X is not None else 0}"}
        
        # Convert dates if needed
        if not isinstance(dates, pd.DatetimeIndex):
            try:
                dates = pd.DatetimeIndex(dates)
            except Exception:
                return {"error": "Invalid dates"}
        
        # Make dates timezone-naive for comparison
        dates_naive = dates.tz_localize(None) if dates.tz is not None else dates
        
        # Split by date
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)
        
        train_mask = (dates_naive >= train_start_ts) & (dates_naive <= train_end_ts)
        test_mask = (dates_naive >= test_start_ts) & (dates_naive <= test_end_ts)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        if len(X_train) < 50 or len(X_test) < 10:
            return {"error": f"Insufficient train/test data: {len(X_train)}/{len(X_test)}"}
        
        # Fit model
        if hasattr(model, 'fit') and callable(model.fit):
            if isinstance(model, (XGBLSTMEnsemble, XGBLSTMRFEnsemble)):
                model.fit(X_train, y_train, epochs=30, verbose=0)
            else:
                model.fit(X_train, y_train)
        
        # Predict
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_pred, y_train)
        test_metrics = calculate_metrics(test_pred, y_test)
        
        return {
            "train": train_metrics,
            "test": test_metrics,
            "overfit_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
            "sharpe_gap": train_metrics["sharpe"] - test_metrics["sharpe"],
        }
        
    except Exception as e:
        import traceback
        return {"error": f"{str(e)[:100]}"}


# ==============================================================================
# MAIN TEST
# ==============================================================================

def main():
    """Run comprehensive ensemble comparison."""
    print("=" * 80)
    print("XGB vs XGB+LSTM vs XGB+LSTM+RF ENSEMBLE COMPARISON")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test tickers (good quality, diverse)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM"]
    
    # Test periods (bull, bear, neutral)
    periods = [
        ("bull_2021", "2020-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
        ("bear_2022", "2021-01-01", "2021-12-31", "2022-01-01", "2022-06-30"),
        ("recovery_2023", "2022-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
        ("ai_boom_2024", "2023-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
        ("recent_2025", "2024-01-01", "2024-12-31", "2025-01-01", "2025-01-15"),
    ]
    
    # Models to test
    model_configs = [
        ("XGB Only", XGBOnlyModel),
        ("XGB+LSTM (70/30)", lambda: XGBLSTMEnsemble(xgb_weight=0.7)),
        ("XGB+LSTM (50/50)", lambda: XGBLSTMEnsemble(xgb_weight=0.5)),
        ("XGB+LSTM+RF (60/25/15)", lambda: XGBLSTMRFEnsemble(xgb_weight=0.6, lstm_weight=0.25, rf_weight=0.15)),
        ("XGB+LSTM+RF (50/30/20)", lambda: XGBLSTMRFEnsemble(xgb_weight=0.5, lstm_weight=0.3, rf_weight=0.2)),
    ]
    
    all_results = []
    
    for model_name, model_factory in model_configs:
        print(f"\n{'=' * 40}")
        print(f"Testing: {model_name}")
        print("=" * 40)
        
        model_results = []
        
        for ticker in tickers:
            for period_name, train_start, train_end, test_start, test_end in periods:
                print(f"  {ticker} - {period_name}...", end=" ", flush=True)
                
                # Create fresh model
                model = model_factory() if callable(model_factory) else model_factory
                
                result = backtest_model(
                    model, ticker,
                    train_start, train_end,
                    test_start, test_end
                )
                
                if "error" not in result:
                    model_results.append({
                        "model": model_name,
                        "ticker": ticker,
                        "period": period_name,
                        "test_accuracy": result["test"]["accuracy"],
                        "test_sharpe": result["test"]["sharpe"],
                        "test_cum_return": result["test"]["cum_return"],
                        "test_max_dd": result["test"]["max_drawdown"],
                        "overfit_gap": result["overfit_gap"],
                        "sharpe_gap": result["sharpe_gap"],
                    })
                    print(f"Acc: {result['test']['accuracy']:.1%}, Sharpe: {result['test']['sharpe']:.2f}")
                else:
                    print(f"Error: {result['error']}")
        
        all_results.extend(model_results)
        
        # Summary for this model
        if model_results:
            df = pd.DataFrame(model_results)
            print(f"\n  {model_name} Summary:")
            print(f"    Avg Test Accuracy: {df['test_accuracy'].mean():.1%}")
            print(f"    Avg Test Sharpe:   {df['test_sharpe'].mean():.2f}")
            print(f"    Avg Cum Return:    {df['test_cum_return'].mean():.1%}")
            print(f"    Avg Overfit Gap:   {df['overfit_gap'].mean():.1%}")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        summary = df.groupby("model").agg({
            "test_accuracy": ["mean", "std"],
            "test_sharpe": ["mean", "std", "min", "max"],
            "test_cum_return": "mean",
            "test_max_dd": "mean",
            "overfit_gap": "mean",
        }).round(3)
        
        print("\n" + summary.to_string())
        
        # Best model by Sharpe
        sharpe_by_model = df.groupby("model")["test_sharpe"].mean().sort_values(ascending=False)
        print(f"\n🏆 BEST BY AVG SHARPE: {sharpe_by_model.index[0]} ({sharpe_by_model.iloc[0]:.2f})")
        
        # Best by accuracy
        acc_by_model = df.groupby("model")["test_accuracy"].mean().sort_values(ascending=False)
        print(f"🏆 BEST BY ACCURACY:   {acc_by_model.index[0]} ({acc_by_model.iloc[0]:.1%})")
        
        # Lowest overfitting
        overfit_by_model = df.groupby("model")["overfit_gap"].mean().sort_values()
        print(f"🏆 LOWEST OVERFIT:     {overfit_by_model.index[0]} ({overfit_by_model.iloc[0]:.1%})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"ensemble_comparison_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to: {csv_path}")
        
        # Per-period analysis
        print("\n" + "-" * 60)
        print("PERFORMANCE BY MARKET REGIME")
        print("-" * 60)
        
        for period in df["period"].unique():
            period_df = df[df["period"] == period]
            print(f"\n{period}:")
            for model_name in period_df["model"].unique():
                model_period = period_df[period_df["model"] == model_name]
                print(f"  {model_name}: Sharpe={model_period['test_sharpe'].mean():.2f}, "
                      f"Acc={model_period['test_accuracy'].mean():.1%}")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
