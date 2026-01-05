"""
Backtest Service
================

Walk-forward backtesting, prediction tracking, and performance analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from ..config import (
    FEATURE_COLUMNS, MACRO_COLUMNS,
    USE_ELASTICNET_SELECT, ELASTICNET_L1_RATIO, ELASTICNET_CV_FOLDS,
    USE_OLSSIGSELECT, OLSSIG_ALPHA, OLSSIG_TOPK, OLSSIG_MINFEATURES,
)
from ..core.features import build_all_features, add_gbm_features, build_target
from ..core.models import make_model, select_features_elasticnet, select_features_ols_pvalue
from ..core.metrics import compute_sharpe, compute_drawdown
from ..data.market import get_price_history
from ..data.macro import get_macro_df
from ..data.fundamentals import get_fundamental_features


def _prepare_features(
    hist: pd.DataFrame,
    ticker: str,
    period: str,
    horizon: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare feature DataFrame with all engineering and macro data.
    Returns (df, feat_cols) where df has target and features.
    """
    hist = build_all_features(hist)
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1, horizon) if horizon > 1 else (1,))
    
    # Macro data (optional)
    try:
        macro_df = get_macro_df(period=period)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"[_prepare_features] Macro data unavailable: {e}")
    
    # Fundamentals (optional)
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[_prepare_features] Fundamental data unavailable: {e}")
    
    # Target
    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)
    
    # Collect available features
    feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
    macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
    feat_cols = feat_cols_available + macro_cols_available
    
    # Quality filter
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    feat_cols = [c for c in feat_cols if data_quality[c] < 0.5]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
    df = hist[cols_needed].dropna().copy()
    
    return df, feat_cols


def track_predictions(
    ticker: str,
    period: str = "1y",
    model_type: str = "rf",
    horizon: int = 1,
) -> tuple[pd.DataFrame, float]:
    """
    Track model predictions over historical test period.
    
    Returns:
        (results_df, accuracy) where results_df has date, actual_close, predicted_return, etc.
    """
    try:
        hist = get_price_history(ticker, period=period, interval="1d")
        if hist is None or hist.empty or len(hist) < 50:
            print(f"[track_predictions] Insufficient data for {ticker}")
            return pd.DataFrame(), 0.0
        
        df, feat_cols = _prepare_features(hist, ticker, period, horizon)
        
        if len(df) < 50:
            print(f"[track_predictions] Not enough data after feature engineering")
            return pd.DataFrame(), 0.0
        
        # Split
        n = len(df)
        min_test = 60
        max_test = 252
        proposed_test = int(n * 0.2)
        test_size = max(min_test, min(proposed_test, max_test, n - 1))
        
        if test_size < 5:
            return pd.DataFrame(), 0.0
        
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
        
        X_train = train_df[feat_cols].values
        y_train = train_df["ftarget_ret_horizon_ahead"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["ftarget_ret_horizon_ahead"].values
        
        # Feature selection (optional)
        if USE_ELASTICNET_SELECT:
            try:
                X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                    X_train, y_train,
                    feature_names=list(feat_cols),
                    dates=train_df.index,
                    horizon=horizon,
                    n_splits=ELASTICNET_CV_FOLDS,
                    l1_ratio=ELASTICNET_L1_RATIO,
                    min_features=10,
                )
                X_train = X_train_sel
                X_test = X_test[:, sel_mask]
                feat_cols = sel_names
            except Exception as e:
                print(f"[track_predictions] ElasticNet failed: {e}")
        
        # Train and predict
        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Build results DataFrame
        results = pd.DataFrame({
            "date": test_df.index,
            "actual_close": hist.loc[test_df.index, "Close"],
            "predicted_return": y_pred,
            "actual_return": y_test,
            "pred_direction": np.sign(y_pred),
            "actual_direction": np.sign(y_test),
            "correct_direction": (np.sign(y_pred) == np.sign(y_test)),
        })
        results["predicted_price"] = results["actual_close"] * (1 + results["predicted_return"])
        
        # GBM confidence bands (if available)
        try:
            from scipy.stats import norm
            T = horizon / 252.0
            mu = hist.loc[test_df.index, "gbm_mu_60d"].astype(float)
            sig = hist.loc[test_df.index, "gbm_sig_60d"].astype(float)
            S0 = results["actual_close"].astype(float)
            m = (mu - 0.5 * sig**2) * T
            s = sig * np.sqrt(T)
            results["gbm_med_price"] = S0 * np.exp(m)
            results["gbm_p05_price"] = S0 * np.exp(m + s * norm.ppf(0.05))
            results["gbm_p95_price"] = S0 * np.exp(m + s * norm.ppf(0.95))
        except Exception:
            pass
        
        accuracy = float(results["correct_direction"].mean())
        return results, accuracy
    
    except Exception as e:
        print(f"[track_predictions] Error for {ticker}: {e}")
        return pd.DataFrame(), 0.0


def backtest_one_ticker(
    ticker: str = "AAPL",
    period: str = "10y",
    test_years: int = 1,
    threshold: float = 0.002,
    model_type: str = "rf",
    horizon: int = 1,
) -> dict[str, Any]:
    """
    Single train/test backtest for a ticker.
    
    Returns dict with metrics: sharpe, accuracy, num_trades, etc.
    """
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return {"error": "No data"}
    
    df, feat_cols = _prepare_features(hist, ticker, period, horizon)
    
    if len(df) < 100:
        return {"error": "Insufficient data"}
    
    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_df = df.loc[df.index < cutoff_date].copy()
    test_df = df.loc[df.index >= cutoff_date].copy()
    
    if len(train_df) < 50 or len(test_df) < 20:
        return {"error": "Insufficient train/test data"}
    
    X_train = train_df[feat_cols].values
    y_train = train_df["ftarget_ret_horizon_ahead"].values
    X_test = test_df[feat_cols].values
    y_test = test_df["ftarget_ret_horizon_ahead"].values
    
    # Feature selection (optional)
    if USE_ELASTICNET_SELECT:
        try:
            X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                X_train, y_train,
                feature_names=list(feat_cols),
                dates=train_df.index,
                horizon=horizon,
                n_splits=ELASTICNET_CV_FOLDS,
                l1_ratio=ELASTICNET_L1_RATIO,
                min_features=10,
            )
            X_train = X_train_sel
            X_test = X_test[:, sel_mask]
        except Exception as e:
            print(f"[backtest_one_ticker] ElasticNet failed: {e}")
    
    # Train and predict
    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Simulate trading
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    pnl = positions * y_test
    
    accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
    sharpe = compute_sharpe(pnl)
    
    # Convert to Series for drawdown
    pnl_series = pd.Series(pnl)
    dd_df = compute_drawdown(pnl_series)
    max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
    
    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": int(np.count_nonzero(np.diff(positions))),
        "total_return": float(pnl.sum()),
    }


def walk_forward_backtest(
    ticker: str = "AAPL",
    period: str = "10y",
    horizon: int = 1,
    model_type: str = "rf",
    train_years: int = 4,
    test_years: int = 1,
    threshold: float = 0.002,
    cost_per_trade: float = 0.0005,
    step_days: int | None = None,
) -> list[dict[str, Any]]:
    """
    Walk-forward cross-validation backtest.
    
    Returns list of fold metrics dicts with: train_start, train_end, test_start, test_end,
    sharpe, accuracy, num_trades, avg_daily_pnl.
    """
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return []
    
    df, feat_cols = _prepare_features(hist, ticker, period, horizon)
    
    if len(df) < 100:
        return []
    
    fold_metrics = []
    
    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    if step_days is None:
        step_days = test_days
    
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        test_start_idx = train_end
        test_end_idx = test_start_idx + test_days
        
        if test_end_idx > len(df):
            break
        
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start_idx:test_end_idx]
        
        if len(train_df) < 50 or len(test_df) < 20:
            start += step_days
            continue
        
        X_train = train_df[feat_cols].values
        y_train = train_df["ftarget_ret_horizon_ahead"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["ftarget_ret_horizon_ahead"].values
        
        # Feature selection (optional)
        current_feat_cols = list(feat_cols)
        if USE_ELASTICNET_SELECT:
            try:
                X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                    X_train, y_train,
                    feature_names=current_feat_cols,
                    dates=train_df.index,
                    horizon=horizon,
                    n_splits=ELASTICNET_CV_FOLDS,
                    l1_ratio=ELASTICNET_L1_RATIO,
                    min_features=10,
                )
                X_train = X_train_sel
                X_test = X_test[:, sel_mask]
                current_feat_cols = sel_names
            except Exception as e:
                print(f"[walk_forward_backtest] ElasticNet failed: {e}")
        
        # Train
        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Simulate trading with costs
        positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
        
        pnl = []
        prev_pos = 0
        for pos, ret in zip(positions, y_test):
            trade = abs(pos - prev_pos)
            pnl.append(pos * ret - cost_per_trade * trade)
            prev_pos = pos
        pnl = np.array(pnl)
        
        accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
        avg_daily = float(pnl.mean())
        std_daily = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily > 0 else 0.0
        
        num_trades = int(np.count_nonzero(np.diff(np.concatenate([[0], (positions != 0).astype(int)]))))
        
        fold_metrics.append({
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "sharpe": float(sharpe),
            "accuracy": accuracy,
            "num_trades": num_trades,
            "avg_daily_pnl": avg_daily,
            "total_pnl": float(pnl.sum()),
            "num_features": len(current_feat_cols),
        })
        
        start += step_days
    
    return fold_metrics
