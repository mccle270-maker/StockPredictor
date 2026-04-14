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
from ..core.metrics import compute_sharpe, compute_calmar, compute_drawdown
from ..core.feature_monitor import FeatureImportanceTracker
from ..data.market import get_price_history, get_spx
from ..data.macro import get_macro_df, get_vix
from ..data.fundamentals import get_fundamental_features


def _prepare_features(
    hist: pd.DataFrame,
    ticker: str,
    period: str,
    horizon: int,
    use_vol_scaled_target: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare feature DataFrame with all engineering and macro data.
    Returns (df, feat_cols) where df has target and features.
    """
    spx_df = None
    vix_series = None
    try:
        spx_df = get_spx(hist.index.min(), hist.index.max())
    except Exception as e:
        print(f"[_prepare_features] SPX data unavailable: {e}")
    
    try:
        vix_series = get_vix(period=period)
    except Exception as e:
        print(f"[_prepare_features] VIX data unavailable: {e}")
    
    hist = build_all_features(hist, spx_df=spx_df, vix_series=vix_series)
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1, horizon) if horizon > 1 else (1,))
    
    # Macro data (optional)
    try:
        spx_returns = None
        if spx_df is not None and not spx_df.empty and "Close" in spx_df.columns:
            spx_returns = spx_df["Close"].pct_change()
        macro_df = get_macro_df(period=period, spx_returns=spx_returns)
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
    target_scale = hist.get("vol_20d", pd.Series(index=hist.index, dtype=float)).astype(float)
    target_scale = target_scale.replace([np.inf, -np.inf], np.nan).clip(lower=1e-4)
    hist["target_vol_scale"] = target_scale
    hist["ftarget_ret_horizon_ahead_vol_scaled"] = hist["ftarget_ret_horizon_ahead"] / hist["target_vol_scale"]
    
    # Collect available features (deduplicate)
    feat_cols_available = list(dict.fromkeys([c for c in FEATURE_COLUMNS if c in hist.columns]))
    macro_cols_available = list(dict.fromkeys([c for c in MACRO_COLUMNS if c in hist.columns]))
    feat_cols = feat_cols_available + [c for c in macro_cols_available if c not in feat_cols_available]
    
    # Quality filter
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    # Convert to dict for safe scalar access
    data_quality_dict = data_quality.to_dict() if hasattr(data_quality, 'to_dict') else {}
    feat_cols = [c for c in feat_cols if data_quality_dict.get(c, 0) < 0.5]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    cols_needed = feat_cols + [
        "ftarget_ret_horizon_ahead",
        "ftarget_ret_horizon_ahead_vol_scaled",
        "target_vol_scale",
    ]
    df = hist[cols_needed].dropna().copy()
    
    return df, feat_cols


def track_predictions(
    ticker: str,
    period: str = "1y",
    model_type: str = "rf",
    horizon: int = 1,
    use_vol_scaled_target: bool = False,
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
        
        df, feat_cols = _prepare_features(hist, ticker, period, horizon, use_vol_scaled_target=use_vol_scaled_target)
        
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
        train_target_col = "ftarget_ret_horizon_ahead_vol_scaled" if use_vol_scaled_target else "ftarget_ret_horizon_ahead"
        is_binary_xgb = model_type == "xgb_binary"
        y_train = train_df[train_target_col].values
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
        if is_binary_xgb:
            model = make_model(
                model_type="xgb",
                random_state=42,
                task="clf",
                n_estimators=250,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=20,
                reg_alpha=0.5,
                reg_lambda=12.0,
            )
            model.fit(X_train, (train_df["ftarget_ret_horizon_ahead"].values > 0).astype(int))
            prob_up = model.predict_proba(X_test)[:, 1]
            y_pred = (prob_up - 0.5) * 2.0 * np.maximum(test_df["target_vol_scale"].values, 0.005)
        else:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        if use_vol_scaled_target and not is_binary_xgb:
            y_pred = y_pred * test_df["target_vol_scale"].values
        
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
    use_vol_scaled_target: bool = False,
) -> dict[str, Any]:
    """
    Single train/test backtest for a ticker.
    
    Returns dict with metrics: sharpe, accuracy, num_trades, etc.
    """
    # --- Config snapshot integration ---
    from ..core.versioning import create_config_snapshot, save_config_snapshot
    from ..config import get_model_version_info, FEATURE_COLUMNS
    import os
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return {"error": "No data"}

    df, feat_cols = _prepare_features(hist, ticker, period, horizon, use_vol_scaled_target=use_vol_scaled_target)

    if len(df) < 100:
        return {"error": "Insufficient data"}

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_df = df.loc[df.index < cutoff_date].copy()
    test_df = df.loc[df.index >= cutoff_date].copy()

    if len(train_df) < 50 or len(test_df) < 20:
        return {"error": "Insufficient train/test data"}

    X_train = train_df[feat_cols].values
    train_target_col = "ftarget_ret_horizon_ahead_vol_scaled" if use_vol_scaled_target else "ftarget_ret_horizon_ahead"
    is_binary_xgb = model_type == "xgb_binary"
    y_train = train_df[train_target_col].values
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
    if is_binary_xgb:
        model = make_model(
            model_type="xgb",
            random_state=42,
            task="clf",
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=20,
            reg_alpha=0.5,
            reg_lambda=12.0,
        )
        model.fit(X_train, (train_df["ftarget_ret_horizon_ahead"].values > 0).astype(int))
        prob_up = model.predict_proba(X_test)[:, 1]
        y_pred = (prob_up > 0.5).astype(float) - (prob_up <= 0.5).astype(float)
    else:
        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    if use_vol_scaled_target and not is_binary_xgb:
        y_pred = y_pred * test_df["target_vol_scale"].values

    # Simulate trading
    if is_binary_xgb:
        positions = y_pred
    else:
        positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    pnl = positions * y_test

    accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
    pnl_series = pd.Series(pnl, index=test_df.index)
    sharpe = compute_sharpe(pnl_series)
    calmar = compute_calmar(pnl_series)

    # Convert to Series for drawdown
    dd_df = compute_drawdown(pnl_series)
    max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0

    # --- Create and save config snapshot ---
    model_info = get_model_version_info(model_type)
    feature_set = {"features": list(FEATURE_COLUMNS)}
    snapshot = create_config_snapshot(
        model_version=model_info.get("version", model_type),
        feature_set=feature_set,
    )
    snap_path = save_config_snapshot(snapshot)
    run_id = snapshot["run_id"]
    print(f"[Config Snapshot] Saved: {os.path.relpath(snap_path)} (run_id={run_id})")

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "test_start": str(test_df.index[0].date()),
        "test_end": str(test_df.index[-1].date()),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "num_trades": int(np.count_nonzero(np.diff(positions))),
        "total_return": float(pnl.sum()),
        "run_id": run_id,
        "config_snapshot": str(snap_path),
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
    track_feature_importance: bool = True,
    purge_gap_days: int = 0,
    embargo_days: int = 0,
    use_vol_scaled_target: bool = False,
) -> list[dict[str, Any]]:
    """
    Walk-forward cross-validation backtest.
    
    Returns list of fold metrics dicts with: train_start, train_end, test_start, test_end,
    sharpe, accuracy, num_trades, avg_daily_pnl.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., '10y')
        horizon: Prediction horizon in days
        model_type: Model type ('rf', 'xgb', etc.)
        train_years: Years of training data per fold
        test_years: Years of test data per fold
        threshold: Minimum prediction for position entry
        cost_per_trade: Transaction cost per trade
        step_days: Days to step forward between folds (default: test_days)
        track_feature_importance: If True, track and save feature importances
        purge_gap_days: Number of days to skip between train end and test start
        embargo_days: Number of days to skip after each test fold before next fold
    
    Returns:
        List of fold metrics dicts, with 'stable_features' key if tracking enabled
    """
    # --- Config snapshot integration ---
    from ..core.versioning import create_config_snapshot, save_config_snapshot
    from ..config import get_model_version_info, FEATURE_COLUMNS, CACHE_DIR
    import os
    
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return []

    df, feat_cols = _prepare_features(hist, ticker, period, horizon, use_vol_scaled_target=use_vol_scaled_target)

    if len(df) < 100:
        return []

    # Create config snapshot ONCE per run
    model_info = get_model_version_info(model_type)
    feature_set = {"features": list(FEATURE_COLUMNS)}
    snapshot = create_config_snapshot(
        model_version=model_info.get("version", model_type),
        feature_set=feature_set,
    )
    snap_path = save_config_snapshot(snapshot)
    run_id = snapshot["run_id"]
    print(f"[Config Snapshot] Saved: {os.path.relpath(snap_path)} (run_id={run_id})")

    # Initialize feature importance tracker
    feature_tracker = None
    if track_feature_importance:
        tracker_path = CACHE_DIR / f"feature_importance_{ticker}_{model_type}.json"
        feature_tracker = FeatureImportanceTracker(save_path=tracker_path, auto_save=False)
        feature_tracker.clear()  # Start fresh for this run

    fold_metrics = []

    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    if step_days is None:
        step_days = test_days

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        test_start_idx = train_end + max(0, purge_gap_days)
        test_end_idx = test_start_idx + test_days

        if test_end_idx > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start_idx:test_end_idx]

        if len(train_df) < 50 or len(test_df) < 20:
            start += step_days
            continue

        X_train = train_df[feat_cols].values
        train_target_col = "ftarget_ret_horizon_ahead_vol_scaled" if use_vol_scaled_target else "ftarget_ret_horizon_ahead"
        is_binary_xgb = model_type == "xgb_binary"
        y_train = train_df[train_target_col].values
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
        if is_binary_xgb:
            model = make_model(
                model_type="xgb",
                random_state=42,
                task="clf",
                n_estimators=250,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                min_child_weight=20,
                reg_alpha=0.5,
                reg_lambda=12.0,
            )
            model.fit(X_train, (train_df["ftarget_ret_horizon_ahead"].values > 0).astype(int))
            prob_up = model.predict_proba(X_test)[:, 1]
            y_pred = (prob_up > 0.5).astype(float) - (prob_up <= 0.5).astype(float)
        else:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        if use_vol_scaled_target and not is_binary_xgb:
            y_pred = y_pred * test_df["target_vol_scale"].values

        # Track feature importances
        fold_id = len(fold_metrics)
        if feature_tracker is not None and hasattr(model, "feature_importances_"):
            feature_tracker.update_importance(
                fold_id=fold_id,
                feature_names=current_feat_cols,
                importances=model.feature_importances_,
                ticker=ticker,
                model_type=model_type,
                extra={
                    "train_start": str(train_df.index[0].date()),
                    "train_end": str(train_df.index[-1].date()),
                }
            )

        # Simulate trading with costs
        if is_binary_xgb:
            positions = y_pred
        else:
            positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))

        pnl = []
        prev_pos = 0
        for pos, ret in zip(positions, y_test):
            trade = abs(pos - prev_pos)
            pnl.append(pos * ret - cost_per_trade * trade)
            prev_pos = pos
        pnl = np.array(pnl)

        accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
        pnl_series = pd.Series(pnl, index=test_df.index)
        avg_daily = float(pnl.mean())
        std_daily = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily > 0 else 0.0
        calmar = compute_calmar(pnl_series)

        num_trades = int(np.count_nonzero(np.diff(np.concatenate([[0], (positions != 0).astype(int)]))))

        fold_metrics.append({
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
            "sharpe": float(sharpe),
            "calmar": float(calmar) if calmar is not None else None,
            "accuracy": accuracy,
            "num_trades": num_trades,
            "avg_daily_pnl": avg_daily,
            "total_pnl": float(pnl.sum()),
            "num_features": len(current_feat_cols),
            "purge_gap_days": int(max(0, purge_gap_days)),
            "embargo_days": int(max(0, embargo_days)),
            "use_vol_scaled_target": bool(use_vol_scaled_target),
            "run_id": run_id,
            "config_snapshot": str(snap_path),
        })

        start += step_days + max(0, embargo_days)

    # Save feature importance tracker and get stable features
    stable_features = []
    if feature_tracker and len(feature_tracker) > 0:
        feature_tracker.save()
        # Require features to appear in at least half the folds (min 2)
        min_folds_required = max(2, len(fold_metrics) // 2)
        stable_features = feature_tracker.get_stable_features(
            min_folds=min_folds_required,
            min_avg_importance=0.01,
            ticker=ticker,
        )
        print(f"[Feature Tracker] Saved {len(feature_tracker)} folds, {len(stable_features)} stable features")
        
        # Add stable features and summary to each fold metric
        summary_df = feature_tracker.get_summary_df(ticker=ticker)
        top_features = feature_tracker.get_top_features(n=10, ticker=ticker)
        
        for fold in fold_metrics:
            fold["stable_features"] = stable_features
            fold["top_features"] = top_features
    
    return fold_metrics
