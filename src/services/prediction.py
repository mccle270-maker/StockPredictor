"""
Prediction Service
==================

Orchestrates data fetching + feature engineering + model prediction.
Main entrypoints: predict_next_for_ticker, predict_long_horizon_for_ticker
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from ..config import (
    FEATURE_COLUMNS, MACRO_COLUMNS,
    USE_ELASTICNET_SELECT, ELASTICNET_L1_RATIO, ELASTICNET_CV_FOLDS,
    USE_OLSSIGSELECT, OLSSIG_ALPHA, OLSSIG_TOPK, OLSSIG_MINFEATURES,
    is_elasticnet_enabled,  # Runtime check function
)
from ..core.features import build_all_features, add_gbm_features, build_target, get_available_features
from ..core.models import make_model, select_features_elasticnet, select_features_ols_pvalue
from ..data.market import get_price_history
from ..data.macro import get_macro_df
from ..data.fundamentals import get_fundamental_features

# Optional: long horizon module
try:
    from long_horizon import predict_long_horizon
    HAS_LONG_HORIZON = True
except ImportError:
    HAS_LONG_HORIZON = False

# Optional: GAF-CNN model
try:
    import tensorflow as tf
    gafcnn = tf.keras.models.load_model("gaf_cnn_updown.keras")
    from pyts.image import GramianAngularField
    HAS_GAF = True
except Exception:
    gafcnn = None
    HAS_GAF = False


def build_features_and_target(
    ticker: str,
    period: str = "5y",
    horizon: int = 1,
    use_vol_scaled_target: bool = False,
    run_gaf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float | None, float | None, pd.DatetimeIndex]:
    """
    Build feature matrix X, target y, and the last row x_last for prediction.
    
    Returns:
        (X, y, x_last, last_close, last_vol_20d, prob_up_gaf, last_rsi, dates)
    """
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No price history for {ticker}")
    
    # Add all features (technical, momentum, volatility, etc.)
    hist = build_all_features(hist)
    
    # Add GBM probabilistic features (if not already added)
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1, horizon) if horizon > 1 else (1,))
    
    # Macro data (optional)
    try:
        macro_df = get_macro_df(period=period)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"[build_features_and_target] Macro data unavailable: {e}")
    
    # Fundamentals (optional)
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[build_features_and_target] Fundamental data unavailable: {e}")
    
    # Target: forward return
    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)
    
    # Volatility-scaled target (optional)
    vol_20d = hist["Close"].pct_change().rolling(20).std()
    if use_vol_scaled_target:
        hist["ftarget_ret_horizon_ahead"] = hist["ftarget_ret_horizon_ahead"] / (vol_20d + 1e-9)
    
    last_vol_20d = float(vol_20d.iloc[-1]) if not vol_20d.empty else 0.01
    last_close = float(hist["Close"].iloc[-1])
    
    # Extract RSI for signal filtering
    last_rsi = None
    if "rsi14" in hist.columns:
        rsi_vals = hist["rsi14"].dropna()
        if len(rsi_vals) > 0:
            last_rsi = float(rsi_vals.iloc[-1])
    
    # Collect available feature columns (deduplicate)
    feat_cols_available = list(dict.fromkeys([c for c in FEATURE_COLUMNS if c in hist.columns]))
    macro_cols_available = list(dict.fromkeys([c for c in MACRO_COLUMNS if c in hist.columns]))
    feat_cols = feat_cols_available + [c for c in macro_cols_available if c not in feat_cols_available]
    
    # Quality filter: drop mostly-missing features
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    # Convert to dict for safe scalar access
    data_quality_dict = data_quality.to_dict() if hasattr(data_quality, 'to_dict') else {}
    feat_cols = [c for c in feat_cols if data_quality_dict.get(c, 0) < 0.5]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
    df = hist[cols_needed].dropna().copy()
    
    if len(df) < 60:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows after cleaning")
    
    # GAF-CNN probability (optional)
    prob_up_gaf = None
    if run_gaf and HAS_GAF and gafcnn is not None:
        try:
            rets = hist["Close"].pct_change().dropna()
            if len(rets) >= 30:
                window_vals = rets.values[-30:].reshape(1, -1)
                gaf = GramianAngularField(image_size=30, method="summation")
                Xgaf = gaf.fit_transform(window_vals)
                prob_up_gaf = float(gafcnn.predict(Xgaf[..., np.newaxis], verbose=0)[0][0])
        except Exception as e:
            print(f"[build_features_and_target] GAF-CNN failed: {e}")
    
    X = df[feat_cols].values[:-1]  # All but last (no target for last)
    y = df["ftarget_ret_horizon_ahead"].values[:-1]
    x_last = df[feat_cols].values[-1]
    dates = df.index[:-1]
    
    return X, y, x_last, last_close, last_vol_20d, prob_up_gaf, last_rsi, dates


def predict_next_for_ticker(
    ticker: str = "^GSPC",
    period: str = "5y",
    model_type: str = "rf",
    horizon: int = 1,
    use_vol_scaled_target: bool = False,
    auto_optimize: bool = True,
    run_gaf: bool = False,
) -> dict[str, Any]:
    """
    Generate prediction for next N days (horizon).
    
    Returns dict with: ticker, model_type, horizon, last_close, vol_20d, pe_ratio,
    pred_next_ret, pred_next_price, prob_up, prob_down, prob_up_gaf, confidence_score,
    num_features, top_features, rsi14, elasticnet metadata.
    """
    X, y, x_last, last_close, last_vol_20d, prob_up_gaf, last_rsi, dates = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=use_vol_scaled_target,
        run_gaf=run_gaf,
    )
    
    n = len(X)
    actual_feat_cols = [f"feat_{i}" for i in range(X.shape[1])]
    
    # OLS significance selection (optional)
    if USE_OLSSIGSELECT:
        train_end = int(n * 0.8)
        X_train_ols = X[:train_end]
        y_train_ols = y[:train_end]
        
        X_train_sel, ols_names, ols_mask = select_features_ols_pvalue(
            X_train_ols, y_train_ols,
            feature_names=list(actual_feat_cols),
            alpha=OLSSIG_ALPHA,
            top_k=OLSSIG_TOPK,
            min_features=OLSSIG_MINFEATURES,
        )
        X = X[:, ols_mask]
        x_last = x_last[ols_mask]
        actual_feat_cols = ols_names
    
    # ElasticNet selection (optional) - check at RUNTIME
    use_elasticnet_now = is_elasticnet_enabled()
    if use_elasticnet_now:
        try:
            train_end = int(n * 0.8)
            X_en_train = X[:train_end]
            y_en_train = y[:train_end]
            
            X_en_sel, en_names, en_mask = select_features_elasticnet(
                X_en_train, y_en_train,
                feature_names=list(actual_feat_cols),
                dates=dates[:train_end],
                horizon=horizon,
                n_splits=ELASTICNET_CV_FOLDS,
                l1_ratio=ELASTICNET_L1_RATIO,
                min_features=10,
            )
            X = X[:, en_mask]
            x_last = x_last[en_mask]
            actual_feat_cols = en_names
            print(f"✂️ {ticker} ElasticNet selected {len(actual_feat_cols)} features")
        except Exception as e:
            print(f"{ticker} ElasticNet selection failed: {e}")
    else:
        print(f"📊 {ticker} Using ALL {X.shape[1]} features (ElasticNet disabled)")
    
    # Auto-optimize: prune weak features
    train_end = int(n * 0.8)
    X_train = X[:train_end]
    y_train = y[:train_end]
    
    if auto_optimize:
        model_init = make_model(model_type=model_type, random_state=42, task="reg")
        model_init.fit(X_train, y_train)
        
        if hasattr(model_init, "feature_importances_"):
            importance = model_init.feature_importances_
            important_mask = importance > 0.001
            X_train = X_train[:, important_mask]
            x_last = x_last[important_mask]
            actual_feat_cols = [actual_feat_cols[i] for i in range(len(actual_feat_cols)) if important_mask[i]]
            print(f"{ticker} Using {len(actual_feat_cols)} features after pruning")
    
    # Final model training
    model = make_model(model_type=model_type, random_state=42, task="reg")
    model.fit(X_train, y_train)
    
    # Prediction
    pred_ret = float(model.predict(x_last.reshape(1, -1))[0])
    if use_vol_scaled_target:
        pred_ret = pred_ret * float(last_vol_20d)
    
    confidence_score = float(abs(pred_ret))
    pred_price = float(last_close * (1 + pred_ret))
    
    # Classification for probabilities
    prob_up = None
    prob_down = None
    try:
        y_dir = (y > 0).astype(int)
        y_dir_train = y_dir[:train_end]
        clf = make_model(model_type=model_type, random_state=42, task="clf")
        clf.fit(X_train, y_dir_train)
        
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x_last.reshape(1, -1))[0]
            if hasattr(clf, "classes_") and 1 in clf.classes_:
                idx_up = list(clf.classes_).index(1)
                prob_up = float(proba[idx_up])
                prob_down = float(1.0 - prob_up)
            else:
                prob_up = float(proba.max())
                prob_down = float(1.0 - prob_up)
    except Exception:
        pass
    
    # Fundamentals for display
    fund_feats = get_fundamental_features(ticker)
    pe_ratio = fund_feats.get("fund_pe_trailing", None)
    
    # Compute prediction z-score (how extreme is this prediction vs training predictions)
    pred_zscore = 0.0
    try:
        # Get predictions for training data to compute mean/std
        train_preds = model.predict(X_train)
        train_mean = float(np.mean(train_preds))
        train_std = float(np.std(train_preds))
        if train_std > 1e-9:
            pred_zscore = float((pred_ret - train_mean) / train_std)
    except Exception:
        pass
    
    # Feature importance
    top_features_str = "NA"
    if hasattr(model, "feature_importances_"):
        importance_dict = dict(zip(actual_feat_cols, model.feature_importances_))
        top_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        top_features_str = ", ".join([f"{f}:{v:.3f}" for f, v in top_feats])
    
    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "last_close": last_close,
        "vol_20d": last_vol_20d,
        "pe_ratio": pe_ratio,
        "pred_next_ret": pred_ret,
        "pred_zscore": pred_zscore,  # Z-score of prediction
        "confidence_score": confidence_score,
        "pred_next_price": pred_price,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_up_gaf": prob_up_gaf,
        "rsi14": last_rsi,  # For signal filtering
        "num_features": len(actual_feat_cols),
        "top_features": top_features_str,
        "elasticnet_enabled": use_elasticnet_now,  # Runtime check value
        "elasticnet_l1_ratio": float(ELASTICNET_L1_RATIO),
        "elasticnet_cv_folds": int(ELASTICNET_CV_FOLDS),
        "elasticnet_selected_n": len(actual_feat_cols) if use_elasticnet_now else None,
    }


def predict_long_horizon_for_ticker(
    ticker: str,
    period: str = "5y",
    k: int = 200,
) -> dict[str, Any]:
    """
    Return analog/regime 20-30d view using long_horizon module.
    
    Gracefully falls back across periods and returns error dict if unavailable.
    """
    if not HAS_LONG_HORIZON:
        return {"ticker": ticker, "error": "long_horizon module unavailable"}
    
    fallback_periods = ["5y", "3y", "2y", "1y", "6mo", "3mo"]
    periods_to_try = [period] + [p for p in fallback_periods if p != period]
    
    last_error: Exception | None = None
    for per in periods_to_try:
        try:
            hist = get_price_history(ticker, period=per, interval="1d")
            if hist is None or hist.empty or len(hist) < 120:
                raise ValueError(f"Insufficient history for {ticker} period={per}")
            
            res = predict_long_horizon(hist)
            if res is None:
                raise ValueError("long_horizon returned None")
            
            return {
                "ticker": ticker,
                "period": per,
                "p_up_30d": res.p_up_30d,
                "ret_p10_30d": res.ret_p10_30d,
                "ret_p50_30d": res.ret_p50_30d,
                "ret_p90_30d": res.ret_p90_30d,
                "vol_expansion_prob": res.vol_expansion_prob,
                "flags": res.flags,
                "effective_sample_size": res.effective_sample_size,
                "analog_count": res.analog_count,
            }
        except Exception as e:
            last_error = e
            continue
    
    return {"ticker": ticker, "error": str(last_error) if last_error else "unknown"}
