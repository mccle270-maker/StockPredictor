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

from sklearn.calibration import CalibratedClassifierCV

from ..config import (
    FEATURE_COLUMNS, MACRO_COLUMNS,
    USE_ELASTICNET_SELECT, ELASTICNET_L1_RATIO, ELASTICNET_CV_FOLDS,
    USE_OLSSIGSELECT, OLSSIG_ALPHA, OLSSIG_TOPK, OLSSIG_MINFEATURES,
    is_elasticnet_enabled,  # Runtime check function
    is_regime_models_enabled, get_min_samples_per_regime,  # Regime models config
    is_ff_residual_enabled,  # Fama-French residual target config
)
from ..core.features import build_all_features, add_gbm_features, build_target, get_available_features, fetch_ff3_factors, build_ff_residual_target
from ..core.models import make_model, select_features_elasticnet, select_features_ols_pvalue, predict_with_uncertainty
from ..core.regime_predictor import RegimeAwarePredictor
from ..core.regime_filter import MarketRegime, get_current_regime
from ..data.market import get_price_history, get_spx
from ..data.macro import get_macro_df, get_vix
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
    
    spx_df = None
    vix_series = None
    try:
        spx_df = get_spx(hist.index.min(), hist.index.max())
    except Exception as e:
        print(f"[build_features_and_target] SPX data unavailable: {e}")
    
    try:
        vix_series = get_vix(period=period)
    except Exception as e:
        print(f"[build_features_and_target] VIX data unavailable: {e}")
    
    # Add all features (technical, momentum, volatility, etc.)
    hist = build_all_features(hist, spx_df=spx_df, vix_series=vix_series)
    
    # Add GBM probabilistic features (if not already added)
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
        print(f"[build_features_and_target] Macro data unavailable: {e}")
    
    # Fundamentals (optional)
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[build_features_and_target] Fundamental data unavailable: {e}")
    
    # Target: forward return (uses winsorization from config by default)
    hist["ftarget_ret_horizon_ahead"] = build_target(hist, horizon=horizon)
    
    # Volatility-scaled target (optional, applied after winsorization)
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
    use_regime_models: bool | None = None,
    use_calibrated_probs: bool = True,
) -> dict[str, Any]:
    """
    Generate prediction for next N days (horizon).
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for training
        model_type: Base model type ('rf', 'xgb', etc.)
        horizon: Prediction horizon in days
        use_vol_scaled_target: Scale target by volatility
        auto_optimize: Prune weak features
        run_gaf: Run GAF-CNN model
        use_regime_models: If True, train separate models for BULL/BEAR/NEUTRAL regimes.
                           If None, reads from REGIME_MODELS_CONFIG.
        use_calibrated_probs: If True, use CalibratedClassifierCV for better probability 
                              estimates (isotonic regression, cv=5).
    
    Returns dict with: ticker, model_type, horizon, last_close, vol_20d, pe_ratio,
    pred_next_ret, pred_next_price, prob_up, prob_down, prob_up_gaf, confidence_score,
    num_features, top_features, rsi14, elasticnet metadata, regime info.
    """
    # Check config if not explicitly set
    if use_regime_models is None:
        use_regime_models = is_regime_models_enabled()
    
    X, y, x_last, last_close, last_vol_20d, prob_up_gaf, last_rsi, dates = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=use_vol_scaled_target,
        run_gaf=run_gaf,
    )

    n = len(X)
    actual_feat_cols = [f"feat_{i}" for i in range(X.shape[1])]
    is_binary_xgb = model_type == "xgb_binary"
    
    # OLS significance selection (optional)
    if USE_OLSSIGSELECT:
        train_end = int(n * 0.8)
        X_train_ols = pd.DataFrame(X[:train_end], columns=actual_feat_cols)
        y_train_ols = pd.Series(y[:train_end], index=dates[:train_end])

        ols_names = select_features_ols_pvalue(
            X_train_ols,
            y_train_ols,
            alpha=OLSSIG_ALPHA,
            top_k=OLSSIG_TOPK,
            min_features=OLSSIG_MINFEATURES,
        )
        if ols_names:
            ols_mask = np.array([name in set(ols_names) for name in actual_feat_cols], dtype=bool)
            X = X[:, ols_mask]
            x_last = x_last[ols_mask]
            actual_feat_cols = [name for name in actual_feat_cols if name in set(ols_names)]
    
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
        init_model_type = "xgb" if is_binary_xgb else model_type
        init_task = "clf" if is_binary_xgb else "reg"
        init_kwargs = {}
        if is_binary_xgb:
            init_kwargs = {
                "n_estimators": 250,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "reg_alpha": 0.5,
                "reg_lambda": 12.0,
            }
        model_init = make_model(model_type=init_model_type, random_state=42, task=init_task, **init_kwargs)
        init_y_train = (y_train > 0).astype(int) if is_binary_xgb else y_train
        model_init.fit(X_train, init_y_train)
        
        if hasattr(model_init, "feature_importances_"):
            importance = model_init.feature_importances_
            important_mask = importance > 0.001
            X_train = X_train[:, important_mask]
            x_last = x_last[important_mask]
            actual_feat_cols = [actual_feat_cols[i] for i in range(len(actual_feat_cols)) if important_mask[i]]
            print(f"{ticker} Using {len(actual_feat_cols)} features after pruning")

    if is_binary_xgb:
        y_dir = (y > 0).astype(int)
        y_train_dir = y_dir[:train_end]
        clf = make_model(
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
        clf.fit(X_train, y_train_dir)
        proba = clf.predict_proba(x_last.reshape(1, -1))[0]
        prob_down = float(proba[0])
        prob_up = float(proba[1])
        pred_ret = float((prob_up - prob_down) * max(last_vol_20d, 0.005))
        confidence_score = float(abs(prob_up - 0.5) * 2.0)
        pred_price = float(last_close * (1 + pred_ret))
        calibrated = False
        top_features_str = "NA"
        if hasattr(clf, "feature_importances_"):
            importance_dict = dict(zip(actual_feat_cols, clf.feature_importances_))
            top_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features_str = ", ".join([f"{f}:{v:.3f}" for f, v in top_feats])

        return {
            "ticker": ticker,
            "model_type": model_type,
            "horizon": horizon,
            "last_close": last_close,
            "vol_20d": last_vol_20d,
            "pe_ratio": None,
            "pred_next_ret": pred_ret,
            "pred_next_price": pred_price,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "calibrated_probs": calibrated,
            "prob_up_gaf": prob_up_gaf,
            "rsi14": last_rsi,
            "num_features": len(actual_feat_cols),
            "top_features": top_features_str,
            "confidence_score": confidence_score,
            "pred_zscore": float((prob_up - 0.5) / 0.1),
            "elasticnet_enabled": bool(use_elasticnet_now),
            "elasticnet_l1_ratio": float(ELASTICNET_L1_RATIO),
            "elasticnet_cv_folds": int(ELASTICNET_CV_FOLDS),
            "elasticnet_selected_n": len(actual_feat_cols) if use_elasticnet_now else None,
            "use_regime_models": False,
            "current_regime": None,
            "regime_stats": None,
            "pred_lower_bound": None,
            "pred_upper_bound": None,
        }
    
    # Determine current regime for prediction
    current_regime = MarketRegime.NEUTRAL
    regime_stats = None
    
    if use_regime_models:
        # Build historical regime labels from training data
        # Use regime_bull/regime_bear features if available, otherwise detect from price
        try:
            hist = get_price_history(ticker, period=period, interval="1d")
            close = hist["Close"]
            sma_200 = close.rolling(200).mean()
            
            # Classify each day's regime (simple: above/below 200DMA)
            # Use NEUTRAL for warmup period where 200DMA is unavailable
            regimes_hist = []
            for i in range(len(close)):
                if pd.isna(sma_200.iloc[i]):
                    regimes_hist.append(MarketRegime.NEUTRAL)
                elif close.iloc[i] > sma_200.iloc[i]:
                    regimes_hist.append(MarketRegime.BULL)
                else:
                    regimes_hist.append(MarketRegime.BEAR)
            
            # Align regimes with training data (need to match dates)
            # Training data is hist[:-1] aligned, so use matching indices
            n_hist = len(hist)
            # Map training indices to regimes (accounting for dropna)
            regimes_train = []
            for i in range(train_end):
                # Approximate index mapping (training data may have been filtered)
                hist_idx = min(i + 200, n_hist - 2)  # Account for warmup period
                regimes_train.append(regimes_hist[hist_idx])
            
            regimes_train = np.array(regimes_train)
            
            # Get current regime
            current_regime_state = get_current_regime()
            current_regime = current_regime_state.regime
            
            # Train regime-aware predictor
            min_samples = get_min_samples_per_regime()
            regime_predictor = RegimeAwarePredictor(
                base_model_type=model_type,
                min_samples=min_samples,
                use_optimized=True,
            )
            regime_predictor.fit(X_train, y_train, regimes_train)
            regime_stats = regime_predictor.get_stats()
            
            # Predict using current regime
            # For regime predictor, use the underlying model for uncertainty
            pred_ret = float(regime_predictor.predict(x_last.reshape(1, -1), current_regime)[0])
            
            # Get uncertainty from the active regime's model
            active_model = regime_predictor.get_model_for_regime(current_regime)
            if active_model is not None:
                _, lower_arr, upper_arr = predict_with_uncertainty(active_model, x_last.reshape(1, -1))
                pred_lower_bound = float(lower_arr[0]) if lower_arr is not None else None
                pred_upper_bound = float(upper_arr[0]) if upper_arr is not None else None
            else:
                pred_lower_bound = None
                pred_upper_bound = None
            
            model = regime_predictor  # For feature importance extraction
            
            print(f"🔀 {ticker} Using RegimeAwarePredictor (current regime: {current_regime.value})")
            
        except Exception as e:
            print(f"⚠️ {ticker} Regime model failed, falling back to single model: {e}")
            use_regime_models = False  # Fall back to standard model
    
    if not use_regime_models:
        # Standard single model training
        model = make_model(model_type=model_type, random_state=42, task="reg")
        model.fit(X_train, y_train)
        
        # Prediction with uncertainty bounds
        pred_arr, lower_arr, upper_arr = predict_with_uncertainty(model, x_last.reshape(1, -1))
        pred_ret = float(pred_arr[0])
        pred_lower_bound = float(lower_arr[0]) if lower_arr is not None else None
        pred_upper_bound = float(upper_arr[0]) if upper_arr is not None else None
        
    if use_vol_scaled_target:
        pred_ret = pred_ret * float(last_vol_20d)
        # Also scale uncertainty bounds
        if pred_lower_bound is not None:
            pred_lower_bound = pred_lower_bound * float(last_vol_20d)
        if pred_upper_bound is not None:
            pred_upper_bound = pred_upper_bound * float(last_vol_20d)
    
    confidence_score = float(abs(pred_ret))
    pred_price = float(last_close * (1 + pred_ret))
    
    # Classification for probabilities (with optional calibration)
    prob_up = None
    prob_down = None
    calibrated = False
    try:
        y_dir = (y > 0).astype(int)
        y_dir_train = y_dir[:train_end]
        base_clf = make_model(model_type=model_type, random_state=42, task="clf")
        
        # Use CalibratedClassifierCV for better probability estimates
        # Requires at least 50 samples for meaningful calibration with cv=5
        if use_calibrated_probs and len(X_train) >= 50:
            clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
            calibrated = True
        else:
            clf = base_clf
        
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
        if use_regime_models and isinstance(model, RegimeAwarePredictor):
            train_preds = model.predict_with_regime_array(X_train, regimes_train)
        else:
            train_preds = model.predict(X_train)
        train_mean = float(np.mean(train_preds))
        train_std = float(np.std(train_preds))
        if train_std > 1e-9:
            pred_zscore = float((pred_ret - train_mean) / train_std)
    except Exception:
        pass
    
    # Feature importance
    top_features_str = "NA"
    if use_regime_models and isinstance(model, RegimeAwarePredictor):
        # Get feature importance from current regime's model
        try:
            regime_group = model._normalize_regime(current_regime)
            importances = model.get_feature_importances(regime=regime_group)
            if regime_group in importances:
                importance_dict = dict(zip(actual_feat_cols, importances[regime_group]))
                top_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features_str = ", ".join([f"{f}:{v:.3f}" for f, v in top_feats])
        except Exception:
            pass
    elif hasattr(model, "feature_importances_"):
        importance_dict = dict(zip(actual_feat_cols, model.feature_importances_))
        top_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        top_features_str = ", ".join([f"{f}:{v:.3f}" for f, v in top_feats])
    
    result = {
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
        "calibrated_probs": calibrated,  # Whether probabilities are calibrated
        "prob_up_gaf": prob_up_gaf,
        "rsi14": last_rsi,  # For signal filtering
        "num_features": len(actual_feat_cols),
        "top_features": top_features_str,
        "elasticnet_enabled": use_elasticnet_now,  # Runtime check value
        "elasticnet_l1_ratio": float(ELASTICNET_L1_RATIO),
        "elasticnet_cv_folds": int(ELASTICNET_CV_FOLDS),
        "elasticnet_selected_n": len(actual_feat_cols) if use_elasticnet_now else None,
        # Regime model info
        "use_regime_models": use_regime_models,
        "current_regime": current_regime.value if current_regime else None,
        "regime_stats": regime_stats,
        # Prediction uncertainty bounds (90% CI)
        "pred_lower_bound": pred_lower_bound,
        "pred_upper_bound": pred_upper_bound,
    }
    
    return result


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
