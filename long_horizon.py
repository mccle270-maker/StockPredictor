"""
Long-horizon (20–30d) analog/regime layer.

- Uses slow, structural features only (trend, vol regime, volume/flow proxy, relative-strength proxy).
- Regime-conditioned similarity with time decay; returns probabilities and quantiles.
- Side-car module: does not modify short-horizon models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LongHorizonResult:
    p_up_30d: float
    ret_p10_30d: float
    ret_p50_30d: float
    ret_p90_30d: float
    vol_expansion_prob: float
    flags: Dict[str, bool]
    effective_sample_size: int
    analog_count: int


# -------------------- feature engineering --------------------


def _trend_slope_r2(series: pd.Series, window: int = 50) -> Tuple[pd.Series, pd.Series]:
    idx = np.arange(window)
    slopes = []
    r2s = []
    values = series.values
    for i in range(len(series)):
        if i + 1 < window or np.any(np.isnan(values[i + 1 - window : i + 1])):
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue
        y = values[i + 1 - window : i + 1]
        x = idx
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        beta = cov / (var + 1e-9)
        alpha = y_mean - beta * x_mean
        y_hat = alpha + beta * x
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        slopes.append(beta)
        r2s.append(r2)
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def build_long_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Build slow/structural features for 20–30d horizon.
    Expects daily OHLCV with DateTimeIndex.
    Gracefully handles short histories by relaxing window sizes.
    """
    df = hist.copy()
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(index=df.index, data=np.nan)
    
    n = len(hist)
    
    # Adapt windows to available history (at least 5 data points per window)
    mom_20_window = min(20, max(5, n // 4))
    mom_50_window = min(50, max(5, n // 3))
    mom_100_window = min(100, max(5, n // 2))
    slope_window = min(50, max(5, n // 3))

    # Momentum / trend distance (relaxed windows)
    df["mom_20d"] = close.pct_change(mom_20_window)
    df["mom_50d"] = close.pct_change(mom_50_window)
    df["mom_100d"] = close.pct_change(mom_100_window)

    df["slope_50d"], df["slope_r2_50d"] = _trend_slope_r2(close, slope_window)
    df["slope_100d"], df["slope_r2_100d"] = _trend_slope_r2(close, slope_window)

    ma50 = close.rolling(min(50, max(5, n // 3))).mean()
    ma100 = close.rolling(min(100, max(5, n // 2))).mean()
    df["dist_ma50"] = close / (ma50 + 1e-9) - 1
    df["dist_ma100"] = close / (ma100 + 1e-9) - 1

    # Volatility regime (relaxed windows)
    ret_1d = close.pct_change()
    hv_20_window = min(20, max(5, n // 4))
    hv_60_window = min(60, max(5, n // 3))
    
    df["hv_20"] = ret_1d.rolling(hv_20_window).std()
    df["hv_60"] = ret_1d.rolling(hv_60_window).std()
    df["vol_of_vol"] = df["hv_20"].rolling(hv_20_window).std()
    df["hv_20_pct_1y"] = df["hv_20"].rolling(min(252, n)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(pd.Series(x).dropna()) else np.nan
    )

    # Volume / flow proxies
    vol_ma20 = volume.rolling(min(20, max(5, n // 4))).mean()
    vol_ma60 = volume.rolling(min(60, max(5, n // 3))).mean()
    df["vol_trend_20_60"] = vol_ma20 / (vol_ma60 + 1e-9)
    dollar_vol = close * volume
    df["dollar_vol_20_trend"] = dollar_vol.rolling(min(20, max(5, n // 4))).mean() / (dollar_vol.rolling(min(60, max(5, n // 3))).mean() + 1e-9)

    # Relative strength proxy vs own history
    ret_20d = close.pct_change(min(20, max(5, n // 4)))
    df["z_ret_20d"] = (ret_20d - ret_20d.rolling(min(252, n)).mean()) / (ret_20d.rolling(min(252, n)).std() + 1e-9)

    # Forward returns and vol for labels
    df["fwd_ret_20d"] = close.pct_change(min(20, max(5, n // 4))).shift(-min(20, max(5, n // 4)))
    df["fwd_ret_30d"] = close.pct_change(min(30, max(5, n // 3))).shift(-min(30, max(5, n // 3)))
    df["hv_20_fwd"] = df["hv_20"].shift(-hv_20_window)
    df["vol_expanded"] = (df["hv_20_fwd"] > df["hv_20"]).astype(float)

    return df


# -------------------- regime + similarity --------------------


def _tag_regime(df: pd.DataFrame) -> pd.Series:
    trend_state = np.where(df["slope_50d"] > 0, "up", "down")
    try:
        vol_state = pd.qcut(df["hv_20"], q=3, labels=["low", "med", "high"], duplicates="drop")
        vol_state = vol_state.astype(str).fillna("unk")
    except (ValueError, TypeError):
        # Not enough unique values for qcut; fall back to simple thresholding
        hv_med = df["hv_20"].median()
        hv_q75 = df["hv_20"].quantile(0.75)
        vol_state = pd.Series(index=df.index, dtype=str)
        vol_state[df["hv_20"] <= hv_med] = "low"
        vol_state[(df["hv_20"] > hv_med) & (df["hv_20"] <= hv_q75)] = "med"
        vol_state[df["hv_20"] > hv_q75] = "high"
        vol_state = vol_state.fillna("unk")
    return pd.Series([f"{t}_{v}" for t, v in zip(trend_state, vol_state)], index=df.index)


def _weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles=(0.1, 0.5, 0.9)):
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    if cum_weights[-1] <= 0:
        return [np.nan for _ in quantiles]
    qs = []
    for q in quantiles:
        cutoff = q * cum_weights[-1]
        idx = np.searchsorted(cum_weights, cutoff)
        idx = min(idx, len(values) - 1)
        qs.append(values[idx])
    return qs


def _effective_sample_size(weights: np.ndarray) -> float:
    w = weights[weights > 0]
    if len(w) == 0:
        return 0.0
    return (w.sum() ** 2) / ((w ** 2).sum() + 1e-12)


def analog_infer(df: pd.DataFrame, as_of: pd.Timestamp, k: int = 200, decay_half_life: int = 252) -> Optional[LongHorizonResult]:
    if as_of not in df.index:
        return None

    # Filter history
    hist = df.loc[:as_of].iloc[:-1]
    min_rows = 30  # Ultra-relaxed to support ~3mo data
    if len(hist) < min_rows:
        return None

    feat_cols = [
        "mom_20d",
        "mom_50d",
        "mom_100d",
        "slope_50d",
        "slope_r2_50d",
        "slope_100d",
        "slope_r2_100d",
        "dist_ma50",
        "dist_ma100",
        "hv_20",
        "hv_60",
        "vol_of_vol",
        "hv_20_pct_1y",
        "vol_trend_20_60",
        "dollar_vol_20_trend",
        "z_ret_20d",
    ]

    # Fill NaNs and drop only rows where ALL features are NaN
    hist_feat = hist[feat_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    if hist_feat.isna().all(axis=1).all():
        return None

    means = hist_feat.mean()
    stds = hist_feat.std().replace(0, 1.0)  # Prevent division by 0
    hist_z = (hist_feat - means) / stds

    current = df.loc[as_of, feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    current_z = (current - means) / stds

    df_reg = hist.assign(regime=_tag_regime(hist))
    current_regime = _tag_regime(df.loc[[as_of]]).iloc[0]
    same_regime_idx = df_reg.index[df_reg["regime"] == current_regime]
    cand = hist_z.loc[hist_z.index.intersection(same_regime_idx)]
    if cand.empty:
        cand = hist_z

    curr_vec = current_z.values.astype(float)
    cand_mat = cand.values.astype(float)
    cand_norm = np.linalg.norm(cand_mat, axis=1) + 1e-9
    curr_norm = np.linalg.norm(curr_vec) + 1e-9
    cos_sim = (cand_mat @ curr_vec) / (cand_norm * curr_norm)
    dist = 1 - cos_sim

    ages = pd.Series((as_of - cand.index).days).clip(lower=1).values
    lam = math.log(2) / decay_half_life
    decay = np.exp(-lam * ages)

    inv_dist = 1 / (dist + 1e-6)
    weights = inv_dist * decay

    order = np.argsort(-weights)
    top_idx = cand.index[order][:k]
    w_top = weights[order][:k]

    # Forward returns may be sparse; use any available
    fwd = df.loc[top_idx, "fwd_ret_30d"].fillna(0.0).values
    vol_expanded = df.loc[top_idx, "vol_expanded"].fillna(0.0).values

    p_up = float(np.average((fwd > 0).astype(float), weights=w_top)) if len(fwd) else np.nan
    p10, p50, p90 = _weighted_quantiles(fwd, w_top, quantiles=(0.1, 0.5, 0.9)) if len(fwd) else (np.nan, np.nan, np.nan)
    vol_exp_prob = float(np.average(vol_expanded, weights=w_top)) if len(vol_expanded) else np.nan

    ess = int(round(_effective_sample_size(w_top)))
    flags = {
        "low_confidence": (ess < 20),  # Relaxed from 30
        "high_drawdown_risk": (not np.isnan(p10) and p10 < -0.05),
        "regime_shift_risk": (not np.isnan(vol_exp_prob) and vol_exp_prob > 0.6),
    }

    return LongHorizonResult(
        p_up_30d=float(p_up),
        ret_p10_30d=float(p10),
        ret_p50_30d=float(p50),
        ret_p90_30d=float(p90),
        vol_expansion_prob=float(vol_exp_prob),
        flags=flags,
        effective_sample_size=ess,
        analog_count=int(min(k, len(cand))),
    )


def predict_long_horizon(hist: pd.DataFrame, as_of: Optional[pd.Timestamp] = None, k: int = 200) -> Optional[LongHorizonResult]:
    if hist is None or hist.empty:
        return None
    if as_of is None:
        as_of = hist.index[-1]
    feats = build_long_features(hist)
    return analog_infer(feats, as_of=as_of, k=k)
