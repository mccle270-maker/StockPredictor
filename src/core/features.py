"""
Feature engineering functions.
Pure functions that transform DataFrames - no API calls, no caching.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Optional


# ============================================================================
# RETURNS & VOLATILITY
# ============================================================================

def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add return features to DataFrame."""
    close = df[price_col]
    
    # Simple returns at different horizons (shifted 1 day for look-ahead prevention)
    df["ret_1d"] = close.pct_change(1).shift(1)
    df["ret_3d"] = close.pct_change(3).shift(1)
    df["ret_5d"] = close.pct_change(5).shift(1)
    df["ret_10d"] = close.pct_change(10).shift(1)
    df["ret_20d"] = close.pct_change(20).shift(1)
    
    # Cumulative returns
    df["cumret_3d"] = (1 + df["ret_1d"]).rolling(3).apply(np.prod, raw=True) - 1
    df["cumret_5d"] = (1 + df["ret_1d"]).rolling(5).apply(np.prod, raw=True) - 1
    df["cumret_10d"] = (1 + df["ret_1d"]).rolling(10).apply(np.prod, raw=True) - 1
    
    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility features to DataFrame."""
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df["Close"].pct_change(1).shift(1)
    
    ret = df["ret_1d"]
    
    # Rolling volatility at different windows
    df["vol_10d"] = ret.rolling(10).std().shift(1)
    df["vol_20d"] = ret.rolling(20).std().shift(1)
    df["vol_60d"] = ret.rolling(60).std().shift(1)
    df["vol_ratio_10_60"] = (df["vol_10d"] / df["vol_60d"]).replace([np.inf, -np.inf], np.nan)
    
    # ATR (Average True Range)
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14).mean().shift(1)
    df["atr_pct_14"] = (df["atr_14"] / close).shift(1)
    
    return df


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Close") -> pd.DataFrame:
    """Add RSI (Relative Strength Index) to DataFrame."""
    delta = df[price_col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi{window}"] = (100 - (100 / (1 + rs))).shift(1)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence) to DataFrame."""
    close = df["Close"]
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    df["macd"] = (ema_fast - ema_slow).shift(1)
    df["macdsignal"] = df["macd"].ewm(span=signal).mean()
    df["macdhist"] = df["macd"] - df["macdsignal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands to DataFrame."""
    close = df["Close"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    df["bb_upper"] = upper.shift(1)
    df["bb_lower"] = lower.shift(1)
    df["bb_middle"] = ma.shift(1)
    df["bb_pctb"] = ((close - lower) / (upper - lower)).shift(1)
    df["bb_width"] = ((upper - lower) / ma).shift(1)
    return df


def add_mfi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add MFI (Money Flow Index) to DataFrame."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = typical_price * df["Volume"]
    
    positive_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
    negative_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
    
    mfr = positive_mf.rolling(window).sum() / negative_mf.rolling(window).sum().replace(0, np.nan)
    df[f"mfi{window}"] = (100 - (100 / (1 + mfr))).shift(1)
    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add ADX (Average Directional Index) to DataFrame."""
    high, low, close = df["High"], df["Low"], df["Close"]
    
    # Calculate true range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=window).mean()
    
    # Calculate directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    plus_di = 100 * plus_dm.ewm(span=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=window).mean() / atr
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df[f"adx_{window}"] = dx.ewm(span=window).mean().shift(1)
    df[f"plus_di_{window}"] = plus_di.shift(1)
    df[f"minus_di_{window}"] = minus_di.shift(1)
    
    return df


def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add price pattern features to DataFrame."""
    high, low, close, open_ = df["High"], df["Low"], df["Close"], df["Open"]
    
    df["daily_range"] = ((high - low) / close).shift(1)
    df["high_low_ratio"] = (high / low).shift(1)
    df["close_position"] = ((close - low) / (high - low).replace(0, np.nan)).shift(1)
    df["body_to_range"] = (abs(close - open_) / (high - low).replace(0, np.nan)).shift(1)
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features to DataFrame."""
    vol = df["Volume"]
    close = df["Close"]
    
    vol_ma_20 = vol.rolling(20).mean()
    df["vol_spike_20"] = (vol / vol_ma_20).shift(1)
    df["vol_zscore"] = ((vol - vol_ma_20) / vol.rolling(20).std()).shift(1)
    df["dollar_volume"] = (close * vol).shift(1)
    df["dollar_volume_20d_avg"] = df["dollar_volume"].rolling(20).mean()
    
    return df


def add_all_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to DataFrame."""
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_mfi(df)
    df = add_adx(df)
    df = add_price_patterns(df)
    df = add_volume_features(df)
    return df


# ============================================================================
# GBM (GEOMETRIC BROWNIAN MOTION) FEATURES
# ============================================================================

def add_gbm_features(df: pd.DataFrame, window: int = 60, horizons: Tuple[int, ...] = (1, 5)) -> pd.DataFrame:
    """
    Add GBM-derived probability features.
    Uses log-normal distribution to estimate probability of positive returns.
    """
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df["Close"].pct_change(1).shift(1)
    
    ret = df["ret_1d"]
    trading_days = 252
    
    # Annualized drift and volatility from rolling window
    df["gbm_mu_60d"] = ret.rolling(window).mean() * trading_days
    df["gbm_sig_60d"] = ret.rolling(window).std() * np.sqrt(trading_days)
    
    for h in horizons:
        # Daily drift/vol for horizon
        mu_daily = df["gbm_mu_60d"] / trading_days * h
        sig_daily = df["gbm_sig_60d"] / np.sqrt(trading_days) * np.sqrt(h)
        
        # Log-normal parameters
        log_drift = mu_daily - 0.5 * sig_daily ** 2
        
        # Probability of positive return
        z_thresh = -log_drift / sig_daily.replace(0, np.nan)
        # norm.cdf returns numpy array, wrap in Series to use .shift()
        prob_up = pd.Series(1 - norm.cdf(z_thresh.values), index=df.index)
        df[f"gbm_prob_up_{h}d"] = prob_up.shift(1)
        
        # Expected return
        exp_ret = pd.Series(np.exp(mu_daily.values) - 1, index=df.index)
        df[f"gbm_exp_ret_{h}d"] = exp_ret.shift(1)
        
        # Percentile returns (5th and 95th)
        p05 = pd.Series(np.exp(log_drift.values + sig_daily.values * norm.ppf(0.05)) - 1, index=df.index)
        p95 = pd.Series(np.exp(log_drift.values + sig_daily.values * norm.ppf(0.95)) - 1, index=df.index)
        df[f"gbm_p05_ret_{h}d"] = p05.shift(1)
        df[f"gbm_p95_ret_{h}d"] = p95.shift(1)
    
    return df


# ============================================================================
# RELATIVE STRENGTH (vs SPX)
# ============================================================================

def add_relative_strength(df: pd.DataFrame, spx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative strength features comparing stock to SPX.
    
    Args:
        df: Stock DataFrame with 'ret_1d' column
        spx_df: SPX DataFrame with 'Close' column (already aligned to df's index)
    """
    if spx_df is None or spx_df.empty:
        # Return df unchanged if no SPX data
        df["rel_strength_1d"] = 0.0
        df["rel_momentum_5d"] = 0.0
        df["beta_60_spx"] = 1.0
        df["corr_20_spx"] = 0.0
        return df
    
    # Align SPX to stock's dates
    spx_aligned = spx_df.reindex(df.index, method="ffill")
    spx_ret = spx_aligned["Close"].pct_change(1)
    
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df["Close"].pct_change(1).shift(1)
    
    stock_ret = df["ret_1d"]
    
    # Relative strength (excess return)
    df["rel_strength_1d"] = (stock_ret - spx_ret).shift(1)
    
    # 5-day relative momentum
    spx_ret_5d = spx_aligned["Close"].pct_change(5)
    stock_ret_5d = df["Close"].pct_change(5)
    df["rel_momentum_5d"] = (stock_ret_5d - spx_ret_5d).shift(1)
    
    # 60-day rolling beta
    cov = stock_ret.rolling(60).cov(spx_ret)
    var = spx_ret.rolling(60).var()
    df["beta_60_spx"] = (cov / var.replace(0, np.nan)).shift(1)
    
    # 20-day correlation
    df["corr_20_spx"] = stock_ret.rolling(20).corr(spx_ret).shift(1)
    
    return df


# ============================================================================
# REGIME DETECTION
# ============================================================================

def add_regime_features(df: pd.DataFrame, vix_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """Add regime detection features to DataFrame."""
    close = df["Close"]
    
    # Trend regime (vs 200-day MA)
    sma_200 = close.rolling(200).mean()
    df["regime_bull"] = (close > sma_200).astype(float).shift(1)
    df["regime_bear"] = (close < sma_200).astype(float).shift(1)
    
    # VIX regime (if available)
    if vix_series is not None and len(vix_series) > 0:
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        df["regime_vix_low"] = (vix_aligned < 15).astype(float).shift(1)
        df["regime_vix_medium"] = ((vix_aligned >= 15) & (vix_aligned < 25)).astype(float).shift(1)
        df["regime_vix_high"] = (vix_aligned >= 25).astype(float).shift(1)
    else:
        df["regime_vix_low"] = 0.0
        df["regime_vix_medium"] = 1.0
        df["regime_vix_high"] = 0.0
    
    # Streak counting
    up_days = (close > close.shift(1)).astype(int)
    down_days = (close < close.shift(1)).astype(int)
    
    def count_streak(series):
        """Count consecutive True values."""
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).cumsum()
    
    df["bull_streak"] = count_streak(up_days).shift(1)
    df["bear_streak"] = count_streak(down_days).shift(1)
    
    return df


# ============================================================================
# ALL FEATURES COMBINED
# ============================================================================

def build_all_features(
    df: pd.DataFrame,
    spx_df: Optional[pd.DataFrame] = None,
    vix_series: Optional[pd.Series] = None,
    macro_df: Optional[pd.DataFrame] = None,
    fundamentals: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build all features for a stock DataFrame.
    
    Args:
        df: OHLCV DataFrame
        spx_df: SPX price DataFrame for relative strength
        vix_series: VIX series for regime detection
        macro_df: Macro data (FRED) to join
        fundamentals: Dict of fundamental values (P/E, P/B, etc.)
    
    Returns:
        DataFrame with all features added
    """
    # Price-based features
    df = add_returns(df)
    df = add_volatility(df)
    
    # Technical indicators
    df = add_all_technicals(df)
    
    # GBM probabilities
    df = add_gbm_features(df)
    
    # Relative strength
    df = add_relative_strength(df, spx_df)
    
    # Regime detection
    df = add_regime_features(df, vix_series)
    
    # Join macro data if available
    if macro_df is not None and not macro_df.empty:
        # Forward-fill then backward-fill macro data before joining
        macro_filled = macro_df.ffill().bfill()
        df = df.join(macro_filled, how="left")
        # Fill any remaining NaNs with 0
        for col in macro_df.columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)
    
    # Add fundamentals as constant columns
    if fundamentals:
        for key, value in fundamentals.items():
            df[key] = value if value is not None else 0.0
    
    return df


def build_target(df: pd.DataFrame, horizon: int = 1, price_col: str = "Close") -> pd.Series:
    """
    Build target variable (forward return).
    
    Args:
        df: DataFrame with price data
        horizon: Number of days ahead to predict
        price_col: Column to use for price
    
    Returns:
        Series of forward returns (shifted forward by horizon days)
    """
    return df[price_col].pct_change(horizon).shift(-horizon)


def get_available_features(df: pd.DataFrame, max_nan_pct: float = 0.5) -> list:
    """
    Get list of features that have less than max_nan_pct missing values.
    """
    from ..config import FEATURE_COLUMNS
    
    available = []
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            nan_pct = df[col].isna().mean()
            if nan_pct < max_nan_pct:
                available.append(col)
    return available
