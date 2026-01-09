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
# ADDITIONAL TECHNICAL INDICATORS (for optimized feature set)
# ============================================================================

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators needed for optimized feature set."""
    close = df["Close"]
    
    # OBV (On-Balance Volume)
    if "Volume" in df.columns:
        obv = (np.sign(close.diff()) * df["Volume"]).fillna(0).cumsum()
        df["obv"] = obv.shift(1)
    else:
        df["obv"] = 0.0
    
    # Momentum (rate of change)
    df["momentum"] = close.pct_change(10).shift(1)
    
    # Williams %R
    high_14 = df["High"].rolling(14).max()
    low_14 = df["Low"].rolling(14).min()
    df["williams_r"] = (-100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)).shift(1)
    
    # CCI (Commodity Channel Index)
    typical_price = (df["High"] + df["Low"] + close) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = ((typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))).shift(1)
    
    # Stochastic %K
    df["stoch_k"] = (100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)).shift(1)
    
    # MFI (Money Flow Index) - already exists but ensure it's named correctly
    if "mfi14" not in df.columns and "mfi" not in df.columns:
        typical_price = (df["High"] + df["Low"] + close) / 3
        raw_mf = typical_price * df["Volume"]
        positive_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
        negative_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
        mfr = positive_mf.rolling(14).sum() / negative_mf.rolling(14).sum().replace(0, np.nan)
        df["mfi"] = (100 - (100 / (1 + mfr))).shift(1)
    elif "mfi14" in df.columns:
        df["mfi"] = df["mfi14"]
    
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
    use_optimized_features: bool = False,
) -> pd.DataFrame:
    """
    Build all features for a stock DataFrame.
    
    Args:
        df: OHLCV DataFrame
        spx_df: SPX price DataFrame for relative strength
        vix_series: VIX series for regime detection
        macro_df: Macro data (FRED) to join
        fundamentals: Dict of fundamental values (P/E, P/B, etc.)
        use_optimized_features: If True, only compute features needed for OPTIMIZED_FEATURES
    
    Returns:
        DataFrame with all features added
    """
    import logging
    logger = logging.getLogger("feature_engineering")
    
    # Price-based features
    df = add_returns(df)
    df = add_volatility(df)
    
    # Technical indicators
    df = add_all_technicals(df)
    
    # GBM probabilities
    df = add_gbm_features(df)
    
    # Add momentum indicators (needed for optimized features)
    df = add_momentum_indicators(df)
    
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
    
    # If optimized features requested, filter down to just those columns
    if use_optimized_features:
        from ..config import OPTIMIZED_FEATURES, is_optimized_mode
        if is_optimized_mode():
            available_opt = [f for f in OPTIMIZED_FEATURES if f in df.columns]
            missing_opt = [f for f in OPTIMIZED_FEATURES if f not in df.columns]
            if missing_opt:
                logger.warning(f"Missing optimized features: {missing_opt}")
            logger.info(f"🎯 Using {len(available_opt)}/{len(OPTIMIZED_FEATURES)} optimized features")
    
    return df


def build_optimized_features(
    df: pd.DataFrame,
    spx_df: Optional[pd.DataFrame] = None,
    vix_series: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Build only the optimized feature set for faster inference.
    
    This is an optimized path that only computes the 20 features from
    the Model Improvement Report, skipping unnecessary calculations.
    
    Args:
        df: OHLCV DataFrame
        spx_df: SPX price DataFrame (optional)
        vix_series: VIX series (optional)
    
    Returns:
        (DataFrame with optimized features, list of feature columns)
    """
    from ..config import OPTIMIZED_FEATURES
    import logging
    logger = logging.getLogger("feature_engineering")
    
    logger.info("⚡ Building optimized feature set (20 features)")
    
    # Price-based features
    df = add_returns(df)
    df = add_volatility(df)
    
    # Technical indicators
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_adx(df)
    
    # GBM probabilities
    df = add_gbm_features(df)
    
    # Momentum indicators
    df = add_momentum_indicators(df)
    
    # Filter to only optimized features that exist
    available = [f for f in OPTIMIZED_FEATURES if f in df.columns]
    
    logger.info(f"✅ Built {len(available)}/{len(OPTIMIZED_FEATURES)} optimized features")
    
    return df, available


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



def validate_features(
    df: pd.DataFrame,
    required_features: Optional[list] = None,
    max_row_nan_pct: float = 0.20,
    max_feature_nan_pct: float = 0.10,
    warn_feature_nan_pct: float = 0.02,
    drop_warmup: bool = True,
    min_rows_after_clean: int = 50,
) -> Tuple[pd.DataFrame, dict]:
    """
    Validate and clean feature DataFrame.
    
    Args:
        df: DataFrame with features
        required_features: List of required feature columns (uses FEATURE_COLUMNS if None)
        max_row_nan_pct: Drop rows with more than this % of NaN features (0.20 = 20%)
        max_feature_nan_pct: Error if any feature has >this NaN rate after cleaning
        warn_feature_nan_pct: Warn if any feature has >this NaN rate
        drop_warmup: If True, drop initial rows where many features are NaN (warmup period)
        min_rows_after_clean: Minimum rows required after cleaning
    
    Returns:
        (cleaned_df, quality_report) tuple
    """
    import logging
    logger = logging.getLogger("feature_validation")
    
    if required_features is None:
        from ..config import FEATURE_COLUMNS
        required_features = FEATURE_COLUMNS
    
    # Initialize report
    report = {
        "original_rows": len(df),
        "missing_features": [],
        "nan_rates_before": {},
        "nan_rates_after": {},
        "rows_dropped": 0,
        "warnings": [],
        "status": "OK",
    }
    
    # Check for missing features
    available_features = [f for f in required_features if f in df.columns]
    missing = [f for f in required_features if f not in df.columns]
    
    if missing:
        report["missing_features"] = missing
        logger.warning(f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    if not available_features:
        report["status"] = "ERROR"
        report["warnings"].append("No required features available")
        return df, report
    
    # Calculate NaN rates before cleaning
    for feat in available_features:
        nan_rate = df[feat].isna().mean()
        report["nan_rates_before"][feat] = round(float(nan_rate), 4)
    
    # Step 1: Drop warmup rows (rows where many features are NaN)
    if drop_warmup:
        # Calculate % of NaN per row for required features
        row_nan_pct = df[available_features].isna().mean(axis=1)
        
        # Find first row where NaN% drops below threshold
        valid_mask = row_nan_pct <= max_row_nan_pct
        if valid_mask.any():
            first_valid_idx = valid_mask.idxmax()
            rows_before = len(df)
            df = df.loc[first_valid_idx:].copy()
            report["rows_dropped"] = rows_before - len(df)
            
            if report["rows_dropped"] > 0:
                logger.info(f"Dropped {report['rows_dropped']} warmup rows")
    
    # Step 2: Forward-fill remaining NaNs
    for feat in available_features:
        if df[feat].isna().any():
            df[feat] = df[feat].ffill().bfill()
            # Final fallback to 0 for any remaining NaNs
            if df[feat].isna().any():
                df[feat] = df[feat].fillna(0)
    
    # Step 3: Handle infinities
    for feat in available_features:
        if np.isinf(df[feat]).any():
            # Replace inf with max/min finite values
            max_val = df[feat].replace([np.inf, -np.inf], np.nan).max()
            min_val = df[feat].replace([np.inf, -np.inf], np.nan).min()
            df[feat] = df[feat].replace(np.inf, max_val).replace(-np.inf, min_val)
    
    # Step 4: Calculate NaN rates after cleaning
    high_nan_features = []
    warn_features = []
    
    for feat in available_features:
        nan_rate = df[feat].isna().mean()
        report["nan_rates_after"][feat] = round(float(nan_rate), 4)
        
        if nan_rate > max_feature_nan_pct:
            high_nan_features.append((feat, nan_rate))
        elif nan_rate > warn_feature_nan_pct:
            warn_features.append((feat, nan_rate))
    
    # Log warnings
    for feat, rate in warn_features:
        msg = f"Feature {feat} has {rate*100:.1f}% NaN rate after cleaning"
        report["warnings"].append(msg)
        logger.warning(msg)
    
    # Check minimum rows
    if len(df) < min_rows_after_clean:
        report["status"] = "ERROR"
        report["warnings"].append(f"Only {len(df)} rows after cleaning (need {min_rows_after_clean})")
    
    # Error if any feature has too many NaNs
    if high_nan_features:
        report["status"] = "ERROR"
        for feat, rate in high_nan_features:
            report["warnings"].append(f"Feature {feat} still has {rate*100:.1f}% NaN (>{max_feature_nan_pct*100}%)")
    
    report["final_rows"] = len(df)
    
    return df, report


def get_feature_quality_summary(df: pd.DataFrame) -> dict:
    """Quick summary of feature quality for a DataFrame."""
    from ..config import FEATURE_COLUMNS
    
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    
    summary = {
        "total_features": len(FEATURE_COLUMNS),
        "available_features": len(available),
        "missing_features": len(FEATURE_COLUMNS) - len(available),
        "rows": len(df),
    }
    
    if available:
        nan_rates = {f: df[f].isna().mean() for f in available}
        summary["avg_nan_rate"] = np.mean(list(nan_rates.values()))
        summary["max_nan_rate"] = max(nan_rates.values())
        summary["features_over_5pct_nan"] = sum(1 for r in nan_rates.values() if r > 0.05)
    
    return summary

