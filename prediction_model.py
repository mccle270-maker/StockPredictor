# prediction_model.py
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from xgboost import XGBRegressor

from data_fetch import get_history, get_history_cached


# ---------- Technical indicator helpers ----------

def add_rsi(df, window: int = 14, price_col: str = "Close"):
    """
    Add RSI (Relative Strength Index) column to df using Wilder-style smoothing.
    """
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f"rsi_{window}"] = rsi
    return df


def add_macd(df, price_col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Add MACD (fast-slow EMA), signal line, and histogram.
    """
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    return df


def add_mfi(df, window: int = 14):
    """
    Add Money Flow Index (MFI) column to df. Requires High, Low, Close, Volume.
    """
    # Typical Price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0

    # Raw Money Flow
    rmf = tp * df["Volume"]

    # Positive/negative money flow
    tp_shift = tp.shift(1)
    pos_mf = rmf.where(tp > tp_shift, 0.0)
    neg_mf = rmf.where(tp < tp_shift, 0.0)

    # Sum over window
    pos_mf_sum = pos_mf.rolling(window=window, min_periods=window).sum()
    neg_mf_sum = neg_mf.rolling(window=window, min_periods=window).sum()

    # Avoid division by zero
    money_flow_ratio = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_flow_ratio))

    df[f"mfi_{window}"] = mfi
    return df


def add_technical_indicators(df):
    """
    Convenience wrapper to add RSI, MACD, and MFI to a price DataFrame.
    Assumes df has columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    df = df.copy()
    df = add_rsi(df, window=14, price_col="Close")
    df = add_macd(df, price_col="Close", fast=12, slow=26, signal=9)
    df = add_mfi(df, window=14)
    return df


# ---------- Fundamentals & macro ----------

FUNDAMENTAL_COLUMNS = [
    "fund_pe_trailing",
    "fund_pb",
    "fund_market_cap",
]

MACRO_COLUMNS = ["mkt_ret_1d"]
_macro_cache = {}


def get_macro_df(symbol="^GSPC", period="5y") -> pd.DataFrame:
    """
    Fetch simple market 1-day return to use as a macro regime feature.
    Cached per (symbol, period).
    """
    key = (symbol, period)
    if key in _macro_cache:
        return _macro_cache[key]

    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval="1d")
    df = pd.DataFrame(index=hist.index)
    df["mkt_ret_1d"] = hist["Close"].pct_change()
    _macro_cache[key] = df
    return df


# ---------- Feature columns ----------

FEATURE_COLUMNS = [
    # returns & realized vol
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",

    # lagged returns and rolling stats on returns
    "ret_1d_lag1",
    "ret_1d_lag2",
    "ret_1d_lag5",
    "ret_1d_rollmean_5",
    "ret_1d_rollstd_5",
    "ret_1d_rollmean_10",
    "ret_1d_rollstd_10",

    # moving averages / trend & overbought-oversold
    "sma_ratio_10_50",
    "rsi_14",
    "price_to_ma50",

    # Bollinger / volatility structure
    "bb_width_20",

    # price–volume & volume structure
    "volume_price_corr",
    "volume_trend",
    "vol_ma_20",
    "vol_spike_20",
    "vol_spike_1d_ago",

    # volume rolling stats
    "vol_rollmean_20",
    "vol_rollstd_20",

    # intraday structure
    "high_low_ratio",
    "daily_range",
    "close_position",
    "hl_range",
    "atr_14",

    # calendar effects
    "day_of_week",
    "month",
    "is_month_end",

    # fundamentals
    "fund_pe_trailing",
    "fund_pb",
    "fund_market_cap",

    # new technical indicators
    "macd",
    "macd_signal",
    "macd_hist",
    "mfi_14",
]


# ---------- Price feature engineering ----------

def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with technical indicators, volume features,
    intraday structure, calendar effects, and simple lags/rolling stats.
    Expects columns: Open, High, Low, Close, Volume.
    """
    hist = hist.copy()

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    # Returns and realized volatility
    hist["ret_1d"] = close.pct_change()
    hist["ret_5d"] = close.pct_change(5)
    hist["ret_20d"] = close.pct_change(20)
    hist["vol_20d"] = hist["ret_1d"].rolling(20).std()

    # Simple lags of daily return
    hist["ret_1d_lag1"] = hist["ret_1d"].shift(1)
    hist["ret_1d_lag2"] = hist["ret_1d"].shift(2)
    hist["ret_1d_lag5"] = hist["ret_1d"].shift(5)

    # Rolling mean / std of daily returns
    hist["ret_1d_rollmean_5"] = hist["ret_1d"].rolling(5).mean()
    hist["ret_1d_rollstd_5"] = hist["ret_1d"].rolling(5).std()
    hist["ret_1d_rollmean_10"] = hist["ret_1d"].rolling(10).mean()
    hist["ret_1d_rollstd_10"] = hist["ret_1d"].rolling(10).std()

    # Moving averages and ratio
    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    hist["sma_ratio_10_50"] = sma_10 / (sma_50 + 1e-9)

    # Bollinger Band width (20d)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_20 = sma_20 + 2 * std_20
    lower_20 = sma_20 - 2 * std_20
    bb_width_20 = (upper_20 - lower_20) / (sma_20 + 1e-9)
    hist["bb_width_20"] = bb_width_20

    # Price to 50-day MA ratio
    hist["price_to_ma50"] = close / (sma_50 + 1e-9)

    # Volume-Price Correlation (20-day rolling)
    hist["volume_price_corr"] = (
        hist["ret_1d"].rolling(20).corr(volume.pct_change())
    )

    # Volume Trend (10-day avg / 30-day avg)
    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_30 = volume.rolling(30).mean()
    hist["volume_trend"] = vol_ma_10 / (vol_ma_30 + 1e-9)

    # Volume level and spikes
    hist["vol_ma_20"] = volume.rolling(20).mean()
    hist["vol_spike_20"] = volume / (hist["vol_ma_20"] + 1e-9)
    hist["vol_spike_1d_ago"] = hist["vol_spike_20"].shift(1)

    # Volume rolling stats
    hist["vol_rollmean_20"] = volume.rolling(20).mean()
    hist["vol_rollstd_20"] = volume.rolling(20).std()

    # High-Low Ratio
    hist["high_low_ratio"] = high / (low + 1e-9)

    # Daily Range (normalized by close)
    hist["daily_range"] = (high - low) / (close + 1e-9)

    # Close Position in Daily Range (0=at low, 1=at high)
    hist["close_position"] = (close - low) / (high - low + 1e-9)

    # High-low range vs prior close & ATR-style vol
    hist["hl_range"] = (high - low) / (close.shift(1) + 1e-9)
    hist["atr_14"] = hist["hl_range"].rolling(14).mean()

    # Calendar Effects
    hist["day_of_week"] = hist.index.dayofweek
    hist["month"] = hist.index.month
    hist["is_month_end"] = (hist.index.day >= 25).astype(int)

    # Add RSI, MACD, MFI
    hist = add_technical_indicators(hist)

    return hist


# ---------- Model factory ----------

def make_model(model_type: str = "rf", random_state: int = 42):
    """
    Factory for models:
    - 'rf'   -> RandomForestRegressor
    - 'gbrt' -> GradientBoostingRegressor
    - 'xgb'  -> XGBoost
    """
    if model_type == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    elif model_type == "xgb":
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )
    else:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )


# ---------- Fundamentals fetch ----------

def get_fundamental_features(ticker: str) -> dict:
    """
    Fetch a few slow-moving fundamental metrics for the ticker.
    These are added as constant features across the history window.

    Missing values are filled with 0.0 so that they do not cause the
    entire history to be dropped by dropna().
    """
    feats = {
        "fund_pe_trailing": np.nan,
        "fund_pb": np.nan,
        "fund_market_cap": np.nan,
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info
        feats["fund_pe_trailing"] = info.get("trailingPE", np.nan)
        feats["fund_pb"] = info.get("priceToBook", np.nan)
        feats["fund_market_cap"] = float(info.get("marketCap", np.nan))
    except Exception:
        pass

    # Fill missing fundamentals with a neutral 0.0 instead of NaN
    for k in feats:
        if pd.isna(feats[k]):
            feats[k] = 0.0

    return feats


# ---------- Build features & target ----------

def build_features_and_target(ticker="^GSPC", period="5y", horizon=1):
    """
    Build feature matrix X, target vector y, and the latest row info
    for multi-day prediction.

    If the requested period does not yield enough usable rows after
    feature engineering, automatically fall back to shorter periods.
    """
    fallback_periods = ["5y", "3y", "2y", "1y", "6mo", "3mo"]
    if period in fallback_periods:
        periods_to_try = [period] + [p for p in fallback_periods if p != period]
    else:
        periods_to_try = [period] + fallback_periods

    last_error = None
    min_rows = 60  # safety threshold

    for per in periods_to_try:
        try:
            hist = get_history_cached(ticker, period=per, interval="1d")
            if hist is None or hist.empty:
                raise ValueError(f"No raw history for {ticker} with period={per}")

            hist = add_price_features(hist)

            # add macro factor(s)
            macro_df = get_macro_df(symbol="^GSPC", period=per)
            hist = hist.join(macro_df, how="left")

            # add fundamental features as constant columns
            fund_feats = get_fundamental_features(ticker)
            for k, v in fund_feats.items():
                hist[k] = v

            # Target: multi-day return based on horizon
            hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(
                horizon
            ).shift(-horizon)

            df = hist.dropna().copy()

            if df.empty or len(df) < min_rows:
                raise ValueError(
                    f"Only {len(df)} usable rows for {ticker} with period={per}"
                )

            feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
            X = df[feat_cols].values
            y = df[f"target_ret_{horizon}d_ahead"].values

            last_row = df.iloc[-1]
            last_row_features = last_row[feat_cols].values
            last_close = last_row["Close"]
            last_vol_20d = last_row["vol_20d"]

            return X, y, last_row_features, last_close, last_vol_20d

        except Exception as e:
            last_error = e
            continue

    raise ValueError(
        f"No usable history for {ticker} after trying periods {periods_to_try}. "
        f"Last error: {last_error}"
    )


# ---------- Train & predict ----------

def train_model(X, y, model_type="rf", test_size=0.2, random_state=42):
    """
    Generic train/test with time-ordered split for any supported model.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, r2, rmse


def predict_next_for_ticker(ticker="^GSPC", period="5y", model_type="rf", horizon=1):
    """
    Train a model on the ticker's history and predict multi-day return & price.
    """
    X, y, x_last, last_close, last_vol_20d = build_features_and_target(
        ticker, period=period, horizon=horizon
    )

    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    # predict multi-day return from the latest features
    pred_ret = float(model.predict(x_last.reshape(1, -1))[0])
    pred_price = float(last_close * (1 + pred_ret))

    # get trailing P/E (if available) from Yahoo fundamentals (for display)
    pe_ratio = None
    try:
        t = yf.Ticker(ticker)
        fast = getattr(t, "fast_info", None)
        if fast is not None and hasattr(fast, "trailing_pe"):
            pe_ratio = fast.trailing_pe
        else:
            info = t.info
            pe_ratio = info.get("trailingPE")
    except Exception:
        pe_ratio = None

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    feature_importance = dict(zip(feat_cols, model.feature_importances_))
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]
    top_features_str = "\n".join(
        [f"- **{feat}**: {imp:.3f}" for feat, imp in top_features]
    )

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "last_close": last_close,
        "vol_20d": last_vol_20d,
        "pe_ratio": pe_ratio,
        "pred_next_ret": pred_ret,
        "pred_next_price": pred_price,
        "num_features": len(feat_cols),
        "top_features": top_features_str,
    }


# ---------- Tracking & backtests ----------

def track_predictions(ticker, period="1y", model_type="rf", horizon=1):
    """
    Compare model predictions to actual multi-day returns over the past period.
    """
    try:
        hist = get_history(ticker, period=period, interval="1d")

        if hist.empty or len(hist) < 50:
            print(f"Insufficient data for {ticker}: only {len(hist)} rows")
            return pd.DataFrame(), 0.0

        hist = add_price_features(hist)
        macro_df = get_macro_df(symbol="^GSPC", period=period)
        hist = hist.join(macro_df, how="left")
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v

        hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(
            -horizon
        )

        df = hist.dropna().copy()

        print(f"After dropna for {ticker}: {len(df)} rows")

        if len(df) < 50:
            print(f"Not enough data after feature engineering for {ticker}")
            return pd.DataFrame(), 0.0

        # Use more flexible split - test on last 60 days or 30% of data, whichever is smaller
        test_size = min(60, int(len(df) * 0.3))

        if test_size < 5:
            print(f"Test size too small: {test_size}")
            return pd.DataFrame(), 0.0

        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
        X_train = train_df[feat_cols].values
        y_train = train_df[f"target_ret_{horizon}d_ahead"].values

        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions for test period
        X_test = test_df[feat_cols].values
        y_test = test_df[f"target_ret_{horizon}d_ahead"].values
        y_pred = model.predict(X_test)

        results = pd.DataFrame(
            {
                "date": test_df.index,
                "actual_close": test_df["Close"],
                "predicted_return": y_pred,
                "actual_return": y_test,
                "pred_direction": np.sign(y_pred),
                "actual_direction": np.sign(y_test),
                "correct_direction": np.sign(y_pred) == np.sign(y_test),
            }
        )

        results["predicted_price"] = results["actual_close"] * (
            1 + results["predicted_return"]
        )

        accuracy = results["correct_direction"].mean()

        print(
            f"Success! Generated {len(results)} test predictions with {accuracy*100:.1f}% accuracy"
        )

        return results, accuracy

    except Exception as e:
        print(f"Error in track_predictions for {ticker}: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame(), 0.0


def backtest_one_ticker(
    ticker="AAPL",
    period="10y",
    test_years=1,
    threshold=0.002,
    model_type="rf",
    horizon=1,
):
    """
    Backtest a single model type ('rf', 'gbrt', or 'xgb') on one ticker with multi-day predictions.
    """
    hist = get_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)
    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")
    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v

    hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(
        -horizon
    )

    df = hist.dropna().copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index <= cutoff_date
    test_mask = df.index > cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    X_train = train_df[feat_cols].values
    y_train = train_df[f"target_ret_{horizon}d_ahead"].values

    X_test = test_df[feat_cols].values
    y_test = test_df[f"target_ret_{horizon}d_ahead"].values

    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    # predict multi-day returns over test window
    y_pred = model.predict(X_test)

    # trading rule: long / short / flat
    positions = np.where(
        y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)
    )
    pnl = positions * y_test

    # summary stats
    cum_ret = (1 + pnl).prod() - 1
    hit_rate = (np.sign(y_pred) == np.sign(y_test)).mean()
    avg_daily = pnl.mean()
    std_daily = pnl.std(ddof=1)
    sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily > 0 else 0.0

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "test_days": len(pnl),
        "total_return": cum_ret,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
    }


def backtest_compare_one_ticker(
    ticker="AAPL",
    period="10y",
    test_years=1,
    threshold=0.002,
    horizon=1,
):
    """
    Run backtests for RF, GBRT, and XGBoost on the same ticker with multi-day predictions.
    """
    rf_res = backtest_one_ticker(
        ticker=ticker,
        period=period,
        test_years=test_years,
        threshold=threshold,
        model_type="rf",
        horizon=horizon,
    )
    gbrt_res = backtest_one_ticker(
        ticker=ticker,
        period=period,
        test_years=test_years,
        threshold=threshold,
        model_type="gbrt",
        horizon=horizon,
    )
    xgb_res = backtest_one_ticker(
        ticker=ticker,
        period=period,
        test_years=test_years,
        threshold=threshold,
        model_type="xgb",
        horizon=horizon,
    )
    return {"rf": rf_res, "gbrt": gbrt_res, "xgb": xgb_res}


def tune_xgb_hyperparams(X, y, random_state=42):
    """
    Simple XGBoost hyperparameter tuning with time-series CV.
    Use offline to find good defaults, not inside the live app loop.
    """
    tscv = TimeSeriesSplit(n_splits=3)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        verbosity=0,
    )

    param_distributions = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=tscv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X, y)
    print("Best XGB params:", search.best_params_)
    print("Best CV score (neg MSE):", search.best_score_)

    return search.best_estimator_


if __name__ == "__main__":
    # Example: compare all three models on ^GSPC with different horizons
    print("=" * 60)
    print("Testing 1-Day Predictions - All Models")
    print("=" * 60)
    X, y, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=1)
    best_xgb = tune_xgb_hyperparams(X, y)

    rf_model, rf_r2, rf_rmse = train_model(X, y, model_type="rf")
    print("Random Forest (1-day)")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS) + len(MACRO_COLUMNS)}")
    print(f"  Test R^2:  {rf_r2:.4f}")
    print(f"  Test RMSE: {rf_rmse:.6f}")

    gbrt_model, gbrt_r2, gbrt_rmse = train_model(X, y, model_type="gbrt")
    print("\nGradient Boosting (1-day)")
    print(f"  Test R^2:  {gbrt_r2:.4f}")
    print(f"  Test RMSE: {gbrt_rmse:.6f}")

    xgb_model, xgb_r2, xgb_rmse = train_model(X, y, model_type="xgb")
    print("\nXGBoost (1-day)")
    print(f"  Test R^2:  {xgb_r2:.4f}")
    print(f"  Test RMSE: {xgb_rmse:.6f}")

    print("\n" + "=" * 60)
    print("Testing 2-Day Predictions")
    print("=" * 60)
    X2, y2, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=2)
    rf_model2, rf_r2_2d, rf_rmse_2d = train_model(X2, y2, model_type="rf")

    print("Random Forest (2-day)")
    print(f"  Test R^2:  {rf_r2_2d:.4f}")
    print(f"  Test RMSE: {rf_rmse_2d:.6f}")

    print("\n" + "=" * 60)
    print("Testing 3-Day Predictions")
    print("=" * 60)
    X3, y3, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=3)
    rf_model3, rf_r2_3d, rf_rmse_3d = train_model(X3, y3, model_type="rf")

    print("Random Forest (3-day)")
    print(f"  Test R^2:  {rf_r2_3d:.4f}")
    print(f"  Test RMSE: {rf_rmse_3d:.6f}")
