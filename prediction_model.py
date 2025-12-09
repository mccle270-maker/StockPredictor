# prediction_model.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

from data_fetch import get_history

# Extended feature columns with new indicators
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "sma_ratio_10_50",
    "rsi_14",
    "bb_width_20",
    "MACD",
    "price_to_ma50",
    "volume_price_corr",
    "volume_trend",
    "high_low_ratio",
    "daily_range",
    "close_position",
    "day_of_week",
    "month",
    "is_month_end",
]

def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with technical indicators, volume features,
    and calendar effects.
    """
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    # Original features: Returns and realized volatility
    hist["ret_1d"] = close.pct_change()
    hist["ret_5d"] = close.pct_change(5)
    hist["ret_20d"] = close.pct_change(20)
    hist["vol_20d"] = hist["ret_1d"].rolling(20).std()

    # Moving averages and ratio
    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    hist["sma_ratio_10_50"] = sma_10 / sma_50

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    hist["rsi_14"] = rsi_14

    # Bollinger Band width (20d)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_20 = sma_20 + 2 * std_20
    lower_20 = sma_20 - 2 * std_20
    bb_width_20 = (upper_20 - lower_20) / sma_20
    hist["bb_width_20"] = bb_width_20

    # NEW FEATURES
    
    # MACD (12-26 EMA difference)
    hist["MACD"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    
    # Price to 50-day MA ratio
    hist["price_to_ma50"] = close / sma_50
    
    # Volume-Price Correlation (20-day rolling)
    hist["volume_price_corr"] = hist["ret_1d"].rolling(20).corr(volume.pct_change())
    
    # Volume Trend (10-day avg / 30-day avg)
    hist["volume_trend"] = volume.rolling(10).mean() / volume.rolling(30).mean()
    
    # High-Low Ratio
    hist["high_low_ratio"] = high / low
    
    # Daily Range (normalized by close)
    hist["daily_range"] = (high - low) / close
    
    # Close Position in Daily Range (0=at low, 1=at high)
    hist["close_position"] = (close - low) / (high - low + 1e-9)
    
    # Calendar Effects
    hist["day_of_week"] = hist.index.dayofweek
    hist["month"] = hist.index.month
    hist["is_month_end"] = (hist.index.day >= 25).astype(int)

    return hist

def make_model(model_type: str = "rf", random_state: int = 42):
    """
    Factory for models:
    - 'rf'   -> RandomForestRegressor
    - 'gbrt' -> GradientBoostingRegressor
    """
    if model_type == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )
    else:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )

def build_features_and_target(ticker="^GSPC", period="5y"):
    """
    Build feature matrix X, target vector y, and the latest row info
    for next-day prediction, using price + TA indicators.
    """
    hist = get_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)

    # Target: next-day return
    hist["target_ret_1d_ahead"] = hist["ret_1d"].shift(-1)

    df = hist.dropna().copy()
    X = df[FEATURE_COLUMNS].values
    y = df["target_ret_1d_ahead"].values

    last_row = df.iloc[-1]
    last_row_features = last_row[FEATURE_COLUMNS].values
    last_close = last_row["Close"]
    last_vol_20d = last_row["vol_20d"]

    return X, y, last_row_features, last_close, last_vol_20d

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

def predict_next_for_ticker(ticker="^GSPC", period="5y", model_type="rf"):
    """
    Train a model on the ticker's history and predict next-day return & price,
    plus return the latest 20D vol and trailing P/E (if available).
    model_type: 'rf' or 'gbrt'.
    """
    X, y, x_last, last_close, last_vol_20d = build_features_and_target(
        ticker, period=period
    )

    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    # predict next-day return from the latest features
    pred_ret = float(model.predict(x_last.reshape(1, -1))[0])
    pred_price = float(last_close * (1 + pred_ret))

    # get trailing P/E (if available) from Yahoo fundamentals
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

    # Calculate feature importances for top 5 features
    feature_importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    top_features_str = "\n".join([f"- **{feat}**: {imp:.3f}" for feat, imp in top_features])

    return {
        "ticker": ticker,
        "model_type": model_type,
        "last_close": last_close,
        "vol_20d": last_vol_20d,
        "pe_ratio": pe_ratio,
        "pred_next_ret": pred_ret,
        "pred_next_price": pred_price,
        "num_features": len(FEATURE_COLUMNS),
        "top_features": top_features_str,
    }

def track_predictions(ticker, period="2mo", model_type="rf"):
    """
    Compare model predictions to actual next-day returns over the past period.
    Returns DataFrame with dates, predictions, actual returns, and accuracy.
    """
    hist = get_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)
    hist["target_ret_1d_ahead"] = hist["ret_1d"].shift(-1)
    
    df = hist.dropna().copy()
    
    if len(df) < 30:
        return pd.DataFrame(), 0.0
    
    # Train model on all available data except last 20 days
    train_df = df.iloc[:-20]
    test_df = df.iloc[-20:]
    
    if len(test_df) == 0:
        return pd.DataFrame(), 0.0
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_ret_1d_ahead"].values
    
    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions for test period
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_ret_1d_ahead"].values
    y_pred = model.predict(X_test)
    
    results = pd.DataFrame({
        'date': test_df.index,
        'actual_close': test_df['Close'],
        'predicted_return': y_pred,
        'actual_return': y_test,
        'pred_direction': np.sign(y_pred),
        'actual_direction': np.sign(y_test),
        'correct_direction': np.sign(y_pred) == np.sign(y_test),
    })
    
    results['predicted_price'] = results['actual_close'] * (1 + results['predicted_return'])
    
    accuracy = results['correct_direction'].mean()
    
    return results, accuracy

def backtest_one_ticker(
    ticker="AAPL",
    period="10y",
    test_years=1,
    threshold=0.002,
    model_type="rf",
):
    """
    Backtest a single model type ('rf' or 'gbrt') on one ticker:
    - train on history up to (today - test_years)
    - simulate positions in the last test_years using next-day return predictions.
    threshold is in return units (0.002 = 0.2%).
    """
    hist = get_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)
    hist["target_ret_1d_ahead"] = hist["ret_1d"].shift(-1)

    df = hist.dropna().copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index <= cutoff_date
    test_mask = df.index > cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target_ret_1d_ahead"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target_ret_1d_ahead"].values

    # train model on past only
    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    # predict next-day returns over test window
    y_pred = model.predict(X_test)

    # trading rule: long / short / flat
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
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
):
    """
    Run backtests for BOTH Random Forest ('rf') and Gradient Boosting ('gbrt')
    on the same ticker and return a dict with both results.
    """
    rf_res = backtest_one_ticker(
        ticker=ticker,
        period=period,
        test_years=test_years,
        threshold=threshold,
        model_type="rf",
    )
    gbrt_res = backtest_one_ticker(
        ticker=ticker,
        period=period,
        test_years=test_years,
        threshold=threshold,
        model_type="gbrt",
    )
    return {"rf": rf_res, "gbrt": gbrt_res}

if __name__ == "__main__":
    # Example: compare models on ^GSPC over 10 years
    X, y, _, _, _ = build_features_and_target("^GSPC", period="10y")
    rf_model, rf_r2, rf_rmse = train_model(X, y, model_type="rf")
    gbrt_model, gbrt_r2, gbrt_rmse = train_model(X, y, model_type="gbrt")

    print("Random Forest on ^GSPC")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Test R^2:  {rf_r2:.4f}")
    print(f"  Test RMSE: {rf_rmse:.6f}")

    print("\nGradient Boosting on ^GSPC")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Test R^2:  {gbrt_r2:.4f}")
    print(f"  Test RMSE: {gbrt_rmse:.6f}")
