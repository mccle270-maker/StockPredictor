# prediction_model.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


from data_fetch import get_history, get_history_cached


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


def build_features_and_target(ticker="^GSPC", period="5y", horizon=1):
    """
    Build feature matrix X, target vector y, and the latest row info
    for multi-day prediction.
    
    horizon: Number of days ahead to predict (1, 2, or 3)
    """
    hist = get_history_cached(ticker, period=period, interval="1d")
    hist = add_price_features(hist)


    # Target: multi-day return based on horizon
    hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)


    df = hist.dropna().copy()
    X = df[FEATURE_COLUMNS].values
    y = df[f"target_ret_{horizon}d_ahead"].values


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


def predict_next_for_ticker(ticker="^GSPC", period="5y", model_type="rf", horizon=1):
    """
    Train a model on the ticker's history and predict multi-day return & price.
    
    ticker: Stock symbol
    period: Historical data period
    model_type: 'rf' or 'gbrt'
    horizon: Number of days ahead to predict (1, 2, or 3)
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
        "horizon": horizon,
        "last_close": last_close,
        "vol_20d": last_vol_20d,
        "pe_ratio": pe_ratio,
        "pred_next_ret": pred_ret,
        "pred_next_price": pred_price,
        "num_features": len(FEATURE_COLUMNS),
        "top_features": top_features_str,
    }


def track_predictions(ticker, period="3mo", model_type="rf", horizon=1):
    """
    Compare model predictions to actual multi-day returns over the past period.
    
    horizon: Number of days ahead that was predicted (1, 2, or 3)
    """
    try:
        hist = get_history(ticker, period=period, interval="1d")
        
        if hist.empty or len(hist) < 50:
            print(f"Insufficient data for {ticker}: only {len(hist)} rows")
            return pd.DataFrame(), 0.0
        
        hist = add_price_features(hist)
        hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)
        
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
        
        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[f"target_ret_{horizon}d_ahead"].values
        
        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions for test period
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df[f"target_ret_{horizon}d_ahead"].values
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
        
        print(f"Success! Generated {len(results)} test predictions with {accuracy*100:.1f}% accuracy")
        
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
    Backtest a single model type ('rf' or 'gbrt') on one ticker with multi-day predictions.
    
    horizon: Number of days ahead to predict (1, 2, or 3)
    threshold: Adjusted for multi-day predictions (e.g., 0.004 for 2-day, 0.006 for 3-day)
    """
    hist = get_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)
    hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)


    df = hist.dropna().copy()


    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index <= cutoff_date
    test_mask = df.index > cutoff_date


    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()


    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[f"target_ret_{horizon}d_ahead"].values


    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[f"target_ret_{horizon}d_ahead"].values


    # train model on past only
    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)


    # predict multi-day returns over test window
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
    Run backtests for BOTH Random Forest ('rf') and Gradient Boosting ('gbrt')
    on the same ticker with multi-day predictions.
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
    return {"rf": rf_res, "gbrt": gbrt_res}


if __name__ == "__main__":
    # Example: compare models on ^GSPC with 1-day, 2-day, and 3-day horizons
    print("=" * 60)
    print("Testing 1-Day Predictions")
    print("=" * 60)
    X, y, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=1)
    rf_model, rf_r2, rf_rmse = train_model(X, y, model_type="rf")
    gbrt_model, gbrt_r2, gbrt_rmse = train_model(X, y, model_type="gbrt")

    print("Random Forest (1-day)")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Test R^2:  {rf_r2:.4f}")
    print(f"  Test RMSE: {rf_rmse:.6f}")

    print("\nGradient Boosting (1-day)")
    print(f"  Test R^2:  {gbrt_r2:.4f}")
    print(f"  Test RMSE: {gbrt_rmse:.6f}")
    
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
