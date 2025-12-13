# prediction_model.py
import os
import numpy as np
import pandas as pd
import yfinance as yf

import datetime as dt
import requests

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from xgboost import XGBRegressor, XGBClassifier

from data_fetch import (
    get_history as get_history_yahoo_raw,
    get_history_cached as get_history_yahoo,
    get_fmp_fundamentals,
)

from option_pricing import (
    OptionSpec,
    HestonParams,
    PricingModel,
    price_option,
)

def get_heston_params_for_ticker(ticker: str) -> HestonParams | None:
    params_by_ticker = {
        "AAPL": HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.6),
        "NVDA": HestonParams(v0=0.06, theta=0.05, kappa=1.2, sigma=0.5, rho=-0.7),
    }
    return params_by_ticker.get(ticker.upper())

def price_atm_call_for_ticker(
    ticker: str,
    expiry: pd.Timestamp | str,
    spot: float,
    atm_iv: float | None,
    model: PricingModel = PricingModel.BLACK_SCHOLES,
    risk_free: float = 0.05,
    div_yield: float = 0.0,
) -> float | None:
    try:
        if isinstance(expiry, str):
            expiry_date = pd.to_datetime(expiry).date()
        else:
            expiry_date = expiry.date()

        val_date = pd.Timestamp.today().date()
        vol = float(atm_iv) if atm_iv is not None else 0.2

        opt_spec = OptionSpec(
            spot=float(spot),
            strike=float(spot),
            maturity_date=expiry_date,
            valuation_date=val_date,
            rate=float(risk_free),
            div_yield=float(div_yield),
            vol=vol,
            is_call=True,
        )

        heston_params = None
        if model == PricingModel.HESTON:
            heston_params = get_heston_params_for_ticker(ticker)
            if heston_params is None:
                return None

        return float(price_option(opt_spec, model=model, heston_params=heston_params))
    except Exception as e:
        print(f"[pricing] Error pricing ATM call for {ticker}: {e}")
        return None

from pyts.image import GramianAngularField
import matplotlib.pyplot as plt

gaf_cnn = None

try:
    from tensorflow import keras
    GAF_CNN_MODEL_PATH = "gaf_cnn_updown.keras"

    if os.path.exists(GAF_CNN_MODEL_PATH):
        print(f"[GAF-CNN] Loading model from {GAF_CNN_MODEL_PATH}...")
        gaf_cnn = keras.models.load_model(GAF_CNN_MODEL_PATH)
        print("[GAF-CNN] Loaded successfully.")
    else:
        print(f"[GAF-CNN] Model file not found at {GAF_CNN_MODEL_PATH}; prob_up_gaf will be None.")
except Exception as e:
    print(f"[GAF-CNN] TensorFlow/Keras not available or failed to load model: {e}. prob_up_gaf will be None.")
    gaf_cnn = None

def add_rsi(df, window: int = 14, price_col: str = "Close"):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f"rsi_{window}"] = rsi
    return df

def add_macd(df, price_col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9):
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
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"]

    tp_shift = tp.shift(1)
    pos_mf = rmf.where(tp > tp_shift, 0.0)
    neg_mf = rmf.where(tp < tp_shift, 0.0)

    pos_mf_sum = pos_mf.rolling(window=window, min_periods=window).sum()
    neg_mf_sum = neg_mf.rolling(window=window, min_periods=window).sum()

    money_flow_ratio = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_flow_ratio))

    df[f"mfi_{window}"] = mfi
    return df

def add_technical_indicators(df):
    df = df.copy()
    df = add_rsi(df, window=14, price_col="Close")
    df = add_macd(df, price_col="Close", fast=12, slow=26, signal=9)
    df = add_mfi(df, window=14)
    return df

FUNDAMENTAL_COLUMNS = [
    "fund_pe_trailing",
    "fund_pb",
    "fund_market_cap",
]

MACRO_COLUMNS = ["mkt_ret_1d", "term_spread", "t10y", "vix"]
_macro_cache = {}

FRED_API_KEY = os.environ.get("FRED_API_KEY")

def _get_fred_series(series_id: str, start: dt.date, end: dt.date) -> pd.Series:
    if FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY not set in environment")

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        "&file_type=json"
        f"&observation_start={start.isoformat()}"
        f"&observation_end={end.isoformat()}"
    )
    
    print(f"[DEBUG FRED] Fetching {series_id} from {start} to {end}")
    
    resp = requests.get(url, timeout=10)
    
    print(f"[DEBUG FRED] Status {resp.status_code}: {resp.text[:200]}")  # NEW
    
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    # ... rest unchanged


    dates = []
    values = []
    for obs in data:
        d = obs.get("date")
        v = obs.get("value")
        if v in (".", None):
            continue
        dates.append(pd.to_datetime(d))
        values.append(float(v))

    return pd.Series(values, index=pd.DatetimeIndex(dates))

def get_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = get_history_yahoo(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[get_price_history] Yahoo cached failed for {ticker} ({period}): {e}")

    try:
        if interval != "1d":
            raise ValueError("Stooq fallback only supports daily interval")

        stooq_symbol = f"{ticker.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        raw = pd.read_csv(url)

        if raw.empty:
            raise ValueError("Empty Stooq CSV")

        raw["Date"] = pd.to_datetime(raw["Date"])
        raw = raw.set_index("Date").sort_index()

        years_map = {"10y": 10, "5y": 5, "3y": 3, "2y": 2, "1y": 1}
        months_map = {"6mo": 0.5, "3mo": 0.25}
        today = dt.date.today()

        if period in years_map:
            start_date = today - dt.timedelta(days=365 * years_map[period])
        elif period in months_map:
            start_date = today - dt.timedelta(days=int(365 * months_map[period]))
        else:
            start_date = raw.index.min().date()

        df = raw[raw.index.date >= start_date].copy()

        rename_map = {
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        print(f"[get_price_history] Using Stooq data for {ticker} ({period}), rows={len(df)}")
        return df
    except Exception as e:
        print(f"[get_price_history] Stooq failed for {ticker}: {e}")

    try:
        df = get_history_yahoo_raw(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            print(f"[get_price_history] Fallback to raw Yahoo for {ticker}")
            return df
    except Exception as e:
        print(f"[get_price_history] Raw Yahoo fallback failed for {ticker}: {e}")

    raise ValueError(f"No price history available for {ticker} with period={period}")

def get_macro_df(symbol="^GSPC", period="5y") -> pd.DataFrame:
    key = (symbol, period)
    if key in _macro_cache:
        return _macro_cache[key]

    hist = get_price_history(symbol, period=period, interval="1d")
    df = pd.DataFrame(index=hist.index)
    df["mkt_ret_1d"] = hist["Close"].pct_change()

    if FRED_API_KEY is None:
        print("[get_macro_df] FRED_API_KEY not set; using only mkt_ret_1d")
        _macro_cache[key] = df
        return df

    try:
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        s10 = _get_fred_series("DGS10", start_date, end_date)
        s3m = _get_fred_series("DGS3MO", start_date, end_date)
        vix = _get_fred_series("VIXCLS", start_date, end_date)

        macro = pd.DataFrame(index=df.index)
        macro["t10y"] = s10.reindex(df.index).ffill()
        macro["t3m"] = s3m.reindex(df.index).ffill()
        macro["vix"] = vix.reindex(df.index).ffill()
        macro["term_spread"] = macro["t10y"] - macro["t3m"]

        full = df.join(macro[["t10y", "term_spread", "vix"]], how="left")
        _macro_cache[key] = full
        return full
    except Exception as e:
        print(f"[get_macro_df] FRED fetch failed: {e}")
        _macro_cache[key] = df
        return df

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",

    "sma_ratio_10_50",
    "rsi_14",
    "price_to_ma50",

    "bb_width_20",

    "volume_price_corr",
    "volume_trend",
    "vol_ma_20",
    "vol_spike_20",

    "vol_rollmean_20",
    "vol_rollstd_20",

    "high_low_ratio",
    "daily_range",
    "close_position",
    "hl_range",
    "atr_14",

    "day_of_week",
    "month",
    "is_month_end",

    "fund_pe_trailing",
    "fund_pb",
    "fund_market_cap",

    "macd",
    "macd_signal",
    "macd_hist",
    "mfi_14",
]

def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    hist["ret_1d"] = close.pct_change()
    hist["ret_5d"] = close.pct_change(5)
    hist["ret_20d"] = close.pct_change(20)
    hist["vol_20d"] = hist["ret_1d"].rolling(20).std()

    hist["ret_1d_lag1"] = hist["ret_1d"].shift(1)
    hist["ret_1d_lag2"] = hist["ret_1d"].shift(2)
    hist["ret_1d_lag5"] = hist["ret_1d"].shift(5)

    hist["ret_1d_rollmean_5"] = hist["ret_1d"].rolling(5).mean()
    hist["ret_1d_rollstd_5"] = hist["ret_1d"].rolling(5).std()
    hist["ret_1d_rollmean_10"] = hist["ret_1d"].rolling(10).mean()
    hist["ret_1d_rollstd_10"] = hist["ret_1d"].rolling(10).std()

    sma_10 = close.rolling(10).mean()
    sma_50 = close.rolling(50).mean()
    hist["sma_ratio_10_50"] = sma_10 / (sma_50 + 1e-9)

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_20 = sma_20 + 2 * std_20
    lower_20 = sma_20 - 2 * std_20
    hist["bb_width_20"] = (upper_20 - lower_20) / (sma_20 + 1e-9)

    hist["price_to_ma50"] = close / (sma_50 + 1e-9)

    hist["volume_price_corr"] = hist["ret_1d"].rolling(20).corr(volume.pct_change())

    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_30 = volume.rolling(30).mean()
    hist["volume_trend"] = vol_ma_10 / (vol_ma_30 + 1e-9)

    hist["vol_ma_20"] = volume.rolling(20).mean()
    hist["vol_spike_20"] = volume / (hist["vol_ma_20"] + 1e-9)
    hist["vol_spike_1d_ago"] = hist["vol_spike_20"].shift(1)

    hist["vol_rollmean_20"] = volume.rolling(20).mean()
    hist["vol_rollstd_20"] = volume.rolling(20).std()

    hist["high_low_ratio"] = high / (low + 1e-9)
    hist["daily_range"] = (high - low) / (close + 1e-9)
    hist["close_position"] = (close - low) / (high - low + 1e-9)

    hist["hl_range"] = (high - low) / (close.shift(1) + 1e-9)
    hist["atr_14"] = hist["hl_range"].rolling(14).mean()

    hist["day_of_week"] = hist.index.dayofweek
    hist["month"] = hist.index.month
    hist["is_month_end"] = (hist.index.day >= 25).astype(int)

    hist = add_technical_indicators(hist)

    return hist

def make_model(model_type: str = "rf", random_state: int = 42, task: str = "reg"):
    if task == "clf":
        if model_type == "xgb":
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
                tree_method="hist",
                verbosity=0,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
            )
        else:
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                random_state=random_state,
                n_jobs=-1,
            )

    if model_type == "linreg":
        return LinearRegression()
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
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
        )
    else:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )

def get_fundamental_features(ticker: str) -> dict:
    feats = {
        "fund_pe_trailing": np.nan,
        "fund_pb": np.nan,
        "fund_market_cap": np.nan,
    }

    try:
        fmp_data = get_fmp_fundamentals(ticker)
        if isinstance(fmp_data, dict):
            for k in feats.keys():
                if k in fmp_data:
                    feats[k] = fmp_data.get(k, np.nan)
    except Exception:
        pass

    if any(pd.isna(v) for v in feats.values()):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if pd.isna(feats["fund_pe_trailing"]):
                feats["fund_pe_trailing"] = info.get("trailingPE", np.nan)
            if pd.isna(feats["fund_pb"]):
                feats["fund_pb"] = info.get("priceToBook", np.nan)
            if pd.isna(feats["fund_market_cap"]):
                feats["fund_market_cap"] = float(info.get("marketCap", np.nan))
        except Exception:
            pass

    for k in feats:
        if pd.isna(feats[k]):
            feats[k] = 0.0

    return feats

def build_features_and_target(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    use_vol_scaled_target: bool = False,
):
    fallback_periods = ["5y", "3y", "2y", "1y", "6mo", "3mo"]
    if period in fallback_periods:
        periods_to_try = [period] + [p for p in fallback_periods if p != period]
    else:
        periods_to_try = [period] + fallback_periods

    last_error = None
    min_rows = 60

    for per in periods_to_try:
        try:
            hist = get_price_history(ticker, period=per, interval="1d")
            if hist is None or hist.empty:
                raise ValueError(f"No raw history for {ticker} with period={per}")

            hist = add_price_features(hist)

            macro_df = get_macro_df(symbol="^GSPC", period=per)
            hist = hist.join(macro_df, how="left")

            fund_feats = get_fundamental_features(ticker)
            for k, v in fund_feats.items():
                hist[k] = v

            raw_target = hist["Close"].pct_change(horizon).shift(-horizon)
            if use_vol_scaled_target:
                hist[f"target_ret_{horizon}d_ahead"] = raw_target / (hist["vol_20d"] + 1e-9)
            else:
                hist[f"target_ret_{horizon}d_ahead"] = raw_target

            feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
            cols_needed = feat_cols + [f"target_ret_{horizon}d_ahead"]

            # DEBUG: print what we have before dropna
            print(f"[DEBUG {ticker}] Before dropna: {len(hist)} rows")
            nan_counts = hist[cols_needed].isna().sum()
            print(f"[DEBUG {ticker}] NaNs per column:\n{nan_counts[nan_counts > 0]}")

            df = hist[cols_needed].dropna().copy()

            print(f"[DEBUG {ticker}] After dropna: {len(df)} rows")

            if df.empty or len(df) < min_rows:
                raise ValueError(
                    f"Only {len(df)} usable rows for {ticker} with period={per}"
                )

            X = df[feat_cols].values
            y = df[f"target_ret_{horizon}d_ahead"].values

            last_row = df.iloc[-1]
            last_row_features = last_row[feat_cols].values
            last_close = hist.loc[df.index[-1], "Close"]
            last_vol_20d = last_row["vol_20d"]

            return X, y, last_row_features, last_close, last_vol_20d

        except Exception as e:
            last_error = e
            continue

    raise ValueError(
        f"No usable history for {ticker} after trying periods {periods_to_try}. "
        f"Last error: {last_error}"
    )

def build_features_and_direction_target(
    ticker="^GSPC",
    period="5y",
    horizon=1,
):
    X, y_reg, last_feats, last_close, last_vol_20d = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=False,
    )
    y_dir = (y_reg > 0).astype(int)
    return X, y_dir, last_feats, last_close, last_vol_20d

def train_model(X, y, model_type="rf", test_size=0.2, random_state=42, task="reg"):
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=random_state, task=task)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == "reg":
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return model, r2, rmse
    else:
        acc = accuracy_score(y_test, y_pred)
        return model, acc, None

def predict_next_for_ticker(
    ticker="^GSPC",
    period="5y",
    model_type="rf",
    horizon=1,
    use_vol_scaled_target: bool = False,
):
    X, y, x_last, last_close, last_vol_20d = build_features_and_target(
        ticker, period=period, horizon=horizon, use_vol_scaled_target=use_vol_scaled_target
    )

    n = len(X)
    split_idx = int(n * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=42, task="reg")
    model.fit(X_train, y_train)

    pred_ret = float(model.predict(x_last.reshape(1, -1))[0])
    if use_vol_scaled_target:
        pred_ret = pred_ret * float(last_vol_20d)
    pred_price = float(last_close * (1 + pred_ret))

    prob_up = None
    prob_down = None
    try:
        y_dir = (y > 0).astype(int)
        y_dir_train = y_dir[:split_idx]

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
        else:
            pred_dir = int(clf.predict(x_last.reshape(1, -1))[0])
            prob_up = 1.0 if pred_dir == 1 else 0.0
            prob_down = 1.0 - prob_up
    except Exception:
        prob_up = None
        prob_down = None

    prob_up_gaf = None
    try:
        prob_up_gaf = predict_up_gaf_cnn(ticker)
    except Exception as e:
        print(f"[GAF-CNN] Failed to compute prob_up_gaf for {ticker}: {e}")
        prob_up_gaf = None

    fund_feats = get_fundamental_features(ticker)
    pe_ratio = fund_feats.get("fund_pe_trailing", None)

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(feat_cols, model.feature_importances_))
        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_features_str = "\n".join(
            [f"- **{feat}**: {imp:.3f}" for feat, imp in top_features]
        )
    else:
        top_features_str = "N/A"

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "last_close": last_close,
        "vol_20d": last_vol_20d,
        "pe_ratio": pe_ratio,
        "pred_next_ret": pred_ret,
        "pred_next_price": pred_price,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_up_gaf": prob_up_gaf,
        "num_features": len(feat_cols),
        "top_features": top_features_str,
    }

def track_predictions(ticker, period="1y", model_type="rf", horizon=1):
    try:
        hist = get_price_history(ticker, period=period, interval="1d")

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

        feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
        cols_needed = feat_cols + [f"target_ret_{horizon}d_ahead"]
        df = hist[cols_needed].dropna().copy()

        print(f"After dropna for {ticker}: {len(df)} rows")

        if len(df) < 50:
            print(f"Not enough data after feature engineering for {ticker}")
            return pd.DataFrame(), 0.0

        n_rows = len(df)
        min_test = 60
        max_test = 252
        proposed_test = int(n_rows * 0.2)

        test_size = max(min_test, proposed_test)
        test_size = min(test_size, max_test, n_rows - 1)

        if test_size < 5:
            print(f"Test size too small: {test_size}")
            return pd.DataFrame(), 0.0

        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        X_train = train_df[feat_cols].values
        y_train = train_df[f"target_ret_{horizon}d_ahead"].values

        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)

        X_test = test_df[feat_cols].values
        y_test = test_df[f"target_ret_{horizon}d_ahead"].values
        y_pred = model.predict(X_test)

        results = pd.DataFrame(
            {
                "date": test_df.index,
                "actual_close": hist.loc[test_df.index, "Close"],
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
    hist = get_price_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)
    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")
    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v

    hist[f"target_ret_{horizon}d_ahead"] = hist["Close"].pct_change(horizon).shift(
        -horizon
    )

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    cols_needed = feat_cols + [f"target_ret_{horizon}d_ahead"]
    df = hist[cols_needed].dropna().copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index <= cutoff_date
    test_mask = df.index > cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df[feat_cols].values
    y_train = train_df[f"target_ret_{horizon}d_ahead"].values

    X_test = test_df[feat_cols].values
    y_test = test_df[f"target_ret_{horizon}d_ahead"].values

    model = make_model(model_type=model_type, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    positions = np.where(
        y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)
    )

    cost_per_trade = 0.0005
    pnl = []
    prev_pos = 0
    for pos, ret in zip(positions, y_test):
        trade = abs(pos - prev_pos)
        pnl_t = pos * ret - cost_per_trade * trade
        pnl.append(pnl_t)
        prev_pos = pos
    pnl = np.array(pnl)

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

def walk_forward_backtest(
    ticker="AAPL",
    period="10y",
    horizon=1,
    model_type="rf",
    train_years=4,
    test_years=1,
    threshold=0.002,
    cost_per_trade=0.0005,
):
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return []

    hist = add_price_features(hist)

    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")
    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v

    target_col = f"target_ret_{horizon}d_ahead"
    hist[target_col] = hist["Close"].pct_change(horizon).shift(-horizon)

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    cols_needed = feat_cols + [target_col]
    df = hist[cols_needed].dropna().copy()

    if df.empty:
        return []

    fold_metrics = []
    train_days = int(252 * train_years)
    test_days = int(252 * test_years)

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days
        test_end = train_end + test_days
        if test_end > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[train_end:test_end]
        if len(train_df) < 50 or len(test_df) < 20:
            break

        X_train = train_df[feat_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feat_cols].values
        y_test = test_df[target_col].values

        model = make_model(model_type=model_type, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        positions = np.where(
            y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0)
        )

        pnl = []
        prev_pos = 0
        for pos, ret in zip(positions, y_test):
            trade = abs(pos - prev_pos)
            pnl_t = pos * ret - cost_per_trade * trade
            pnl.append(pnl_t)
            prev_pos = pos
        pnl = np.array(pnl)

        hit_rate = (np.sign(y_pred) == np.sign(y_test)).mean()
        avg_daily = pnl.mean()
        std_daily = pnl.std(ddof=1)
        sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily > 0 else 0.0
        num_trades = int(
            np.count_nonzero(
                np.diff(np.concatenate([[0], positions])) != 0
            )
        )

        fold_metrics.append(
            {
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "test_days": len(pnl),
                "hit_rate": hit_rate,
                "sharpe": sharpe,
                "num_trades": num_trades,
            }
        )

        start += test_days

    return fold_metrics

def analyze_feature_significance(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    use_vol_scaled_target: bool = False,
    alpha: float = 0.05,
):
    X, y, _, _, _ = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=use_vol_scaled_target,
    )

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    X_df = pd.DataFrame(X, columns=feat_cols)

    X_df = sm.add_constant(X_df)
    ols_model = sm.OLS(y, X_df).fit()

    rows = []
    ordered_names = ["const"] + feat_cols
    for name in ordered_names:
        if name in ols_model.params.index:
            p_val = float(ols_model.pvalues[name])
            rows.append(
                {
                    "feature": name,
                    "coef": float(ols_model.params[name]),
                    "p_value": p_val,
                    "significant": bool(p_val < alpha),
                }
            )

    sig_df = pd.DataFrame(rows).sort_values("p_value")
    return ols_model, sig_df

def make_gaf_image_from_returns(returns: pd.Series, window: int = 60, image_size: int = 30):
    r = returns.dropna().values
    if len(r) < window:
        return None, None

    window_vals = r[-window:]
    X = window_vals.reshape(1, -1)

    gaf = GramianAngularField(image_size=image_size, method="summation")
    X_gaf = gaf.fit_transform(X)
    img = X_gaf[0]

    fig, ax = plt.subplots(figsize=(2, 2))
    cax = ax.imshow(img, cmap="rainbow", origin="lower", aspect="equal")
    ax.set_title(f"GAF (last {window} returns)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax

def predict_up_gaf_cnn(
    ticker: str,
    window: int = 30,
    image_size: int = 30,
    period: str = "3y",
) -> float | None:
    if gaf_cnn is None:
        return None

    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty or len(hist) < window + 1:
        return None

    closes = hist["Close"].astype(float).values
    rets = pd.Series(closes).pct_change().dropna()

    if len(rets) < window:
        return None

    window_vals = rets.values[-window:]
    X = window_vals.reshape(1, -1)

    gaf = GramianAngularField(image_size=image_size, method="summation")
    X_gaf = gaf.fit_transform(X)
    X_input = X_gaf[..., np.newaxis]

    try:
        proba = gaf_cnn.predict(X_input, verbose=0)[0]
        prob_up = float(proba[0])
        return prob_up
    except Exception as e:
        print(f"[GAF-CNN] Error during predict for {ticker}: {e}")
        return None

def tune_xgb_hyperparams(X, y, random_state=42):
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
