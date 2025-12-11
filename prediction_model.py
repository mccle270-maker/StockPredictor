# prediction_model.py
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from xgboost import XGBRegressor, XGBClassifier

from data_fetch import get_history, get_history_cached, get_fmp_fundamentals


# ---------- Technical indicator helpers ----------

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


# ---------- Fundamentals & macro ----------

FUNDAMENTAL_COLUMNS = [
    "fund_pe_trailing",
    "fund_pb",
    "fund_market_cap",
]

MACRO_COLUMNS = ["mkt_ret_1d"]
_macro_cache = {}


def get_macro_df(symbol="^GSPC", period="5y") -> pd.DataFrame:
    key = (symbol, period)
    if key in _macro_cache:
        return _macro_cache[key]
    t = yf.Ticker(symbol)
    hist = t.history(period=period, interval="1d")
    df = pd.DataFrame(index=hist.index)
    df["mkt_ret_1d"] = hist["Close"].pct_change()
    _macro_cache[key] = df
    return df


# ---------- Monte Carlo helpers (flexible) ----------

MC_FEATURE_COLUMNS = [
    "mc_pop_gt0",          # P(P/L > 0)
    "mc_pop_gt_thresh",    # P(P/L > profit_thresh)
    "mc_ev",               # E[P/L]
    "mc_pnl_p05",          # 5th percentile P/L
    "mc_pnl_p50",          # 50th percentile P/L
    "mc_pnl_p95",          # 95th percentile P/L
]


def simulate_gbm_paths(S0: float,
                       mu: float,
                       sigma: float,
                       T_years: float,
                       steps: int,
                       n_paths: int,
                       random_state: int | None = None) -> np.ndarray:
    """
    GBM simulation: returns array (steps+1, n_paths) of prices.
    """
    rng = np.random.default_rng(random_state)
    dt = T_years / steps
    z = rng.normal(0.0, 1.0, size=(steps, n_paths))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(increments, axis=0)])
    paths = S0 * np.exp(log_paths)
    return paths


def run_option_mc_for_row(row: pd.Series,
                          ticker: str,
                          horizon: int,
                          profit_thresh: float = 0.0,
                          n_paths: int = 5000,
                          steps_per_year: int = 252,
                          annual_mu: float = 0.0,
                          annual_rf: float = 0.0,
                          iv_col: str | None = None,
                          **mc_kwargs) -> dict:
    """
    Flexible Monte Carlo for a *long European call* on the underlying.

    You can pass in from the UI (via mc_kwargs):
      - strike_price: explicit strike (float)
      - premium: entry debit/credit per contract (float)
      - dte: days to expiry (int); if not given, horizon is used
      - moneyness: if no strike, K = moneyness * S0
    """
    try:
        S0 = float(row.get("Close"))
    except Exception:
        return {k: np.nan for k in MC_FEATURE_COLUMNS}

    # Volatility (annual sigma)
    if iv_col is not None and iv_col in row and pd.notna(row[iv_col]):
        sigma = float(row[iv_col])
    else:
        vol_20d = row.get("vol_20d")
        if pd.isna(vol_20d):
            return {k: np.nan for k in MC_FEATURE_COLUMNS}
        sigma = float(vol_20d) * np.sqrt(252.0)

    # Time to expiry
    dte = mc_kwargs.get("dte", horizon)
    T_years = max(int(dte), 1) / 252.0
    steps = max(int(T_years * steps_per_year), 1)

    # Strike and premium
    strike = mc_kwargs.get("strike_price", row.get("strike_price", None))
    if strike is None:
        moneyness = mc_kwargs.get("moneyness", 1.0)
        strike = float(moneyness) * S0
    else:
        strike = float(strike)

    premium = mc_kwargs.get("premium", row.get("premium", 1.0))
    premium = float(premium)

    # Simulate paths and payoff
    paths = simulate_gbm_paths(
        S0=S0,
        mu=annual_mu,
        sigma=sigma,
        T_years=T_years,
        steps=steps,
        n_paths=n_paths,
    )
    ST = paths[-1]
    payoff = np.maximum(ST - strike, 0.0)  # long call payoff

    df = np.exp(-annual_rf * T_years)
    pnl = df * payoff - premium

    pop_gt0 = float(np.mean(pnl > 0.0))
    pop_gt_thresh = float(np.mean(pnl > profit_thresh))
    ev = float(np.mean(pnl))
    p05 = float(np.percentile(pnl, 5))
    p50 = float(np.percentile(pnl, 50))
    p95 = float(np.percentile(pnl, 95))

    return {
        "mc_pop_gt0": pop_gt0,
        "mc_pop_gt_thresh": pop_gt_thresh,
        "mc_ev": ev,
        "mc_pnl_p05": p05,
        "mc_pnl_p50": p50,
        "mc_pnl_p95": p95,
    }


# ---------- Feature columns ----------

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
] + MC_FEATURE_COLUMNS


# ---------- Price feature engineering ----------

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


# ---------- Model factory ----------

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


# ---------- Fundamentals fetch ----------

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


# ---------- Build features & target ----------

def build_features_and_target(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    use_vol_scaled_target: bool = False,
    use_mc_features: bool = False,
    mc_kwargs: dict | None = None,
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
            hist = get_history_cached(ticker, period=per, interval="1d")
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

            df = hist.dropna().copy()
            if df.empty or len(df) < min_rows:
                raise ValueError(
                    f"Only {len(df)} usable rows for {ticker} with period={per}"
                )

            if use_mc_features:
                mc_kwargs = mc_kwargs or {}
                mc_results = []
                for _, row in df.iterrows():
                    mc_dict = run_option_mc_for_row(
                        row, ticker=ticker, horizon=horizon, **mc_kwargs
                    )
                    mc_results.append(mc_dict)
                mc_df = pd.DataFrame(mc_results, index=df.index)
                for col in MC_FEATURE_COLUMNS:
                    if col in mc_df.columns:
                        df[col] = mc_df[col]
                    else:
                        df[col] = np.nan
                df[MC_FEATURE_COLUMNS] = df[MC_FEATURE_COLUMNS].fillna(0.0)

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


def build_features_and_direction_target(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    use_mc_features: bool = False,
    mc_kwargs: dict | None = None,
):
    X, y_reg, last_feats, last_close, last_vol_20d = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=False,
        use_mc_features=use_mc_features,
        mc_kwargs=mc_kwargs,
    )
    y_dir = (y_reg > 0).astype(int)
    return X, y_dir, last_feats, last_close, last_vol_20d


# ---------- Train & predict helpers ----------

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
    use_mc_features: bool = False,
    mc_kwargs: dict | None = None,
):
    X, y, x_last, last_close, last_vol_20d = build_features_and_target(
        ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=use_vol_scaled_target,
        use_mc_features=use_mc_features,
        mc_kwargs=mc_kwargs,
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

    mc_dict_last = {}
    if use_mc_features:
        row_like = pd.Series({"Close": last_close, "vol_20d": last_vol_20d})
        mc_dict_last = run_option_mc_for_row(
            row_like,
            ticker=ticker,
            horizon=horizon,
            **(mc_kwargs or {}),
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
        "mc_pop_gt0": mc_dict_last.get("mc_pop_gt0"),
        "mc_pop_gt_thresh": mc_dict_last.get("mc_pop_gt_thresh"),
        "mc_ev": mc_dict_last.get("mc_ev"),
        "mc_pnl_p05": mc_dict_last.get("mc_pnl_p05"),
        "mc_pnl_p50": mc_dict_last.get("mc_pnl_p50"),
        "mc_pnl_p95": mc_dict_last.get("mc_pnl_p95"),
    }


# ---------- Tracking & backtests (accuracy test) ----------

def track_predictions(ticker, period="5y", model_type="rf", horizon=1):
    """
    Compare model predictions to actual multi-day returns over the past period.
    Original behavior preserved (no MC features used here).
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

# ---------- Backtests (regression + walk-forward) ----------

def backtest_one_ticker(
    ticker="AAPL",
    period="10y",
    test_years=1,
    threshold=0.002,
    model_type="rf",
    horizon=1,
    cost_per_trade: float = 0.0005,
):
    """
    Backtest a single model type ('rf', 'gbrt', or 'xgb') on one ticker with multi-day predictions.
    Computes hit_rate and Sharpe from a simple long/short/flat strategy.
    """
    hist = get_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return {
            "ticker": ticker,
            "model_type": model_type,
            "horizon": horizon,
            "test_days": 0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
        }

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
    if df.empty:
        return {
            "ticker": ticker,
            "model_type": model_type,
            "horizon": horizon,
            "test_days": 0,
            "total_return": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
        }

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
    y_pred = model.predict(X_test)

    # trading rule: long / short / flat
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
    Returns a dict of {model_name: metrics_dict}.
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
    """
    Walk-forward backtest:
      - Train on rolling train_years
      - Test on following test_years
      - Repeat across the history.
    Returns a list of dicts with Sharpe, hit_rate, trades per fold.
    """
    hist = get_history(ticker, period=period, interval="1d")
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

    df = hist.dropna().copy()
    if df.empty:
        return []

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS

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


# ---------- Hyperparameter tuning ----------

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
    X, y, _, _, _ = build_features_and_target(
        "^GSPC",
        period="10y",
        horizon=1,
        use_mc_features=True,
        mc_kwargs={"n_paths": 2000, "premium": 1.0, "moneyness": 1.0},
    )
    rf_model, rf_r2, rf_rmse = train_model(X, y, model_type="rf")
    print("RF 1-day R^2:", rf_r2, "RMSE:", rf_rmse)
