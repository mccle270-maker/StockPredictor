import os, datetime as dt, requests
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
# ---- SPX cache (avoid repeated yf.download("^GSPC") per ticker) ----
_SPX_CACHE = {}

def _get_spx(start, end, tz=None):
    """
    Fetch SPX once (cached) for a given date range + timezone-ness, and normalize columns/index.

    tz:
      - None => tz-naive index
      - timezone string / tzinfo => tz-aware index localized to tz
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Cache key: normalize dates, also include tz so tz-aware and tz-naive don't clash
    key = (start_ts.tz_localize(None), end_ts.tz_localize(None), str(tz))
    if key in _SPX_CACHE:
        return _SPX_CACHE[key]

    spx = yf.download("^GSPC", start=key[0], end=key[1], progress=False)

    # If yfinance returns MultiIndex columns, flatten them
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Make SPX index match the caller's timezone-ness
    idx = pd.DatetimeIndex(spx.index)
    if tz is not None:
        spx.index = idx.tz_localize(tz)
    else:
        spx.index = idx.tz_localize(None)

    _SPX_CACHE[key] = spx
    return spx


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from xgboost import XGBRegressor, XGBClassifier

try:
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except Exception:
    ElasticNetCV = None
    StandardScaler = None
    Pipeline = None

from data_fetch import get_history as get_history_yahoo_raw, get_history_cached as get_history_yahoo, get_history_intraday_cached, get_fmp_fundamentals
from option_pricing import OptionSpec, HestonParams, PricingModel, price_option
from pyts.image import GramianAngularField

import matplotlib.pyplot as plt


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


USE_ELASTICNET_SELECT = env_bool("USE_ELASTICNET_SELECT", False)
try:
    ELASTICNET_L1_RATIO = float(os.environ.get("ELASTICNET_L1_RATIO", 0.5))
except Exception:
    ELASTICNET_L1_RATIO = 0.5

try:
    ELASTICNET_CV_FOLDS = int(os.environ.get("ELASTICNET_CV_FOLDS", 5))
except Exception:
    ELASTICNET_CV_FOLDS = 5


def get_heston_params_for_ticker(ticker: str) -> HestonParams | None:
    params_by_ticker = {
        "AAPL": HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.6),
        "NVDA": HestonParams(v0=0.06, theta=0.05, kappa=1.2, sigma=0.5, rho=-0.7),
    }
    return params_by_ticker.get(ticker.upper())

from scipy.stats import norm

TRADING_DAYS = 252

def add_gbm_features(hist: pd.DataFrame, window: int = 60, horizons=(1, 5)) -> pd.DataFrame:
    """
    Adds GBM-based distribution features for each horizon in `horizons`.
    Produces columns:
      gbm_mu_{window}d, gbm_sig_{window}d
      gbm_prob_up_{h}d, gbm_exp_ret_{h}d, gbm_p05_ret_{h}d, gbm_p95_ret_{h}d
    """
    hist = hist.copy()

    close = hist["Close"].astype(float)
    logret = np.log(close).diff()

    # lagged rolling estimates (no leakage)
    mu_d = logret.rolling(window).mean().shift(1)
    sig_d = logret.rolling(window).std(ddof=1).shift(1)

    # base params (your FEATURECOLUMNS uses gbm_mu_60d / gbm_sig_60d)
    hist[f"gbm_mu_{window}d"] = mu_d
    hist[f"gbm_sig_{window}d"] = sig_d

    for h in horizons:
        T = h / TRADING_DAYS
        m = (mu_d - 0.5 * sig_d**2) * T
        s = sig_d * np.sqrt(T)

        hist[f"gbm_prob_up_{h}d"] = norm.cdf(m / (s + 1e-12))
        hist[f"gbm_exp_ret_{h}d"] = np.exp(mu_d * T) - 1.0
        hist[f"gbm_p05_ret_{h}d"] = np.exp(m + s * norm.ppf(0.05)) - 1.0
        hist[f"gbm_p95_ret_{h}d"] = np.exp(m + s * norm.ppf(0.95)) - 1.0

    return hist



# Strip timezone so downstream comparisons don't thrash...
def price_atm_call_for_ticker(
    ticker: str,
    expiry: pd.Timestamp | str,
    spot: float,
    atm_iv: float | None = None,
    model: PricingModel = PricingModel.BLACK_SCHOLES,
    risk_free: float = 0.05,
    div_yield: float = 0.0,
) -> float | None:
    try:
        expiry_date = pd.to_datetime(expiry).date() if isinstance(expiry, str) else expiry.date()
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


gafcnn = None
try:
    from tensorflow import keras
    GAF_CNN_MODEL_PATH = "gaf_cnn_updown.keras"
    if os.path.exists(GAF_CNN_MODEL_PATH):
        print(f"[GAF-CNN] Loading model from {GAF_CNN_MODEL_PATH}...")
        gafcnn = keras.models.load_model(GAF_CNN_MODEL_PATH)
        print("[GAF-CNN] Loaded successfully.")
    else:
        print(f"[GAF-CNN] Model file not found at {GAF_CNN_MODEL_PATH}. probup_gaf will be None.")
except Exception as e:
    print(f"[GAF-CNN] TensorFlow/Keras not available or failed to load model: {e}. probup_gaf will be None.")
    gafcnn = None

def make_t1_from_horizon(index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    t1 = pd.Series(index=index, dtype="datetime64[ns]")
    if horizon <= 0:
        t1[:] = index
        return t1
    if len(index) <= horizon:
        t1[:] = index[-1]
        return t1
    t1.iloc[:-horizon] = index[horizon:]
    t1.iloc[-horizon:] = index[-1]
    return t1

class PurgedKFold:
    def __init__(self, nsplits: int = 5, t1: pd.Series = None, pctembargo: float = 0.01, n_splits: int = None):
        if n_splits is not None:
            nsplits = n_splits
        self.nsplits = int(nsplits)
        self.n_splits = self.nsplits
        self.t1 = t1
        self.pctembargo = float(pctembargo)
        self.pct_embargo = self.pctembargo

    def split(self, X: pd.DataFrame, y=None):
        if not X.index.equals(self.t1.index):
            raise ValueError("X.index must equal t1.index")
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        test_starts = np.cumsum(fold_sizes) - fold_sizes
        embargo = int(np.ceil(n * self.pct_embargo))

        all_idx = np.arange(n)
        for k in range(self.n_splits):
            t0 = test_starts[k]
            t1i = t0 + fold_sizes[k] - 1
            test_idx = np.arange(t0, t1i + 1)

            emb_start = t1i + 1
            emb_end = min(n, t1i + 1 + embargo)
            embargo_idx = np.arange(emb_start, emb_end)

            test_start_time = X.index[t0]
            test_end_time = X.index[t1i]

            train_idx = np.setdiff1d(all_idx, np.concatenate([test_idx, embargo_idx]))

            train_start_times = X.index[train_idx]
            train_end_times = self.t1.iloc[train_idx].values
            overlaps = (train_start_times <= test_end_time) & (train_end_times >= test_start_time)
            train_idx = train_idx[~overlaps]

            yield train_idx, test_idx

def sharpe_from_returns(r: pd.Series, periods_per_year: int = 252) -> float | None:
    r = r.dropna()
    if len(r) < 10 or r.std(ddof=1) == 0:
        return None
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))

def build_cross_sectional_portfolio(
    dates,
    tickers,
    preds,
    top_pct: float = 0.1,
    bottom_pct: float | None = None,
) -> pd.DataFrame:
    """
    Build daily cross-sectional portfolio weights from predictions.
    Returns DataFrame indexed by date, columns=tickers, entries=weights.
    """
    df = pd.DataFrame(
        {"date": pd.DatetimeIndex(dates), "ticker": tickers, "pred": preds}
    ).dropna(subset=["date", "ticker", "pred"])
    df = df.sort_values(["date", "pred"])

    weights = []

    for d, grp in df.groupby("date"):
        n = len(grp)
        if n == 0:
            continue
        k_top = max(1, int(n * top_pct))

        if bottom_pct:
            k_bot = max(1, int(n * bottom_pct))
            long = grp.tail(k_top)
            short = grp.head(k_bot)
            w = (
                pd.concat(
                    [
                        pd.Series(1.0 / k_top, index=long["ticker"]),
                        pd.Series(-1.0 / k_bot, index=short["ticker"]),
                    ]
                )
                .groupby(level=0)
                .sum()
            )
        else:
            long = grp.tail(k_top)
            w = pd.Series(1.0 / k_top, index=long["ticker"])

        w.name = d
        weights.append(w)

    if not weights:
        raise ValueError("No portfolio weights generated.")

    W = pd.DataFrame(weights).fillna(0.0)
    W.index = pd.DatetimeIndex(W.index)
    W.index.name = "date"
    return W


def max_drawdown_from_returns(r: pd.Series) -> float | None:
    r = r.dropna()
    if len(r) < 2:
        return None
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())

def turnover_from_positions(pos: pd.Series) -> float:
    return float(pos.diff().abs().fillna(0.0).sum())

# Strip timezone so downstream comparisons don't thrash...
def add_rsi(df, window: int = 14, price_col: str = "Close"):
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    df[f"rsi{window}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, price_col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd
    df["macdsignal"] = macd_signal
    df["macdhist"] = macd - macd_signal
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
    df[f"mfi{window}"] = 100 - (100 / (1 + money_flow_ratio))
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_rsi(df, window=14, price_col="Close")
    df = add_macd(df, price_col="Close", fast=12, slow=26, signal=9)
    df = add_mfi(df, window=14)
    return df


FUNDAMENTAL_COLUMNS = ["fund_pe_trailing", "fund_pb", "fund_marketcap"]
MACRO_COLUMNS = ["mkt_ret_1d", "term_spread", "t10y", "vix"]
macro_cache = {}

FRED_API_KEY = os.environ.get("FRED_API_KEY")


# Strip timezone so downstream comparisons don't thrash...
def get_fred_series(series_id: str, start: dt.date, end: dt.date) -> pd.Series:
    if FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY not set in environment")

    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        f"&observation_start={start.isoformat()}&observation_end={end.isoformat()}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("observations", [])

    dates, values = [], []
    for obs in data:
        d = obs.get("date")
        v = obs.get("value")
        if v in (".", None):
            continue
        dates.append(pd.to_datetime(d))
        values.append(float(v))

    return pd.Series(values, index=pd.DatetimeIndex(dates))


# Strip timezone so downstream comparisons don't thrash...
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
        df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
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


# Strip timezone so downstream comparisons don't thrash...
def get_macro_df(symbol="^GSPC", period="5y") -> pd.DataFrame:
    key = (symbol, period)
    if key in macro_cache:
        return macro_cache[key]

    hist = get_price_history(symbol, period=period, interval="1d")
    df = pd.DataFrame(index=hist.index)
    df["mkt_ret_1d"] = hist["Close"].pct_change()

    if FRED_API_KEY is None:
        print("[get_macro_df] FRED_API_KEY not set; using only mkt_ret_1d")
        macro_cache[key] = df
        return df

    try:
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        s10 = get_fred_series("DGS10", start_date, end_date)
        s3m = get_fred_series("DGS3MO", start_date, end_date)
        vix = get_fred_series("VIXCLS", start_date, end_date)

        df_dates = df.index.normalize().tz_localize(None)
        df["t10y"] = s10.reindex(df_dates).ffill().bfill().values
        df["t3m"] = s3m.reindex(df_dates).ffill().bfill().values
        df["vix"] = vix.reindex(df_dates).ffill().bfill().values
        df["term_spread"] = df["t10y"] - df["t3m"]

        macro_cache[key] = df
        return df
    except Exception as e:
        print(f"[get_macro_df] FRED fetch failed: {e}")
        macro_cache[key] = df
        return df


# Strip timezone so downstream comparisons don't thrash...
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_20d", "vol_20d",
    "sma_ratio_10_50", "rsi14", "price_to_ma50", "bb_width_20",
    "volume_price_corr", "volume_trend", "vol_ma_20", "vol_spike_20",
    "vol_roll_mean_20", "vol_roll_std_20",
    "high_low_ratio", "daily_range", "close_position", "hl_range",
    "day_of_week", "month", "is_month_end",
    "fund_pe_trailing", "fund_pb", "fund_marketcap",
    "macd", "macdsignal", "macdhist", "mfi14",
    "ret_3d", "cumret_3d", "cumret_5d",
    "ret_zscore_20d", "atr_14", "vol_10d", "vol_60d", "vol_ratio_10_60",
    "vol_regime_high", "range_atr_ratio",
    "volume_zscore", "dollar_volume", "dollar_volume_20d_avg",
    "ret_vol_interaction",
    "rel_strength_1d", "rel_strength_3d", "rel_momentum_5d",
    "rsi_change_1d", "rsi_change_3d", "rsi_overbought", "rsi_oversold",
    "bb_upper", "bb_lower", "bb_mid", "bb_pctb",
    "ma_20", "price_minus_20dma", "ma_20_slope",
    "gap_ret_1d", "intraday_ret_1d", "body_to_range", "upper_wick_to_range", "lower_wick_to_range",
    "adx_14",
    "beta_60_spx", "corr_20_spx", "corr_60_spx",
    "vol_20d_std", "gbm_mu_60d",
    "gbm_sig_60d",
    "gbm_prob_up_1d",
    "gbm_exp_ret_1d",
    "gbm_p05_ret_1d",
    "gbm_p95_ret_1d",
    "gbm_prob_up_5d",
    "gbm_exp_ret_5d",
    "gbm_p05_ret_5d",
    "gbm_p95_ret_5d",
    "dist_from_high_20", "dist_from_low_20", "up_days_5", "down_days_5",
]


def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()

    close = hist["Close"]
    hist = add_gbm_features(hist, window=60, horizons=(1, 5))
    high = hist["High"]
    low = hist["Low"]
    open_ = hist["Open"] if "Open" in hist.columns else close
    volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(index=hist.index, data=np.nan)

    # NEW features requested...
    ret_1d_raw = close.pct_change()
    from scipy.stats import norm

# ---- GBM features (lagged; uses log returns) ----
    logret_1d = np.log(close).diff()

    mu_d_60 = logret_1d.rolling(60).mean().shift(1)
    sig_d_60 = logret_1d.rolling(60).std(ddof=1).shift(1)

    hist["gbm_mu_60d"] = mu_d_60
    hist["gbm_sig_60d"] = sig_d_60

    for h in [1, 5]:
        T = h / 252.0
        m = (mu_d_60 - 0.5 * sig_d_60**2) * T
        s = sig_d_60 * np.sqrt(T)

    hist[f"gbm_prob_up_{h}d"] = norm.cdf(m / (s + 1e-12))
    hist[f"gbm_exp_ret_{h}d"]  = np.exp(mu_d_60 * T) - 1.0
    hist[f"gbm_p05_ret_{h}d"]  = np.exp(m + s * norm.ppf(0.05)) - 1.0
    hist[f"gbm_p95_ret_{h}d"]  = np.exp(m + s * norm.ppf(0.95)) - 1.0


    hist["ret_1d"] = ret_1d_raw.shift(1)
    hist["ret_3d"] = close.pct_change(3).shift(1)
    hist["ret_5d"] = close.pct_change(5).shift(1)
    hist["ret_20d"] = close.pct_change(20).shift(1)
    hist["cumret_3d"] = (1 + ret_1d_raw).rolling(3).apply(lambda x: x.prod() - 1, raw=True).shift(1)
    hist["cumret_5d"] = (1 + ret_1d_raw).rolling(5).apply(lambda x: x.prod() - 1, raw=True).shift(1)

    # Volatility...
    hist["vol_10d"] = ret_1d_raw.rolling(10).std().shift(1)
    hist["vol_20d"] = ret_1d_raw.rolling(20).std().shift(1)
    hist["vol_60d"] = ret_1d_raw.rolling(60).std().shift(1)
    hist["vol_ratio_10_60"] = (hist["vol_10d"] / (hist["vol_60d"] + 1e-9)).shift(1)
    hist["vol_regime_high"] = (hist["vol_10d"] > hist["vol_20d"].rolling(60).quantile(0.75)).astype(int).shift(1)

    ret_mean_20d = ret_1d_raw.rolling(20).mean()
    ret_std_20d = ret_1d_raw.rolling(20).std()
    hist["ret_zscore_20d"] = ((ret_1d_raw - ret_mean_20d) / (ret_std_20d + 1e-9)).shift(1)
    hist["vol_20d_std"] = hist["vol_20d"].rolling(20).std().shift(1)

    # ATR...
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    hist["atr_14"] = true_range.rolling(14).mean().shift(1)
    hist["range_atr_ratio"] = ((high - low) / (hist["atr_14"] + 1e-9)).shift(1)

    # Volume...
    vol_mean_20d = volume.rolling(20).mean()
    vol_std_20d = volume.rolling(20).std()
    hist["volume_zscore"] = ((volume - vol_mean_20d) / (vol_std_20d + 1e-9)).shift(1)
    hist["dollar_volume"] = (close * volume).shift(1)
    hist["dollar_volume_20d_avg"] = hist["dollar_volume"].rolling(20).mean().shift(1)
    hist["ret_vol_interaction"] = (ret_1d_raw * hist["volume_zscore"]).shift(1)
    hist["volume_price_corr"] = ret_1d_raw.rolling(20).corr(volume.pct_change()).shift(1)

    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_30 = volume.rolling(30).mean()
    hist["volume_trend"] = (vol_ma_10 / (vol_ma_30 + 1e-9)).shift(1)
    hist["vol_ma_20"] = volume.rolling(20).mean().shift(1)
    hist["vol_spike_20"] = (volume / (hist["vol_ma_20"] + 1e-9)).shift(1)
    hist["vol_roll_mean_20"] = volume.rolling(20).mean().shift(1)
    hist["vol_roll_std_20"] = volume.rolling(20).std().shift(1)

    # Moving averages + BB...
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    sma_10 = close.rolling(10).mean()

    hist["ma_20"] = ma_20.shift(1)
    hist["sma_ratio_10_50"] = (sma_10 / (ma_50 + 1e-9)).shift(1)
    hist["price_to_ma50"] = (close / (ma_50 + 1e-9)).shift(1)
    hist["price_minus_20dma"] = ((close - ma_20) / (ma_20 + 1e-9)).shift(1)
    hist["ma_20_slope"] = ma_20.diff(5).shift(1)

    std_20 = close.rolling(20).std()
    bb_upper = ma_20 + 2 * std_20
    bb_lower = ma_20 - 2 * std_20

    hist["bb_upper"] = bb_upper.shift(1)
    hist["bb_lower"] = bb_lower.shift(1)
    hist["bb_mid"] = ma_20.shift(1)
    hist["bb_width_20"] = ((hist["bb_upper"] - hist["bb_lower"]) / (hist["bb_mid"] + 1e-9))
    hist["bb_pctb"] = ((close - bb_lower) / ((bb_upper - bb_lower) + 1e-9)).shift(1)

    # RSI enhancements...
    hist = add_rsi(hist, window=14, price_col="Close")
    hist["rsi_change_1d"] = hist["rsi14"].diff(1).shift(1)
    hist["rsi_change_3d"] = hist["rsi14"].diff(3).shift(1)
    hist["rsi_overbought"] = (hist["rsi14"] > 70).astype(int).shift(1)
    hist["rsi_oversold"] = (hist["rsi14"] < 30).astype(int).shift(1)
    hist["rsi14"] = hist["rsi14"].shift(1)

    # MACD/MFI...
    hist = add_technical_indicators(hist)
    hist["macd"] = hist["macd"].shift(1)
    hist["macdsignal"] = hist["macdsignal"].shift(1)
    hist["macdhist"] = hist["macdhist"].shift(1)
    hist["mfi14"] = hist["mfi14"].shift(1)

    # ---- SPX: fetch once (cached) and reuse for relative strength + beta/corr ----
    spx_close_aligned = None
    spx_ret_1d = None

    # Relative strength vs SPX (cached + tz-safe)
    try:
        spx_raw = _get_spx(hist.index[0], hist.index[-1], tz=getattr(hist.index, "tz", None))

        if spx_raw is None or spx_raw.empty or "Close" not in spx_raw.columns:
            hist["rel_strength_1d"] = 0.0
            hist["rel_strength_3d"] = 0.0
            hist["rel_momentum_5d"] = 0.0
        else:
            spx_close_aligned = spx_raw["Close"].reindex(hist.index, method="ffill").fillna(0.0)

            spx_ret_1d = spx_close_aligned.pct_change()
            spx_ret_3d = spx_close_aligned.pct_change(3)
            spx_ret_5d = spx_close_aligned.pct_change(5)

            stock_ret_1d = close.pct_change()
            stock_ret_3d = close.pct_change(3)
            stock_ret_5d = close.pct_change(5)

            hist["rel_strength_1d"] = (stock_ret_1d - spx_ret_1d).shift(1)
            hist["rel_strength_3d"] = (stock_ret_3d - spx_ret_3d).shift(1)
            hist["rel_momentum_5d"] = (stock_ret_5d - spx_ret_5d).shift(1)

    except Exception as e:
        print(f"[add_price_features] SPX fetch failed: {e}")
        hist["rel_strength_1d"] = 0.0
        hist["rel_strength_3d"] = 0.0
        hist["rel_momentum_5d"] = 0.0

    # Intraday structure (existing)...
    hist["high_low_ratio"] = (high / (low + 1e-9)).shift(1)
    hist["daily_range"] = ((high - low) / (close + 1e-9)).shift(1)
    hist["close_position"] = ((close - low) / ((high - low) + 1e-9)).shift(1)
    hist["hl_range"] = ((high - low) / (close.shift(1) + 1e-9)).shift(1)

    prev_close = close.shift(1)
    day_range = (high - low).replace(0, np.nan)
    hist["gap_ret_1d"] = ((open_ / (prev_close + 1e-9)) - 1.0).shift(1)
    hist["intraday_ret_1d"] = ((close / (open_ + 1e-9)) - 1.0).shift(1)

    body = (close - open_).abs()
    upper_wick = (high - np.maximum(close, open_)).clip(lower=0)
    lower_wick = (np.minimum(close, open_) - low).clip(lower=0)
    hist["body_to_range"] = (body / (day_range + 1e-9)).shift(1)
    hist["upper_wick_to_range"] = (upper_wick / (day_range + 1e-9)).shift(1)
    hist["lower_wick_to_range"] = (lower_wick / (day_range + 1e-9)).shift(1)

    # ADX14 (Wilder-style, lagged)...
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=hist.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / (atr + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=hist.index).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / (atr + 1e-9))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    hist["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)

    # Distance to recent high/low + streak counts (lagged)...
    roll_high_20 = close.rolling(20).max()
    roll_low_20 = close.rolling(20).min()
    hist["dist_from_high_20"] = ((close / (roll_high_20 + 1e-9)) - 1.0).shift(1)
    hist["dist_from_low_20"] = ((close / (roll_low_20 + 1e-9)) - 1.0).shift(1)

    up = (ret_1d_raw > 0).astype(int)
    down = (ret_1d_raw < 0).astype(int)
    hist["up_days_5"] = up.rolling(5).sum().shift(1)
    hist["down_days_5"] = down.rolling(5).sum().shift(1)

    # Rolling beta/corr to SPX (lagged) - reuse same SPX returns if available
    try:
        if spx_ret_1d is None:
            spx_raw2 = _get_spx(hist.index[0], hist.index[-1], tz=getattr(hist.index, "tz", None))
            if spx_raw2 is None or spx_raw2.empty or "Close" not in spx_raw2.columns:
                raise ValueError("SPX data unavailable for beta/corr")
            spx_close_aligned = spx_raw2["Close"].reindex(hist.index, method="ffill").fillna(0.0)
            spx_ret_1d = spx_close_aligned.pct_change()

        stock_r = ret_1d_raw
        spx_r = pd.Series(spx_ret_1d, index=hist.index)

        cov_60 = stock_r.rolling(60).cov(spx_r)
        var_60 = spx_r.rolling(60).var()

        hist["beta_60_spx"] = (cov_60 / (var_60 + 1e-9)).shift(1)
        hist["corr_20_spx"] = stock_r.rolling(20).corr(spx_r).shift(1)
        hist["corr_60_spx"] = stock_r.rolling(60).corr(spx_r).shift(1)

    except Exception as e:
        print(f"[add_price_features] beta/corr failed: {e}")
        hist["beta_60_spx"] = 0.0
        hist["corr_20_spx"] = 0.0
        hist["corr_60_spx"] = 0.0

    hist["day_of_week"] = hist.index.dayofweek
    hist["month"] = hist.index.month
    hist["is_month_end"] = (hist.index.day > 25).astype(int)

    return hist


def make_model(model_type: str = "rf", random_state: int = 42, task: str = "reg"):
    if task == "clf":
        if model_type == "xgb":
            return XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state,
                tree_method="hist", verbosity=0, subsample=0.8, colsample_bytree=0.7,
                min_child_weight=5, reg_lambda=1.0
            )
        return RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            random_state=random_state, n_jobs=-1
        )

    if model_type == "linreg":
        return LinearRegression()

    if model_type == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state
        )

    if model_type == "xgb":
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, random_state=random_state,
            tree_method="hist", verbosity=0, subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=1.0
        )

    return RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=50,
        random_state=random_state, n_jobs=-1
    )


def prune_weak_features(model, X, y, threshold=0.01):
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_, skipping pruning")
        return X

    importance = model.feature_importances_
    feature_names = X.columns
    important_features = feature_names[importance > threshold]
    print(f"Pruned {len(feature_names) - len(important_features)} weak features; kept {len(important_features)}")
    return X[important_features]


def select_features_elasticnet_timeseries(
    X: np.ndarray,
    y: np.ndarray,
    featurenames: list[str] | None = None,
    dates: pd.DatetimeIndex | None = None,
    horizon: int = 1,
    nsplits: int = 5,
    l1ratio: float = 0.5,
    minfeatures: int = 8,
    randomstate: int = 42,
    pctembargo: float = 0.01,
    # aliases (so any call style works)
    feature_names: list[str] | None = None,
    n_splits: int | None = None,
    l1_ratio: float | None = None,
    min_features: int | None = None,
    random_state: int | None = None,
    pct_embargo: float | None = None,
):
    if featurenames is None and feature_names is not None:
        featurenames = feature_names
    if n_splits is not None:
        nsplits = n_splits
    if l1_ratio is not None:
        l1ratio = l1_ratio
    if min_features is not None:
        minfeatures = min_features
    if random_state is not None:
        randomstate = random_state
    if pct_embargo is not None:
        pctembargo = pct_embargo

    if featurenames is None or dates is None:
        raise ValueError("featurenames(feature_names) and dates are required")

    if ElasticNetCV is None or Pipeline is None or StandardScaler is None:
        raise RuntimeError("scikit-learn not available for ElasticNetCV/StandardScaler/Pipeline")

    nsplits = int(max(3, nsplits))
    nsplits = int(min(nsplits, max(3, len(y) // 50)))

    Xdf = pd.DataFrame(X, index=pd.DatetimeIndex(dates), columns=featurenames)
    t1 = make_t1_from_horizon(Xdf.index, horizon)  # use YOUR old helper name in the file
    cv = PurgedKFold(nsplits=nsplits, t1=t1, pctembargo=pctembargo)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("enet", ElasticNetCV(
                l1_ratio=float(l1ratio),
                alphas=None,
                cv=cv.split(Xdf),
                n_jobs=-1,
                random_state=int(randomstate),
                max_iter=5000,
            )),
        ]
    )
    pipe.fit(Xdf.values, y)

    coefs = pipe.named_steps["enet"].coef_
    mask = np.abs(coefs) > 1e-10
    if mask.sum() < minfeatures:
        idx = np.argsort(np.abs(coefs))[::-1]
        keep = idx[:minfeatures]
        mask = np.zeros_like(mask, dtype=bool)
        mask[keep] = True

    selected_names = [featurenames[i] for i, m in enumerate(mask) if m]
    return X[:, mask], selected_names, mask



def get_fundamental_features(ticker: str) -> dict:
    feats = {
        "fund_pe_trailing": np.nan,
        "fund_pb": np.nan,
        "fund_marketcap": np.nan,
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
            if pd.isna(feats["fund_marketcap"]):
                feats["fund_marketcap"] = float(info.get("marketCap", np.nan))
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
    run_gaf: bool = False,
):
    fallback_periods = ["5y", "3y", "2y", "1y", "6mo", "3mo"]
    periods_to_try = [period] + [p for p in fallback_periods if p != period] if period in fallback_periods else [period] + fallback_periods

    last_error = None
    min_rows = 60

    for per in periods_to_try:
        try:
            hist = get_price_history(ticker, period=per, interval="1d")
            if hist is None or hist.empty:
                raise ValueError(f"No raw history for {ticker} with period={per}")

            hist = add_price_features(hist)
            missing = [c for c in (FEATURE_COLUMNS + MACRO_COLUMNS) if c not in hist.columns]
            print("Missing:", missing[:30])
            print("GBM cols present:", [c for c in hist.columns if c.startswith("gbm_")][:20])



            macro_df = get_macro_df(symbol="^GSPC", period=per)
            hist = hist.join(macro_df, how="left")
            for c in MACRO_COLUMNS:
                if c not in hist.columns:
                    hist[c] = np.nan
            hist[MACRO_COLUMNS] = hist[MACRO_COLUMNS].ffill().bfill()

            fund_feats = get_fundamental_features(ticker)
            for k, v in fund_feats.items():
                hist[k] = v

            raw_target = hist["Close"].pct_change(horizon).shift(-horizon)
            hist["ftarget_ret_horizon_ahead"] = (raw_target / (hist["vol_20d"] + 1e-9)) if use_vol_scaled_target else raw_target

            feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
            cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
            df = hist[cols_needed].dropna().copy()

            print("hist rows:", len(hist), "range:", hist.index.min(), "-", hist.index.max())
            print("df rows:", len(df), "range:", df.index.min(), "-", df.index.max())

            if df.empty or len(df) < min_rows:
                print(f"[build_features_and_target] WARNING: only {len(df)} usable rows for {ticker} with period={per}")
                last_error = ValueError(f"Only {len(df)} usable rows for {ticker} with period={per}")
                continue

            # -------- NEW: compute GAF-CNN prob on SAME usable rows --------
            prob_up_gaf = None
            if run_gaf:
                try:
                    closes_usable = hist.loc[df.index, "Close"].astype(float)
                    rets_usable = closes_usable.pct_change().dropna()
                    prob_up_gaf = predict_up_gafcnn_from_rets(rets_usable, window=30, image_size=30)
                except Exception as e:
                    print(f"[GAF-CNN] Failed on usable rows for {ticker}: {e}")
                    prob_up_gaf = None
            # --------------------------------------------------------------

            X = df[feat_cols].values
            y = df["ftarget_ret_horizon_ahead"].values

            last_row = df.iloc[-1]
            last_row_features = last_row[feat_cols].values
            last_close = hist.loc[df.index[-1], "Close"]
            last_vol_20d = last_row["vol_20d"]

            dates=df.index

            return X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates

        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"No usable history for {ticker} after trying periods={periods_to_try}. Last error={last_error}")

def build_panel_features_and_target(
    tickers,
    period: str = "5y",
    horizon: int = 3,
    usevolscaledtarget: bool = False,
) -> pd.DataFrame:
    """
    Build a cross-sectional panel of features and targets for multiple tickers.
    Returns a DataFrame indexed by date with columns:
    all FEATURECOLUMNS + MACROCOLUMNS + ["target", "ticker"].
    """
    dfs = []
    for tk in tickers:
        try:
            X, y, _, _, _, _, dates = build_features_and_target(...)
        except Exception as e:
            print(f"build_panel_features_and_target: skipping {tk} error={e}")
            continue

        if X is None or y is None or dates is None:
            continue

        featcols = FEATURE_COLUMNS + MACRO_COLUMNS
        if X.shape[1] != len(featcols):
            print(f"{tk}: X.shape[1]={X.shape[1]} != len(featcols)={len(featcols)}")
            continue

        df = pd.DataFrame(X, index=pd.DatetimeIndex(dates), columns=featcols)
        df["target"] = y
        df["ticker"] = tk
        dfs.append(df)

    if not dfs:
        raise ValueError("No usable panel data for any ticker.")

    panel = pd.concat(dfs).sort_index()
    return panel


def build_features_and_direction_target(ticker="^GSPC", period="5y", horizon=1):
    X, yreg, last_feats, last_close, last_vol_20d = build_features_and_target(
        ticker=ticker, period=period, horizon=horizon, use_vol_scaled_target=False,
    )
    ydir = (yreg > 0).astype(int)
    return X, ydir, last_feats, last_close, last_vol_20d


def train_model(X, y, model_type="rf", test_size=0.2, random_state=42, task="reg"):
    n = len(X)
    split_idx = int(n * (1 - test_size))
    Xtrain, Xtest = X[:split_idx], X[split_idx:]
    ytrain, ytest = y[:split_idx], y[split_idx:]

    model = make_model(model_type=model_type, random_state=random_state, task=task)
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)

    if task == "reg":
        return model, r2_score(ytest, ypred), ypred, np.sqrt(mean_squared_error(ytest, ypred))
    return model, accuracy_score(ytest, ypred), ypred, None


def predict_next_for_ticker(
    ticker="^GSPC",
    period="5y",
    model_type="rf",
    modeltype=None,
    horizon=1,
    use_vol_scaled_target: bool = False,
    auto_optimize: bool = True,
    run_gaf: bool = False,
):
    if modeltype is not None:
        model_type = modeltype
    X, y, x_last, last_close, last_vol_20d, prob_up_gaf, dates = build_features_and_target(
        ticker=ticker, period=period, horizon=horizon, use_vol_scaled_target=use_vol_scaled_target, run_gaf=run_gaf,
    )
    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS

    n = len(X)

    if USE_ELASTICNET_SELECT:
        try:
            train_end_for_en = int(n * 0.8)
            X_en_train = X[:train_end_for_en]
            y_en_train = y[:train_end_for_en]

            X_en_train_sel, en_selected_names, en_mask = select_features_elasticnet_timeseries(
                X=X_en_train,
                y=y_en_train,
                feature_names=list(feat_cols),
                dates=dates[:train_end_for_en],
                horizon=horizon,
                n_splits=ELASTICNET_CV_FOLDS,
                l1_ratio=ELASTICNET_L1_RATIO,
                min_features=10,
            )
            X = X[:, en_mask]
            x_last = x_last[en_mask]
            feat_cols = en_selected_names
            print(f"{ticker} ElasticNet selected {len(feat_cols)} features")
        except Exception as e:
            print(f"{ticker} ElasticNet selection failed; continuing without it. Error: {e}")

    if auto_optimize:
        train_end = int(n * 0.8)
        Xtrain = X[:train_end]
        ytrain = y[:train_end]
        model_init = make_model(model_type=model_type, random_state=42, task="reg")
        model_init.fit(Xtrain, ytrain)

        if hasattr(model_init, "feature_importances_"):
            importance = model_init.feature_importances_
            important_mask = importance > 0.001
        else:
            important_mask = np.ones(X.shape[1], dtype=bool)

        important_features = [feat_cols[i] for i in range(len(feat_cols)) if important_mask[i]]
        print(f"{ticker} Using {len(important_features)}/{len(feat_cols)} features for prediction")

        Xtrain_full = X[:train_end][:, important_mask]
        ytrain_full = y[:train_end]
        x_last_pruned = x_last[important_mask]
        feat_cols = important_features
    else:
        split_idx = int(n * 0.8)
        Xtrain_full = X[:split_idx]
        ytrain_full = y[:split_idx]
        x_last_pruned = x_last

    model = make_model(model_type=model_type, random_state=42, task="reg")
    model.fit(Xtrain_full, ytrain_full)

    pred_ret = float(model.predict(x_last_pruned.reshape(1, -1))[0])
    if use_vol_scaled_target:
        pred_ret = pred_ret * float(last_vol_20d)

    pred_price = float(last_close * (1 + pred_ret))

    prob_up = None
    prob_down = None
    try:
        ydir = (y > 0).astype(int)
        ydir_train = ydir[:len(Xtrain_full)]
        clf = make_model(model_type=model_type, random_state=42, task="clf")
        clf.fit(Xtrain_full, ydir_train)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x_last_pruned.reshape(1, -1))[0]
            if hasattr(clf, "classes_") and 1 in clf.classes_:
                idx_up = list(clf.classes_).index(1)
                prob_up = float(proba[idx_up])
                prob_down = float(1.0 - prob_up)
            else:
                prob_up = float(proba.max())
                prob_down = float(1.0 - prob_up)
        else:
            pred_dir = int(clf.predict(x_last_pruned.reshape(1, -1))[0])
            prob_up = 1.0 if pred_dir == 1 else 0.0
            prob_down = 1.0 - prob_up
    except Exception:
        prob_up = None
        prob_down = None

    fund_feats = get_fundamental_features(ticker)
    pe_ratio = fund_feats.get("fund_pe_trailing", None)

    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(feat_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top_features_str = ", ".join([f"{feat}:{imp:.3f}" for feat, imp in top_features])
    else:
        top_features_str = "NA"

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
        "elasticnet_enabled": bool(USE_ELASTICNET_SELECT),
        "elasticnet_l1_ratio": float(ELASTICNET_L1_RATIO),
        "elasticnet_cv_folds": int(ELASTICNET_CV_FOLDS),
        "elasticnet_selected_n": int(len(feat_cols)) if USE_ELASTICNET_SELECT else None,
    }



def track_predictions(ticker, period="1y", model_type="rf", horizon=1):
    try:
        hist = get_price_history(ticker, period=period, interval="1d")
        if hist.empty or len(hist) < 50:
            print(f"[track_predictions] Insufficient data for {ticker}; only {len(hist)} rows")
            return pd.DataFrame(), 0.0

        hist = add_price_features(hist)

        macro_df = get_macro_df(symbol="^GSPC", period=period)
        hist = hist.join(macro_df, how="left")

        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v

        hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

        feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
        cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
        df = hist[cols_needed].dropna().copy()

        print(f"[track_predictions] After dropna for {ticker}: {len(df)} rows")
        if len(df) < 50:
            print("[track_predictions] Not enough data after feature engineering")
            return pd.DataFrame(), 0.0

        nrows = len(df)
        min_test = 60
        max_test = 252
        proposed_test = int(nrows * 0.2)
        test_size = max(min_test, proposed_test)
        test_size = min(test_size, max_test, nrows - 1)

        if test_size < 5:
            print("[track_predictions] Test size too small:", test_size)
            return pd.DataFrame(), 0.0

        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        Xtrain = train_df[feat_cols].values
        ytrain = train_df["ftarget_ret_horizon_ahead"].values

        selected_mask = None
        if USE_ELASTICNET_SELECT:
            try:
                Xtrain_sel, selected_names, selected_mask = select_features_elasticnet_timeseries(
                    X=Xtrain,
                    y=ytrain,
                    feature_names=list(feat_cols),
                    dates=train_df.index,
                    horizon=horizon,
                    n_splits=ELASTICNET_CV_FOLDS,
                    l1_ratio=ELASTICNET_L1_RATIO,
                    min_features=10,
                )
                Xtrain = Xtrain_sel
                feat_cols = selected_names
                print(f"{ticker} track_predictions ElasticNet selected {len(feat_cols)} features")
            except Exception as e:
                print(f"{ticker} track_predictions ElasticNet failed; skipping. Error: {e}")

        model = make_model(model_type=model_type, random_state=42)
        model.fit(Xtrain, ytrain)

        Xtest = test_df[FEATURE_COLUMNS + MACRO_COLUMNS].values
        ytest = test_df["ftarget_ret_horizon_ahead"].values
        if selected_mask is not None:
            Xtest = Xtest[:, selected_mask]

        ypred = model.predict(Xtest)

        results = pd.DataFrame({
            "date": test_df.index,
            "actual_close": hist.loc[test_df.index, "Close"],
            "predicted_return": ypred,
            "actual_return": ytest,
            "pred_direction": np.sign(ypred),
            "actual_direction": np.sign(ytest),
            "correct_direction": (np.sign(ypred) == np.sign(ytest)),
        })
        results["predicted_price"] = results["actual_close"] * (1 + results["predicted_return"])

        from scipy.stats import norm

        T = horizon / 252.0

        mu = hist.loc[test_df.index, "gbm_mu_60d"].astype(float)     # daily log-return drift
        sig = hist.loc[test_df.index, "gbm_sig_60d"].astype(float)   # daily log-return vol
        S0 = results["actual_close"].astype(float)

        m = (mu - 0.5 * sig**2) * T
        s = sig * np.sqrt(T)

        results["gbm_med_price"] = S0 * np.exp(m)
        results["gbm_p05_price"] = S0 * np.exp(m + s * norm.ppf(0.05))
        results["gbm_p95_price"] = S0 * np.exp(m + s * norm.ppf(0.95))

        accuracy = results["correct_direction"].mean()
        print(f"[track_predictions] Success: {len(results)} test preds; direction accuracy={100*accuracy:.1f}%")
        return results, accuracy

    except Exception as e:
        print(f"[track_predictions] Error for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), 0.0


def backtest_one_ticker(ticker="AAPL", period="10y", test_years=1, threshold=0.002, model_type="rf", horizon=1):
    hist = get_price_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)

    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")

    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v

    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
    df = hist[cols_needed].dropna().copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index < cutoff_date
    test_mask = df.index >= cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    Xtrain = train_df[feat_cols].values
    ytrain = train_df["ftarget_ret_horizon_ahead"].values
    Xtest = test_df[FEATURE_COLUMNS + MACRO_COLUMNS].values
    ytest = test_df["ftarget_ret_horizon_ahead"].values

    selected_mask = None
    selected_feats = None
    if USE_ELASTICNET_SELECT:
        try:
            Xtrain_sel, selected_feats, selected_mask = select_features_elasticnet_timeseries(
                X=Xtrain,
                y=ytrain,
                feature_names=list(feat_cols),
                dates=train_df.index,
                horizon=horizon,
                n_splits=ELASTICNET_CV_FOLDS,
                l1_ratio=ELASTICNET_L1_RATIO,
                min_features=10,
            )
            Xtrain = Xtrain_sel
            feat_cols = selected_feats
            print(f"{ticker} backtest_one_ticker ElasticNet selected {len(feat_cols)} features")
        except Exception as e:
            print(f"{ticker} backtest_one_ticker ElasticNet failed; skipping. Error: {e}")

    if selected_mask is not None:
        Xtest = Xtest[:, selected_mask]

    model = make_model(model_type=model_type, random_state=42)
    model.fit(Xtrain, ytrain)

    ypred = model.predict(Xtest)

    positions = np.where(ypred > threshold, 1, np.where(ypred < -threshold, -1, 0))
    cost_per_trade = 0.0005

    pnl = []
    prev_pos = 0
    for pos, ret in zip(positions, ytest):
        trade = abs(pos - prev_pos)
        pnl.append(pos * ret - cost_per_trade * trade)
        prev_pos = pos

    pnl = np.array(pnl)
    cumret = (1 + pnl).prod() - 1
    hitrate = (np.sign(ypred) == np.sign(ytest)).mean()
    avg_daily = pnl.mean()
    std_daily = pnl.std(ddof=1)
    sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily != 0 else 0.0

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "test_days": len(pnl),
        "total_return": cumret,
        "hitrate": hitrate,
        "sharpe": sharpe,
        "elasticnet_enabled": bool(USE_ELASTICNET_SELECT),
        "elasticnet_selected_features": selected_feats,
        "elasticnet_selected_n": len(selected_feats) if selected_feats is not None else None,
    }


def backtest_one_ticker_auto_optimized(
    ticker="AAPL",
    period="10y",
    test_years=2,
    threshold=0.002,
    model_type="rf",
    horizon=5,
    importance_threshold=0.001,
):
    hist = get_price_history(ticker, period=period, interval="1d")
    hist = add_price_features(hist)

    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")

    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v

    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
    df = hist[cols_needed].dropna().copy()

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    Xtrain = train_df[feat_cols].values
    ytrain = train_df["ftarget_ret_horizon_ahead"].values

    selected_mask = None
    if USE_ELASTICNET_SELECT:
        try:
            Xtrain_sel, selected_feats, selected_mask = select_features_elasticnet_timeseries(
                X=Xtrain,
                y=ytrain,
                feature_names=list(feat_cols),
                dates=train_df.index,
                horizon=horizon,
                n_splits=ELASTICNET_CV_FOLDS,
                l1_ratio=ELASTICNET_L1_RATIO,
                min_features=10,
            )
            Xtrain = Xtrain_sel
            feat_cols = selected_feats
            print(f"{ticker} auto_optimized ElasticNet selected {len(feat_cols)} features")
        except Exception as e:
            print(f"{ticker} auto_optimized ElasticNet failed; skipping. Error: {e}")
            selected_mask = None

    model_init = make_model(model_type=model_type, random_state=42)
    model_init.fit(Xtrain, ytrain)

    if hasattr(model_init, "feature_importances_"):
        importance = model_init.feature_importances_
        important_mask = importance > importance_threshold
    else:
        important_mask = np.ones(len(feat_cols), dtype=bool)

    important_features = [feat_cols[i] for i in range(len(feat_cols)) if important_mask[i]]
    dropped_count = len(feat_cols) - len(important_features)
    print(f"{ticker} Kept {len(important_features)}/{len(feat_cols)} features; dropped {dropped_count} weak")

    train_val_df = df.iloc[:val_end]
    Xtrain_val_full = train_val_df[FEATURE_COLUMNS + MACRO_COLUMNS].values
    ytrain_val = train_val_df["ftarget_ret_horizon_ahead"].values

    if selected_mask is not None:
        Xtrain_val_full = Xtrain_val_full[:, selected_mask]

    feat_to_idx = {f: i for i, f in enumerate(feat_cols)}
    imp_idx = [feat_to_idx[f] for f in important_features if f in feat_to_idx]

    Xtrain_val = Xtrain_val_full[:, imp_idx]

    model_final = make_model(model_type=model_type, random_state=42)
    model_final.fit(Xtrain_val, ytrain_val)

    Xtest_full = test_df[FEATURE_COLUMNS + MACRO_COLUMNS].values
    ytest = test_df["ftarget_ret_horizon_ahead"].values
    if selected_mask is not None:
        Xtest_full = Xtest_full[:, selected_mask]
    Xtest = Xtest_full[:, imp_idx]

    ypred = model_final.predict(Xtest)

    positions = np.where(ypred > threshold, 1, np.where(ypred < -threshold, -1, 0))
    cost_per_trade = 0.0005

    pnl = []
    prev_pos = 0
    for pos, ret in zip(positions, ytest):
        trade = abs(pos - prev_pos)
        pnl.append(pos * ret - cost_per_trade * trade)
        prev_pos = pos

    pnl = np.array(pnl)
    cumret = (1 + pnl).prod() - 1
    hitrate = (np.sign(ypred) == np.sign(ytest)).mean()
    avg_daily = pnl.mean()
    std_daily = pnl.std(ddof=1)
    sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily != 0 else 0.0

    return {
        "ticker": ticker,
        "model_type": model_type,
        "horizon": horizon,
        "num_features_original": len(FEATURE_COLUMNS) + len(MACRO_COLUMNS),
        "num_features_used": len(important_features),
        "features_dropped": dropped_count,
        "test_days": len(pnl),
        "total_return": cumret,
        "hitrate": hitrate,
        "sharpe": sharpe,
        "elasticnet_enabled": bool(USE_ELASTICNET_SELECT),
        "elasticnet_selected_n": int(selected_mask.sum()) if selected_mask is not None else None,
    }


def analyze_feature_significance(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    use_vol_scaled_target: bool = False,
    alpha: float = 0.05,
):
    X, y, *_ = build_features_and_target(
        ticker=ticker, period=period, horizon=horizon, use_vol_scaled_target=use_vol_scaled_target
    )
    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS

    Xdf = pd.DataFrame(X, columns=feat_cols)
    Xdf = sm.add_constant(Xdf)

    ols_model = sm.OLS(y, Xdf).fit()

    rows = []
    ordered_names = ["const"] + feat_cols
    for name in ordered_names:
        if name in ols_model.params.index:
            pval = float(ols_model.pvalues[name])
            rows.append({
                "feature": name,
                "coef": float(ols_model.params[name]),
                "pvalue": pval,
                "significant": bool(pval < alpha),
            })

    sig_df = pd.DataFrame(rows).sort_values("pvalue")
    return ols_model, sig_df


def make_gaf_image_from_returns(returns: pd.Series, window: int = 60, image_size: int = 30):
    r = returns.dropna().values
    if len(r) < window:
        return None, None

    window_vals = r[-window:]
    X = window_vals.reshape(1, -1)

    gaf = GramianAngularField(image_size=image_size, method="summation")
    Xgaf = gaf.fit_transform(X)
    img = Xgaf[0]

    fig, ax = plt.subplots(figsize=(2, 2))
    cax = ax.imshow(img, cmap="rainbow", origin="lower", aspect="equal")
    ax.set_title("GAF (last window returns)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def predict_up_gafcnn(ticker: str, window: int = 30, image_size: int = 30, period: str = "3y") -> float | None:
    if gafcnn is None:
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
    Xgaf = gaf.fit_transform(X)
    Xinput = Xgaf[..., np.newaxis]

    try:
        proba = gafcnn.predict(Xinput, verbose=0)[0][0]
        return float(proba)
    except Exception as e:
        print(f"[GAF-CNN] Error during predict for {ticker}: {e}")
        return None
    
def predict_up_gafcnn_from_rets(
    rets: pd.Series,
    window: int = 30,
    image_size: int = 30
) -> float | None:
    if gafcnn is None:
        return None

    r = rets.dropna().astype(float)
    if len(r) < window:
        return None

    window_vals = r.values[-window:]
    X = window_vals.reshape(1, -1)

    gaf = GramianAngularField(image_size=image_size, method="summation")
    Xgaf = gaf.fit_transform(X)
    Xinput = Xgaf[..., np.newaxis]

    try:
        proba = gafcnn.predict(Xinput, verbose=0)[0][0]
        return float(proba)
    except Exception as e:
        print(f"[GAF-CNN] Error during predict (from_rets): {e}")
        return None


def tune_xgb_hyperparams(X, y, random_state=42):
    tscv = TimeSeriesSplit(n_splits=3)
    base_model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=random_state, verbosity=0)

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


def walk_forward_backtest(
    ticker="AAPL",
    period="10y",
    horizon=1,
    model_type="rf",
    train_years=4,
    test_years=1,
    threshold=0.002,
    cost_per_trade=0.0005,
    step_days: int | None = None,
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

    target_col = "ftarget_ret_horizon_ahead"
    hist[target_col] = hist["Close"].pct_change(horizon).shift(-horizon)

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    cols_needed = feat_cols + [target_col]
    df = hist[cols_needed].dropna().copy()
    if df.empty:
        return []

    fold_metrics = []

    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    if step_days is None:
        step_days = test_days

    start = 0
    while True:
        train_start = start
        train_end = train_start + train_days

        test_start = train_end
        test_end = test_start + test_days

        # stop when you can't form a full fold
        if test_end > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df  = df.iloc[test_start:test_end]

        # don't kill the entire backtest if one fold is too small
        if len(train_df) < 50 or len(test_df) < 20:
            start += step_days
            continue

        Xtrain = train_df[feat_cols].values
        ytrain = train_df[target_col].values
        Xtest  = test_df[feat_cols].values
        ytest  = test_df[target_col].values

        # ... your existing: fit model, predict, simulate trades, compute metrics ...
        # fold_metrics.append({...})

        if USE_ELASTICNET_SELECT:
            try:
                Xtrain_sel, fold_feats, fold_mask = select_features_elasticnet_timeseries(
                    X=Xtrain,
                    y=ytrain,
                    feature_names=list(feat_cols),
                    dates=train_df.index,
                    horizon=horizon,
                    n_splits=ELASTICNET_CV_FOLDS,
                    l1_ratio=ELASTICNET_L1_RATIO,
                    min_features=10,
                )
                Xtrain = Xtrain_sel
                Xtest = Xtest[:, fold_mask]
                print(f"{ticker} WF fold ElasticNet selected {len(fold_feats)} features")
            except Exception as e:
                print(f"{ticker} WF fold ElasticNet failed; skipping. Error: {e}")

        model = make_model(model_type=model_type, random_state=42)
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)

        positions = np.where(ypred > threshold, 1, np.where(ypred < -threshold, -1, 0))

        pnl = []
        prev_pos = 0
        for pos, ret in zip(positions, ytest):
            trade = abs(pos - prev_pos)
            pnl.append(pos * ret - cost_per_trade * trade)
            prev_pos = pos

        pnl = np.array(pnl)
        hitrate = (np.sign(ypred) == np.sign(ytest)).mean()
        avg_daily = pnl.mean()
        std_daily = pnl.std(ddof=1)
        sharpe = np.sqrt(252) * avg_daily / std_daily if std_daily != 0 else 0.0

        num_trades = int(np.count_nonzero(np.diff(np.concatenate([[0], (positions != 0).astype(int)]))))

        fold_metrics.append({
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "test_days": len(pnl),
            "hitrate": hitrate,
            "sharpe": sharpe,
            "num_trades": num_trades,
        })

        start += step_days

    return fold_metrics

def walkforward_cross_sectional(
    tickers,
    period: str = "5y",
    horizon: int = 3,
    modeltype: str = "rf",
    trainyears: int = 3,
    testyears: int = 1,
    top_pct: float = 0.1,
    bottom_pct: float | None = None,
) -> list[dict]:
    """
    Walk-forward validation on a cross-sectional portfolio built from model predictions.
    Returns list of dicts with per-fold metrics.
    """
    panel = build_panel_features_and_target(
        tickers=tickers,
        period=period,
        horizon=horizon,
        usevolscaledtarget=False,
    )

    featcols = FEATURE_COLUMNS + MACRO_COLUMNS
    df = panel.dropna(subset=featcols + ["target"]).copy()
    df = df.sort_index()

    if df.empty:
        raise ValueError("walkforward_cross_sectional: no usable rows after dropna.")

    # convert to deterministic row index for simple slicing
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    traindays = 252 * trainyears
    testdays = 252 * testyears

    n = len(df)
    foldmetrics = []
    start = 0

    while True:
        train_start = start
        train_end = train_start + traindays
        test_start = train_end
        test_end = test_start + testdays

        if test_end >= n:
            break

        traindf = df.iloc[train_start:train_end]
        testdf = df.iloc[test_start:test_end]

        if len(traindf) < 500 or len(testdf) < 100:
            start += testdays
            continue

        Xtrain = traindf[featcols].values
        ytrain = traindf["target"].values
        Xtest = testdf[featcols].values
        ytest = testdf["target"].values

        model = make_model(modeltype, randomstate=42, task="reg")
        model.fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)

        dates = pd.DatetimeIndex(testdf["date"].values)
        tickers_test = testdf["ticker"].values

        W = build_cross_sectional_portfolio(
            dates=dates,
            tickers=tickers_test,
            preds=ypred,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
        )

        # build matrix of realized future returns aligned to dates,tickers
        mt = pd.MultiIndex.from_arrays(
            [dates, tickers_test], names=["date", "ticker"]
        )
        rets_series = pd.Series(ytest, index=mt)
        rets_df = (
            rets_series.unstack("ticker")
            .reindex(W.index)
            .reindex(columns=W.columns)
            .fillna(0.0)
        )

        port_rets = (W * rets_df).sum(axis=1)
        sharpe = sharpe_from_returns(port_rets)
        hitrate = float(
            np.mean(np.sign(ypred) * np.sign(ytest) > 0)
        ) if len(ypred) else None

        foldmetrics.append(
            {
                "train_start": traindf["date"].iloc[0],
                "train_end": traindf["date"].iloc[-1],
                "test_start": testdf["date"].iloc[0],
                "test_end": testdf["date"].iloc[-1],
                "sharpe": float(sharpe) if sharpe is not None else None,
                "hitrate": hitrate,
                "testdays": int(len(port_rets)),
                "numtrades": int((W.abs() > 0).sum().sum()),
            }
        )

        start += testdays

    return foldmetrics


def backtest_compare_one_ticker(ticker="AAPL", period="10y", test_years=1, threshold=0.002, horizon=1, auto_optimize=True):
    if auto_optimize:
        rf_res = backtest_one_ticker_auto_optimized(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="rf", horizon=horizon
        )
        gbrt_res = backtest_one_ticker_auto_optimized(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="gbrt", horizon=horizon
        )
        xgb_res = backtest_one_ticker_auto_optimized(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="xgb", horizon=horizon
        )
    else:
        rf_res = backtest_one_ticker(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="rf", horizon=horizon
        )
        gbrt_res = backtest_one_ticker(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="gbrt", horizon=horizon
        )
        xgb_res = backtest_one_ticker(
            ticker=ticker, period=period, test_years=test_years, threshold=threshold, model_type="xgb", horizon=horizon
        )

    return {"rf": rf_res, "gbrt": gbrt_res, "xgb": xgb_res}

predictnextforticker = predict_next_for_ticker

if __name__ == "__main__":
    print("=" * 60)
    print("Testing 1-Day Predictions - All Models")
    print("=" * 60)

    X, y, _, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=1)

    best_xgb = tune_xgb_hyperparams(X, y)

    rf_model, rf_r2, _, rf_rmse = train_model(X, y, model_type="rf")
    print("Random Forest 1-day")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS) + len(MACRO_COLUMNS)}")
    print(f"  Test R2: {rf_r2:.4f}")
    print(f"  Test RMSE: {rf_rmse:.6f}")

    gbrt_model, gbrt_r2, _, gbrt_rmse = train_model(X, y, model_type="gbrt")
    print("Gradient Boosting 1-day")
    print(f"  Test R2: {gbrt_r2:.4f}")
    print(f"  Test RMSE: {gbrt_rmse:.6f}")

    xgb_model, xgb_r2, _, xgb_rmse = train_model(X, y, model_type="xgb")
    print("XGB 1-day")
    print(f"  Test R2: {xgb_r2:.4f}")
    print(f"  Test RMSE: {xgb_rmse:.6f}")

    print("=" * 60)
    print("Testing 2-Day Predictions")
    print("=" * 60)

    X2, y2, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=2)
    rf_model2, rf_r2_2d, _, rf_rmse_2d = train_model(X2, y2, model_type="rf")
    print("Random Forest 2-day")
    print(f"  Test R2: {rf_r2_2d:.4f}")
    print(f"  Test RMSE: {rf_rmse_2d:.6f}")

    print("=" * 60)
    print("Testing 3-Day Predictions")
    print("=" * 60)

    X3, y3, _, _, _ = build_features_and_target("^GSPC", period="10y", horizon=3)
    rf_model3, rf_r2_3d, _, rf_rmse_3d = train_model(X3, y3, model_type="rf")
    print("Random Forest 3-day")
    print(f"  Test R2: {rf_r2_3d:.4f}")
    print(f"  Test RMSE: {rf_rmse_3d:.6f}")
