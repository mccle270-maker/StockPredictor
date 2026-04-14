import os, datetime as dt, requests
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
# ---- SPX cache (avoid repeated yf.download("^GSPC") per ticker) ----
_SPX_CACHE = {}
BASE_DIR = Path(__file__).resolve().parent
MACRO_CACHE_DIR = BASE_DIR / ".cache" / "macro"
MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _spx_days_to_period(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    """Map a date span to an approximate yf period string with a small buffer."""
    days = max(1, (end_ts - start_ts).days + 5)
    if days >= 3650:
        return "10y"
    if days >= 1825:
        return "5y"
    if days >= 1095:
        return "3y"
    if days >= 730:
        return "2y"
    if days >= 365:
        return "1y"
    if days >= 180:
        return "6mo"
    return "3mo"


def _get_spx(start, end, tz=None):
    """
    Fetch SPX once (cached) for a given date range + timezone-ness, and normalize columns/index.

    Robust fallback chain:
    1) ^GSPC (primary)
    2) ^SPX / SPX proxies
    3) SPY / VOO ETFs as market proxy
    Uses existing get_price_history fallbacks (Stooq, raw Yahoo) to survive 429s.
    """

    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)

    # Cache key includes tz so tz-aware and tz-naive don't clash
    key = (start_ts, end_ts, str(tz))
    if key in _SPX_CACHE:
        return _SPX_CACHE[key]

    period = _spx_days_to_period(start_ts, end_ts)
    candidates = ["^GSPC", "^SPX", "SPX", "SPY", "VOO"]

    spx = pd.DataFrame()
    for sym in candidates:
        try:
            df = get_price_history(sym, period=period, interval="1d")
            if df is None or df.empty:
                continue

            # Normalize columns if yfinance returned multi-index (handled in get_price_history but keep safe)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Restrict to requested window (handle both tz-aware and tz-naive)
            idx_naive = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df = df.loc[(idx_naive >= start_ts) & (idx_naive <= end_ts)]
            if df.empty:
                continue

            spx = df.copy()
            break
        except Exception as e:
            print(f"[_get_spx] {sym} failed: {e}")
            continue

    # If nothing worked, return empty DataFrame (caller will handle)
    if spx is None or spx.empty:
        print("[_get_spx] Warning: all SPX proxies failed; returning empty DataFrame")
        spx = pd.DataFrame()
    else:
        # Make index match caller tz
        idx = pd.DatetimeIndex(spx.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        if tz is not None:
            spx.index = idx.tz_localize(tz)
        else:
            spx.index = idx

    _SPX_CACHE[key] = spx
    return spx


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from xgboost import XGBRegressor, XGBClassifier

# Import model improvements
try:
    from model_improvements import (
        add_enhanced_features,
        optimize_threshold_for_fold,
        apply_volatility_weighting,
        DirectionClassifier,
        apply_position_holding,
        ModelEnsemble,
        apply_kelly_sizing,
        apply_all_improvements,
    )
    HAS_IMPROVEMENTS = True
except ImportError:
    print("[prediction_model] Warning: model_improvements module not found, enhancements disabled")
    HAS_IMPROVEMENTS = False

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

# Long-horizon analog/regime layer (side-car)
try:
    from long_horizon import predict_long_horizon, LongHorizonResult
    HAS_LONG_HORIZON = True
except Exception as e:
    print(f"[prediction_model] Warning: long_horizon module unavailable: {e}")
    HAS_LONG_HORIZON = False

def vol_target_position_size(signal, vol_20d, target_vol=0.15):
    """Vectorized volatility targeting - FIXED."""
    import numpy as np
    
    # Handle scalar inputs
    if isinstance(vol_20d, (int, float)):
        if vol_20d == 0 or pd.isna(vol_20d): 
            return signal
        return signal * (target_vol / vol_20d)
    
    # Vectorized Series/DataFrame (FIXED)
    if pd.isna(vol_20d).all():
        return pd.Series(np.ones_like(vol_20d), index=vol_20d.index) * signal
    
    # Safe division: replace 0/NaN with 1.0
    safe_vol = vol_20d.copy()
    safe_vol = safe_vol.replace([0, np.nan], 1.0)
    weights = target_vol / safe_vol
    
    # Broadcast signal if scalar
    if isinstance(signal, (int, float)):
        return signal * weights
    return signal * weights

    
    # Handle Series inputs (vectorized)
    weights = pd.Series(np.ones_like(vol_20d), index=vol_20d.index)
    valid_mask = (vol_20d != 0) & (~pd.isna(vol_20d))
    weights[valid_mask] = target_vol / vol_20d[valid_mask]
    
    if hasattr(signal, '__len__'):
        return signal * weights
    return signal * weights.iloc[0]


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def is_elasticnet_enabled() -> bool:
    """Check if ElasticNet feature selection is enabled - called at RUNTIME, not import time."""
    return env_bool("USE_ELASTICNET_SELECT", False)  # Default to OFF when not set


# Keep module-level for backward compat, but prefer is_elasticnet_enabled() for runtime checks
USE_ELASTICNET_SELECT = env_bool("USE_ELASTICNET_SELECT", False)  # Changed default to False
try:
    ELASTICNET_L1_RATIO = float(os.environ.get("ELASTICNET_L1_RATIO", 0.5))
    ELASTICNET_MINFEATURES = int(os.environ.get("ELASTICNET_MINFEATURES", 12))
except Exception:
    ELASTICNET_L1_RATIO = 0.5
    ELASTICNET_MINFEATURES = 12

try:
    ELASTICNET_CV_FOLDS = int(os.environ.get("ELASTICNET_CV_FOLDS", 5))
except Exception:
    ELASTICNET_CV_FOLDS = 5

USE_OLSSIGSELECT = env_bool("USE_OLSSIGSELECT", False)

try:
    OLSSIG_ALPHA = float(os.environ.get("OLSSIG_ALPHA", "0.05"))
except Exception:
    OLSSIG_ALPHA = 0.05

# --- XGBoost Feature Selection Config ---
USE_XGB_FEATURE_SELECTION = env_bool("USE_XGB_FEATURE_SELECTION", False)
try:
    XGB_TOP_FEATURES = int(os.environ.get("XGB_TOP_FEATURES", 30))
except Exception:
    XGB_TOP_FEATURES = 30

try:
    OLSSIG_TOPK = int(os.environ.get("OLSSIG_TOPK", "50"))  # 0 => no cap
except Exception:
    OLSSIG_TOPK = 50

try:
    OLSSIG_MINFEATURES = int(os.environ.get("OLSSIG_MINFEATURES", "10"))
except Exception:
    OLSSIG_MINFEATURES = 10


def get_heston_params_for_ticker(ticker: str, hist: pd.DataFrame = None) -> HestonParams | None:
    """
    Get Heston model parameters for a ticker.
    
    First checks hardcoded calibrated params, then estimates from historical data.
    This allows Heston pricing for ANY ticker, not just AAPL/NVDA.
    
    Args:
        ticker: Stock ticker symbol
        hist: Optional historical price DataFrame with 'Close' column
              If not provided, will attempt to fetch
    
    Returns:
        HestonParams or None if estimation fails
    """
    # Hardcoded calibrated parameters for popular tickers
    calibrated_params = {
        "AAPL": HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.6),
        "NVDA": HestonParams(v0=0.06, theta=0.05, kappa=1.2, sigma=0.5, rho=-0.7),
        "MSFT": HestonParams(v0=0.035, theta=0.035, kappa=1.8, sigma=0.25, rho=-0.55),
        "GOOGL": HestonParams(v0=0.045, theta=0.04, kappa=1.6, sigma=0.35, rho=-0.5),
        "AMZN": HestonParams(v0=0.05, theta=0.045, kappa=1.4, sigma=0.4, rho=-0.55),
        "META": HestonParams(v0=0.055, theta=0.05, kappa=1.3, sigma=0.45, rho=-0.6),
        "TSLA": HestonParams(v0=0.08, theta=0.07, kappa=1.0, sigma=0.6, rho=-0.65),
        "SPY": HestonParams(v0=0.02, theta=0.02, kappa=2.0, sigma=0.2, rho=-0.7),
        "QQQ": HestonParams(v0=0.03, theta=0.025, kappa=1.8, sigma=0.25, rho=-0.65),
    }
    
    ticker_upper = ticker.upper()
    if ticker_upper in calibrated_params:
        return calibrated_params[ticker_upper]
    
    # Estimate parameters from historical data
    try:
        if hist is None:
            hist = get_price_history(ticker, period="2y", interval="1d")
        
        if hist is None or len(hist) < 60:
            return None
        
        close = hist["Close"].astype(float)
        log_returns = np.log(close / close.shift(1)).dropna()
        
        if len(log_returns) < 30:
            return None
        
        # Estimate current variance (v0) from recent 20-day realized vol
        recent_var = log_returns.tail(20).var() * 252  # Annualized
        v0 = float(np.clip(recent_var, 0.01, 0.25))  # Clamp to reasonable range
        
        # Estimate long-term variance (theta) from full history
        long_term_var = log_returns.var() * 252
        theta = float(np.clip(long_term_var, 0.01, 0.20))
        
        # Estimate mean reversion speed (kappa)
        # Higher vol stocks tend to have slower mean reversion
        kappa = float(np.clip(2.0 - theta * 10, 0.5, 3.0))
        
        # Estimate vol of vol (sigma) from rolling volatility changes
        rolling_vol = log_returns.rolling(20).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.pct_change().dropna().std()
        sigma = float(np.clip(vol_of_vol * 5, 0.1, 0.8))
        
        # Estimate correlation (rho) between returns and vol changes
        # Most stocks have negative correlation (leverage effect)
        try:
            vol_changes = rolling_vol.diff().dropna()
            aligned_returns = log_returns.loc[vol_changes.index]
            if len(aligned_returns) > 10:
                rho = float(np.clip(np.corrcoef(aligned_returns, vol_changes)[0, 1], -0.9, -0.1))
                if np.isnan(rho):
                    rho = -0.5
            else:
                rho = -0.5
        except Exception:
            rho = -0.5  # Default negative correlation
        
        return HestonParams(v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)
        
    except Exception as e:
        print(f"[Heston] Failed to estimate params for {ticker}: {e}")
        return None

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


import importlib

gafcnn = None
GAF_CNN_MODEL_PATH = "gaf_cnn_updown.keras"

try:
    tf_keras = importlib.import_module("tensorflow.keras")
    keras = tf_keras
    if os.path.exists(GAF_CNN_MODEL_PATH):
        print(f"[GAF-CNN] Loading model from {GAF_CNN_MODEL_PATH}...")
        gafcnn = keras.models.load_model(GAF_CNN_MODEL_PATH)
        print("[GAF-CNN] Loaded successfully.")
    else:
        print(f"[GAF-CNN] Model file not found at {GAF_CNN_MODEL_PATH}. probup_gaf will be None.")
except ModuleNotFoundError:
    print("[GAF-CNN] TensorFlow/Keras not installed; probup_gaf will be None.")
    gafcnn = None
except Exception as e:
    print(f"[GAF-CNN] TensorFlow/Keras failed to load model: {e}. probup_gaf will be None.")
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

def add_basket_stress_from_z(
    df: pd.DataFrame,
    z_col: str = "retzscore20d",
    pct_window: int = 252,
    min_periods: int = 60,
    shift_days: int = 1,
) -> pd.DataFrame:
    """
    Adds per-date basket distance (Euclidean norm) and rolling percentile rank.
    Expects df columns: date, ticker, and z_col.
    shift_days=1 prevents lookahead (today's signal uses yesterday's stress).
    """
    if z_col not in df.columns:
        raise ValueError(f"Missing z_col={z_col}. Available columns: {list(df.columns)[:20]}...")

    z_wide = df.pivot_table(index="date", columns="ticker", values=z_col, aggfunc="mean")
    D = np.sqrt((z_wide ** 2).sum(axis=1))

    def pct_rank_last(x):
        s = pd.Series(x)
        return float(s.rank(pct=True).iloc[-1])

    D_pct = D.rolling(pct_window, min_periods=min_periods).apply(pct_rank_last, raw=False)

    stress = pd.DataFrame(
        {
            "date": D.index,
            "basket_D": D.values,
            "basket_D_pct": D_pct.shift(shift_days).values,
        }
    )
    return df.merge(stress, on="date", how="left")


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
MACRO_COLUMNS = ["mkt_ret_1d", "term_spread", "t10y", "vix", "unrate", "cpi", "oas", "fed_funds"]
macro_cache = {}

FRED_API_KEY = os.environ.get("FRED_API_KEY")


def _ordered_feature_groups(all_feature_names: list[str]) -> list[str]:
    """
    Group similar features into logical vectors (momentum, volatility, trend, volume, pattern, regime,
    macro, fundamentals, gbm/ARIMA/news), then flatten into a final ordered list.
    This preserves (n_samples, n_features) shape and keeps compatibility with RF/XGB.
    """

    group_defs: dict[str, list[str]] = {
        # Momentum / returns
        "momentum": [
            "ret_1d", "ret_3d", "ret_5d", "ret_20d",
            "cumret_3d", "cumret_5d", "ret_zscore_20d",
            "rel_strength_1d", "rel_strength_3d", "rel_momentum_5d",
            "gap_ret_1d", "intraday_ret_1d",
            "rsi_change_1d", "rsi_change_3d", "rsi_overbought", "rsi_oversold",
        ],
        # Volatility / range
        "volatility": [
            "vol_10d", "vol_20d", "vol_60d", "vol_ratio_10_60",
            "vol_roll_mean_20", "vol_roll_std_20", "vol_regime_high",
            "atr_14", "range_atr_ratio", "vol_20d_std", "ret_vol_interaction",
        ],
        # Trend / moving averages / oscillators
        "trend": [
            "sma_ratio_10_50", "price_to_ma50", "ma_20", "ma_20_slope",
            "bb_mid", "bb_upper", "bb_lower", "bb_pctb",
            "macd", "macdsignal", "macdhist", "adx_14", "rsi14",
        ],
        # Volume
        "volume": [
            "volume_price_corr", "volume_trend", "vol_ma_20", "vol_spike_20",
            "volume_zscore", "dollar_volume", "dollar_volume_20d_avg",
        ],
        # Price pattern / seasonality
        "pattern": [
            "high_low_ratio", "daily_range", "close_position", "hl_range",
            "body_to_range", "upper_wick_to_range", "lower_wick_to_range",
            "day_of_week", "month", "is_month_end",
        ],
        # Regime/state flags (these may be added by model_improvements)
        "regime": [
            "regime_bull", "regime_bear", "regime_vix_low", "regime_vix_medium", "regime_vix_high",
            "regime_covid", "regime_high_corr", "regime_low_corr", "regime_bull_streak", "regime_bear_streak",
        ],
        # Fundamentals
        "fundamentals": FUNDAMENTAL_COLUMNS,
        # Macro
        "macro": MACRO_COLUMNS,
        # GBM-derived
        "gbm": [c for c in all_feature_names if c.startswith("gbm_")],
        # ARIMA-derived
        "arima": [c for c in all_feature_names if c.startswith("arima_")],
        # News
        "news": ["news_sentiment", "news_count"],
    }

    ordered: list[str] = []
    for cols in group_defs.values():
        ordered.extend(cols)

    # Keep any remaining features (e.g., enhanced features) in original order
    remaining = [c for c in all_feature_names if c not in ordered]
    ordered.extend(remaining)
    # Deduplicate while preserving order
    seen = set()
    final_order = []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            final_order.append(c)
    return final_order




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
    """
    Fetch price history using the fixed multi-source pipeline.
    
    Fallback chain: yfinance → Tiingo → Alpaca (lazy-loaded)
    Removed: Stooq (empty data), Alpha Vantage (premium-only)
    """
    # Use the fixed src.data.market module (has proper fallback chain)
    try:
        from src.data.market import get_price_history as _market_get_price
        df = _market_get_price(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[get_price_history] src.data.market failed for {ticker} ({period}): {e}")

    # Legacy fallback: data_fetch cached Yahoo
    try:
        df = get_history_yahoo(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        print(f"[get_price_history] Yahoo cached failed for {ticker} ({period}): {e}")

    # Last resort: raw Yahoo download
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

    safe_sym = symbol.replace("^", "").replace("/", "-").upper()
    cache_file = MACRO_CACHE_DIR / f"{safe_sym}_{period}.pkl"

    def _save_cache(df: pd.DataFrame):
        try:
            df.to_pickle(cache_file)
        except Exception as e:
            print(f"[get_macro_df] Warning: failed to persist macro cache {cache_file}: {e}")

    # Try on-disk cache first to avoid repeated API pulls across runs
    if cache_file.exists():
        try:
            cached = pd.read_pickle(cache_file)
            if cached is not None and not cached.empty:
                macro_cache[key] = cached
                return cached
        except Exception as e:
            print(f"[get_macro_df] Warning: failed to read macro cache {cache_file}: {e}")

    # Robust fallback: try primary index then proxies
    candidates = [symbol, "^SPX", "SPX", "SPY", "VOO"]
    hist = pd.DataFrame()
    for sym in candidates:
        try:
            hist = get_price_history(sym, period=period, interval="1d")
            if hist is not None and not hist.empty:
                break
        except Exception as e:
            print(f"[get_macro_df] {sym} failed: {e}")
            continue

    if hist is None or hist.empty:
        # Synthetic zero series fallback to keep macro columns populated during outages/429s
        print(f"[get_macro_df] Warning: price history unavailable for proxies {candidates} (period={period}); using synthetic zero macro series")

        # Approximate business-day length for the requested period
        period_days_map = {
            "10y": 252 * 10,
            "5y": 252 * 5,
            "3y": 252 * 3,
            "2y": 252 * 2,
            "1y": 252,
            "6mo": 252 // 2,
            "3mo": 252 // 4,
        }
        days = period_days_map.get(period, 252)
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="B")
        hist = pd.DataFrame(index=idx, data={"Close": 100.0})

    df = pd.DataFrame(index=hist.index)
    # Always normalize to tz-naive to prevent join errors with callers
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["mkt_ret_1d"] = hist.get("Close", pd.Series(index=hist.index, data=100.0)).pct_change().fillna(0.0)

    # Pre-seed macro columns with zeros so downstream filtering retains them even if FRED is unavailable
    for base_col in ["t10y", "t3m", "vix", "unrate", "cpi", "oas", "fed_funds", "term_spread"]:
        df[base_col] = 0.0

    if FRED_API_KEY is None:
        print("[get_macro_df] FRED_API_KEY not set; using zero-filled macro features + mkt_ret_1d")
        macro_cache[key] = df
        _save_cache(df)
        return df

    try:
        start_date = df.index.min().date()
        end_date = df.index.max().date()

        s10 = get_fred_series("DGS10", start_date, end_date)
        s3m = get_fred_series("DGS3MO", start_date, end_date)
        vix = get_fred_series("VIXCLS", start_date, end_date)
        
        # === NEW: TIER 1 Macro Expansion (FRED API) ===
        unrate = get_fred_series("UNRATE", start_date, end_date)  # Unemployment rate
        cpi = get_fred_series("CPIAUCSL", start_date, end_date)   # CPI (all urban consumers)
        oas = get_fred_series("BAMLH0A0HYM2", start_date, end_date)  # HY OAS spread
        fed_funds = get_fred_series("FEDFUNDS", start_date, end_date)  # Fed Funds Rate

        # PHASE 1 FIX: Fill NaNs BEFORE reindex to prevent look-ahead bias at fold boundaries
        s10_filled = s10.fillna(method='ffill').fillna(method='bfill')
        s3m_filled = s3m.fillna(method='ffill').fillna(method='bfill')
        vix_filled = vix.fillna(method='ffill').fillna(method='bfill')
        unrate_filled = unrate.fillna(method='ffill').fillna(method='bfill')
        cpi_filled = cpi.fillna(method='ffill').fillna(method='bfill')
        oas_filled = oas.fillna(method='ffill').fillna(method='bfill')
        fed_funds_filled = fed_funds.fillna(method='ffill').fillna(method='bfill')

        df_dates = df.index.normalize().tz_localize(None)
        df["t10y"] = s10_filled.reindex(df_dates).values
        df["t3m"] = s3m_filled.reindex(df_dates).values
        df["vix"] = vix_filled.reindex(df_dates).values
        df["unrate"] = unrate_filled.reindex(df_dates).values
        df["cpi"] = cpi_filled.reindex(df_dates).values
        df["oas"] = oas_filled.reindex(df_dates).values
        df["fed_funds"] = fed_funds_filled.reindex(df_dates).values
        df["term_spread"] = df["t10y"] - df["t3m"]

        macro_cache[key] = df
        _save_cache(df)
        return df
    except Exception as e:
        print(f"[get_macro_df] FRED fetch failed: {e}")
        # Keep zero-filled macro placeholders so callers retain columns
        macro_cache[key] = df
        _save_cache(df)
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
    # Advanced features (momentum, volatility clustering, regime transitions)
    "momentum_3m_zscore", "momentum_6m_zscore", "momentum_ratio_10_20",
    "volatility_cluster", "volatility_skew_ratio",
    "regime_transition_score", "correlation_with_vol",
    "tail_risk_20d", "rsi_divergence_5d", "macd_reversal_strength",
    "vol_mean_reversion_score", "price_mean_reversion_score",
    "liquidity_ratio", "spread_zscore", "momentum_accelerator",
    # PHASE 2: Regime detection features (lagged by 1 day)
    "regime_bull", "regime_bear",
    "regime_vix_low", "regime_vix_medium", "regime_vix_high",
    "regime_covid",
    "regime_high_corr", "regime_low_corr",
    "bull_streak", "bear_streak",
    # PHASE 3: Enhanced features (volatility-adjusted, cross-sectional, lagged)
    "ret_vol_adjusted", "vol_percentile_rank", "vol_regime_strength",
    "momentum_5d_vol_adj", "momentum_20d_vol_adj",
    "mean_reversion_signal", "momentum_confirmation",
    "ret_1d_lag_2", "ret_1d_lag_3", "ret_1d_lag_4", "ret_1d_lag_5",
    "vol_20d_lag_2", "vol_20d_lag_3", "vol_20d_lag_4", "vol_20d_lag_5",
    "rsi14_lag_2", "rsi14_lag_3", "rsi14_lag_4", "rsi14_lag_5",
    "ret_1d_rolling_mean_10", "ret_1d_rolling_std_10", "vol_20d_rolling_mean_10",
    "ret_vol_correlation", "price_action_strength",
    # TIER 1: Support/Resistance Features
    "dist_from_50d_high", "dist_from_50d_low", "dist_from_52w_high",
    # TIER 1: Divergence Detection Features
    "rsi_price_divergence", "macd_price_divergence",
    # TIER 1: News Sentiment Features
    "news_sentiment", "news_count",
    # Energy/oil sensitivity features
    "crude_ret_1d", "crude_ret_5d", "brent_wti_spread",
    "dxy_ret_5d", "crude_corr_20d", "crude_beta_60d", "crude_rel_strength_20d",
]

# Ordered feature list grouped by logical vectors for consistent stacking
FEAT_GROUP_ORDER = _ordered_feature_groups(FEATURE_COLUMNS + MACRO_COLUMNS)


def add_advanced_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Adds advanced momentum, volatility clustering, and regime transition features.
    All features are lagged by 1 day to prevent look-ahead bias.
    """
    hist = hist.copy()
    close = hist["Close"]
    ret_1d = close.pct_change()
    
    # ---- Multi-scale momentum features ----
    # 3-month (63 days) and 6-month (126 days) momentum
    mom_3m = close / close.rolling(63).mean() - 1
    mom_6m = close / close.rolling(126).mean() - 1
    
    # Standardize momentum scores
    mom_3m_mean = mom_3m.rolling(60).mean()
    mom_3m_std = mom_3m.rolling(60).std()
    hist["momentum_3m_zscore"] = ((mom_3m - mom_3m_mean) / (mom_3m_std + 1e-9)).shift(1)
    
    mom_6m_mean = mom_6m.rolling(60).mean()
    mom_6m_std = mom_6m.rolling(60).std()
    hist["momentum_6m_zscore"] = ((mom_6m - mom_6m_mean) / (mom_6m_std + 1e-9)).shift(1)
    
    # Cross-scale momentum ratio (short-term vs long-term)
    ret_10d = close.pct_change(10)
    ret_20d = close.pct_change(20)
    hist["momentum_ratio_10_20"] = (ret_10d / (ret_20d + 1e-9)).shift(1)
    
    # ---- Volatility clustering ----
    vol_20d = ret_1d.rolling(20).std()
    vol_60d = ret_1d.rolling(60).std()
    
    # High vol followed by high vol (clustering)
    vol_above_mean = (vol_20d > vol_60d).astype(float)
    hist["volatility_cluster"] = vol_above_mean.rolling(3).sum().shift(1)
    
    # Volatility skew: ratio of recent to long-term vol changes
    vol_recent_change = vol_20d.diff(5)
    hist["volatility_skew_ratio"] = (vol_recent_change / (vol_60d + 1e-9)).shift(1)
    
    # ---- Regime transition detection ----
    # Score based on crossing moving averages (potential regime change)
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    above_50 = (close > ma_50).astype(int)
    above_200 = (close > ma_200).astype(int)
    regime_changes = above_50.diff().abs() + above_200.diff().abs()
    hist["regime_transition_score"] = regime_changes.rolling(5).sum().shift(1)
    
    # ---- Correlation with volatility (volatility feedback) ----
    ret_vol_corr = ret_1d.rolling(30).corr(vol_20d)
    hist["correlation_with_vol"] = ret_vol_corr.shift(1)
    
    # ---- Tail risk (extreme moves) ----
    # Proportion of days with returns in bottom 5% of distribution
    ret_p5 = ret_1d.rolling(20).quantile(0.05)
    tail_days = (ret_1d < ret_p5).astype(int)
    hist["tail_risk_20d"] = tail_days.rolling(10).mean().shift(1)
    
    # ---- RSI divergence (momentum reversal signal) ----
    # When price makes new high but RSI doesn't
    rsi14 = hist["rsi14"] if "rsi14" in hist.columns else pd.Series(index=hist.index, data=np.nan)
    price_high = close.rolling(5).max()
    rsi_high = rsi14.rolling(5).max()
    rsi_divergence = ((price_high - close.shift(1)) > 0) & ((rsi_high - rsi14.shift(1)) <= 0)
    hist["rsi_divergence_5d"] = rsi_divergence.astype(int).shift(1)
    
    # ---- MACD strength and reversals ----
    if "macdhist" in hist.columns:
        macd_hist = hist["macdhist"]
        macd_strength = macd_hist.abs()
        hist["macd_reversal_strength"] = macd_hist.rolling(3).apply(
            lambda x: 1 if x.iloc[-1] * x.iloc[0] < 0 else 0, raw=False
        ).shift(1)
    else:
        hist["macd_reversal_strength"] = pd.Series(index=hist.index, data=0.0)
    
    # ---- Mean reversion signals ----
    # Volatility mean reversion: when vol is high relative to average
    vol_zscore = (vol_20d - vol_60d.rolling(60).mean()) / (vol_60d.rolling(60).std() + 1e-9)
    hist["vol_mean_reversion_score"] = vol_zscore.shift(1)
    
    # Price mean reversion: distance from moving average
    ma_20 = close.rolling(20).mean()
    price_deviation = (close - ma_20) / (ma_20 + 1e-9)
    hist["price_mean_reversion_score"] = price_deviation.shift(1)
    
    # ---- Liquidity features ----
    # High volume relative to recent average = good liquidity
    vol_avg = hist["Volume"].rolling(20).mean() if "Volume" in hist.columns else pd.Series(index=hist.index, data=1.0)
    hist["liquidity_ratio"] = (hist["Volume"] / (vol_avg + 1e-9)).shift(1) if "Volume" in hist.columns else 1.0
    
    # Bid-ask proxy: range as % of price
    high = hist["High"]
    low = hist["Low"]
    hl_range = (high - low) / (close + 1e-9)
    hl_mean = hl_range.rolling(20).mean()
    hist["spread_zscore"] = ((hl_range - hl_mean) / (hl_mean.rolling(20).std() + 1e-9)).shift(1)
    
    # ---- Momentum acceleration (second derivative) ----
    ret_5d = close.pct_change(5)
    ret_5d_prev = ret_5d.shift(5)
    hist["momentum_accelerator"] = ((ret_5d - ret_5d_prev) / (ret_5d_prev.abs() + 1e-9)).shift(1)
    
    return hist


def add_regime_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    PHASE 2: Add regime detection features to identify market conditions.
    
    4 Regime Types:
    1. Bull/Bear: Based on 20-day rolling returns
    2. VIX Regimes: Low volatility, medium, high volatility
    3. COVID Crisis: Manual date-based crisis period
    4. Correlation Regimes: Stock-to-market correlation regimes
    
    All features are LAGGED by 1 day to prevent look-ahead bias.
    """
    hist = hist.copy()
    
    # ===== REGIME 1: Bull vs Bear (20-day rolling return) =====
    close = hist["Close"]
    ret_1d = close.pct_change()
    rolling_ret_20d = (1 + ret_1d).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    
    hist["regime_bull"] = (rolling_ret_20d > 0).astype(int).shift(1)
    hist["regime_bear"] = (rolling_ret_20d <= 0).astype(int).shift(1)
    
    # ===== REGIME 2: VIX-based volatility regimes =====
    if "vix" in hist.columns:
        vix = hist["vix"]
        hist["regime_vix_low"] = (vix < 12).astype(int).shift(1)
        hist["regime_vix_medium"] = ((vix >= 12) & (vix <= 20)).astype(int).shift(1)
        hist["regime_vix_high"] = (vix > 20).astype(int).shift(1)
    else:
        # Fallback: Use realized volatility if VIX not available
        vol_20d = ret_1d.rolling(20).std()
        vol_50d = ret_1d.rolling(50).std()
        vol_percentile_75 = vol_20d.rolling(50).quantile(0.75)
        vol_percentile_25 = vol_20d.rolling(50).quantile(0.25)
        
        hist["regime_vix_low"] = (vol_20d < vol_percentile_25).astype(int).shift(1)
        hist["regime_vix_medium"] = ((vol_20d >= vol_percentile_25) & (vol_20d <= vol_percentile_75)).astype(int).shift(1)
        hist["regime_vix_high"] = (vol_20d > vol_percentile_75).astype(int).shift(1)
    
    # ===== REGIME 3: COVID/Crisis Period =====
    # Define crisis periods (historical market stress events)
    covid_start = pd.Timestamp("2020-02-15")
    covid_end = pd.Timestamp("2020-06-30")
    
    # Handle timezone-aware indices
    hist_idx = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    
    regime_covid_series = pd.Series(
        ((hist_idx >= covid_start) & (hist_idx <= covid_end)).astype(int),
        index=hist.index
    )
    hist["regime_covid"] = regime_covid_series.shift(1).fillna(0).astype(int)
    
    # ===== REGIME 4: Correlation with market =====
    # Compute rolling correlation between stock and SPX (market)
    if "Close" in hist.columns and len(hist) >= 60:
        try:
            # Get date range - normalize timezone for consistency
            min_date = hist.index.min()
            max_date = hist.index.max()
            
            # Remove timezone info if present for fetching
            if hasattr(min_date, 'tz_localize'):
                min_date = pd.Timestamp(min_date.date())
                max_date = pd.Timestamp(max_date.date())
            
            # Try to get SPX for correlation
            spx = _get_spx(min_date, max_date, tz=None)
            if not spx.empty and "Close" in spx.columns:
                spx_close = spx["Close"]
                stock_rets = close.pct_change()
                spx_rets = spx_close.pct_change()
                
                # Normalize indices to be tz-naive for alignment
                stock_rets_index = stock_rets.index.tz_localize(None) if stock_rets.index.tz is not None else stock_rets.index
                spx_rets_index = spx_rets.index.tz_localize(None) if spx_rets.index.tz is not None else spx_rets.index
                
                stock_rets = stock_rets.set_axis(stock_rets_index)
                spx_rets = spx_rets.set_axis(spx_rets_index)
                
                # Align indices
                combined = pd.DataFrame({
                    "stock": stock_rets,
                    "spx": spx_rets
                }).dropna()
                
                if len(combined) >= 20:
                    corr_20d = combined["stock"].rolling(20).corr(combined["spx"])
                    corr_median = corr_20d.rolling(60).median()
                    
                    # High correlation = market-driven, Low correlation = idiosyncratic
                    hist["regime_high_corr"] = (corr_20d > corr_median).astype(int).shift(1).reindex(hist.index, fill_value=0)
                    hist["regime_low_corr"] = (corr_20d <= corr_median).astype(int).shift(1).reindex(hist.index, fill_value=0)
                else:
                    hist["regime_high_corr"] = 0
                    hist["regime_low_corr"] = 0
            else:
                hist["regime_high_corr"] = 0
                hist["regime_low_corr"] = 0
        except Exception as e:
            print(f"[add_regime_features] Warning: Could not compute correlation regimes: {e}")
            hist["regime_high_corr"] = 0
            hist["regime_low_corr"] = 0
    else:
        hist["regime_high_corr"] = 0
        hist["regime_low_corr"] = 0
    
    # ===== Regime Duration (how long we've been in current regime) =====
    hist["bull_streak"] = (hist["regime_bull"] == 1).astype(int)
    hist["bull_streak"] = hist["bull_streak"].groupby((hist["bull_streak"] != hist["bull_streak"].shift()).cumsum()).cumcount() + 1
    hist["bull_streak"] = hist["bull_streak"] * hist["regime_bull"]
    
    bear_streak = (hist["regime_bear"] == 1).astype(int)
    hist["bear_streak"] = bear_streak.groupby((bear_streak != bear_streak.shift()).cumsum()).cumcount() + 1
    hist["bear_streak"] = hist["bear_streak"] * hist["regime_bear"]
    
    return hist


# Cache for external series (crude, brent, dollar proxy) to avoid repeat fetches
ENERGY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _get_energy_series(symbol: str, period: str = "5y") -> pd.DataFrame:
    key = (symbol, period)
    if key in ENERGY_CACHE:
        return ENERGY_CACHE[key]
    try:
        df = get_price_history(symbol, period=period, interval="1d")
        ENERGY_CACHE[key] = df if df is not None else pd.DataFrame()
    except Exception as e:
        print(f"[add_energy_features] Fetch failed for {symbol} ({period}): {e}")
        ENERGY_CACHE[key] = pd.DataFrame()
    return ENERGY_CACHE[key]


def add_energy_features(hist: pd.DataFrame, ticker: str, period: str = "5y") -> pd.DataFrame:
    """Adds crude/dollar sensitivity features for energy-heavy tickers (e.g., XOM, CVX).

    All features are lagged by 1 day to avoid look-ahead bias and are safely filled to 0 when
    external series are unavailable.
    """
    hist = hist.copy()
    if hist.empty:
        return hist

    crude = _get_energy_series("CL=F", period)
    brent = _get_energy_series("BZ=F", period)
    dxy = _get_energy_series("UUP", period)  # USD proxy (inverse relationship)

    # If crude missing, create zero placeholders and exit early
    if crude is None or crude.empty or "Close" not in crude.columns:
        for col in [
            "crude_ret_1d", "crude_ret_5d", "brent_wti_spread",
            "dxy_ret_5d", "crude_corr_20d", "crude_beta_60d", "crude_rel_strength_20d",
        ]:
            hist[col] = 0.0
        return hist

    crude_close = crude["Close"].copy()
    crude_ret = crude_close.pct_change()
    brent_close = brent["Close"] if brent is not None and "Close" in brent.columns else pd.Series(dtype=float)
    dxy_close = dxy["Close"] if dxy is not None and "Close" in dxy.columns else pd.Series(dtype=float)

    idx = hist.index
    crude_ret_1d = crude_ret.shift(1).reindex(idx).ffill().bfill()
    crude_ret_5d = crude_close.pct_change(5).shift(1).reindex(idx).ffill().bfill()
    hist["crude_ret_1d"] = crude_ret_1d
    hist["crude_ret_5d"] = crude_ret_5d

    if not brent_close.empty:
        spread = (brent_close - crude_close) / (crude_close.abs() + 1e-9)
        hist["brent_wti_spread"] = spread.shift(1).reindex(idx).ffill().bfill()
    else:
        hist["brent_wti_spread"] = 0.0

    if not dxy_close.empty:
        hist["dxy_ret_5d"] = dxy_close.pct_change(5).shift(1).reindex(idx).ffill().bfill()
    else:
        hist["dxy_ret_5d"] = 0.0

    # Rolling correlation/beta to crude
    sym_ret = hist["Close"].pct_change()
    combined = pd.DataFrame({"sym": sym_ret, "crude": crude_ret}).dropna()
    if not combined.empty:
        rolling_corr = combined["sym"].rolling(20).corr(combined["crude"])
        rolling_cov = combined["sym"].rolling(60).cov(combined["crude"])
        rolling_var = combined["crude"].rolling(60).var()
        beta = (rolling_cov / (rolling_var + 1e-9))

        hist["crude_corr_20d"] = rolling_corr.shift(1).reindex(idx).ffill().bfill()
        hist["crude_beta_60d"] = beta.shift(1).reindex(idx).ffill().bfill()
    else:
        hist["crude_corr_20d"] = 0.0
        hist["crude_beta_60d"] = 0.0

    # Relative strength: equity 20d cumulative vs crude 20d cumulative
    sym_cum20 = (1 + sym_ret.fillna(0)).rolling(20).apply(lambda x: np.prod(x) - 1, raw=True)
    crude_cum20 = (1 + crude_ret.fillna(0)).rolling(20).apply(lambda x: np.prod(x) - 1, raw=True)
    rel_strength = (sym_cum20 - crude_cum20).shift(1)
    hist["crude_rel_strength_20d"] = rel_strength.reindex(idx).ffill().bfill()

    return hist


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

    # === NEW: Support/Resistance Features (TIER 1 Implementation) ===
    # Distance to 50-day high (mean reversion signal)
    high_50d = close.rolling(50).max()
    hist["dist_from_50d_high"] = ((close - high_50d) / (high_50d + 1e-9)).shift(1)
    
    # Distance to 50-day low (support level)
    low_50d = close.rolling(50).min()
    hist["dist_from_50d_low"] = ((close - low_50d) / (low_50d + 1e-9)).shift(1)
    
    # Distance to 52-week high (longer-term trend)
    high_252d = close.rolling(252).max()
    hist["dist_from_52w_high"] = ((close - high_252d) / (high_252d + 1e-9)).shift(1)

    # === NEW: Divergence Detection Features (TIER 1 Implementation) ===
    # RSI divergence: RSI going up but price going down (or vice versa)
    ret_5d_change = close.pct_change(5)
    rsi_5d_change = hist["rsi14"].diff(5) if "rsi14" in hist.columns else 0
    hist["rsi_price_divergence"] = (rsi_5d_change * -ret_5d_change).shift(1)
    
    # MACD divergence: MACD strength vs price momentum
    if "macdhist" in hist.columns:
        macd_5d_change = hist["macdhist"].diff(5)
        hist["macd_price_divergence"] = (macd_5d_change * -ret_5d_change).shift(1)
    else:
        hist["macd_price_divergence"] = 0.0

    # Add advanced momentum/volatility clustering/regime features
    hist = add_advanced_features(hist)

    return hist


def make_model(model_type: str = "rf", random_state: int = 42, task: str = "reg", log_version: bool = True, **kwargs):
    """
    Create a model with hyperparameters.
    
    Args:
        model_type: "rf", "xgb", "gbrt", "linreg", "ensemble"
        random_state: Random seed
        task: "reg" (regression) or "clf" (classification)
        log_version: If True, log the model version being used
        **kwargs: Additional hyperparameters (max_depth, n_estimators, learning_rate, etc.)
    """
    # Handle ensemble model type
    if model_type == "ensemble":
        if HAS_IMPROVEMENTS:
            print("🔧 Creating ModelEnsemble (RF + GB + XGB)")
            return ModelEnsemble(include_xgb=True)
        else:
            print("⚠️ model_improvements.py not available, falling back to RF")
            model_type = "rf"
    
    # Handle xgb_lstm ensemble (for bear markets)
    if model_type == "xgb_lstm":
        try:
            from src.core.lstm_xgb_ensemble import LSTMXGBEnsemble
            print("🐻 Creating XGB+LSTM Ensemble (optimized for bear markets)")
            return LSTMXGBEnsemble(lstm_weight=0.5, xgb_weight=0.5)
        except ImportError:
            print("⚠️ LSTMXGBEnsemble not available, falling back to XGB")
            model_type = "xgb"
    
    # Log model version if enabled
    if log_version:
        try:
            from src.config import log_model_version, get_model_version_info
            version_log = log_model_version(model_type)
            print(f"🔧 Creating model: {version_log}")
        except ImportError:
            pass  # Graceful fallback if config not available
    
    if task == "clf":
        if model_type == "xgb":
            params = {
                'n_estimators': kwargs.get('n_estimators', 300),
                'learning_rate': kwargs.get('learning_rate', 0.05),
                'max_depth': kwargs.get('max_depth', 3),           # Reduced from 4 for regularization
                'random_state': random_state,
                'tree_method': "hist",
                'verbosity': 0,
                'subsample': kwargs.get('subsample', 0.8),
                'colsample_bytree': kwargs.get('colsample_bytree', 0.7),
                'min_child_weight': kwargs.get('min_child_weight', 100),  # Increased for regularization
                'reg_lambda': kwargs.get('reg_lambda', 10.0),      # L2 regularization
                'reg_alpha': kwargs.get('reg_alpha', 1.0),         # L1 regularization
            }
            return XGBClassifier(**params)
        
        # RandomForest classifier
        params = {
            'n_estimators': kwargs.get('n_estimators', 300),
            'max_depth': kwargs.get('max_depth', 6),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 50),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'random_state': random_state,
            'n_jobs': -1
        }
        return RandomForestClassifier(**params)

    if model_type == "linreg":
        return LinearRegression()

    if model_type == "gbrt":
        params = {
            'n_estimators': kwargs.get('n_estimators', 300),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'max_depth': kwargs.get('max_depth', 4),
            'subsample': kwargs.get('subsample', 1.0),
            'random_state': random_state,
        }
        return GradientBoostingRegressor(**params)

    if model_type == "xgb":
        # Use centralized config from src/config.py
        try:
            from src.config import get_model_config
            default_params = get_model_config("xgb")
        except ImportError:
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 3,
                'subsample': 0.6,
                'colsample_bytree': 0.5,
                'min_child_weight': 50,
                'reg_lambda': 10.0,
                'reg_alpha': 1.0,
            }
        
        params = {
            'n_estimators': kwargs.get('n_estimators', default_params.get('n_estimators', 100)),
            'learning_rate': kwargs.get('learning_rate', default_params.get('learning_rate', 0.01)),
            'max_depth': kwargs.get('max_depth', default_params.get('max_depth', 3)),
            'random_state': random_state,
            'tree_method': "hist",
            'verbosity': 0,
            'subsample': kwargs.get('subsample', default_params.get('subsample', 0.6)),
            'colsample_bytree': kwargs.get('colsample_bytree', default_params.get('colsample_bytree', 0.5)),
            'min_child_weight': kwargs.get('min_child_weight', default_params.get('min_child_weight', 50)),
            'reg_lambda': kwargs.get('reg_lambda', default_params.get('reg_lambda', 10.0)),
            'reg_alpha': kwargs.get('reg_alpha', default_params.get('reg_alpha', 1.0)),
        }
        return XGBRegressor(**params)

    # RandomForest regressor (default)
    params = {
        'n_estimators': kwargs.get('n_estimators', 300),
        'max_depth': kwargs.get('max_depth', 8),
        'min_samples_leaf': kwargs.get('min_samples_leaf', 50),
        'min_samples_split': kwargs.get('min_samples_split', 2),
        'random_state': random_state,
        'n_jobs': -1
    }
    return RandomForestRegressor(**params)


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

    # Allow env override to keep more near-cutoff features
    try:
        minfeatures = int(os.environ.get("ELASTICNET_MINFEATURES", minfeatures))
    except Exception:
        pass

    Xdf = pd.DataFrame(X, index=pd.DatetimeIndex(dates), columns=featurenames)
    t1 = make_t1_from_horizon(Xdf.index, horizon)  # use YOUR old helper name in the file
    cv = PurgedKFold(nsplits=nsplits, t1=t1, pctembargo=pctembargo)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("enet", ElasticNetCV(
                l1_ratio=float(l1ratio),
                alphas=np.logspace(-4, 1, 100),  # Use explicit alpha range instead of None (deprecated)
                cv=cv.split(Xdf),
                n_jobs=-1,
                random_state=int(randomstate),
                max_iter=10000,  # Increased from 5000 for better convergence
                tol=1e-3,  # Looser tolerance to aid convergence
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

def selectfeaturesols_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    featurenames: list[str],
    alpha: float = 0.05,
    topk: int = 50,          # 0 => no cap
    minfeatures: int = 10,
):
    """
    Returns: (X_selected, selected_names, selected_mask_full)
    selected_mask_full is boolean mask aligned to the *input* featurenames.
    """
    if X is None or y is None or len(y) < 30:
        mask = np.ones(len(featurenames), dtype=bool)
        return X, list(featurenames), mask

    try:
        Xdf = pd.DataFrame(X, columns=featurenames)
        Xdf = sm.add_constant(Xdf, has_constant="add")
        ols = sm.OLS(y, Xdf).fit()

        # p-values for features (exclude constant)
        pvals = ols.pvalues.drop(labels=["const"], errors="ignore")
        pvals = pvals.replace([np.inf, -np.inf], np.nan).dropna()

        if pvals.empty:
            mask = np.ones(len(featurenames), dtype=bool)
            return X, list(featurenames), mask

        # primary rule: keep p <= alpha
        keep = pvals[pvals <= float(alpha)].sort_values()

        # fallback: if too few, take best minfeatures by p-value
        if len(keep) < int(minfeatures):
            keep = pvals.sort_values().head(int(minfeatures))

        # optional cap: keep only topk “best” p-values
        if topk is not None and int(topk) > 0:
            keep = keep.head(int(topk))

        selected = list(keep.index)
        mask = np.array([f in set(selected) for f in featurenames], dtype=bool)

        # guardrail: never return empty
        if mask.sum() == 0:
            mask[:] = True
            selected = list(featurenames)

        return X[:, mask], selected, mask

    except Exception as e:
        print("OLS significance selection failed, skipping. Error:", e)
        mask = np.ones(len(featurenames), dtype=bool)
        return X, list(featurenames), mask


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
            hist = add_regime_features(hist)  # PHASE 2: Add regime detection
            # Energy/oil sensitivity features (esp. for XOM/CVX)
            try:
                hist = add_energy_features(hist, ticker=ticker, period=per)
            except Exception as e:
                print(f"[build_features_and_target] Warning: energy feature calc failed: {e}")
            
            # Add enhanced features (volatility-adjusted, cross-sectional, lagged)
            if HAS_IMPROVEMENTS:
                hist = add_enhanced_features(hist)
            
            # PHASE 3: Add TA-Lib and Pandas-TA indicators
            try:
                from talib_integration import add_talib_indicators
                hist = add_talib_indicators(hist)
            except Exception as e:
                print(f"[build_features_and_target] Warning: TA-Lib indicators failed: {e}")
            
            try:
                from pandas_ta_integration import add_pandas_ta_indicators
                hist = add_pandas_ta_indicators(hist, categories=["momentum", "trend", "volatility", "volume"])
            except Exception as e:
                print(f"[build_features_and_target] Warning: Pandas-TA indicators failed: {e}")
            
            # PHASE 4: Add ARIMA ensemble features
            try:
                from arima_integration import add_arima_features
                hist = add_arima_features(hist, target_col="ret_1d", arima_horizons=[1, 5, 20])
            except Exception as e:
                print(f"[build_features_and_target] Warning: ARIMA features failed: {e}")
            
            missing = [c for c in (FEATURE_COLUMNS + MACRO_COLUMNS) if c not in hist.columns]
            print("Missing:", missing[:30])
            print("GBM cols present:", [c for c in hist.columns if c.startswith("gbm_")][:20])

            # Try to get macro data, but don't fail if unavailable
            try:
                macro_df = get_macro_df(symbol="^GSPC", period=per)
                # Normalize both indices to tz-naive to avoid join errors
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                if macro_df.index.tz is not None:
                    macro_df.index = macro_df.index.tz_localize(None)
                hist = hist.join(macro_df, how="left")
            except Exception as e:
                print(f"[build_features_and_target] Warning: Could not fetch macro data: {e}")
                # Ensure macro columns exist so downstream filters keep them
                for c in MACRO_COLUMNS:
                    if c not in hist.columns:
                        hist[c] = 0.0
            
            # Fill missing macro columns with NaN, then forward/backward fill
            for c in MACRO_COLUMNS:
                if c not in hist.columns:
                    hist[c] = np.nan
            # Only fill if column exists
            macro_cols_present = [c for c in MACRO_COLUMNS if c in hist.columns]
            if macro_cols_present:
                hist[macro_cols_present] = hist[macro_cols_present].ffill().bfill()

            # Try to get fundamental data, but don't fail if unavailable
            try:
                fund_feats = get_fundamental_features(ticker)
                for k, v in fund_feats.items():
                    hist[k] = v
            except Exception as e:
                print(f"[build_features_and_target] Warning: Could not fetch fundamental data: {e}")

            # === NEW: TIER 1 News Sentiment (Marketaux API) ===
            try:
                from data_fetch import get_news_sentiment
                sentiment_data = get_news_sentiment(ticker, lookback_days=7)
                hist["news_sentiment"] = sentiment_data.get("sentiment_score", 0.0)
                hist["news_count"] = sentiment_data.get("article_count", 0)
                # Forward fill sentiment scores within the period
                hist["news_sentiment"] = hist["news_sentiment"].ffill().bfill().fillna(0.0)
                hist["news_count"] = hist["news_count"].ffill().bfill().fillna(0)
            except Exception as e:
                print(f"[build_features_and_target] Warning: Could not fetch news sentiment: {e}")
                hist["news_sentiment"] = 0.0
                hist["news_count"] = 0

            raw_target = hist["Close"].pct_change(horizon).shift(-horizon)
            hist["ftarget_ret_horizon_ahead"] = (raw_target / (hist["vol_20d"] + 1e-9)) if use_vol_scaled_target else raw_target

            # Use only columns that actually exist AND have data
            # First, only include feature columns that exist
            feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
            # Then, filter to only columns with < 50% NaN
            data_quality = hist[feat_cols_available].isna().sum() / len(hist)
            feat_cols_available = [c for c in feat_cols_available if data_quality[c] < 0.5]
            
            macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
            # Filter macro columns with < 50% NaN
            data_quality_macro = hist[macro_cols_available].isna().sum() / len(hist)
            macro_cols_available = [c for c in macro_cols_available if data_quality_macro[c] < 0.5]
            
            # Order features by logical vectors (momentum, volatility, trend, volume, pattern, regime, macro, fundamentals, gbm/arima/news)
            feat_cols = feat_cols_available + macro_cols_available
            ordered_feat_cols = [c for c in FEAT_GROUP_ORDER if c in feat_cols] + [c for c in feat_cols if c not in FEAT_GROUP_ORDER]
            
            # SAFEGUARD: Ensure we have minimum features before proceeding
            MIN_REQUIRED_FEATURES = 5
            if len(ordered_feat_cols) < MIN_REQUIRED_FEATURES:
                print(f"⚠️ {ticker} only has {len(ordered_feat_cols)} features after quality filter, relaxing threshold...")
                # Relax the NaN threshold to 70%
                feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
                data_quality = hist[feat_cols_available].isna().sum() / len(hist)
                feat_cols_available = [c for c in feat_cols_available if data_quality[c] < 0.7]
                macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
                data_quality_macro = hist[macro_cols_available].isna().sum() / len(hist)
                macro_cols_available = [c for c in macro_cols_available if data_quality_macro[c] < 0.7]
                feat_cols = feat_cols_available + macro_cols_available
                ordered_feat_cols = [c for c in FEAT_GROUP_ORDER if c in feat_cols] + [c for c in feat_cols if c not in FEAT_GROUP_ORDER]
                
                if len(ordered_feat_cols) < MIN_REQUIRED_FEATURES:
                    raise ValueError(f"{ticker}: Only {len(ordered_feat_cols)} features available. Insufficient data quality.")
            
            # Fill remaining NaNs with forward fill, then backward fill
            hist[ordered_feat_cols] = hist[ordered_feat_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # IMPORTANT: Get the actual last close BEFORE dropna
            # This is the most recent price in the raw data
            actual_last_close = hist["Close"].iloc[-1]
            actual_last_date = hist.index[-1]

            cols_needed = ordered_feat_cols + ["ftarget_ret_horizon_ahead"]
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

            # Concatenate grouped vectors into final feature matrix (n_samples, n_features)
            X = df[ordered_feat_cols].values
            y = df["ftarget_ret_horizon_ahead"].values

            last_row = df.iloc[-1]
            last_row_features = last_row[ordered_feat_cols].values
            # Use the actual most recent close price from raw history, not df (which is dropna'd)
            last_close = float(actual_last_close)
            last_vol_20d = last_row["vol_20d"]

            dates=df.index

            return X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates

        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"No usable history for {ticker} after trying periods={periods_to_try}. Last error={last_error}")

def build_panel_features_and_target(tickers, period="5y", horizon=1, use_vol_scaled_target=False):
    """Build cross-sectional panel from multiple tickers, skipping failures."""
    dfs = []
    
    for ticker in tickers:
        try:
            print(f"[panel] Building features for {ticker}...")
            X, y, _, _, _, _, dates = build_features_and_target(
                ticker=ticker, period=period, horizon=horizon, 
                use_vol_scaled_target=use_vol_scaled_target
            )
            if X is None or len(X) < 50:
                print(f"[panel] Skipping {ticker}: insufficient data ({len(X) if X is not None else 0} rows)")
                continue
            
            # Use the actual number of features returned by build_features_and_target
            actual_feat_count = X.shape[1] if len(X.shape) > 1 else len(X)
            
            # Get ALL available feature column names from FEATURE_COLUMNS and MACRO_COLUMNS
            all_possible_cols = FEATURE_COLUMNS + MACRO_COLUMNS
            
            # Use up to the number of features we have, use generic names if we need more
            if len(all_possible_cols) >= actual_feat_count:
                feat_cols = all_possible_cols[:actual_feat_count]
            else:
                # More features returned than we have names for (shouldn't happen)
                feat_cols = all_possible_cols + [f"feat_{i}" for i in range(len(all_possible_cols), actual_feat_count)]
            
            # Ensure exact match
            feat_cols = feat_cols[:actual_feat_count]
            
            df = pd.DataFrame(X, index=pd.DatetimeIndex(dates), columns=feat_cols)
            df['target'] = y
            df['ticker'] = ticker
            dfs.append(df)
            print(f"[panel] {ticker}: {len(df)} rows OK ({actual_feat_count} features)")
            
        except Exception as e:
            print(f"[panel] Skipping {ticker}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No usable panel data for any ticker.")
    
    panel = pd.concat(dfs, axis=0).sort_index()
    print(f"[panel] Combined {len(panel)} rows across {len(dfs)} tickers")
    return panel


def _select_features_for_fold(
    train_df: pd.DataFrame,
    feat_cols: list,
    horizon: int,
    selection_mode: str = "best",
) -> list | None:
    """
    Apply feature selection to a training fold.
    
    Args:
        train_df: Training data for this fold
        feat_cols: All available feature columns
        horizon: Prediction horizon
        selection_mode: "elasticnet", "ols", or "best" (compare both)
    
    Returns:
        Selected feature columns, or None if selection fails
    """
    try:
        X = train_df[feat_cols].fillna(0).values
        y = train_df["target"].values
        
        if len(X) < 30:
            return None
        
        selected_cols = None
        
        if selection_mode == "elasticnet":
            try:
                dates = pd.to_datetime(train_df["date"]).values if "date" in train_df.columns else None
                X_sel, en_names, en_mask = select_features_elasticnet_timeseries(
                    X=X,
                    y=y,
                    feature_names=list(feat_cols),
                    dates=dates,
                    horizon=horizon,
                    n_splits=min(5, len(X) // 30),
                    l1_ratio=0.5,
                    min_features=10,
                )
                selected_cols = en_names
            except Exception as e:
                print(f"[FS] ElasticNet failed: {e}")
        
        elif selection_mode == "ols":
            try:
                X_sel, ols_names, ols_mask = selectfeaturesols_pvalues(
                    X, y,
                    featurenames=list(feat_cols),
                    alpha=0.05,
                    topk=50,
                    minfeatures=10,
                )
                selected_cols = ols_names
            except Exception as e:
                print(f"[FS] OLS failed: {e}")
        
        elif selection_mode == "best":
            # Try ElasticNet first, fallback to OLS, fallback to all
            elasticnet_cols = None
            ols_cols = None
            
            try:
                dates = pd.to_datetime(train_df["date"]).values if "date" in train_df.columns else None
                X_sel, en_names, en_mask = select_features_elasticnet_timeseries(
                    X=X,
                    y=y,
                    feature_names=list(feat_cols),
                    dates=dates,
                    horizon=horizon,
                    n_splits=min(5, len(X) // 30),
                    l1_ratio=0.5,
                    min_features=10,
                )
                elasticnet_cols = en_names
            except Exception as e:
                print(f"[FS] ElasticNet failed in 'best' mode: {e}")
            
            try:
                X_sel, ols_names, ols_mask = selectfeaturesols_pvalues(
                    X, y,
                    featurenames=list(feat_cols),
                    alpha=0.05,
                    topk=50,
                    minfeatures=10,
                )
                ols_cols = ols_names
            except Exception as e:
                print(f"[FS] OLS failed in 'best' mode: {e}")
            
            # Pick ElasticNet if available (more aggressive regularization), else OLS, else all
            if elasticnet_cols:
                selected_cols = elasticnet_cols
                print(f"[FS] Using ElasticNet ({len(elasticnet_cols)} features)")
            elif ols_cols:
                selected_cols = ols_cols
                print(f"[FS] Using OLS ({len(ols_cols)} features)")
            else:
                print(f"[FS] Both methods failed, using all {len(feat_cols)} features")
        
        return selected_cols
    
    except Exception as e:
        print(f"[FS] Feature selection error: {e}")
        return None


def walkforward_cross_sectional(
    tickers,
    period: str = "5y",
    horizon: int = 1,
    model_type: str = "rf",
    train_years: float = 1,
    test_years: float = 0.25,
    top_pct_long: float = 0.15,
    top_pct_short: float = 0.35,
    vix_filter: float | None = None,
    basket_gate: bool = False,
    basket_entry_pct: float = 0.95,
    basket_pct_window: int = 252,
    basket_z_col: str = "retzscore20d",
    basket_mode: str = "gate",
    feature_selection: str = "best",  # "none", "elasticnet", "ols", "best" (compare both, pick winner)
    # NEW IMPROVEMENT PARAMETERS
    enable_threshold_optimization: bool = True,
    enable_volatility_weighting: bool = True,
    enable_position_holding: bool = True,
    enable_kelly_criterion: bool = False,
    position_holding_days: int = 3,
    use_ensemble: bool = False,
    use_classification: bool = False,
) -> pd.DataFrame:
    print(f"[WF] Building panel for {len(tickers)} tickers...")
    panel = build_panel_features_and_target(tickers, period=period, horizon=horizon)

    # Use actual available features from the panel
    # The panel contains all columns created by build_features_and_target
    all_cols = set(panel.columns) - {"target", "ticker"}
    feat_cols = [c for c in all_cols if c != "ticker"]
    
    print(f"[WF] Available features: {len(feat_cols)}")
    if len(feat_cols) == 0:
        raise ValueError("[WF] No features available in panel. Check build_features_and_target output.")
    
    df = panel.dropna(subset=feat_cols + ["target"]).copy()
    
    print(f"[WF] After dropna: {len(df)} rows with {len(feat_cols)} features")

    # INDEX HANDLING
    df_reset = df.reset_index()
    date_col = "Date" if "Date" in df_reset.columns else "index"
    df_reset = df_reset.rename(columns={date_col: "date"})
    df = df_reset.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Basket stress (computed BEFORE VIX filter so percentile rank is based on full history)
    if basket_gate:
        df = add_basket_stress_from_z(
            df,
            z_col=basket_z_col,
            pct_window=basket_pct_window,
            min_periods=max(30, int(basket_pct_window * 0.25)),
            shift_days=1,  # prevents lookahead
        )

    # VIX regime filter (date-level macro => removes whole dates)
    if vix_filter and "vix" in df.columns:
        orig_rows = len(df)
        df = df[df["vix"] < vix_filter].copy()
        print(f"[VIX] Kept {len(df)}/{orig_rows} rows (VIX<{vix_filter})")

    # IMPORTANT: split by UNIQUE DATES (not by raw rows), so dates don't leak across folds
    all_dates = np.array(sorted(df["date"].unique()))
    n_dates = len(all_dates)

    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    print(f"[WF DEBUG] Rows: {len(df)}, Unique dates: {n_dates}, Train days: {train_days}, Test days: {test_days}")

    fold_metrics: list[dict] = []
    start = 0

    while True:
        train_start = start
        train_end = min(start + train_days, n_dates)
        test_start = train_end
        test_end = min(test_start + test_days, n_dates)

        if test_start >= n_dates:
            break

        train_dates = all_dates[train_start:train_end]
        test_dates = all_dates[test_start:test_end]

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()

        if len(train_df) < 30 or len(test_df) < 5:
            start += test_days
            continue

        fold_idx = len(fold_metrics)
        print(f"[WF] Fold {fold_idx}: train_rows={len(train_df)}, test_rows={len(test_df)}, "
              f"train_dates={len(train_dates)}, test_dates={len(test_dates)}")

        # GUARD: If we have 0 features available, skip this fold
        if len(feat_cols) == 0:
            print(f"[WF] Fold {fold_idx}: No features available (feat_cols is empty), skipping fold")
            start += test_days
            continue

        # FEATURE SELECTION (if enabled)
        fold_feat_cols = feat_cols.copy()
        if feature_selection != "none" and len(feat_cols) > 0:
            fold_feat_cols = _select_features_for_fold(
                train_df=train_df,
                feat_cols=feat_cols,
                horizon=horizon,
                selection_mode=feature_selection,
            )
            if fold_feat_cols is not None and len(fold_feat_cols) > 0:
                print(f"[WF] Fold {fold_idx}: Selected {len(fold_feat_cols)}/{len(feat_cols)} features")
            else:
                fold_feat_cols = feat_cols
                print(f"[WF] Fold {fold_idx}: Feature selection failed, using all {len(feat_cols)} features")

        # MODEL TRAINING
        X_train = train_df[fold_feat_cols].fillna(0).values
        y_train = train_df["target"].values

        np.random.seed(42)
        X_train = X_train + np.random.normal(0, 1e-8, X_train.shape)

        # Use ensemble if requested
        if use_ensemble and HAS_IMPROVEMENTS:
            print(f"[WF] Fold {fold_idx}: Using model ensemble (RF + GB + XGB)")
            model = ModelEnsemble(include_xgb=True)
        else:
            model = make_model(model_type, random_state=42)
        
        model.fit(X_train, y_train)

        # PREDICTION
        X_test = test_df[fold_feat_cols].fillna(0).values
        X_test = X_test + np.random.normal(0, 1e-8, X_test.shape)
        y_test = test_df["target"].values
        y_pred = model.predict(X_test)

        print(f"[DEBUG] Pred stats: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")

        test_df = test_df.copy()
        test_df["pred"] = y_pred
        test_df["predicted_return"] = y_pred  # Alias for improvements module
        test_df["actual_return"] = test_df["target"]  # Alias for improvements module
        test_df["rank_pct"] = test_df.groupby("date")["pred"].rank(pct=True, method="average")
        
        # APPLY IMPROVEMENTS if enabled and module available
        if HAS_IMPROVEMENTS and (enable_threshold_optimization or enable_volatility_weighting or 
                                  enable_position_holding or enable_kelly_criterion):
            print(f"[WF] Fold {fold_idx}: Applying improvements (threshold_opt={enable_threshold_optimization}, "
                  f"vol_weight={enable_volatility_weighting}, holding={enable_position_holding}, kelly={enable_kelly_criterion})")
            
            positions_improved, metrics_improved = apply_all_improvements(
                test_df,
                pred_col="predicted_return",
                actual_col="actual_return",
                vol_col="vol_20d" if "vol_20d" in test_df.columns else None,
                enable_threshold_opt=enable_threshold_optimization,
                enable_vol_weighting=enable_volatility_weighting,
                enable_holding=enable_position_holding,
                enable_kelly=enable_kelly_criterion,
                hold_days=position_holding_days,
            )
            print(f"[WF] Fold {fold_idx}: Improved Sharpe={metrics_improved['sharpe']:.3f}, "
                  f"Hit Rate={metrics_improved['hit_rate']:.1%}, Threshold={metrics_improved['threshold']:.4f}")


        # INITIAL VERY LENIENT MASKS JUST TO AVOID EMPTY DAYS
        base_long_mask = test_df["rank_pct"] <= 0.30
        base_short_mask = test_df["rank_pct"] >= 0.70
        print(f"[WF] Base Long: {base_long_mask.sum()}, Base Short: {base_short_mask.sum()}")

        # FINAL MASKS USING USER THRESHOLDS
        long_mask = test_df["rank_pct"] <= top_pct_long
        short_mask = test_df["rank_pct"] >= (1 - top_pct_short)
        print(f"[WF] Final Long: {long_mask.sum()}, Final Short: {short_mask.sum()}")

        # VOL-TARGETED WEIGHTS (15% target vol)
        test_df["vol_weight"] = vol_target_position_size(1.0, test_df["vol_20d"], target_vol=0.15)

        # VOL-WEIGHTED RETURNS
        long_rets = test_df[long_mask].groupby("date", as_index=True).apply(
            lambda x: (x["target"] * x["vol_weight"]).mean(),
            include_groups=False
        )
        short_rets = -test_df[short_mask].groupby("date", as_index=True).apply(
            lambda x: (x["target"] * x["vol_weight"]).mean(),
            include_groups=False
        )

        # Basket gate/throttle applied at the DATE level
        if basket_gate:
            if "basket_D_pct" not in test_df.columns:
                raise ValueError("basket_gate=True but basket_D_pct column is missing. "
                                 "Ensure add_basket_stress_from_z ran and returned basket_D_pct.")

            stress_by_date = test_df.groupby("date")["basket_D_pct"].first()

            if basket_mode == "gate":
                active = (stress_by_date >= basket_entry_pct).fillna(False).astype(float)
                long_rets = long_rets.reindex(active.index).fillna(0.0) * active
                short_rets = short_rets.reindex(active.index).fillna(0.0) * active

            elif basket_mode == "throttle":
                throttle = ((stress_by_date - 0.50) / 0.50).clip(0.0, 1.0).fillna(0.0)
                long_rets = long_rets.reindex(throttle.index).fillna(0.0) * throttle
                short_rets = short_rets.reindex(throttle.index).fillna(0.0) * throttle

            else:
                raise ValueError(f"Unknown basket_mode={basket_mode}. Use 'gate' or 'throttle'.")

        # HANDLE EMPTY SERIES
        if long_rets.empty and short_rets.empty:
            print("[WF] WARNING: empty long/short rets; skipping fold")
            start += test_days
            continue
        if long_rets.empty:
            long_rets = pd.Series(0.0, index=short_rets.index)
        if short_rets.empty:
            short_rets = pd.Series(0.0, index=long_rets.index)

        port_rets = (long_rets + short_rets) / 2
        port_rets = port_rets.dropna()

        print(f"[WF] Portfolio: {len(port_rets)} days (15% target vol)")
        if len(port_rets) == 0:
            start += test_days
            continue

        # STORE PER-TICKER SIGNALS FOR LATEST FOLD (for options overlay)
        per_ticker = (
            test_df.assign(long=long_mask, short=short_mask)
            .groupby("ticker")
            .agg(
                avg_pred=("pred", "mean"),
                any_long=("long", "any"),
                any_short=("short", "any"),
            )
            .reset_index()
        )
        per_ticker["fold"] = fold_idx

        # FOLD METRICS
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_start": str(train_df["date"].min())[:10],
                "train_end": str(train_df["date"].max())[:10],
                "test_start": str(test_df["date"].min())[:10],
                "test_end": str(test_df["date"].max())[:10],
                "test_days": int(len(port_rets)),
                "sharpe": float(sharpe_from_returns(port_rets) or 0.0),
                "ann_return": float(port_rets.mean() * 252),
                "max_dd": float(max_drawdown_from_returns(port_rets) or 0.0),
                "avg_n_long": float(long_mask.sum() / test_df["date"].nunique()),
                "avg_n_short": float(short_mask.sum() / test_df["date"].nunique()),
                "hit_rate": float((np.sign(y_pred) == np.sign(y_test)).mean()),
            }
        )

        start += test_days

    print(f"[WF] Completed {len(fold_metrics)} folds")
    return pd.DataFrame(fold_metrics)


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
    use_arima: bool = True,
    arima_weight: float = 0.3,
):
    if modeltype is not None:
        model_type = modeltype
    X, y, x_last, last_close, last_vol_20d, prob_up_gaf, dates = build_features_and_target(
        ticker=ticker, period=period, horizon=horizon, use_vol_scaled_target=use_vol_scaled_target, run_gaf=run_gaf,
    )
    
    # Dynamically determine available features (don't use hardcoded FEATURE_COLUMNS + MACRO_COLUMNS)
    actual_feat_cols = [f"feat_{i}" for i in range(X.shape[1])]

    n = len(X)

# --- OLS significance selection (train-only to avoid lookahead) ---
    if USE_OLSSIGSELECT:
        trainend = int(n * 0.8)
        Xtrain = X[:trainend]
        ytrain = y[:trainend]

        Xtrain_sel, ols_names, ols_mask = selectfeaturesols_pvalues(
            Xtrain, ytrain,
            featurenames=list(actual_feat_cols),
            alpha=OLSSIG_ALPHA,
            topk=OLSSIG_TOPK,
            minfeatures=OLSSIG_MINFEATURES,
        )

        # Apply same mask to full X + last row
        X = X[:, ols_mask]
        x_last = x_last[ols_mask]
        actual_feat_cols = ols_names

    # Check ElasticNet at RUNTIME (not module import time)
    use_elasticnet_now = is_elasticnet_enabled()
    if use_elasticnet_now:
        try:
            # Re-read config at runtime
            en_l1_ratio = float(os.environ.get("ELASTICNET_L1_RATIO", 0.5))
            en_cv_folds = int(os.environ.get("ELASTICNET_CV_FOLDS", 5))
            en_min_features = int(os.environ.get("ELASTICNET_MINFEATURES", 12))
            
            train_end_for_en = int(n * 0.8)
            X_en_train = X[:train_end_for_en]
            y_en_train = y[:train_end_for_en]

            X_en_train_sel, en_selected_names, en_mask = select_features_elasticnet_timeseries(
                X=X_en_train,
                y=y_en_train,
                feature_names=list(actual_feat_cols),
                dates=dates[:train_end_for_en],
                horizon=horizon,
                n_splits=en_cv_folds,
                l1_ratio=en_l1_ratio,
                min_features=en_min_features,
            )
            
            # SAFEGUARD: Ensure we have at least 1 feature after ElasticNet
            if np.sum(en_mask) == 0 or len(en_selected_names) == 0:
                print(f"⚠️ {ticker} ElasticNet selected 0 features! Falling back to all features.")
                # Don't apply the mask - keep all features
            else:
                X = X[:, en_mask]
                x_last = x_last[en_mask]
                actual_feat_cols = en_selected_names
                print(f"✂️ {ticker} ElasticNet selected {len(actual_feat_cols)} features (enabled via env)")
        except Exception as e:
            print(f"{ticker} ElasticNet selection failed; continuing without it. Error: {e}")
    else:
        print(f"📊 {ticker} Using ALL {X.shape[1]} features (ElasticNet disabled)")

    # SAFEGUARD: Check if we have valid features before proceeding
    if X.shape[1] == 0:
        raise ValueError(f"{ticker}: No features available after preprocessing. Check data quality.")


    # --- XGBoost Feature Importance Selection (optional) ---
    Xtest_for_zscore = None  # Will be set in each branch for z-score calculation
    
    if USE_XGB_FEATURE_SELECTION and model_type == "xgb":
        train_end = int(n * 0.8)
        Xtrain = X[:train_end]
        ytrain = y[:train_end]
        model_init = make_model(model_type=model_type, random_state=42, task="reg")
        model_init.fit(Xtrain, ytrain)
        if hasattr(model_init, "feature_importances_"):
            importances = model_init.feature_importances_
            # Get indices of top N features
            topn = min(XGB_TOP_FEATURES, len(importances))
            top_idx = np.argsort(importances)[::-1][:topn]
            important_mask = np.zeros_like(importances, dtype=bool)
            important_mask[top_idx] = True
            important_features = [actual_feat_cols[i] for i in top_idx]
            print(f"{ticker} XGB feature selection: top {topn} features: {important_features}")
        else:
            important_mask = np.ones(X.shape[1], dtype=bool)
            important_features = actual_feat_cols
        Xtrain_full = X[:train_end][:, important_mask]
        Xtest_for_zscore = X[train_end:][:, important_mask]  # Store test set with same mask for z-score
        ytrain_full = y[:train_end]
        x_last_pruned = x_last[important_mask]
        actual_feat_cols = important_features
    elif auto_optimize:
        train_end = int(n * 0.8)
        Xtrain = X[:train_end]
        ytrain = y[:train_end]
        model_init = make_model(model_type=model_type, random_state=42, task="reg")
        model_init.fit(Xtrain, ytrain)

        if hasattr(model_init, "feature_importances_"):
            importance = model_init.feature_importances_
            important_mask = importance > 0.001
            # Safety check: ensure we keep at least min_features features
            min_features = min(20, X.shape[1])  # Keep at least 20 features
            if np.sum(important_mask) < min_features:
                # If too few features selected, take top N by importance instead
                top_indices = np.argsort(importance)[::-1][:min_features]
                important_mask = np.zeros(X.shape[1], dtype=bool)
                important_mask[top_indices] = True
                print(f"  ⚠️ Too few features above threshold, using top {min_features} by importance")
        else:
            important_mask = np.ones(X.shape[1], dtype=bool)

        important_features = [actual_feat_cols[i] for i in range(len(actual_feat_cols)) if important_mask[i]]
        print(f"{ticker} Using {len(important_features)}/{len(actual_feat_cols)} features for prediction")

        # SAFEGUARD: Ensure we have at least 1 feature
        if len(important_features) == 0:
            print(f"⚠️ {ticker} auto_optimize selected 0 features! Using all available features.")
            important_mask = np.ones(X.shape[1], dtype=bool)
            important_features = list(actual_feat_cols)

        Xtrain_full = X[:train_end][:, important_mask]
        Xtest_for_zscore = X[train_end:][:, important_mask]  # Store test set with same mask for z-score
        ytrain_full = y[:train_end]
        x_last_pruned = x_last[important_mask]
        actual_feat_cols = important_features
    else:
        # No feature pruning - use all features
        split_idx = int(n * 0.8)
        Xtrain_full = X[:split_idx]
        Xtest_for_zscore = X[split_idx:]  # Store test set for z-score
        ytrain_full = y[:split_idx]
        x_last_pruned = x_last
        print(f"📊 {ticker} auto_optimize=OFF, using ALL {X.shape[1]} features")

    # FINAL SAFEGUARD: Ensure training data has features
    if Xtrain_full.shape[1] == 0:
        raise ValueError(f"{ticker}: No features available for training. X.shape={X.shape}")

    model = make_model(model_type=model_type, random_state=42, task="reg")
    model.fit(Xtrain_full, ytrain_full)

    pred_ret = float(model.predict(x_last_pruned.reshape(1, -1))[0])
    if use_vol_scaled_target:
        pred_ret = pred_ret * float(last_vol_20d)

    # === ARIMA ENSEMBLE INTEGRATION ===
    arima_pred = None
    arima_order = None
    ensemble_pred = pred_ret  # Default to ML-only prediction
    ensemble_weights = {"ml": 1.0, "arima": 0.0}  # Default weights
    
    # === ARIMA Volatility & Trend Structure Signals ===
    vol_forecast_result = None
    trend_structure_result = None
    arima_signals = {}
    
    if use_arima:
        try:
            from arima_integration import ARIMAPredictor, VolatilityForecaster, TrendStructureDetector
            
            # Fetch returns series for ARIMA
            hist_for_arima = get_price_history(ticker, period=period, interval="1d")
            if hist_for_arima is not None and not hist_for_arima.empty and "Close" in hist_for_arima.columns:
                # Calculate returns
                returns_series = hist_for_arima["Close"].pct_change().dropna()
                
                if len(returns_series) >= 60:  # Need enough data for ARIMA
                    # --- Original ARIMA on returns (for blending) ---
                    arima_predictor = ARIMAPredictor(max_p=3, max_d=1, max_q=3, verbose=False)
                    
                    if arima_predictor.fit(returns_series):
                        # Get ARIMA forecast for the horizon
                        arima_forecast = arima_predictor.predict(steps=horizon)
                        arima_order = arima_predictor.get_fitted_order()
                        
                        if arima_forecast is not None and len(arima_forecast) > 0:
                            # Sum the forecasted returns for multi-day horizon
                            arima_pred = float(np.sum(arima_forecast))
                            
                            # Blend ML and ARIMA predictions
                            ml_weight = 1.0 - arima_weight
                            ensemble_pred = ml_weight * pred_ret + arima_weight * arima_pred
                            ensemble_weights = {"ml": ml_weight, "arima": arima_weight}
                            
                            print(f"  📈 {ticker} ARIMA{arima_order}: pred={arima_pred:.4f}, "
                                  f"blended={ensemble_pred:.4f} (ML={pred_ret:.4f} @ {ml_weight:.0%})")
                        else:
                            print(f"  ⚠️ {ticker} ARIMA forecast returned None")
                    else:
                        print(f"  ⚠️ {ticker} ARIMA fit failed, using ML-only")
                    
                    # --- NEW: Volatility Forecaster (for options strategies) ---
                    try:
                        vol_forecaster = VolatilityForecaster(lookback=60, vol_window=20, verbose=False)
                        # fit_and_forecast takes PRICE series, not returns
                        price_series = hist_for_arima["Close"]
                        vol_forecast_result = vol_forecaster.fit_and_forecast(price_series, horizon=5)
                        if vol_forecast_result and vol_forecast_result.get("success"):
                            arima_signals["vol_current"] = vol_forecast_result.get("vol_current")
                            arima_signals["vol_forecast"] = vol_forecast_result.get("vol_forecast")
                            arima_signals["vol_direction"] = vol_forecast_result.get("vol_direction")
                            arima_signals["vol_regime"] = vol_forecast_result.get("vol_regime")
                            
                            # Derive options signal from vol direction and regime
                            vol_regime = vol_forecast_result.get("vol_regime", "normal")
                            vol_dir = vol_forecast_result.get("vol_direction", "neutral")
                            if vol_regime in ("low", "normal") and vol_dir == "down":
                                arima_signals["options_signal"] = "sell_vol"  # Vol low/falling, sell premium
                            elif vol_regime in ("high", "extreme") or vol_dir == "up":
                                arima_signals["options_signal"] = "buy_vol"  # Vol high/rising, buy protection
                            else:
                                arima_signals["options_signal"] = "neutral"
                                
                            print(f"  📊 {ticker} Vol: {vol_forecast_result.get('vol_current', 0)*100:.1f}% → "
                                  f"{vol_forecast_result.get('vol_forecast', 0)*100:.1f}% ({vol_forecast_result.get('vol_direction', 'n/a')}) "
                                  f"→ {arima_signals.get('options_signal', 'neutral')}")
                    except Exception as vol_err:
                        print(f"  ⚠️ {ticker} VolatilityForecaster error: {vol_err}")
                    
                    # --- NEW: Trend Structure Detector (for direction confirmation) ---
                    try:
                        trend_detector = TrendStructureDetector(momentum_window=10, smooth_window=5, verbose=False)
                        # analyze_trend takes PRICE series, not momentum
                        price_series = hist_for_arima["Close"]
                        trend_structure_result = trend_detector.analyze_trend(price_series, horizon=5)
                        if trend_structure_result and trend_structure_result.get("success"):
                            arima_signals["trend_direction"] = trend_structure_result.get("trend_direction")
                            arima_signals["trend_strength"] = trend_structure_result.get("trend_strength")
                            arima_signals["has_structure"] = trend_structure_result.get("structure_detected")
                            arima_signals["trend_order"] = trend_structure_result.get("arima_order")
                            
                            # Check for direction confirmation with ML prediction
                            ml_direction = "up" if pred_ret > 0 else "down" if pred_ret < 0 else "neutral"
                            trend_dir = trend_structure_result.get("trend_direction", "neutral")
                            direction_confirmed = (ml_direction == trend_dir) or (trend_dir == "neutral")
                            arima_signals["direction_confirmed"] = direction_confirmed
                            
                            print(f"  🔍 {ticker} Trend: {trend_dir} @ {trend_structure_result.get('trend_strength', 0):.0%} strength, "
                                  f"structure={trend_structure_result.get('structure_detected')}, "
                                  f"ML confirms: {direction_confirmed}")
                    except Exception as trend_err:
                        print(f"  ⚠️ {ticker} TrendStructureDetector error: {trend_err}")
                else:
                    print(f"  ⚠️ {ticker} Not enough data for ARIMA ({len(returns_series)} < 60)")
            else:
                print(f"  ⚠️ {ticker} Could not fetch price history for ARIMA")
        except ImportError:
            print(f"  ⚠️ ARIMA not available (pmdarima not installed)")
        except Exception as arima_err:
            print(f"  ⚠️ {ticker} ARIMA error: {arima_err}")

    # NEW: Calculate confidence score (absolute prediction magnitude)
    # Higher |pred_ret| = higher confidence model has in the prediction
    confidence_score = float(abs(pred_ret))
    
    # Calculate prediction z-score using model's out-of-sample predictions
    # This tells us how unusual this prediction is compared to historical predictions
    pred_zscore = 0.0
    try:
        # FIXED 2026-01-08: Xtest_for_zscore is now pre-computed with correct features
        # in each branch (XGB feature selection, auto_optimize, or plain)
        
        if Xtest_for_zscore is not None and len(Xtest_for_zscore) >= 5:
            # Verify dimensions match model's expectation
            expected_features = Xtrain_full.shape[1]
            if Xtest_for_zscore.shape[1] != expected_features:
                print(f"  ⚠️ Z-score: Feature mismatch ({Xtest_for_zscore.shape[1]} vs {expected_features}), using fallback")
                # Fallback: use training predictions to estimate distribution
                train_preds = model.predict(Xtrain_full)
                pred_mean = float(np.mean(train_preds))
                pred_std = float(np.std(train_preds, ddof=1))
            else:
                test_preds = model.predict(Xtest_for_zscore)
                pred_mean = float(np.mean(test_preds))
                pred_std = float(np.std(test_preds, ddof=1))
            
            # Calculate z-score
            if pred_std > 1e-9:
                pred_zscore = float((pred_ret - pred_mean) / pred_std)
            else:
                # If std is 0, use a simple sign-based z-score
                pred_zscore = 1.0 if pred_ret > pred_mean else -1.0 if pred_ret < pred_mean else 0.0
                print(f"  ⚠️ Z-score: Zero std, using sign-based z-score")
            
            print(f"  📊 {ticker} Z-score: {pred_zscore:.3f} (pred={pred_ret:.4f}, mean={pred_mean:.4f}, std={pred_std:.4f})")
        else:
            # Fallback: use training data to estimate prediction distribution
            print(f"  ⚠️ Z-score: Not enough test data, using training predictions")
            train_preds = model.predict(Xtrain_full)
            pred_mean = float(np.mean(train_preds))
            pred_std = float(np.std(train_preds, ddof=1))
            if pred_std > 1e-9:
                pred_zscore = float((pred_ret - pred_mean) / pred_std)
            else:
                pred_zscore = 1.0 if pred_ret > pred_mean else -1.0 if pred_ret < pred_mean else 0.0
            print(f"  📊 {ticker} Z-score (from train): {pred_zscore:.3f}")
    except Exception as zscore_err:
        print(f"Z-score calculation failed: {zscore_err}")
        pred_zscore = 0.0

    pred_price = float(last_close * (1 + pred_ret))

    prob_up = None
    prob_down = None
    try:
        ydir = (y > 0).astype(int)
        ydir_train = ydir[:len(Xtrain_full)]
        
        # For ensemble model, use RF classifier for probability (ensemble is regressor only)
        clf_model_type = "rf" if model_type == "ensemble" else model_type
        clf = make_model(model_type=clf_model_type, random_state=42, task="clf", log_version=False)
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
    except Exception as prob_err:
        print(f"Probability calculation failed: {prob_err}")
        prob_up = None
        prob_down = None

    fund_feats = get_fundamental_features(ticker)
    pe_ratio = fund_feats.get("fund_pe_trailing", None)

    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(actual_feat_cols, model.feature_importances_))
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
        "pred_zscore": pred_zscore,  # Z-score of prediction vs historical predictions
        "confidence_score": confidence_score,  # Prediction confidence (|prediction magnitude|)
        "pred_next_price": pred_price,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "prob_up_gaf": prob_up_gaf,
        "num_features": len(actual_feat_cols),
        "top_features": top_features_str,
        "elasticnet_enabled": use_elasticnet_now,  # Runtime check, not module-level constant
        "elasticnet_l1_ratio": float(os.environ.get("ELASTICNET_L1_RATIO", 0.5)),
        "elasticnet_cv_folds": int(os.environ.get("ELASTICNET_CV_FOLDS", 5)),
        "elasticnet_selected_n": int(len(actual_feat_cols)) if use_elasticnet_now else None,
        # ARIMA ensemble fields
        "arima_pred": arima_pred,  # ARIMA-only prediction (or None if failed)
        "arima_order": arima_order,  # ARIMA order tuple (p,d,q) or None
        "ensemble_pred": ensemble_pred,  # Blended ML + ARIMA prediction
        "ensemble_weights": ensemble_weights,  # {"ml": 0.7, "arima": 0.3}
        "use_arima": use_arima,  # Whether ARIMA was requested
        # NEW: ARIMA volatility & trend signals for options strategies
        "arima_signals": arima_signals,  # Dict with vol/trend signals
        "vol_current": arima_signals.get("vol_current"),  # Current 20d annualized vol
        "vol_forecast": arima_signals.get("vol_forecast"),  # Forecasted 5d ahead vol
        "vol_direction": arima_signals.get("vol_direction"),  # "up", "down", "neutral"
        "vol_regime": arima_signals.get("vol_regime"),  # "low", "normal", "high"
        "options_signal": arima_signals.get("options_signal"),  # "sell_vol", "buy_vol", "neutral"
        "trend_direction": arima_signals.get("trend_direction"),  # ARIMA trend direction
        "trend_strength": arima_signals.get("trend_strength"),  # 0-100% strength
        "has_structure": arima_signals.get("has_structure"),  # True if ARIMA found structure
        "direction_confirmed": arima_signals.get("direction_confirmed"),  # ML + ARIMA agree
    }



# -------------------- Long-horizon side-car (20–30d) --------------------

def predict_long_horizon_for_ticker(
    ticker: str,
    period: str = "5y",
    k: int = 200,
):
    """Return analog/regime 20–30d view without touching short-horizon model.

    Returns a dict with probabilities/quantiles or an error payload. Gracefully
    falls back across periods and returns None if module unavailable.
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


def track_predictions(ticker, period="1y", model_type="rf", horizon=1):
    try:
        hist = get_price_history(ticker, period=period, interval="1d")
        if hist.empty or len(hist) < 50:
            print(f"[track_predictions] Insufficient data for {ticker}; only {len(hist)} rows")
            return pd.DataFrame(), 0.0

        hist = add_price_features(hist)

        # Try to get macro data, but don't fail if unavailable
        try:
            macro_df = get_macro_df(symbol="^GSPC", period=period)
            hist = hist.join(macro_df, how="left")
        except Exception as e:
            print(f"[track_predictions] Warning: Could not fetch macro data: {e}")

        # Try to get fundamental data, but don't fail if unavailable
        try:
            fund_feats = get_fundamental_features(ticker)
            for k, v in fund_feats.items():
                hist[k] = v
        except Exception as e:
            print(f"[track_predictions] Warning: Could not fetch fundamental data: {e}")

        hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

        # Use only columns that actually exist and have reasonable coverage
        feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
        macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
        feat_cols = feat_cols_available + macro_cols_available

        # Drop features that are mostly missing, then fill remaining gaps to avoid wiping out the dataset
        data_quality = hist[feat_cols].isna().sum() / len(hist)
        feat_cols = [c for c in feat_cols if data_quality[c] < 0.5]

        hist[feat_cols] = hist[feat_cols].fillna(method="ffill").fillna(method="bfill").fillna(0)

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

        if USE_OLSSIGSELECT:
            Xtrain, ols_names, ols_mask = selectfeaturesols_pvalues(
                Xtrain, ytrain,
                featurenames=list(featcols),
                alpha=OLSSIG_ALPHA,
                topk=OLSSIG_TOPK,
                minfeatures=OLSSIG_MINFEATURES,
            )
            Xtest = Xtest[:, ols_mask]
            featcols = ols_names

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

        Xtest = test_df[feat_cols].values
        ytest = test_df["ftarget_ret_horizon_ahead"].values

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

    # Make macro data optional
    try:
        macro_df = get_macro_df(symbol="^GSPC", period=period)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"[backtest_one_ticker] Warning: Could not fetch macro data: {e}")

    # Make fundamental data optional
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[backtest_one_ticker] Warning: Could not fetch fundamental data: {e}")

    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

    # Use actual available features, not hardcoded list
    feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
    macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
    feat_cols = feat_cols_available + macro_cols_available
    
    # Filter by data quality (< 50% NaN)
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    feat_cols = [c for c in feat_cols if data_quality[c] < 0.5]
    
    # Fill remaining NaNs
    hist[feat_cols] = hist[feat_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    cols_needed = feat_cols + ["ftarget_ret_horizon_ahead"]
    df = hist[cols_needed].dropna().copy()

    cutoff_date = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_mask = df.index < cutoff_date
    test_mask = df.index >= cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    Xtrain = train_df[feat_cols].values
    ytrain = train_df["ftarget_ret_horizon_ahead"].values
    Xtest = test_df[feat_cols].values
    ytest = test_df["ftarget_ret_horizon_ahead"].values

    selected_mask = None
    selected_feats = None

    if USE_OLSSIGSELECT:
                Xtrain, ols_names, ols_mask = selectfeaturesols_pvalues(
                    Xtrain, ytrain,
                    featurenames=list(feat_cols),
                    alpha=OLSSIG_ALPHA,
                    topk=OLSSIG_TOPK,
                    minfeatures=OLSSIG_MINFEATURES,
                )
                Xtest = Xtest[:, ols_mask]
                featcols = ols_names

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

    # Make macro data optional
    try:
        macro_df = get_macro_df(symbol="^GSPC", period=period)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"[backtest_one_ticker_auto_optimized] Warning: Could not fetch macro data: {e}")

    # Make fundamental data optional
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[backtest_one_ticker_auto_optimized] Warning: Could not fetch fundamental data: {e}")

    hist["ftarget_ret_horizon_ahead"] = hist["Close"].pct_change(horizon).shift(-horizon)

    # Use actual available features, not hardcoded list
    feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
    macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
    feat_cols = feat_cols_available + macro_cols_available
    
    # Filter by data quality (< 50% NaN)
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    feat_cols = [c for c in feat_cols if data_quality[c] < 0.5]
    
    # Fill remaining NaNs
    hist[feat_cols] = hist[feat_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
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

    if USE_OLSSIGSELECT:
            Xtrain, ols_names, ols_mask = selectfeaturesols_pvalues(
                Xtrain, ytrain,
                featurenames=list(featcols),
                alpha=OLSSIG_ALPHA,
                topk=OLSSIG_TOPK,
                minfeatures=OLSSIG_MINFEATURES,
            )
            Xtest = Xtest[:, ols_mask]
            featcols = ols_names

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
    Xtrain_val_full = train_val_df[feat_cols].values
    ytrain_val = train_val_df["ftarget_ret_horizon_ahead"].values

    if selected_mask is not None:
        Xtrain_val_full = Xtrain_val_full[:, selected_mask]

    feat_to_idx = {f: i for i, f in enumerate(feat_cols)}
    imp_idx = [feat_to_idx[f] for f in important_features if f in feat_to_idx]

    Xtrain_val = Xtrain_val_full[:, imp_idx]

    model_final = make_model(model_type=model_type, random_state=42)
    model_final.fit(Xtrain_val, ytrain_val)

    Xtest_full = test_df[feat_cols].values
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
    # Log model version at start of backtest
    try:
        from src.config import log_model_version
        print(f"📈 Starting walk-forward backtest: {ticker}")
        print(f"   Model: {log_model_version(model_type)}")
    except ImportError:
        print(f"📈 Starting walk-forward backtest: {ticker}, model={model_type}")
    
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        return []

    hist = add_price_features(hist)
    
    # Make macro data optional
    try:
        macro_df = get_macro_df(symbol="^GSPC", period=period)
        # Normalize both indices to tz-naive to avoid join errors
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"[walk_forward_backtest] Warning: Could not fetch macro data: {e}")

    # Make fundamental data optional
    try:
        fund_feats = get_fundamental_features(ticker)
        for k, v in fund_feats.items():
            hist[k] = v
    except Exception as e:
        print(f"[walk_forward_backtest] Warning: Could not fetch fundamental data: {e}")

    target_col = "ftarget_ret_horizon_ahead"
    hist[target_col] = hist["Close"].pct_change(horizon).shift(-horizon)

    # Use actual available features, not hardcoded list
    feat_cols_available = [c for c in FEATURE_COLUMNS if c in hist.columns]
    macro_cols_available = [c for c in MACRO_COLUMNS if c in hist.columns]
    feat_cols = feat_cols_available + macro_cols_available
    
    # Filter by data quality (< 50% NaN)
    data_quality = hist[feat_cols].isna().sum() / len(hist)
    feat_cols = [c for c in feat_cols if data_quality[c] < 0.5]
    
    # Fill remaining NaNs
    hist[feat_cols] = hist[feat_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
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


        # --- OLS significance selection ---
        if USE_OLSSIGSELECT:
            Xtrain, ols_names, ols_mask = selectfeaturesols_pvalues(
                Xtrain, ytrain,
                featurenames=list(featcols),
                alpha=OLSSIG_ALPHA,
                topk=OLSSIG_TOPK,
                minfeatures=OLSSIG_MINFEATURES,
            )
            Xtest = Xtest[:, ols_mask]
            featcols = ols_names

        # --- ElasticNet selection ---
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

        # --- XGBoost feature importance selection (optional) ---
        if USE_XGB_FEATURE_SELECTION and model_type == "xgb":
            model_init = make_model(model_type=model_type, random_state=42)
            model_init.fit(Xtrain, ytrain)
            if hasattr(model_init, "feature_importances_"):
                importances = model_init.feature_importances_
                topn = min(XGB_TOP_FEATURES, len(importances))
                top_idx = np.argsort(importances)[::-1][:topn]
                important_mask = np.zeros_like(importances, dtype=bool)
                important_mask[top_idx] = True
                important_features = [featcols[i] for i in top_idx]
                print(f"{ticker} WF fold XGB top {topn} features: {important_features}")
            else:
                important_mask = np.ones(Xtrain.shape[1], dtype=bool)
                important_features = featcols
            Xtrain = Xtrain[:, important_mask]
            Xtest = Xtest[:, important_mask]
            featcols = important_features

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
