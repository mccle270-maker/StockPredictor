"""
Centralized configuration for Stock Predictor.
All constants, presets, defaults, and environment settings live here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / ".cache"
MACRO_CACHE_DIR = CACHE_DIR / "macro"
SIGNALS_PATH = BASE_DIR / "signals.json"
GAF_MODEL_PATH = BASE_DIR / "gaf_cnn_updown.keras"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API KEYS (from environment or secrets)
# ============================================================================
def get_api_key(name: str, default: str = "") -> str:
    """Get API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)

FRED_API_KEY = get_api_key("FRED_API_KEY")
FMP_API_KEY = get_api_key("FMP_API_KEY")
MARKETAUX_API_KEY = get_api_key("MARKETAUX_API_KEY")
ALPHAVANTAGE_API_KEY = get_api_key("ALPHAVANTAGE_API_KEY")
APCA_API_KEY_ID = get_api_key("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = get_api_key("APCA_API_SECRET_KEY")

# ============================================================================
# TRADING CONSTANTS
# ============================================================================
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05
DEFAULT_HORIZON = 5
DEFAULT_PERIOD = "5y"
DEFAULT_MODEL_TYPE = "rf"

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
MODEL_DEFAULTS = {
    "rf": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_leaf": 50,
        "random_state": 42,
    },
    "xgb": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    },
    "gbrt": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    },
}

# Heston model parameters for specific tickers
HESTON_PARAMS = {
    "AAPL": {"v0": 0.04, "theta": 0.04, "kappa": 1.5, "sigma": 0.3, "rho": -0.6},
    "NVDA": {"v0": 0.06, "theta": 0.05, "kappa": 1.2, "sigma": 0.5, "rho": -0.7},
}

# ============================================================================
# FEATURE COLUMNS
# ============================================================================
PRICE_FEATURE_COLUMNS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "cumret_3d", "cumret_5d", "cumret_10d",
    "vol_10d", "vol_20d", "vol_60d", "vol_ratio_10_60",
    "atr_14", "atr_pct_14",
]

TECHNICAL_COLUMNS = [
    "rsi14", "macd", "macdsignal", "macdhist",
    "bb_pctb", "bb_width", "mfi14", "adx_14",
    "close_position", "daily_range", "high_low_ratio",
]

GBM_COLUMNS = [
    "gbm_mu_60d", "gbm_sig_60d",
    "gbm_prob_up_1d", "gbm_exp_ret_1d", "gbm_p05_ret_1d", "gbm_p95_ret_1d",
    "gbm_prob_up_5d", "gbm_exp_ret_5d", "gbm_p05_ret_5d", "gbm_p95_ret_5d",
]

MACRO_COLUMNS = [
    "mkt_ret_1d", "vix", "t10y", "term_spread",
    "unrate", "cpi", "oas", "fed_funds",
]

REGIME_COLUMNS = [
    "regime_bull", "regime_bear",
    "regime_vix_low", "regime_vix_medium", "regime_vix_high",
    "bull_streak", "bear_streak",
]

FUNDAMENTAL_COLUMNS = [
    "fund_pe_trailing", "fund_pb", "fund_marketcap",
]

RELATIVE_STRENGTH_COLUMNS = [
    "rel_strength_1d", "rel_momentum_5d",
    "beta_60_spx", "corr_20_spx",
]

# All features combined
FEATURE_COLUMNS = (
    PRICE_FEATURE_COLUMNS +
    TECHNICAL_COLUMNS +
    GBM_COLUMNS +
    MACRO_COLUMNS +
    REGIME_COLUMNS +
    FUNDAMENTAL_COLUMNS +
    RELATIVE_STRENGTH_COLUMNS
)

# ============================================================================
# FRED SERIES IDS
# ============================================================================
FRED_SERIES = {
    "vix": "VIXCLS",
    "t10y": "DGS10",
    "t3m": "DGS3MO",
    "unrate": "UNRATE",
    "cpi": "CPIAUCSL",
    "oas": "BAMLH0A0HYM2",
    "fed_funds": "FEDFUNDS",
}

# ============================================================================
# EXECUTION PRESETS
# ============================================================================
@dataclass
class ExecutionModel:
    """Trading execution parameters."""
    delay_days: int = 1
    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    fee_bps: float = 0.0
    
    @property
    def total_cost_bps(self) -> float:
        return self.half_spread_bps + self.slippage_bps + self.fee_bps

FRICTION_PRESETS = {
    "Default": ExecutionModel(delay_days=1, half_spread_bps=2.0, slippage_bps=3.0, fee_bps=0.0),
    "Loose (optimistic)": ExecutionModel(delay_days=0, half_spread_bps=1.0, slippage_bps=1.0, fee_bps=0.0),
    "Strict (pessimistic)": ExecutionModel(delay_days=1, half_spread_bps=5.0, slippage_bps=8.0, fee_bps=1.0),
}

# ============================================================================
# OPTIONS PRESETS
# ============================================================================
@dataclass
class OptionsConfig:
    """Options trading parameters."""
    dte_min: int = 3
    dte_max: int = 45
    width_pct: float = 0.05
    prefer_spreads: bool = True
    max_strike: float = 500.0
    max_premium: float = 500.0

OPTIONS_PRESETS = {
    "Default": OptionsConfig(dte_min=3, dte_max=45, width_pct=0.05, prefer_spreads=True),
    "Loose": OptionsConfig(dte_min=3, dte_max=90, width_pct=0.10, prefer_spreads=True),
    "Strict": OptionsConfig(dte_min=3, dte_max=21, width_pct=0.03, prefer_spreads=True),
}

# ============================================================================
# WALK-FORWARD PRESETS
# ============================================================================
@dataclass
class WalkForwardConfig:
    """Walk-forward backtest parameters."""
    train_years: float = 2.0
    test_years: float = 0.15
    step_days: int = 21
    threshold: float = 0.002
    
    @property
    def description(self) -> str:
        return f"{self.train_years}y train, ~{int(self.test_years * 252)}d test"

WALKFORWARD_PRESETS = {
    "Conservative (Anti-Overfit)": WalkForwardConfig(train_years=2.0, test_years=0.15),
    "Balanced": WalkForwardConfig(train_years=1.5, test_years=0.2),
    "Aggressive (More Data)": WalkForwardConfig(train_years=1.0, test_years=0.1),
}

# ============================================================================
# QUICK UNIVERSE SELECTIONS
# ============================================================================
UNIVERSE_PRESETS = {
    "Top 10": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "WMT"],
    "Mag 7": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"],
    "Tech": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ADBE"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO"],
}

# ============================================================================
# SIGNAL THRESHOLDS
# ============================================================================
SIGNAL_THRESHOLDS = {
    "strong_buy": {"min_return": 0.012, "min_prob": 0.58},
    "buy": {"min_return": 0.003, "min_prob": 0.52},
    "strong_sell": {"max_return": -0.012, "max_prob": 0.42},
    "sell": {"max_return": -0.003, "max_prob": 0.48},
}

# Horizon multipliers for options strategy thresholds
HORIZON_MULTIPLIERS = {1: 1.0, 2: 1.4, 3: 1.7, 4: 2.0, 5: 2.3}

# ============================================================================
# CANDIDATE FILTER DEFAULTS
# ============================================================================
@dataclass
class CandidateFilters:
    """Stock screening filter parameters."""
    max_tickers: int = 10
    min_recent_return_pct: float = 3.0
    min_vol_spike: float = 1.5
    min_predicted_move_pct: float = 1.0
    min_iv: float = 0.20
    max_iv: float = 0.80
    exclude_disagree: bool = True

# ============================================================================
# CACHE TTLs (in seconds)
# ============================================================================
CACHE_TTL = {
    "intraday": 30,
    "daily": 10 * 60,
    "prediction": 15 * 60,
    "backtest": 30 * 60,
    "macro": 60 * 60,
}

# ============================================================================
# ENVIRONMENT HELPERS
# ============================================================================
def env_bool(name: str, default: bool = False) -> bool:
    """Read boolean from environment variable."""
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default

def env_float(name: str, default: float) -> float:
    """Read float from environment variable."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def env_int(name: str, default: int) -> int:
    """Read int from environment variable."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

# Feature selection settings (from env)
USE_ELASTICNET_SELECT = env_bool("USE_ELASTICNET_SELECT", True)  # Default enabled per original
ELASTICNET_L1_RATIO = env_float("ELASTICNET_L1_RATIO", 0.5)
ELASTICNET_CV_FOLDS = env_int("ELASTICNET_CV_FOLDS", 5)
ELASTICNET_MIN_FEATURES = env_int("ELASTICNET_MINFEATURES", 12)

# OLS significance selection
USE_OLSSIGSELECT = env_bool("USE_OLSSIGSELECT", False)
OLSSIG_ALPHA = env_float("OLSSIG_ALPHA", 0.05)
OLSSIG_TOPK = env_int("OLSSIG_TOPK", 50)
OLSSIG_MINFEATURES = env_int("OLSSIG_MINFEATURES", 8)

# Aliases for backward compatibility
USE_OLS_SIG_SELECT = USE_OLSSIGSELECT
OLS_SIG_ALPHA = OLSSIG_ALPHA
OLS_SIG_TOPK = OLSSIG_TOPK
OLS_SIG_MIN_FEATURES = OLSSIG_MINFEATURES

# ============================================================================
# NON-US STOCK MARKERS
# ============================================================================
NON_US_EXCHANGE_SUFFIXES = [".AX", ".L", ".TO", ".V", ".NZ", ".AS", ".KL", ".SG", ".HK"]

def is_us_tradeable(ticker: str) -> bool:
    """Check if symbol is tradeable on US exchanges (Alpaca)."""
    ticker_clean = str(ticker).upper().strip()
    if "." in ticker_clean:
        return not any(ticker_clean.endswith(suffix) for suffix in NON_US_EXCHANGE_SUFFIXES)
    return True
