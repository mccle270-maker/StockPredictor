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
# MODEL VERSIONS - LOCKED CONFIGURATIONS
# ============================================================================
# These are tested, validated configurations. DO NOT MODIFY without versioning.
# To add a new version, create a new entry (e.g., "xgb_regularized_v2")

MODEL_VERSIONS = {
    # XGBoost Regularized V1 - Baseline (Locked 2026-01-05)
    # Backtest results: Sharpe 0.75, Win Rate 55%, Return +22.7%
    # Reduces overfitting: train/test accuracy gap from 50% to 3%
    "xgb_regularized_v1": {
        "model_type": "xgb",
        "version": "v1",
        "created": "2026-01-05",
        "status": "stable",  # stable | experimental | deprecated
        "params": {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "min_child_weight": 100,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.7,
            "random_state": 42,
        },
        "metrics": {
            "avg_sharpe": 0.75,
            "avg_win_rate": 0.55,
            "avg_return": 0.227,
            "train_test_gap": 0.031,
        },
    },
    
    # RandomForest Default - Baseline
    "rf_default_v1": {
        "model_type": "rf",
        "version": "v1",
        "created": "2026-01-05",
        "status": "stable",
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "min_samples_leaf": 50,
            "random_state": 42,
        },
        "metrics": {},
    },
    
    # GradientBoosting Default - Baseline
    "gbrt_default_v1": {
        "model_type": "gbrt",
        "version": "v1",
        "created": "2026-01-05",
        "status": "stable",
        "params": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
        },
        "metrics": {},
    },
}

# Current active model versions (change these to switch versions)
ACTIVE_MODEL_VERSIONS = {
    "xgb": "xgb_regularized_v1",
    "rf": "rf_default_v1",
    "gbrt": "gbrt_default_v1",
}

def get_model_config(model_type: str) -> dict:
    """
    Get the LOCKED configuration for a model type.
    Returns params from the active versioned config.
    """
    version_key = ACTIVE_MODEL_VERSIONS.get(model_type)
    if version_key and version_key in MODEL_VERSIONS:
        return MODEL_VERSIONS[version_key]["params"].copy()
    # Fallback to MODEL_DEFAULTS
    return MODEL_DEFAULTS.get(model_type, {}).copy()

def get_model_version_info(model_type: str) -> dict:
    """Get full version info including metrics for logging."""
    version_key = ACTIVE_MODEL_VERSIONS.get(model_type)
    if version_key and version_key in MODEL_VERSIONS:
        return MODEL_VERSIONS[version_key].copy()
    return {"model_type": model_type, "version": "default", "status": "unversioned"}

def log_model_version(model_type: str) -> str:
    """Return a log string for the model version being used."""
    info = get_model_version_info(model_type)
    version_key = ACTIVE_MODEL_VERSIONS.get(model_type, "default")
    return f"[{version_key}] {model_type.upper()} {info.get('version', '?')} ({info.get('status', 'unknown')})"

# ============================================================================
# MODEL PARAMETERS (derived from versioned configs)
# ============================================================================
# NOTE: These are now populated from MODEL_VERSIONS for consistency
MODEL_DEFAULTS = {
    "rf": get_model_config("rf"),
    "xgb": get_model_config("xgb"),
    "gbrt": get_model_config("gbrt"),
}

# ============================================================================
# TICKER ELIGIBILITY FILTERING
# ============================================================================
# Based on walk-forward backtest results from xgb_regularized_v1
# Only tickers meeting these thresholds are eligible for auto-trading

TICKER_ELIGIBILITY_THRESHOLDS = {
    "min_sharpe": 0.75,      # Minimum average Sharpe ratio
    "min_hitrate": 0.55,     # Minimum hit rate (55%)
}

# Walk-forward results from xgb_regularized_v1 baseline (2026-01-05)
# These metrics determine which tickers are eligible for trading
TICKER_WALKFORWARD_METRICS = {
    "AAPL": {"avg_sharpe": 1.49, "avg_hitrate": 0.560, "status": "eligible"},
    "MSFT": {"avg_sharpe": 1.39, "avg_hitrate": 0.571, "status": "eligible"},
    "SPY":  {"avg_sharpe": 2.04, "avg_hitrate": 0.587, "status": "eligible"},
    "NVDA": {"avg_sharpe": -1.48, "avg_hitrate": 0.484, "status": "disabled"},
    "GOOGL": {"avg_sharpe": -0.82, "avg_hitrate": 0.476, "status": "disabled"},
    "AMZN": {"avg_sharpe": -1.22, "avg_hitrate": 0.468, "status": "disabled"},
    "META": {"avg_sharpe": -0.51, "avg_hitrate": 0.444, "status": "disabled"},
    "QQQ":  {"avg_sharpe": -1.75, "avg_hitrate": 0.508, "status": "disabled"},
    "TSLA": {"avg_sharpe": -0.04, "avg_hitrate": 0.456, "status": "disabled"},
    "AMD":  {"avg_sharpe": -1.92, "avg_hitrate": 0.472, "status": "disabled"},
}

def is_ticker_eligible(ticker: str) -> tuple[bool, str]:
    """
    Check if a ticker is eligible for auto-trading based on walk-forward metrics.
    
    Returns:
        (is_eligible, reason) tuple
    """
    ticker = ticker.upper().strip()
    thresholds = TICKER_ELIGIBILITY_THRESHOLDS
    
    # Check if we have metrics for this ticker
    if ticker not in TICKER_WALKFORWARD_METRICS:
        # Unknown ticker - allow with warning (will be evaluated in production)
        return True, "no_metrics_available"
    
    metrics = TICKER_WALKFORWARD_METRICS[ticker]
    sharpe = metrics.get("avg_sharpe", 0)
    hitrate = metrics.get("avg_hitrate", 0)
    
    # Check thresholds
    if sharpe < thresholds["min_sharpe"]:
        return False, f"sharpe={sharpe:.2f} < {thresholds['min_sharpe']}"
    
    if hitrate < thresholds["min_hitrate"]:
        return False, f"hitrate={hitrate*100:.1f}% < {thresholds['min_hitrate']*100:.0f}%"
    
    return True, "meets_all_thresholds"

def get_eligible_tickers() -> list[str]:
    """Return list of tickers that meet eligibility thresholds."""
    return [tk for tk, m in TICKER_WALKFORWARD_METRICS.items() 
            if m.get("status") == "eligible"]

def get_disabled_tickers() -> list[str]:
    """Return list of tickers that are disabled due to poor metrics."""
    return [tk for tk, m in TICKER_WALKFORWARD_METRICS.items() 
            if m.get("status") == "disabled"]

def log_ticker_eligibility(ticker: str) -> str:
    """Return a log string for ticker eligibility status."""
    eligible, reason = is_ticker_eligible(ticker)
    status = "✅ ELIGIBLE" if eligible else "❌ DISABLED"
    return f"[{ticker}] {status}: {reason}"

# ============================================================================
# Z-SCORE GATING CONFIGURATION
# ============================================================================
# Trade only when predictions are statistically significant
# NOTE: hard_filter=False means we TAG weak signals but DO NOT exclude them
#       Set to True to actually filter out weak signals from trading
ZSCORE_GATING_CONFIG = {
    "min_zscore": 1.0,          # Minimum |z-score| to trade (1.0 = 1 std dev)
    "rolling_window": 20,       # Days for rolling mean/std calculation
    "min_data_points": 10,      # Minimum data points required for z-score
    "boost_threshold": 2.0,     # Z-score above which to boost position size
    "hard_filter": False,       # If True, weak signals excluded; if False, logged only
    "log_weak_signals": True,   # Log weak signals for analysis
    "weak_signal_log": "weak_signals.jsonl",  # Filename in CACHE_DIR
}

# Per-ticker z-score overrides (optional)
# Use this to set different thresholds for specific tickers
ZSCORE_TICKER_OVERRIDES = {
    # "SPY": {"min_zscore": 0.8},  # Example: lower threshold for SPY
    # "TSLA": {"min_zscore": 1.5},  # Example: higher threshold for volatile stocks
}

def get_zscore_threshold(ticker: str) -> float:
    """
    Get z-score threshold for a specific ticker.
    Checks ZSCORE_TICKER_OVERRIDES first, then falls back to default.
    """
    ticker = ticker.upper().strip()
    override = ZSCORE_TICKER_OVERRIDES.get(ticker, {})
    return override.get("min_zscore", ZSCORE_GATING_CONFIG["min_zscore"])

def is_zscore_hard_filter_enabled() -> bool:
    """Check if hard z-score filtering is enabled."""
    return ZSCORE_GATING_CONFIG.get("hard_filter", False)

def get_zscore_log_path() -> Path:
    """Get path for weak signals log file."""
    log_file = ZSCORE_GATING_CONFIG.get("weak_signal_log", "weak_signals.jsonl")
    return CACHE_DIR / log_file

# ============================================================================
# TRADE LIMITING CONFIGURATION
# ============================================================================
# Limit number of trades per ticker per period to avoid overtrading
TRADE_LIMIT_CONFIG = {
    "enabled": True,                    # Enable trade limiting
    "max_trades_per_ticker": 1,         # Max trades per ticker per period
    "period": "day",                    # "day", "week", or "session"
    "ranking_method": "zscore",         # "zscore", "confidence", or "return"
    "include_skipped_in_output": True,  # Include skipped signals in output with flag
    "log_skipped_signals": True,        # Log skipped signals for analysis
    "skipped_log": "skipped_signals.jsonl",  # Filename in CACHE_DIR
}

# Per-ticker trade limit overrides
TRADE_LIMIT_OVERRIDES = {
    # "SPY": {"max_trades_per_ticker": 2},  # Allow more trades for SPY
    # "TSLA": {"max_trades_per_ticker": 0},  # Disable trading for TSLA
}

def get_trade_limit(ticker: str) -> int:
    """Get max trades per period for a specific ticker."""
    ticker = ticker.upper().strip()
    override = TRADE_LIMIT_OVERRIDES.get(ticker, {})
    return override.get("max_trades_per_ticker", TRADE_LIMIT_CONFIG["max_trades_per_ticker"])

def is_trade_limiting_enabled() -> bool:
    """Check if trade limiting is enabled."""
    return TRADE_LIMIT_CONFIG.get("enabled", True)

def get_skipped_log_path() -> Path:
    """Get path for skipped signals log file."""
    log_file = TRADE_LIMIT_CONFIG.get("skipped_log", "skipped_signals.jsonl")
    return CACHE_DIR / log_file

# ============================================================================
# MARKET REGIME FILTER
# ============================================================================
# Filter trades based on market regime (SPY 200DMA, VIX, etc.)
REGIME_FILTER_CONFIG = {
    "enabled": True,                    # Enable regime-based filtering
    "spy_dma_period": 200,              # SPY moving average period
    "vix_high_threshold": 25.0,         # VIX level considered "high"
    "vix_extreme_threshold": 35.0,      # VIX level for CRASH regime
    "rsi_oversold": 30.0,               # RSI oversold level
    "rsi_overbought": 70.0,             # RSI overbought level
    "min_conviction_override": 2.0,     # Min |z-score| to override regime block
    "log_blocked_trades": True,         # Log blocked trades for analysis
    "blocked_log": "blocked_trades.jsonl",  # Filename in CACHE_DIR
}

def is_regime_filter_enabled() -> bool:
    """Check if regime filtering is enabled."""
    return REGIME_FILTER_CONFIG.get("enabled", True)

def get_blocked_trades_log_path() -> Path:
    """Get path for blocked trades log file."""
    log_file = REGIME_FILTER_CONFIG.get("blocked_log", "blocked_trades.jsonl")
    return CACHE_DIR / log_file

# ============================================================================
# VOLATILITY-SCALED POSITION SIZING
# ============================================================================
# Scale positions to target consistent daily volatility exposure
POSITION_SIZING_CONFIG = {
    "target_daily_vol": 0.01,       # Target 1% daily volatility per position
    "vol_lookback_days": 20,        # Rolling window for volatility calculation
    "max_leverage": 2.0,            # Maximum leverage per ticker (2x)
    "min_position_pct": 0.25,       # Minimum position as % of base (25%)
    "use_atr": False,               # Use ATR instead of std dev (False = use std dev)
    "atr_period": 14,               # ATR period if use_atr=True
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
