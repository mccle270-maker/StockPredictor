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

# New Data Providers (Added 2026-01-07)
TIINGO_API_KEY = get_api_key("TIINGO_API_KEY")
FINNHUB_API_KEY = get_api_key("FINNHUB_API_KEY")
FINNHUB_SECRET = get_api_key("FINNHUB_SECRET")

# ============================================================================
# TRADING CONSTANTS
# ============================================================================
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.05
DEFAULT_HORIZON = 5
DEFAULT_PERIOD = "5y"
DEFAULT_MODEL_TYPE = "rf"

# ============================================================================
# TRADING STRATEGIES (Signal Filters)
# ============================================================================
# Tested on 512 holdout days (2026-01-08)
# Higher filters = Higher accuracy but fewer trades
# 
# Strategy comparison (5 tickers, 3 years data):
#   - Baseline:           Sharpe +1.55, Accuracy 54.5%, 512 days active
#   - Z>1.5 OR RSI:       Sharpe +1.94, Accuracy 58.0%, 205 days active  <- BEST SHARPE
#   - RSI Extreme:        Sharpe +1.84, Accuracy 59.3%, 140 days active
#   - Z>1.5 AND RSI:      Sharpe +1.30, Accuracy 64.7%, 34 days active   <- BEST ACCURACY

TRADING_STRATEGIES = {
    # Baseline - No filtering, trade on all model predictions
    # Sharpe: +1.55 | Accuracy: 54.5% | Active: 100%
    "baseline": {
        "name": "Baseline (No Filter)",
        "description": "Trade on all model predictions. Simple and robust.",
        "filters": {},
        "metrics": {"sharpe": 1.55, "accuracy": 0.545, "active_pct": 1.0},
    },
    
    # RSI Extreme - Mean reversion on oversold/overbought
    # Sharpe: +1.84 | Accuracy: 59.3% | Active: 27%
    "rsi_extreme": {
        "name": "RSI Extreme",
        "description": "Trade only when RSI < 30 (oversold) or RSI > 70 (overbought).",
        "filters": {"rsi_low": 30, "rsi_high": 70},
        "metrics": {"sharpe": 1.84, "accuracy": 0.593, "active_pct": 0.27},
    },
    
    # Z-Score High Conviction - Momentum extremes
    # Sharpe: +1.44 | Accuracy: 58.6% | Active: 19%
    "zscore_high": {
        "name": "Z-Score > 1.5",
        "description": "Trade only when momentum z-score exceeds 1.5.",
        "filters": {"zscore_threshold": 1.5},
        "metrics": {"sharpe": 1.44, "accuracy": 0.586, "active_pct": 0.19},
    },
    
    # Combined OR - Best Sharpe (Z>1.5 OR RSI Extreme)
    # Sharpe: +1.94 | Accuracy: 58.0% | Active: 40%
    "combined_or": {
        "name": "Z>1.5 OR RSI Extreme",
        "description": "Trade when z-score > 1.5 OR RSI is extreme. Best Sharpe ratio.",
        "filters": {"zscore_threshold": 1.5, "rsi_low": 30, "rsi_high": 70, "mode": "OR"},
        "metrics": {"sharpe": 1.94, "accuracy": 0.580, "active_pct": 0.40},
    },
    
    # Combined AND - Highest Accuracy (Z>1.5 AND RSI Extreme)
    # Sharpe: +1.30 | Accuracy: 64.7% | Active: 7%
    "combined_and": {
        "name": "Z>1.5 AND RSI Extreme",
        "description": "Trade only when BOTH z-score > 1.5 AND RSI is extreme. Highest accuracy but few trades.",
        "filters": {"zscore_threshold": 1.5, "rsi_low": 30, "rsi_high": 70, "mode": "AND"},
        "metrics": {"sharpe": 1.30, "accuracy": 0.647, "active_pct": 0.07},
    },
}

# Default strategy for production
DEFAULT_TRADING_STRATEGY = "baseline"

def get_trading_strategy(name: str = None) -> dict:
    """Get trading strategy configuration."""
    if name is None:
        name = DEFAULT_TRADING_STRATEGY
    return TRADING_STRATEGIES.get(name, TRADING_STRATEGIES["baseline"])

def get_strategy_names() -> list:
    """Get list of available strategy names for UI dropdown."""
    return list(TRADING_STRATEGIES.keys())

def get_strategy_display_names() -> dict:
    """Get mapping of strategy keys to display names for UI."""
    return {k: v["name"] for k, v in TRADING_STRATEGIES.items()}

# ============================================================================
# MODEL VERSIONS - LOCKED CONFIGURATIONS
# ============================================================================
# These are tested, validated configurations. DO NOT MODIFY without versioning.
# To add a new version, create a new entry (e.g., "xgb_regularized_v2")

MODEL_VERSIONS = {
    # XGBoost Regularized V4 - PRODUCTION (2026-01-08)
    # Balanced regularization - prevents overfitting while maintaining prediction variance
    # Test Acc: 51.6%, Gap: 8.8%, Has prediction variance for z-score calculation
    # See session notes from 2026-01-08
    "xgb_regularized_v4": {
        "model_type": "xgb",
        "version": "v4",
        "created": "2026-01-08",
        "status": "production",  # Active
        "params": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "min_child_weight": 30,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "colsample_bytree": 0.6,
            "random_state": 42,
        },
        "metrics": {
            "avg_accuracy": 0.516,
            "train_test_gap": 0.088,
            "has_variance": True,  # Important for z-score
        },
    },
    
    # XGBoost Regularized V3 - DEPRECATED (too regularized, constant predictions)
    # Test Sharpe: 0.84, Accuracy: 53.2%, but predictions are constant (std=0)
    # Z-score calculation fails due to zero variance
    "xgb_regularized_v3": {
        "model_type": "xgb",
        "version": "v3",
        "created": "2026-01-08",
        "status": "deprecated",  # Constant predictions - z-score broken
        "params": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.01,
            "subsample": 0.6,
            "min_child_weight": 50,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "colsample_bytree": 0.5,
            "random_state": 42,
        },
        "metrics": {
            "avg_sharpe": 0.84,
            "avg_accuracy": 0.532,
            "train_test_gap": -0.03,  # Nearly zero overfitting!
        },
    },
    
    # XGBoost Optimized V2 - From Model Improvement Pipeline (2026-01-07)
    # 50 Optuna trials: Sharpe +2.77, Accuracy 56.1%
    # WARNING: SEVERELY OVERFIT - train/test gap of 16+
    # See experiments/hyperparameter_optimization_report.json
    "xgb_optimized_v2": {
        "model_type": "xgb",
        "version": "v2",
        "created": "2026-01-07",
        "status": "deprecated",  # OVERFIT - train/test gap 16.22
        "params": {
            "n_estimators": 450,
            "max_depth": 7,
            "learning_rate": 0.048,
            "subsample": 0.998,
            "min_child_weight": 19,
            "reg_alpha": 0.012,
            "reg_lambda": 9.3,
            "colsample_bytree": 0.67,
            "random_state": 42,
        },
        "metrics": {
            "avg_sharpe": 2.77,
            "avg_accuracy": 0.561,
            "improvement_pct": 127.0,
            "train_test_gap": 16.22,  # SEVERE OVERFITTING
        },
    },
    
    # XGBoost Regularized V1 - Previous Baseline (Locked 2026-01-05)
    # Backtest results: Sharpe 0.75, Win Rate 55%, Return +22.7%
    # Reduces overfitting: train/test accuracy gap from 50% to 3%
    "xgb_regularized_v1": {
        "model_type": "xgb",
        "version": "v1",
        "created": "2026-01-05",
        "status": "deprecated",  # Replaced by xgb_optimized_v2
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
    
    # XGBoost Heavy Regularized V5 - TESTING (2026-01-08)
    # Dramatically outperformed all others on out-of-sample holdout:
    # Test Sharpe: 1.00 (vs -0.82 for Prod v4)
    # Test Accuracy: 54.4%
    # Sharpe Gap: 0.13 (virtually zero overfitting!)
    # Test Total Return: +591.7%
    # Uses only 10 core features for robustness
    # 
    # CRITICAL: gamma MUST be 0.0 (or omitted) for z-score to work!
    # Any gamma > 0 causes predictions to collapse to constant values.
    # 
    # reg_lambda/reg_alpha MUST stay below 7.0/0.7 threshold for z-score
    # Heavy reg (8.0/0.8) causes std < 0.0001 which breaks z-score
    # See experiments/final_model_comparison.py
    "xgb_heavy_reg_v5": {
        "model_type": "xgb",
        "version": "v5",
        "created": "2026-01-08",
        "status": "testing",  # Maximum regularization that still allows z-score
        "params": {
            "n_estimators": 75,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "min_child_weight": 30,
            "reg_alpha": 0.7,            # Max safe value (0.8 breaks z-score)
            "reg_lambda": 7.0,           # Max safe value (8.0 breaks z-score)
            # gamma MUST BE 0 - omitted uses default 0
            "random_state": 42,
        },
        "features": [
            # 10 core robust features
            "ret_5d", "vol_20d", "rsi14", "macd", "atr_14",
            "momentum", "ret_1d", "vol_10d", "bb_width", "adx_14"
        ],
        "metrics": {
            "pred_std": 0.00022,  # Just above threshold
            "zscore_works": True,
            "notes": "Maximum regularization that still allows z-score calculation",
        },
    },
    
    # RandomForest Regularized V1 - TESTED (2026-01-08)
    # Tested but performed poorly on holdout:
    # Test Sharpe: -0.16
    # NOT RECOMMENDED - XGBoost heavy reg is much better
    "rf_regularized_v1": {
        "model_type": "rf",
        "version": "v1",
        "created": "2026-01-08",
        "status": "deprecated",  # Poor test performance
        "params": {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_split": 50,
            "min_samples_leaf": 25,
            "max_features": 0.3,
            "random_state": 42,
        },
        "metrics": {
            "test_sharpe": -0.16,
            "test_accuracy": 0.514,
            "train_test_gap": 10.64,
        },
    },
    
    # XGBoost Balanced V6 - PRODUCTION (2026-01-08)
    # Fixed the "all predictions UP" issue from V4/V5
    # Root cause: min_child_weight and reg_lambda too high → predictions collapse to mean
    # V6 uses lighter regularization for prediction variance while still preventing overfit
    # 
    # Key changes from V5:
    # - min_child_weight: 30 → 5 (allows more splits)
    # - reg_lambda: 7.0 → 1.0 (lighter regularization)
    # - reg_alpha: 0.7 → 0.1 (lighter L1)
    # 
    # Result: Predictions have variance (std > 0.002) → both UP and DOWN signals
    "xgb_balanced_v6": {
        "model_type": "xgb",
        "version": "v6",
        "created": "2026-01-08",
        "status": "production",  # Fixed all-UP issue
        "params": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,      # Was 30 in V5 - too restrictive
            "reg_alpha": 0.1,            # Was 0.7 in V5 - too high
            "reg_lambda": 1.0,           # Was 7.0 in V5 - caused constant preds
            "random_state": 42,
        },
        "metrics": {
            "pred_std": 0.002340,        # Good variance for z-score
            "zscore_works": True,
            "up_down_split": "55/45",    # No longer 100% UP
            "notes": "Balanced regularization - enough variance for both signals",
        },
    },
}

# Current active model versions (change these to switch versions)
ACTIVE_MODEL_VERSIONS = {
    "xgb": "xgb_balanced_v6",     # Updated 2026-01-08 - V6 fixes all-UP issue (pred std 0.002 vs 0.00003)
    "rf": "rf_default_v1",         # RF deprecated - negative test Sharpe
    "gbrt": "gbrt_default_v1",     # GBRT deprecated - severe overfitting
    # Previous versions (kept for reference):
    # "xgb": "xgb_heavy_reg_v5",   # V5 - too regularized, all predictions UP
    # "xgb": "xgb_regularized_v4", # V4 - also too regularized
}

def get_model_config(model_type: str, version_override: str = None) -> dict:
    """
    Get the LOCKED configuration for a model type.
    Returns params from the active versioned config.
    
    Args:
        model_type: 'xgb', 'rf', or 'gbrt'
        version_override: Optional version key to use instead of active version
                         e.g., 'xgb_heavy_reg_v5' for testing
    """
    if version_override and version_override in MODEL_VERSIONS:
        return MODEL_VERSIONS[version_override]["params"].copy()
    
    version_key = ACTIVE_MODEL_VERSIONS.get(model_type)
    if version_key and version_key in MODEL_VERSIONS:
        return MODEL_VERSIONS[version_key]["params"].copy()
    # Fallback to MODEL_DEFAULTS
    return MODEL_DEFAULTS.get(model_type, {}).copy()


def get_heavy_reg_config() -> dict:
    """
    Get the Heavy Regularized XGB config (v5) for testing.
    
    This config dramatically outperformed production v4 on holdout data:
    - Test Sharpe: 1.00 (vs -0.82 for v4)
    - Test Accuracy: 54.4%
    - Sharpe Gap: 0.13 (virtually zero overfitting)
    
    Usage:
        from src.config import get_heavy_reg_config
        params = get_heavy_reg_config()
        model = XGBRegressor(**params)
    """
    return MODEL_VERSIONS["xgb_heavy_reg_v5"]["params"].copy()


def get_heavy_reg_features() -> list:
    """
    Get the 10 core features for Heavy Regularized XGB.
    
    These features were selected for robustness:
    ret_5d, vol_20d, rsi14, macd, atr_14, momentum, ret_1d, vol_10d, bb_width, adx_14
    """
    return MODEL_VERSIONS["xgb_heavy_reg_v5"]["features"].copy()

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
# OPTIMIZED MODEL CONFIGURATION (From Model Improvement Report)
# ============================================================================
# Results from 6-experiment pipeline (2026-01-07):
# - Experiment 3: Hyperparameter optimization (50 Optuna trials) -> Sharpe +2.77
# - Experiment 6: Temperature scaling calibration -> +7.2% Sharpe improvement
# - Experiment 2: Feature selection -> 20 optimal features from 150
#
# UPDATED 2026-01-08 (v6): Balanced regularization with REAL prediction variance
# V4/V5 were too regularized (constant predictions → all UP)
# V6 uses lighter regularization so model can predict both UP and DOWN

OPTIMIZED_MODEL_CONFIG = {
    # XGBoost Balanced V6 Hyperparameters (2026-01-08)
    # Fixed the "all predictions UP" issue
    # Key: min_child_weight=5 and reg_lambda=1.0 allow prediction variance
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,      # Was 0.6 (use more features)
    "min_child_weight": 5,        # Was 30 (CRITICAL: lower = more splits = more variance)
    "reg_alpha": 0.1,             # Was 0.5 (lighter L1 reg)
    "reg_lambda": 1.0,            # Was 5.0 (CRITICAL: lower = more variance in predictions)
    "random_state": 42,
    
    # Calibration disabled for regularized model
    "use_temperature_scaling": False,
    "temperature": 1.0,
}

# ============================================================================
# OPTIMIZED RANDOM FOREST CONFIGURATION
# ============================================================================
# Random Forest improvements to reduce overfitting and improve stability:
# 1. Deeper trees with more regularization (min_samples_leaf, min_samples_split)
# 2. More trees for ensemble stability (n_estimators=500)
# 3. Bootstrap aggregation with max_features tuning
# 4. Out-of-bag scoring for validation
#
# Key RF parameters for financial data:
# - max_depth: Controls tree complexity (deeper = more overfitting risk)
# - min_samples_leaf: Minimum samples per leaf (higher = more regularization)
# - max_features: Features per split ("sqrt" or 0.3-0.5 works well)
# - n_estimators: More trees = more stable predictions (diminishing returns after 300)

# OPTIMIZED 2026-01-08: Via Optuna (50 trials) - Best Sharpe 11.32
# See experiments/RF_OPTIMIZATION_REPORT.md and experiments/optimized_rf_config.json
OPTIMIZED_RF_CONFIG = {
    "n_estimators": 100,        # Optuna optimal (was 500)
    "max_depth": None,          # No limit optimal (was 10)
    "min_samples_split": 2,     # Default optimal (was 20)
    "min_samples_leaf": 4,      # Light regularization (was 10)
    "max_features": 0.7,        # Use 70% of features per split (was 0.5)
    "bootstrap": True,          # Use bootstrap samples
    "oob_score": False,         # Disable for speed (not needed)
    "random_state": 42,
    "n_jobs": -1,               # Use all cores
}

# RF Optimization Results (2026-01-08):
# - Best Sharpe: 11.32 (via Optuna 50 trials)
# - Best Ensemble: xgb_rf_equal (Sharpe 11.14)
# - Most Stable: rf_only (Worst 3M Sharpe: 7.26)
# - Temperature Scaling: 1.0 (no scaling needed)
# 
# Key findings from optimization:
# 1. max_depth=None performs better than constrained depth
# 2. max_features=0.7 (70%) better than 0.5 (50%)
# 3. Fewer trees (100) with less regularization works better
# 4. Equal-weight ensemble (XGB+RF 50/50) is best overall

def get_optimized_rf_config() -> dict:
    """Get the optimized Random Forest configuration."""
    return OPTIMIZED_RF_CONFIG.copy()

# Top 20 features from Experiment 2 (Recursive Feature Elimination)
# These achieved Sharpe 2.119 vs 1.847 baseline
OPTIMIZED_FEATURES = [
    "gbm_exp_ret_5d", "gbm_prob_up_5d", "ret_5d", "gbm_exp_ret_1d",
    "vol_20d", "rsi14", "gbm_prob_up_1d", "macd", "atr_14", "adx_14",
    "ret_1d", "ret_10d", "vol_10d", "obv", "momentum", "williams_r",
    "cci", "stoch_k", "bb_width", "mfi",
]

# A/B Testing Configuration
AB_TEST_CONFIG = {
    "enabled": True,
    "default_variant": "optimized",  # "optimized" or "legacy"
    "log_predictions": True,
    "comparison_window_days": 14,
}

def get_optimized_config() -> dict:
    """Get the optimized model configuration."""
    return OPTIMIZED_MODEL_CONFIG.copy()

def get_optimized_features() -> list:
    """Get the list of optimized features."""
    return OPTIMIZED_FEATURES.copy()

def is_optimized_mode() -> bool:
    """Check if we should use optimized configuration."""
    import os
    # Can be overridden via environment variable
    override = os.environ.get("USE_LEGACY_MODEL", "").lower()
    if override in ("1", "true", "yes"):
        return False
    return AB_TEST_CONFIG.get("default_variant", "optimized") == "optimized"

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

# ============================================================================
# TARGET PREPROCESSING
# ============================================================================
# Settings for how the target variable (forward returns) is processed
# Tested 2026-01-09: Winsorization improves Sharpe by +0.44 across 5 tickers
TARGET_PREPROCESSING_CONFIG = {
    "winsorize_enabled": True,      # Clip extreme returns at percentiles
    "winsorize_percentile": 0.01,   # Clips at 1st/99th percentile (~2% of days)
    "use_log_returns": False,       # Use log returns instead of simple returns (not tested)
}

# ============================================================================
# FAMA-FRENCH RESIDUAL TARGET CONFIG
# ============================================================================
# When enabled, the target becomes the stock-specific alpha (residual) from
# regressing returns on Fama-French 3 factors (MKT-RF, SMB, HML).
# This strips out systematic risk, letting the model focus on stock-specific signal.
# Data source: Kenneth French's Data Library (free, no API key)
# Added 2026-01-12
FF_RESIDUAL_TARGET_CONFIG = {
    "enabled": False,               # Set True to use FF alpha as target (start False for testing)
    "ff_url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "regression_window": 252,       # Rolling window for factor regression (1 year)
    "min_regression_obs": 60,       # Minimum observations for valid regression
    "cache_ttl_hours": 168,         # Cache FF data for 1 week (data updates monthly)
    "fallback_to_raw": True,        # If FF data unavailable, fall back to raw returns
}


def is_ff_residual_enabled() -> bool:
    """Check if Fama-French residual target is enabled."""
    return FF_RESIDUAL_TARGET_CONFIG.get("enabled", False)


def get_ff_config() -> dict:
    """Get Fama-French residual target configuration."""
    return FF_RESIDUAL_TARGET_CONFIG.copy()


def get_target_preprocessing_config() -> dict:
    """Get target preprocessing configuration."""
    return TARGET_PREPROCESSING_CONFIG.copy()


def is_winsorization_enabled() -> bool:
    """Check if target winsorization is enabled."""
    return TARGET_PREPROCESSING_CONFIG.get("winsorize_enabled", True)


def get_winsorize_percentile() -> float:
    """Get the winsorization percentile (e.g., 0.01 = 1st/99th)."""
    return TARGET_PREPROCESSING_CONFIG.get("winsorize_percentile", 0.01)


# ============================================================================
# REGIME-SPECIFIC MODELS CONFIG
# ============================================================================
# When enabled, trains separate models for BULL, BEAR, and NEUTRAL market regimes
# Each model learns patterns specific to its regime (above/below 200DMA)
# Added 2026-01-09
REGIME_MODELS_CONFIG = {
    "enabled": False,                # Off by default (experimental)
    "min_samples_per_regime": 100,   # Minimum training samples per regime
}


def get_regime_models_config() -> dict:
    """Get regime models configuration."""
    return REGIME_MODELS_CONFIG.copy()


def is_regime_models_enabled() -> bool:
    """Check if regime-specific models are enabled."""
    return REGIME_MODELS_CONFIG.get("enabled", False)


def get_min_samples_per_regime() -> int:
    """Get minimum samples required per regime for training."""
    return REGIME_MODELS_CONFIG.get("min_samples_per_regime", 100)


def set_regime_models_enabled(enabled: bool) -> None:
    """Enable or disable regime-specific models at runtime."""
    REGIME_MODELS_CONFIG["enabled"] = enabled


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
    # NOTE: Credit spreads, VIX term structure, and 2s10s are computed in macro.py
    # but EXCLUDED from active features — walk-forward test (2026-03-24) showed
    # they hurt accuracy (-0.8pp hit rate, -0.106 Sharpe) due to curse of dimensionality.
    # Uncomment below to re-enable for experimentation:
    # "credit_spread",       # BAA - AAA (widens in stress)
    # "term_spread_2s10s",   # 10Y - 2Y (inversion = recession signal)
    # "vix_term_structure",  # VIX / VIX3M ratio (<1 = contango/calm, >1 = backwardation/fear)
    # "credit_spread_chg_5d", # 5-day change in credit spread (momentum of stress)
    # "vix_ts_chg_5d",       # 5-day change in VIX term structure
]

REGIME_COLUMNS = [
    "regime_bull", "regime_bear",
    "regime_vix_low", "regime_vix_medium", "regime_vix_high",
    "bull_streak", "bear_streak",
    # NOTE: HMM regime indicators computed in regime_filter.py but EXCLUDED
    # from active features — walk-forward test (2026-03-24) showed they hurt
    # accuracy. Uncomment to re-enable:
    # "hmm_regime_bull", "hmm_regime_bear", "hmm_regime_neutral",
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
    # NEW: Credit spreads & VIX term structure
    "baa_yield": "DBAA",           # Moody's Baa corporate bond yield
    "aaa_yield": "DAAA",           # Moody's Aaa corporate bond yield
    "t2y": "DGS2",                 # 2-year treasury (for 2s10s curve)
    "vix3m": "VXVCLS",             # VIX 3-month (for term structure)
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
# TRADING STRATEGY PRESETS (Master Presets - Simplify UI)
# ============================================================================
@dataclass
class TradingStrategyPreset:
    """
    Master preset combining all trading parameters.
    Simplifies UI by providing Strict/Default/Loose options.
    """
    # Z-Score filtering
    min_zscore: float = 1.0
    
    # Signal filters
    min_signal_pct: float = 0.25  # Minimum prediction % to show
    min_predicted_move_pct: float = 1.0
    
    # Candidate filters
    max_tickers: int = 10
    min_iv: float = 0.20
    max_iv: float = 0.80
    
    # Position sizing
    use_vol_scaling: bool = True
    use_regime_filter: bool = True
    
    # Model settings
    model_type: str = "rf"
    period: str = "5y"
    horizon: int = 1
    auto_optimize: bool = True
    
    # Execution
    friction_preset: str = "Default"
    options_preset: str = "Default"
    
    @property
    def description(self) -> str:
        return f"z≥{self.min_zscore}, {self.model_type.upper()}, {self.period}"


# Based on experiment results: BASELINE_004 (Sharpe +0.129) used z=1.6, regime ON
TRADING_STRATEGY_PRESETS = {
    "Default (Balanced)": TradingStrategyPreset(
        min_zscore=1.0,
        min_signal_pct=0.25,
        min_predicted_move_pct=1.0,
        max_tickers=10,
        min_iv=0.20,
        max_iv=0.80,
        use_vol_scaling=True,
        use_regime_filter=True,
        model_type="rf",
        period="5y",
        horizon=1,
        auto_optimize=True,
        friction_preset="Default",
        options_preset="Default",
    ),
    "Strict (Fewer, Stronger Signals)": TradingStrategyPreset(
        min_zscore=1.6,  # From experiments: best z-score threshold
        min_signal_pct=0.5,
        min_predicted_move_pct=1.5,
        max_tickers=5,
        min_iv=0.25,
        max_iv=0.70,
        use_vol_scaling=True,
        use_regime_filter=True,  # From experiments: ON improved drawdown
        model_type="rf",
        period="5y",
        horizon=1,
        auto_optimize=True,
        friction_preset="Strict (pessimistic)",
        options_preset="Strict",
    ),
    "Loose (More Signals)": TradingStrategyPreset(
        min_zscore=0.5,
        min_signal_pct=0.1,
        min_predicted_move_pct=0.5,
        max_tickers=20,
        min_iv=0.15,
        max_iv=0.90,
        use_vol_scaling=True,
        use_regime_filter=False,
        model_type="rf",
        period="5y",
        horizon=1,
        auto_optimize=True,
        friction_preset="Loose (optimistic)",
        options_preset="Loose",
    ),
    "Aggressive (XGBoost, More Trades)": TradingStrategyPreset(
        min_zscore=0.8,
        min_signal_pct=0.2,
        min_predicted_move_pct=0.75,
        max_tickers=15,
        min_iv=0.15,
        max_iv=0.85,
        use_vol_scaling=True,
        use_regime_filter=True,
        model_type="xgb",
        period="5y",
        horizon=1,
        auto_optimize=True,
        friction_preset="Default",
        options_preset="Default",
    ),
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
# IMPORTANT: Default to False - only enable when explicitly requested via env or UI
USE_ELASTICNET_SELECT = env_bool("USE_ELASTICNET_SELECT", False)  # Changed default to False
ELASTICNET_L1_RATIO = env_float("ELASTICNET_L1_RATIO", 0.5)
ELASTICNET_CV_FOLDS = env_int("ELASTICNET_CV_FOLDS", 5)
ELASTICNET_MIN_FEATURES = env_int("ELASTICNET_MINFEATURES", 12)


def is_elasticnet_enabled() -> bool:
    """Check if ElasticNet feature selection is enabled at RUNTIME.
    
    This function reads the environment variable each time it's called,
    allowing UI toggles to take effect without restarting the app.
    """
    return env_bool("USE_ELASTICNET_SELECT", False)

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
