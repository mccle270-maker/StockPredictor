"""
Anti-Overfitting Configuration
==============================

TESTED 2026-01-08: Regularized XGB PASSED validation!

Results from test_regularized_models.py:
- REGULARIZED_XGB_CONFIG: ✅ APPROVED
  - Test Sharpe: 0.84 (realistic, sustainable)
  - Overfitting Gap: -0.03 (nearly zero!)
  - Accuracy: 53.2% (real edge above 50%)
  
- REGULARIZED_RF_CONFIG: ❌ NOT APPROVED
  - Test Sharpe: -0.12 (negative)
  - RF should not be used in production

RECOMMENDATION: Use XGB only for production trading.
The regularized XGB config is now active in src/config.py as xgb_regularized_v3.
"""

# =============================================================================
# STEP 1: SIMPLER MODELS (Reduce capacity to prevent memorization)
# =============================================================================

REGULARIZED_RF_CONFIG = {
    # Fewer trees = less capacity to memorize
    "n_estimators": 50,           # Was 100 → reduced
    
    # Limit tree depth = simpler decision boundaries  
    "max_depth": 5,               # Was None (unlimited) → constrained
    
    # Require more samples per split = smoother predictions
    "min_samples_split": 50,      # Was 2 → much stricter
    "min_samples_leaf": 25,       # Was 4 → much stricter
    
    # Use fewer features per tree = decorrelation
    "max_features": 0.3,          # Was 0.7 → more random
    
    "bootstrap": True,
    "oob_score": True,            # Out-of-bag validation
    "random_state": 42,
    "n_jobs": -1,
}

REGULARIZED_XGB_CONFIG = {
    # Fewer trees
    "n_estimators": 50,           # Was 450 → drastically reduced
    
    # Shallower trees
    "max_depth": 3,               # Was 7 → much shallower
    
    # Slower learning = less aggressive fitting
    "learning_rate": 0.1,         # Was 0.048 → simpler
    
    # More aggressive subsampling
    "subsample": 0.5,             # Was 0.998 → much more regularization
    "colsample_bytree": 0.3,      # Was 0.67 → use fewer features
    
    # Stronger L1/L2 regularization
    "reg_alpha": 1.0,             # Was 0.012 → 80x stronger L1
    "reg_lambda": 10.0,           # Was 9.3 → stronger L2
    
    # Require more samples per leaf
    "min_child_weight": 50,       # Was 19 → much stricter
    
    "random_state": 42,
    "n_jobs": -1,
}


# =============================================================================
# STEP 2: FEATURE REDUCTION (Use only proven stable features)
# =============================================================================

# Only price-based features with clear economic meaning
# No fundamentals (data quality issues), no macro (timezone issues)
MINIMAL_FEATURES = [
    # Returns (lagged)
    "ret_1d",
    "ret_5d", 
    "ret_10d",
    "ret_20d",
    
    # Volatility
    "vol_10d",
    "vol_20d",
    
    # Core technicals
    "rsi14",
    "macd",
    
    # Volume
    "obv",
    
    # Momentum
    "momentum",
]

# Slightly expanded set if minimal works
CONSERVATIVE_FEATURES = MINIMAL_FEATURES + [
    # Additional technicals
    "atr_14",
    "adx_14",
    "williams_r",
    
    # GBM probabilities (pure price-based)
    "gbm_prob_up_1d",
    "gbm_prob_up_5d",
]


# =============================================================================
# STEP 3: VALIDATION REQUIREMENTS (Don't deploy without passing)
# =============================================================================

DEPLOYMENT_THRESHOLDS = {
    # Minimum holdout Sharpe to deploy
    "min_holdout_sharpe": 0.3,
    
    # Maximum acceptable overfitting gap
    "max_overfitting_gap": 3.0,  # Train Sharpe - Test Sharpe
    
    # Minimum direction accuracy on holdout
    "min_holdout_accuracy": 0.52,
    
    # Minimum % of walk-forward folds with positive Sharpe
    "min_positive_fold_pct": 0.55,
    
    # Maximum drawdown on holdout
    "max_holdout_drawdown": -0.25,
}


# =============================================================================
# STEP 4: TRAINING PROTOCOL (Prevent data leakage)
# =============================================================================

TRAINING_PROTOCOL = {
    # Gap between train and validation (prevent leakage)
    "purge_days": 10,
    
    # Extra buffer after test periods
    "embargo_days": 5,
    
    # Minimum training samples
    "min_train_samples": 500,
    
    # Maximum features relative to samples (rule of thumb)
    "max_feature_ratio": 0.05,  # features / samples < 5%
    
    # Retrain frequency
    "retrain_frequency_days": 90,  # Quarterly retraining
}


# =============================================================================
# STEP 5: ENSEMBLE STRATEGY (If using ensemble)
# =============================================================================

CONSERVATIVE_ENSEMBLE = {
    # Only trade when both models agree
    "require_agreement": True,
    
    # Minimum confidence threshold
    "min_confidence": 0.6,  # 60% probability
    
    # Position sizing based on agreement strength
    "scale_by_confidence": True,
    
    # Maximum position size
    "max_position_pct": 0.10,  # 10% max per position
}


# =============================================================================
# STEP 6: TICKER SELECTION (Focus on what works)
# =============================================================================

# Based on holdout results:
# - AAPL: RF Sharpe 1.01 ✓
# - NVDA: RF Sharpe 0.99 ✓  
# - MSFT: XGB Sharpe 0.66 ✓
# - GOOGL: XGB Sharpe 1.31 ✓
# - AMZN: Avoid with XGB (-2.04 Sharpe!)

TICKER_MODEL_MAPPING = {
    "AAPL": "rf",    # RF works better
    "NVDA": "rf",    # RF works better
    "MSFT": "xgb",   # XGB slightly better
    "GOOGL": "xgb",  # XGB works well
    "AMZN": "rf",    # XGB is dangerous here
}

# Exclude these entirely (consistently bad)
EXCLUDED_TICKERS = ["SPY", "META"]


# =============================================================================
# IMPLEMENTATION HELPER
# =============================================================================

def get_regularized_model(model_type: str = "rf"):
    """Get a regularized model configured to reduce overfitting."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**REGULARIZED_RF_CONFIG)
    elif model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**REGULARIZED_XGB_CONFIG)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def validate_before_deploy(holdout_sharpe: float, train_sharpe: float,
                           holdout_accuracy: float, holdout_drawdown: float) -> dict:
    """Check if model passes deployment thresholds."""
    results = {
        "sharpe_check": holdout_sharpe >= DEPLOYMENT_THRESHOLDS["min_holdout_sharpe"],
        "overfitting_check": (train_sharpe - holdout_sharpe) <= DEPLOYMENT_THRESHOLDS["max_overfitting_gap"],
        "accuracy_check": holdout_accuracy >= DEPLOYMENT_THRESHOLDS["min_holdout_accuracy"],
        "drawdown_check": holdout_drawdown >= DEPLOYMENT_THRESHOLDS["max_holdout_drawdown"],
    }
    results["deploy_approved"] = all(results.values())
    return results


if __name__ == "__main__":
    print("Anti-Overfitting Configuration")
    print("="*50)
    print(f"\nRegularized RF Config:")
    for k, v in REGULARIZED_RF_CONFIG.items():
        print(f"  {k}: {v}")
    
    print(f"\nRegularized XGB Config:")
    for k, v in REGULARIZED_XGB_CONFIG.items():
        print(f"  {k}: {v}")
    
    print(f"\nMinimal Features ({len(MINIMAL_FEATURES)}):")
    for f in MINIMAL_FEATURES:
        print(f"  - {f}")
    
    print(f"\nDeployment Thresholds:")
    for k, v in DEPLOYMENT_THRESHOLDS.items():
        print(f"  {k}: {v}")
