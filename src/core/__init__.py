# Core module - pure business logic (no I/O, no UI dependencies)

from .features import (
    build_all_features,
    build_optimized_features,
    add_returns,
    add_volatility,
    add_gbm_features,
    add_regime_features,
    add_momentum_indicators,
    build_target,
)
from .models import (
    make_model,
    train_model,
    select_features_elasticnet,
    select_features_ols_pvalue,
    get_feature_importance,
)

# LSTM model (optional - requires TensorFlow)
try:
    from .lstm_model import LSTMPredictor, LSTMWrapper, create_lstm_model, HAS_TF
except ImportError:
    HAS_TF = False
    LSTMPredictor = None
    LSTMWrapper = None
    create_lstm_model = None

from .calibration import (
    temperature_scale,
    logit_temperature_scale,
    calibrate_predictions,
    PredictionCalibrator,
    get_default_temperature,
    should_use_calibration,
)
from .metrics import (
    compute_sharpe,
    compute_sortino,
    compute_drawdown,
    compute_max_drawdown,
    summarize_risk,
    deflated_sharpe_ratio,
)
from .pricing import (
    black_scholes_price,
    black_scholes_greeks,
    suggest_options_strategy,
    normalize_strategy,
    OptionSpec,
    Greeks,
)
from .option_strategy import (
    generate_option_strategy,
    classify_delta_strike,
    determine_strategy,
    get_strategy_summary,
)
from .zscore_filter import (
    ZScoreFilter,
    ZScoreResult,
    get_zscore_filter,
    reset_zscore_filter,
    compute_prediction_zscore,
    evaluate_ticker_zscore,
)
from .trade_limiter import (
    TradeLimiter,
    SkippedSignal,
    TradeCount,
    get_trade_limiter,
    reset_trade_limiter,
    apply_trade_limits,
    rank_signals_by_conviction,
)
from .regime_filter import (
    RegimeFilter,
    MarketRegime,
    RegimeState,
    BlockedTrade,
    get_regime_filter,
    reset_regime_filter,
    apply_regime_filter,
    get_current_regime,
)
from .signal_filter import (
    apply_signal_filter,
    calculate_momentum_zscore,
    get_filter_stats,
    should_trade,
)
from .ab_testing import (
    ABTestConfig,
    get_ab_config,
    set_ab_variant,
    get_active_variant,
    is_optimized_active,
    log_prediction,
    compare_variants,
    generate_ab_report,
    variant_context,
)
from .feature_monitor import (
    FeatureImportanceTracker,
    get_feature_tracker,
    reset_feature_tracker,
)
from .regime_predictor import (
    RegimeAwarePredictor,
    REGIME_GROUPS,
    MIN_SAMPLES_PER_REGIME,
)
from .production_predictor import (
    ProductionPredictor,
    PredictionResult,
    TradingMode,
    TRADING_MODES,
    quick_predict,
)

__all__ = [
    # Features
    "build_all_features",
    "build_optimized_features",
    "add_returns",
    "add_volatility",
    "add_gbm_features",
    "add_regime_features",
    "add_momentum_indicators",
    "build_target",
    # Models
    "make_model",
    "train_model",
    "select_features_elasticnet",
    "select_features_ols_pvalue",
    "get_feature_importance",
    # Calibration
    "temperature_scale",
    "logit_temperature_scale",
    "calibrate_predictions",
    "PredictionCalibrator",
    "get_default_temperature",
    "should_use_calibration",
    # Metrics
    "compute_sharpe",
    "compute_sortino",
    "compute_drawdown",
    "compute_max_drawdown",
    "summarize_risk",
    "deflated_sharpe_ratio",
    # Pricing
    "black_scholes_price",
    "black_scholes_greeks",
    "suggest_options_strategy",
    "normalize_strategy",
    "OptionSpec",
    "Greeks",
    # Option Strategy Generator
    "generate_option_strategy",
    "classify_delta_strike",
    "determine_strategy",
    "get_strategy_summary",
    # Z-Score Filtering
    "ZScoreFilter",
    "ZScoreResult",
    "get_zscore_filter",
    "reset_zscore_filter",
    "compute_prediction_zscore",
    "evaluate_ticker_zscore",
    # Trade Limiting
    "TradeLimiter",
    "SkippedSignal",
    "TradeCount",
    "get_trade_limiter",
    "reset_trade_limiter",
    "apply_trade_limits",
    "rank_signals_by_conviction",
    # A/B Testing
    "ABTestConfig",
    "get_ab_config",
    "set_ab_variant",
    "get_active_variant",
    "is_optimized_active",
    "log_prediction",
    "compare_variants",
    "generate_ab_report",
    "variant_context",
    # Regime Filtering
    "RegimeFilter",
    "MarketRegime",
    "RegimeState",
    "BlockedTrade",
    "get_regime_filter",
    "reset_regime_filter",
    "apply_regime_filter",
    "get_current_regime",
    # Signal Filtering (Trading Strategies)
    "apply_signal_filter",
    "calculate_momentum_zscore",
    "get_filter_stats",
    "should_trade",
    # Feature Monitoring
    "FeatureImportanceTracker",
    "get_feature_tracker",
    "reset_feature_tracker",
    # Regime-Aware Prediction
    "RegimeAwarePredictor",
    "REGIME_GROUPS",
    "MIN_SAMPLES_PER_REGIME",
    # Production Predictor (Adaptive Model)
    "ProductionPredictor",
    "PredictionResult",
    "TradingMode",
    "TRADING_MODES",
    "quick_predict",
]
