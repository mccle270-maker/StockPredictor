"""
A/B Testing for Model Configurations.

This module provides infrastructure for comparing the performance of
different model configurations (optimized vs legacy) in production.

Features:
- Configuration switching via flags
- Prediction logging with variant tracking
- Performance comparison over time windows
- Automatic variant selection based on rolling performance
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger("ab_testing")

# ============================================================================
# CONFIGURATION
# ============================================================================

AB_TEST_LOG_DIR = Path(__file__).parent.parent / ".ab_tests"
AB_TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    enabled: bool = True
    default_variant: str = "optimized"  # "optimized" or "legacy"
    log_predictions: bool = True
    comparison_window_days: int = 14
    auto_select_variant: bool = False  # If True, automatically use best performing variant
    min_samples_for_comparison: int = 20


# Global config instance
_ab_config = ABTestConfig()


def get_ab_config() -> ABTestConfig:
    """Get current A/B test configuration."""
    return _ab_config


def set_ab_variant(variant: str) -> None:
    """
    Set the active variant.
    
    Args:
        variant: "optimized" or "legacy"
    """
    if variant not in ("optimized", "legacy"):
        raise ValueError(f"Invalid variant: {variant}. Must be 'optimized' or 'legacy'")
    _ab_config.default_variant = variant
    logger.info(f"🔀 Switched to {variant.upper()} configuration")


def get_active_variant() -> str:
    """Get the currently active variant."""
    # Check environment override
    env_variant = os.environ.get("AB_TEST_VARIANT", "").lower()
    if env_variant in ("optimized", "legacy"):
        return env_variant
    return _ab_config.default_variant


def is_optimized_active() -> bool:
    """Check if optimized variant is active."""
    return get_active_variant() == "optimized"


# ============================================================================
# PREDICTION LOGGING
# ============================================================================

@dataclass
class PredictionLog:
    """Log entry for a single prediction."""
    timestamp: str
    ticker: str
    variant: str
    prediction: float
    actual: Optional[float] = None
    position: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


def log_prediction(
    ticker: str,
    prediction: float,
    variant: Optional[str] = None,
    position: Optional[float] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Log a prediction for A/B test comparison.
    
    Args:
        ticker: Stock ticker
        prediction: Model prediction (return or probability)
        variant: Which variant made this prediction (auto-detected if None)
        position: Position taken based on prediction
        metadata: Additional metadata to log
    """
    if not _ab_config.log_predictions:
        return
    
    if variant is None:
        variant = get_active_variant()
    
    log_entry = PredictionLog(
        timestamp=datetime.now().isoformat(),
        ticker=ticker,
        variant=variant,
        prediction=float(prediction),
        position=float(position) if position is not None else None,
        metadata=metadata or {},
    )
    
    # Write to log file
    log_file = AB_TEST_LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry.__dict__) + "\n")


def update_prediction_outcome(
    ticker: str,
    timestamp: str,
    actual: float,
    pnl: Optional[float] = None,
) -> None:
    """
    Update a prediction log entry with actual outcome.
    
    Args:
        ticker: Stock ticker
        timestamp: Original prediction timestamp
        actual: Actual return that occurred
        pnl: Realized P&L (if applicable)
    """
    # Read today's log file
    log_file = AB_TEST_LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    if not log_file.exists():
        return
    
    # Find and update the entry
    entries = []
    updated = False
    with open(log_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry["ticker"] == ticker and entry["timestamp"].startswith(timestamp[:10]):
                entry["actual"] = actual
                if pnl is not None:
                    entry["pnl"] = pnl
                updated = True
            entries.append(entry)
    
    if updated:
        with open(log_file, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")


# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

def load_prediction_logs(days: int = 14) -> pd.DataFrame:
    """
    Load prediction logs for the last N days.
    
    Args:
        days: Number of days to look back
    
    Returns:
        DataFrame with prediction logs
    """
    entries = []
    start_date = datetime.now() - timedelta(days=days)
    
    for log_file in AB_TEST_LOG_DIR.glob("predictions_*.jsonl"):
        # Parse date from filename
        try:
            file_date = datetime.strptime(log_file.stem.split("_")[1], "%Y%m%d")
            if file_date < start_date:
                continue
        except (IndexError, ValueError):
            continue
        
        with open(log_file, "r") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not entries:
        return pd.DataFrame()
    
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def compare_variants(days: int = 14) -> Dict:
    """
    Compare performance of optimized vs legacy variants.
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Dictionary with comparison metrics
    """
    df = load_prediction_logs(days)
    
    if df.empty or "actual" not in df.columns:
        return {
            "status": "insufficient_data",
            "message": "Not enough prediction logs with outcomes",
            "optimized": {},
            "legacy": {},
        }
    
    # Filter to predictions with actual outcomes
    df_with_outcomes = df[df["actual"].notna()].copy()
    
    if len(df_with_outcomes) < _ab_config.min_samples_for_comparison:
        return {
            "status": "insufficient_data",
            "message": f"Need at least {_ab_config.min_samples_for_comparison} predictions with outcomes",
            "samples": len(df_with_outcomes),
        }
    
    results = {}
    
    for variant in ["optimized", "legacy"]:
        variant_df = df_with_outcomes[df_with_outcomes["variant"] == variant]
        
        if len(variant_df) < 5:
            results[variant] = {"samples": len(variant_df), "status": "insufficient_data"}
            continue
        
        # Calculate metrics
        predictions = variant_df["prediction"].values
        actuals = variant_df["actual"].values
        
        # Direction accuracy
        correct_direction = np.sign(predictions) == np.sign(actuals)
        direction_accuracy = correct_direction.mean()
        
        # Correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        
        # If PnL available
        if "pnl" in variant_df.columns and variant_df["pnl"].notna().any():
            pnl_series = variant_df["pnl"].fillna(0)
            total_pnl = pnl_series.sum()
            win_rate = (pnl_series > 0).mean()
            sharpe = pnl_series.mean() / pnl_series.std() * np.sqrt(252) if pnl_series.std() > 0 else 0
        else:
            total_pnl = None
            win_rate = None
            sharpe = None
        
        results[variant] = {
            "samples": len(variant_df),
            "direction_accuracy": float(direction_accuracy),
            "correlation": float(correlation) if not np.isnan(correlation) else None,
            "total_pnl": float(total_pnl) if total_pnl is not None else None,
            "win_rate": float(win_rate) if win_rate is not None else None,
            "sharpe": float(sharpe) if sharpe is not None else None,
        }
    
    # Determine winner
    winner = None
    if "optimized" in results and "legacy" in results:
        opt = results["optimized"]
        leg = results["legacy"]
        
        if isinstance(opt, dict) and isinstance(leg, dict):
            opt_acc = opt.get("direction_accuracy", 0) or 0
            leg_acc = leg.get("direction_accuracy", 0) or 0
            
            if opt_acc > leg_acc + 0.02:  # 2% threshold
                winner = "optimized"
            elif leg_acc > opt_acc + 0.02:
                winner = "legacy"
            else:
                winner = "tie"
    
    return {
        "status": "complete",
        "comparison_days": days,
        "total_samples": len(df_with_outcomes),
        "winner": winner,
        "optimized": results.get("optimized", {}),
        "legacy": results.get("legacy", {}),
        "timestamp": datetime.now().isoformat(),
    }


def get_recommended_variant() -> str:
    """
    Get the recommended variant based on recent performance.
    
    Returns:
        "optimized" or "legacy" based on comparison
    """
    comparison = compare_variants(_ab_config.comparison_window_days)
    
    if comparison.get("status") != "complete":
        # Default to optimized if no data
        return "optimized"
    
    winner = comparison.get("winner")
    if winner == "legacy":
        return "legacy"
    
    # Default to optimized (including tie)
    return "optimized"


def auto_select_variant() -> str:
    """
    Automatically select and set the best variant based on performance.
    
    Returns:
        The selected variant
    """
    if not _ab_config.auto_select_variant:
        return get_active_variant()
    
    recommended = get_recommended_variant()
    if recommended != _ab_config.default_variant:
        logger.info(f"🤖 Auto-selecting {recommended.upper()} variant based on performance")
        _ab_config.default_variant = recommended
    
    return recommended


# ============================================================================
# CONTEXT MANAGER FOR TESTING
# ============================================================================

class variant_context:
    """
    Context manager for temporarily using a specific variant.
    
    Usage:
        with variant_context("legacy"):
            # All predictions here use legacy config
            make_prediction(...)
    """
    
    def __init__(self, variant: str):
        self.variant = variant
        self.previous_variant = None
    
    def __enter__(self):
        self.previous_variant = _ab_config.default_variant
        _ab_config.default_variant = self.variant
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _ab_config.default_variant = self.previous_variant
        return False


# ============================================================================
# REPORTING
# ============================================================================

def generate_ab_report() -> str:
    """Generate a human-readable A/B test report."""
    comparison = compare_variants()
    
    lines = [
        "=" * 60,
        "A/B TEST REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Comparison Window: {_ab_config.comparison_window_days} days",
        f"Active Variant: {get_active_variant().upper()}",
        "",
    ]
    
    if comparison.get("status") != "complete":
        lines.append(f"Status: {comparison.get('message', 'Insufficient data')}")
        return "\n".join(lines)
    
    lines.append(f"Total Samples: {comparison.get('total_samples', 0)}")
    lines.append(f"Winner: {comparison.get('winner', 'N/A').upper()}")
    lines.append("")
    
    for variant in ["optimized", "legacy"]:
        data = comparison.get(variant, {})
        lines.append(f"--- {variant.upper()} ---")
        lines.append(f"  Samples: {data.get('samples', 0)}")
        lines.append(f"  Direction Accuracy: {data.get('direction_accuracy', 0)*100:.1f}%")
        if data.get("sharpe") is not None:
            lines.append(f"  Sharpe Ratio: {data.get('sharpe', 0):.3f}")
        if data.get("win_rate") is not None:
            lines.append(f"  Win Rate: {data.get('win_rate', 0)*100:.1f}%")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
