"""
Z-Score Filtering Module
========================

Provides rolling z-score calculation and filtering for predictions.
Supports configurable thresholds without hard-excluding tickers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# Set up logging for weak signals
logger = logging.getLogger("zscore_filter")
logger.setLevel(logging.INFO)


@dataclass
class ZScoreResult:
    """Result of z-score calculation for a single prediction."""
    ticker: str
    prediction: float
    z_score: float
    rolling_mean: float
    rolling_std: float
    history_length: int
    passes_threshold: bool
    threshold: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ZScoreResult":
        return cls(**data)
    
    @property
    def is_strong_signal(self) -> bool:
        """True if |z-score| >= threshold."""
        return self.passes_threshold
    
    @property
    def signal_strength(self) -> str:
        """Categorize signal strength."""
        z_abs = abs(self.z_score)
        if z_abs >= 2.0:
            return "very_strong"
        elif z_abs >= 1.5:
            return "strong"
        elif z_abs >= 1.0:
            return "moderate"
        elif z_abs >= 0.5:
            return "weak"
        else:
            return "very_weak"


class ZScoreFilter:
    """
    Rolling z-score filter for predictions with logging.
    
    Features:
    - Configurable threshold
    - Tracks weak signals for analysis
    - Does NOT hard-exclude tickers by default
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        rolling_window: int = 20,
        min_data_points: int = 10,
        log_path: Optional[Path] = None,
        hard_filter: bool = False,
    ):
        """
        Initialize z-score filter.
        
        Args:
            threshold: Minimum |z-score| to pass filter (default 1.0)
            rolling_window: Rolling window for mean/std calculation
            min_data_points: Minimum data points required for z-score
            log_path: Path to log weak signals (JSON lines format)
            hard_filter: If True, weak signals are filtered out; if False, logged only
        """
        self.threshold = threshold
        self.rolling_window = rolling_window
        self.min_data_points = min_data_points
        self.log_path = Path(log_path) if log_path else None
        self.hard_filter = hard_filter
        
        # In-memory weak signal log (for dashboard display)
        self._weak_signals: list[ZScoreResult] = []
        self._max_weak_signals = 1000  # Rolling buffer
    
    def compute_zscore(
        self,
        prediction: float,
        history: list[float] | pd.Series,
    ) -> tuple[float, float, float]:
        """
        Compute z-score for a prediction relative to history.
        
        Args:
            prediction: Current prediction value
            history: Historical predictions for this ticker
            
        Returns:
            (z_score, rolling_mean, rolling_std)
        """
        if isinstance(history, pd.Series):
            history = history.tolist()
        
        if len(history) < self.min_data_points:
            # Not enough data - return z-score of 0 (neutral)
            return 0.0, prediction, 0.0
        
        # Use recent window for mean/std
        recent = history[-self.rolling_window:] if len(history) > self.rolling_window else history
        roll_mean = float(np.mean(recent))
        roll_std = float(np.std(recent, ddof=1))  # Sample std
        
        if roll_std == 0 or np.isnan(roll_std):
            # No variance - can't compute z-score
            return 0.0, roll_mean, 0.0
        
        z_score = (prediction - roll_mean) / roll_std
        return float(z_score), roll_mean, roll_std
    
    def evaluate(
        self,
        ticker: str,
        prediction: float,
        history: list[float] | pd.Series,
    ) -> ZScoreResult:
        """
        Evaluate a prediction and return ZScoreResult.
        
        Args:
            ticker: Stock ticker
            prediction: Current prediction value
            history: Historical predictions for this ticker
            
        Returns:
            ZScoreResult with all z-score data
        """
        z_score, roll_mean, roll_std = self.compute_zscore(prediction, history)
        
        result = ZScoreResult(
            ticker=ticker,
            prediction=prediction,
            z_score=z_score,
            rolling_mean=roll_mean,
            rolling_std=roll_std,
            history_length=len(history) if isinstance(history, list) else len(history.tolist()),
            passes_threshold=abs(z_score) >= self.threshold,
            threshold=self.threshold,
        )
        
        # Log weak signals
        if not result.passes_threshold:
            self._log_weak_signal(result)
        
        return result
    
    def _log_weak_signal(self, result: ZScoreResult):
        """Log a weak signal for later analysis."""
        # In-memory log (rolling buffer)
        self._weak_signals.append(result)
        if len(self._weak_signals) > self._max_weak_signals:
            self._weak_signals = self._weak_signals[-self._max_weak_signals:]
        
        # File log (optional)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(result.to_dict()) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log weak signal: {e}")
        
        # Console log
        logger.info(
            f"WEAK SIGNAL: {result.ticker} | z={result.z_score:.2f} < {result.threshold} | "
            f"pred={result.prediction*100:.2f}% | hist={result.history_length}"
        )
    
    def get_weak_signals(
        self,
        ticker: Optional[str] = None,
        last_n: int = 100,
    ) -> list[ZScoreResult]:
        """
        Get recent weak signals for analysis.
        
        Args:
            ticker: Filter by ticker (None = all)
            last_n: Return last N signals
            
        Returns:
            List of ZScoreResult for weak signals
        """
        signals = self._weak_signals[-last_n:]
        
        if ticker:
            signals = [s for s in signals if s.ticker.upper() == ticker.upper()]
        
        return signals
    
    def get_weak_signals_df(
        self,
        ticker: Optional[str] = None,
        last_n: int = 100,
    ) -> pd.DataFrame:
        """Get weak signals as DataFrame for dashboard display."""
        signals = self.get_weak_signals(ticker, last_n)
        if not signals:
            return pd.DataFrame()
        
        return pd.DataFrame([s.to_dict() for s in signals])
    
    def filter_signals(
        self,
        signals_dict: dict[str, dict[str, Any]],
        prediction_history: dict[str, list[float]],
    ) -> tuple[dict[str, dict[str, Any]], list[ZScoreResult]]:
        """
        Apply z-score filtering to a signals dictionary.
        
        Args:
            signals_dict: Dict of ticker -> signal data
            prediction_history: Dict of ticker -> list of historical predictions
            
        Returns:
            (filtered_signals, weak_signal_results)
            
        Note: If hard_filter=False, signals are tagged but not removed.
        """
        filtered = {}
        weak_results = []
        
        for ticker, signal in signals_dict.items():
            pred = signal.get("pred_next_ret", 0.0)
            history = prediction_history.get(ticker, [])
            
            result = self.evaluate(ticker, pred, history)
            
            # Add z-score data to signal
            signal["z_score"] = result.z_score
            signal["z_score_passes"] = result.passes_threshold
            signal["z_score_strength"] = result.signal_strength
            signal["z_score_history_len"] = result.history_length
            
            if not result.passes_threshold:
                weak_results.append(result)
                if self.hard_filter:
                    # Skip weak signals
                    continue
            
            filtered[ticker] = signal
        
        return filtered, weak_results


# Default global filter instance (can be configured via config.py)
_default_filter: Optional[ZScoreFilter] = None


def get_zscore_filter() -> ZScoreFilter:
    """Get or create the default z-score filter instance."""
    global _default_filter
    
    if _default_filter is None:
        # Import config here to avoid circular imports
        try:
            from ..config import ZSCORE_GATING_CONFIG, CACHE_DIR
            _default_filter = ZScoreFilter(
                threshold=ZSCORE_GATING_CONFIG.get("min_zscore", 1.0),
                rolling_window=ZSCORE_GATING_CONFIG.get("rolling_window", 20),
                min_data_points=ZSCORE_GATING_CONFIG.get("min_data_points", 10),
                log_path=CACHE_DIR / "weak_signals.jsonl",
                hard_filter=ZSCORE_GATING_CONFIG.get("hard_filter", False),
            )
        except ImportError:
            # Fallback defaults
            _default_filter = ZScoreFilter(threshold=1.0)
    
    return _default_filter


def reset_zscore_filter():
    """Reset the default filter (e.g., after config changes)."""
    global _default_filter
    _default_filter = None


def compute_prediction_zscore(
    predictions: pd.Series | list[float],
    current_pred: float,
    window: int = 20,
    min_periods: int = 10,
) -> tuple[float, float, float]:
    """
    Compute z-score for a prediction relative to recent predictions.
    
    This is a convenience function matching the original signals.py API.
    
    Args:
        predictions: Historical predictions for this ticker
        current_pred: Current prediction value
        window: Rolling window size for mean/std
        min_periods: Minimum periods required
        
    Returns:
        (z_score, rolling_mean, rolling_std)
    """
    fltr = ZScoreFilter(
        threshold=1.0,  # Threshold not used in compute_zscore
        rolling_window=window,
        min_data_points=min_periods,
    )
    return fltr.compute_zscore(current_pred, predictions if isinstance(predictions, list) else predictions.tolist())


def evaluate_ticker_zscore(
    ticker: str,
    prediction: float,
    history: list[float],
    threshold: float = 1.0,
) -> ZScoreResult:
    """
    Evaluate a single ticker's prediction z-score.
    
    Convenience function for quick evaluation.
    """
    fltr = get_zscore_filter()
    # Temporarily override threshold for this call
    old_threshold = fltr.threshold
    fltr.threshold = threshold
    result = fltr.evaluate(ticker, prediction, history)
    fltr.threshold = old_threshold
    return result
