"""
Probability calibration for model predictions.

This module implements temperature scaling and other calibration methods
to improve the reliability of prediction probabilities.

From Model Improvement Report (Experiment 6):
- Temperature T=2.9 achieved +7.2% Sharpe improvement
- Calibration helps convert raw model outputs to well-calibrated probabilities
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger("calibration")


# ============================================================================
# TEMPERATURE SCALING
# ============================================================================

def temperature_scale(
    probs: Union[np.ndarray, pd.Series, float],
    temperature: float = 2.9,
) -> Union[np.ndarray, pd.Series, float]:
    """
    Apply temperature scaling to probabilities.
    
    Temperature scaling adjusts the "sharpness" of probabilities:
    - T > 1: Softens probabilities (less confident, better calibrated)
    - T < 1: Sharpens probabilities (more confident)
    - T = 1: No change
    
    For regression outputs interpreted as return predictions:
    - Divides the prediction by temperature to scale magnitude
    
    Args:
        probs: Raw model output (probabilities or predictions)
        temperature: Scaling factor (default 2.9 from Experiment 6)
    
    Returns:
        Calibrated probabilities/predictions
    
    Example:
        >>> raw_pred = 0.05  # 5% predicted return
        >>> calibrated = temperature_scale(raw_pred, temperature=2.9)
        >>> # calibrated ≈ 0.017 (1.7% return - more conservative)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    if temperature == 1.0:
        return probs
    
    # For scalar values
    if isinstance(probs, (int, float)):
        return probs / temperature
    
    # For numpy arrays
    if isinstance(probs, np.ndarray):
        return probs / temperature
    
    # For pandas Series
    if isinstance(probs, pd.Series):
        return probs / temperature
    
    # For pandas DataFrame (apply to all columns)
    if isinstance(probs, pd.DataFrame):
        return probs / temperature
    
    # Fallback
    return probs / temperature


def logit_temperature_scale(
    probs: Union[np.ndarray, pd.Series],
    temperature: float = 2.9,
    eps: float = 1e-7,
) -> Union[np.ndarray, pd.Series]:
    """
    Apply temperature scaling in logit space (for true probabilities 0-1).
    
    This is the mathematically proper way to apply temperature scaling
    to probability distributions:
    
    1. Convert probabilities to logits: logit = log(p / (1-p))
    2. Divide logits by temperature
    3. Convert back to probabilities: p = sigmoid(logit/T)
    
    Args:
        probs: Probabilities in range [0, 1]
        temperature: Scaling factor (T > 1 softens, T < 1 sharpens)
        eps: Small value to avoid log(0)
    
    Returns:
        Calibrated probabilities in range [0, 1]
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    # Clip to avoid numerical issues
    if isinstance(probs, pd.Series):
        p = probs.clip(lower=eps, upper=1-eps)
    else:
        p = np.clip(probs, eps, 1-eps)
    
    # Convert to logits
    logits = np.log(p / (1 - p))
    
    # Scale logits
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    calibrated = 1 / (1 + np.exp(-scaled_logits))
    
    return calibrated


# ============================================================================
# PREDICTION CALIBRATOR CLASS
# ============================================================================

class PredictionCalibrator:
    """
    Calibrates model predictions using temperature scaling.
    
    Usage:
        calibrator = PredictionCalibrator(temperature=2.9)
        raw_preds = model.predict(X)
        calibrated_preds = calibrator.calibrate(raw_preds)
    """
    
    def __init__(
        self,
        temperature: float = 2.9,
        method: str = "linear",
    ):
        """
        Initialize calibrator.
        
        Args:
            temperature: Temperature scaling factor (default 2.9 from experiments)
            method: Calibration method - "linear" or "logit"
        """
        self.temperature = temperature
        self.method = method
        self._calibration_stats = {}
        
    def calibrate(
        self,
        predictions: Union[np.ndarray, pd.Series, float],
        is_probability: bool = False,
    ) -> Union[np.ndarray, pd.Series, float]:
        """
        Calibrate predictions.
        
        Args:
            predictions: Raw model predictions
            is_probability: If True, use logit-space scaling for probabilities [0,1]
        
        Returns:
            Calibrated predictions
        """
        if is_probability and self.method == "logit":
            return logit_temperature_scale(predictions, self.temperature)
        else:
            return temperature_scale(predictions, self.temperature)
    
    def fit(
        self,
        raw_predictions: np.ndarray,
        true_values: np.ndarray,
        optimize_temperature: bool = True,
    ) -> "PredictionCalibrator":
        """
        Optionally fit the optimal temperature on validation data.
        
        Args:
            raw_predictions: Raw model predictions
            true_values: Actual observed values
            optimize_temperature: If True, find optimal T via grid search
        
        Returns:
            self (fitted calibrator)
        """
        if not optimize_temperature:
            return self
        
        # Grid search for optimal temperature
        best_temp = self.temperature
        best_sharpe = -np.inf
        
        temps_to_try = np.arange(0.5, 5.1, 0.1)
        
        for temp in temps_to_try:
            calibrated = temperature_scale(raw_predictions, temp)
            
            # Calculate Sharpe from predictions
            returns = calibrated * np.sign(true_values)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_temp = temp
        
        self.temperature = best_temp
        self._calibration_stats = {
            "optimal_temperature": best_temp,
            "sharpe_improvement": best_sharpe,
        }
        
        logger.info(f"📊 Fitted optimal temperature: T={best_temp:.2f} (Sharpe: {best_sharpe:.3f})")
        
        return self
    
    def get_stats(self) -> dict:
        """Get calibration statistics."""
        return {
            "temperature": self.temperature,
            "method": self.method,
            **self._calibration_stats,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calibrate_predictions(
    predictions: Union[np.ndarray, pd.Series],
    use_calibration: bool = True,
    temperature: Optional[float] = None,
) -> Tuple[Union[np.ndarray, pd.Series], dict]:
    """
    Convenience function to calibrate predictions with logging.
    
    Args:
        predictions: Raw model predictions
        use_calibration: Whether to apply calibration
        temperature: Temperature to use (default from config)
    
    Returns:
        (calibrated_predictions, metadata_dict)
    """
    from ..config import OPTIMIZED_MODEL_CONFIG
    
    if not use_calibration:
        return predictions, {"calibrated": False, "temperature": None}
    
    # Get temperature from config if not specified
    if temperature is None:
        temperature = OPTIMIZED_MODEL_CONFIG.get("temperature", 2.9)
    
    calibrated = temperature_scale(predictions, temperature)
    
    metadata = {
        "calibrated": True,
        "temperature": temperature,
        "method": "linear",
    }
    
    logger.debug(f"Applied temperature scaling with T={temperature}")
    
    return calibrated, metadata


def get_default_temperature() -> float:
    """Get default temperature from config."""
    from ..config import OPTIMIZED_MODEL_CONFIG
    return OPTIMIZED_MODEL_CONFIG.get("temperature", 2.9)


def should_use_calibration() -> bool:
    """Check if calibration should be used based on config."""
    from ..config import OPTIMIZED_MODEL_CONFIG, is_optimized_mode
    if not is_optimized_mode():
        return False
    return OPTIMIZED_MODEL_CONFIG.get("use_temperature_scaling", True)
