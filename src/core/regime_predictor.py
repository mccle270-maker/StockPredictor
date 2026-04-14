"""
Regime-Aware Predictor
======================

A meta-model that maintains separate sub-models for different market regimes
(BULL, BEAR, NEUTRAL). Each sub-model is trained only on data from its
respective regime, allowing the predictor to learn regime-specific patterns.

Usage:
    from src.core.regime_predictor import RegimeAwarePredictor
    from src.core.regime_filter import MarketRegime
    
    predictor = RegimeAwarePredictor(base_model_type="xgb")
    predictor.fit(X_train, y_train, regimes_train)
    
    predictions = predictor.predict(X_test, current_regime=MarketRegime.BULL)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from copy import deepcopy

import numpy as np
import pandas as pd

from .regime_filter import MarketRegime
from .models import make_model

logger = logging.getLogger(__name__)


# Simplified regime mapping for training
# Combines STRONG_BULL with BULL, STRONG_BEAR/CRASH with BEAR
REGIME_GROUPS = {
    MarketRegime.STRONG_BULL: "BULL",
    MarketRegime.BULL: "BULL",
    MarketRegime.NEUTRAL: "NEUTRAL",
    MarketRegime.BEAR: "BEAR",
    MarketRegime.STRONG_BEAR: "BEAR",
    MarketRegime.CRASH: "BEAR",
}

# Minimum samples required to train a regime-specific model
MIN_SAMPLES_PER_REGIME = 50


class RegimeAwarePredictor:
    """
    A meta-model that trains separate sub-models for each market regime.
    
    This allows the model to learn different patterns during bull markets
    vs bear markets, which may have fundamentally different dynamics.
    
    Attributes:
        base_model_type: Type of base model ('rf', 'xgb', etc.)
        models: Dict mapping regime group to fitted model
        fallback_model: Model trained on NEUTRAL data (or all data if insufficient)
        min_samples: Minimum samples required per regime
        regime_stats: Training statistics per regime
    """
    
    def __init__(
        self,
        base_model_type: str = "xgb",
        min_samples: int = MIN_SAMPLES_PER_REGIME,
        model_kwargs: Optional[Dict[str, Any]] = None,
        use_optimized: bool = True,
    ):
        """
        Initialize the regime-aware predictor.
        
        Args:
            base_model_type: Type of model to use ('rf', 'xgb', 'gbrt', 'linreg')
            min_samples: Minimum samples required to train a regime-specific model
            model_kwargs: Additional kwargs to pass to make_model()
            use_optimized: Whether to use optimized hyperparameters
        """
        self.base_model_type = base_model_type
        self.min_samples = min_samples
        self.model_kwargs = model_kwargs or {}
        self.use_optimized = use_optimized
        
        # Models dict: {"BULL": model, "BEAR": model, "NEUTRAL": model}
        self.models: Dict[str, Any] = {}
        self.fallback_model: Optional[Any] = None
        
        # Training statistics
        self.regime_stats: Dict[str, Dict[str, Any]] = {}
        self.is_fitted = False
        
    def _create_base_model(self) -> Any:
        """Create a fresh instance of the base model."""
        return make_model(
            model_type=self.base_model_type,
            task="reg",
            use_optimized=self.use_optimized,
            **self.model_kwargs
        )
    
    def _normalize_regime(self, regime: Union[MarketRegime, str]) -> str:
        """
        Convert a MarketRegime enum (or string) to a simplified group.
        
        Args:
            regime: MarketRegime enum or string like "bull", "BULL", etc.
            
        Returns:
            One of "BULL", "BEAR", "NEUTRAL"
        """
        if isinstance(regime, MarketRegime):
            return REGIME_GROUPS.get(regime, "NEUTRAL")
        
        # Handle string input
        regime_str = str(regime).upper()
        
        if "STRONG_BULL" in regime_str or regime_str == "BULL":
            return "BULL"
        elif "STRONG_BEAR" in regime_str or "CRASH" in regime_str or regime_str == "BEAR":
            return "BEAR"
        else:
            return "NEUTRAL"
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        regimes: Union[np.ndarray, pd.Series, List[MarketRegime]],
    ) -> "RegimeAwarePredictor":
        """
        Fit separate models for each regime.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            regimes: Market regime for each sample (n_samples,)
                     Can be MarketRegime enums, strings, or a Series
        
        Returns:
            self (fitted predictor)
        """
        # Convert to numpy arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
        
        # Normalize regimes to strings
        if isinstance(regimes, pd.Series):
            regimes_list = regimes.tolist()
        else:
            regimes_list = list(regimes)
        
        regime_groups = np.array([self._normalize_regime(r) for r in regimes_list])
        
        # Reset state
        self.models = {}
        self.regime_stats = {}
        
        # Train a model for each regime group
        for regime_name in ["BULL", "BEAR", "NEUTRAL"]:
            mask = regime_groups == regime_name
            n_samples = mask.sum()
            
            self.regime_stats[regime_name] = {
                "n_samples": int(n_samples),
                "trained": False,
                "fallback_used": False,
            }
            
            if n_samples >= self.min_samples:
                # Enough data - train regime-specific model
                X_regime = X_arr[mask]
                y_regime = y_arr[mask]
                
                model = self._create_base_model()
                model.fit(X_regime, y_regime)
                self.models[regime_name] = model
                
                self.regime_stats[regime_name]["trained"] = True
                logger.info(f"Trained {regime_name} model on {n_samples} samples")
            else:
                logger.warning(
                    f"Insufficient data for {regime_name} regime ({n_samples} < {self.min_samples}). "
                    "Will use fallback model."
                )
        
        # Train fallback model on NEUTRAL data (or all data if NEUTRAL is insufficient)
        neutral_mask = regime_groups == "NEUTRAL"
        n_neutral = neutral_mask.sum()
        
        if n_neutral >= self.min_samples:
            # Use NEUTRAL data for fallback
            X_fallback = X_arr[neutral_mask]
            y_fallback = y_arr[neutral_mask]
            fallback_source = "NEUTRAL"
        else:
            # Use all data for fallback
            X_fallback = X_arr
            y_fallback = y_arr
            fallback_source = "ALL"
            logger.warning(
                f"NEUTRAL regime has insufficient data ({n_neutral}). "
                "Fallback model trained on ALL data."
            )
        
        self.fallback_model = self._create_base_model()
        self.fallback_model.fit(X_fallback, y_fallback)
        
        self.regime_stats["_fallback"] = {
            "source": fallback_source,
            "n_samples": len(X_fallback),
        }
        
        # Mark models that will use fallback
        for regime_name in ["BULL", "BEAR", "NEUTRAL"]:
            if regime_name not in self.models:
                self.regime_stats[regime_name]["fallback_used"] = True
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        current_regime: Union[MarketRegime, str],
    ) -> np.ndarray:
        """
        Generate predictions using the appropriate regime-specific model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            current_regime: Current market regime (single value for all samples)
        
        Returns:
            Predictions array (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeAwarePredictor must be fitted before predict()")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        regime_group = self._normalize_regime(current_regime)
        
        # Get appropriate model
        if regime_group in self.models:
            model = self.models[regime_group]
            logger.debug(f"Using {regime_group} model for prediction")
        else:
            model = self.fallback_model
            logger.debug(f"Using fallback model for {regime_group} (regime model not available)")
        
        return model.predict(X_arr)
    
    def predict_with_regime_array(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        regimes: Union[np.ndarray, pd.Series, List[MarketRegime]],
    ) -> np.ndarray:
        """
        Generate predictions where each sample may have a different regime.
        
        This is useful for backtesting where the regime changes over time.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            regimes: Regime for each sample (n_samples,)
        
        Returns:
            Predictions array (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeAwarePredictor must be fitted before predict()")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        
        # Normalize regimes
        if isinstance(regimes, pd.Series):
            regimes_list = regimes.tolist()
        else:
            regimes_list = list(regimes)
        
        regime_groups = np.array([self._normalize_regime(r) for r in regimes_list])
        
        # Initialize predictions
        predictions = np.zeros(len(X_arr))
        
        # Predict in batches by regime
        for regime_name in np.unique(regime_groups):
            mask = regime_groups == regime_name
            
            if regime_name in self.models:
                model = self.models[regime_name]
            else:
                model = self.fallback_model
            
            predictions[mask] = model.predict(X_arr[mask])
        
        return predictions
    
    def get_feature_importances(
        self,
        regime: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get feature importances from regime-specific models.
        
        Args:
            regime: If provided, return only that regime's importances.
                    Otherwise, return all regimes.
        
        Returns:
            Dict mapping regime name to feature importance array
        """
        if not self.is_fitted:
            raise RuntimeError("RegimeAwarePredictor must be fitted first")
        
        importances = {}
        
        for regime_name, model in self.models.items():
            if regime is not None and regime_name != regime:
                continue
            
            if hasattr(model, "feature_importances_"):
                importances[regime_name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances[regime_name] = np.abs(model.coef_)
        
        # Include fallback if requested
        if regime is None or regime == "_fallback":
            if self.fallback_model is not None:
                if hasattr(self.fallback_model, "feature_importances_"):
                    importances["_fallback"] = self.fallback_model.feature_importances_
                elif hasattr(self.fallback_model, "coef_"):
                    importances["_fallback"] = np.abs(self.fallback_model.coef_)
        
        return importances
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dict with per-regime statistics
        """
        return {
            "base_model_type": self.base_model_type,
            "min_samples": self.min_samples,
            "is_fitted": self.is_fitted,
            "regimes": self.regime_stats.copy(),
        }
    
    def get_model_for_regime(
        self,
        regime: Union[MarketRegime, str],
    ) -> Optional[Any]:
        """
        Get the underlying model for a specific regime.
        
        Useful for accessing model internals like feature_importances_
        or for computing prediction uncertainty.
        
        Args:
            regime: Market regime to get model for
            
        Returns:
            The fitted model for that regime, or fallback if not available,
            or None if not fitted
        """
        if not self.is_fitted:
            return None
        
        regime_group = self._normalize_regime(regime)
        
        if regime_group in self.models:
            return self.models[regime_group]
        else:
            return self.fallback_model
    
    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"RegimeAwarePredictor(base_model={self.base_model_type}, fitted=False)"
        
        trained = [k for k, v in self.regime_stats.items() 
                   if k != "_fallback" and v.get("trained", False)]
        return (
            f"RegimeAwarePredictor(base_model={self.base_model_type}, "
            f"trained_regimes={trained})"
        )
