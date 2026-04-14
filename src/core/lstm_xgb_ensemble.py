"""
Hybrid LSTM + XGB Ensemble Model.

Combines the temporal pattern learning of LSTM with the 
feature-based predictions of XGBoost for improved accuracy.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

# Check for TensorFlow
try:
    from .lstm_model import LSTMWrapper, create_lstm_model, HAS_TF
except ImportError:
    HAS_TF = False

# Check for XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class LSTMXGBEnsemble:
    """
    Ensemble combining LSTM and XGBoost predictions.
    
    This model leverages:
    - LSTM: Captures temporal patterns and sequences
    - XGBoost: Captures feature interactions and non-linear relationships
    
    The ensemble averages predictions with configurable weights.
    
    Attributes:
        lstm_weight: Weight for LSTM predictions (default 0.5)
        xgb_weight: Weight for XGBoost predictions (default 0.5)
        lstm_config: Configuration dict for LSTM model
        xgb_config: Configuration dict for XGBoost model
        
    Example:
        >>> ensemble = LSTMXGBEnsemble(lstm_weight=0.6, xgb_weight=0.4)
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        lstm_weight: float = 0.5,
        xgb_weight: float = 0.5,
        lstm_config: Optional[Dict] = None,
        xgb_config: Optional[Dict] = None,
        random_state: int = 42,
    ):
        """
        Initialize the LSTM+XGB ensemble.
        
        Args:
            lstm_weight: Weight for LSTM predictions (0-1)
            xgb_weight: Weight for XGBoost predictions (0-1)
            lstm_config: Override LSTM hyperparameters
            xgb_config: Override XGBoost hyperparameters  
            random_state: Random seed for reproducibility
        """
        if not HAS_TF:
            raise ImportError("TensorFlow required for LSTM. Install: pip install tensorflow")
        if not HAS_XGB:
            raise ImportError("XGBoost required. Install: pip install xgboost")
        
        # Normalize weights
        total = lstm_weight + xgb_weight
        self.lstm_weight = lstm_weight / total
        self.xgb_weight = xgb_weight / total
        self.random_state = random_state
        
        # Default LSTM config (optimized small version)
        self.lstm_config = {
            'lookback': 30,
            'lstm_units': 32,
            'dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'patience': 10,
            'learning_rate': 0.001,
            'n_layers': 2,
            'verbose': 0,
            'random_state': random_state,
        }
        if lstm_config:
            self.lstm_config.update(lstm_config)
        
        # Default XGBoost config (optimized)
        self.xgb_config = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': random_state,
            'n_jobs': -1,
        }
        if xgb_config:
            self.xgb_config.update(xgb_config)
        
        self.lstm_model: Optional[LSTMWrapper] = None
        self.xgb_model: Optional[XGBRegressor] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "LSTMXGBEnsemble":
        """
        Fit both LSTM and XGBoost models.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional arguments
            
        Returns:
            self
        """
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        
        # Convert to numpy if needed
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y
        
        # Train LSTM
        logger.info("Training LSTM component...")
        print("🧠 Training LSTM component...")
        self.lstm_model = create_lstm_model(**self.lstm_config)
        try:
            self.lstm_model.fit(X, y)
            print("  ✅ LSTM trained successfully")
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            print(f"  ⚠️ LSTM training failed: {e}")
            self.lstm_model = None
        
        # Train XGBoost
        logger.info("Training XGBoost component...")
        print("🌲 Training XGBoost component...")
        self.xgb_model = XGBRegressor(**self.xgb_config)
        try:
            self.xgb_model.fit(X_arr, y_arr)
            print("  ✅ XGBoost trained successfully")
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
            print(f"  ⚠️ XGBoost training failed: {e}")
            self.xgb_model = None
        
        # Check at least one model trained
        if self.lstm_model is None and self.xgb_model is None:
            raise RuntimeError("Both LSTM and XGBoost training failed")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Combines LSTM and XGBoost predictions using weighted average.
        If one model fails, uses the other model's prediction alone.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_arr = X.values if hasattr(X, 'values') else X
        predictions = []
        weights = []
        
        # Get LSTM predictions
        if self.lstm_model is not None:
            try:
                lstm_pred = self.lstm_model.predict(X)
                if len(lstm_pred) == 1 and len(X) > 1:
                    # LSTM returns single prediction, broadcast
                    lstm_pred = np.full(len(X), lstm_pred[0])
                predictions.append(lstm_pred)
                weights.append(self.lstm_weight)
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
        
        # Get XGBoost predictions
        if self.xgb_model is not None:
            try:
                xgb_pred = self.xgb_model.predict(X_arr)
                predictions.append(xgb_pred)
                weights.append(self.xgb_weight)
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
        
        if not predictions:
            raise RuntimeError("Both models failed to predict")
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)  # Normalize
        ensemble_pred = np.zeros(len(predictions[0]))
        for pred, weight in zip(predictions, weights):
            # Handle length mismatch
            if len(pred) != len(ensemble_pred):
                pred = np.resize(pred, len(ensemble_pred))
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict_single(self, X: pd.DataFrame) -> float:
        """
        Predict for a single sample (most recent data point).
        
        Args:
            X: Feature DataFrame with lookback rows
            
        Returns:
            Single prediction value
        """
        preds = self.predict(X)
        return float(preds[-1]) if len(preds) > 0 else 0.0
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (sklearn compatibility)."""
        return {
            'lstm_weight': self.lstm_weight,
            'xgb_weight': self.xgb_weight,
            'lstm_config': self.lstm_config,
            'xgb_config': self.xgb_config,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params) -> "LSTMXGBEnsemble":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from XGBoost component."""
        if self.xgb_model is None or self.feature_names is None:
            return None
        
        importance = self.xgb_model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


def create_lstm_xgb_ensemble(
    lstm_weight: float = 0.5,
    xgb_weight: float = 0.5,
    **kwargs
) -> LSTMXGBEnsemble:
    """
    Factory function to create LSTM+XGB ensemble.
    
    Args:
        lstm_weight: Weight for LSTM (default 0.5)
        xgb_weight: Weight for XGBoost (default 0.5)
        **kwargs: Additional config overrides
        
    Returns:
        LSTMXGBEnsemble instance
    """
    return LSTMXGBEnsemble(
        lstm_weight=lstm_weight,
        xgb_weight=xgb_weight,
        **kwargs
    )
