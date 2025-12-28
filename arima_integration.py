"""
ARIMA Ensemble Integration
Auto-ARIMA for time series predictions blended with ML models
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None


class ARIMAPredictor:
    """Automatic ARIMA for single stock prediction"""
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5, 
                 seasonal: bool = False, verbose: bool = False):
        """
        Args:
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to use seasonal ARIMA
            verbose: Print fitting details
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.verbose = verbose
        self.model = None
        self.fitted = False
    
    def fit(self, returns_series: pd.Series) -> bool:
        """
        Fit ARIMA model to return series
        
        Args:
            returns_series: pd.Series of returns (e.g., daily log returns)
            
        Returns:
            True if fit successful, False otherwise
        """
        if auto_arima is None:
            print("⚠️  pmdarima not installed, cannot fit ARIMA")
            return False
        
        try:
            clean_returns = returns_series.dropna()
            
            if len(clean_returns) < 20:
                print(f"⚠️  Not enough data to fit ARIMA (need 20+, have {len(clean_returns)})")
                return False
            
            self.model = auto_arima(
                clean_returns,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal=self.seasonal,
                stepwise=True,
                trace=self.verbose,
                error_action="ignore",
                suppress_warnings=True,
                information_criterion="aic"
            )
            
            self.fitted = True
            
            if self.verbose:
                print(f"✅ ARIMA{self.model.order} fitted successfully")
            
            return True
            
        except Exception as e:
            print(f"⚠️  ARIMA fit failed: {e}")
            return False
    
    def predict(self, steps: int = 1) -> Optional[np.ndarray]:
        """
        Forecast next N days
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predictions, or None if not fitted
        """
        if not self.fitted or self.model is None:
            return None
        
        try:
            forecast, conf_int = self.model.get_forecast(steps=steps)
            return forecast.values
        except Exception as e:
            print(f"⚠️  ARIMA prediction failed: {e}")
            return None
    
    def get_fitted_order(self) -> Optional[Tuple[int, int, int]]:
        """Get the fitted ARIMA order (p, d, q)"""
        if self.model is None:
            return None
        return self.model.order


class EnsemblePredictor:
    """Blend ML predictions with ARIMA forecasts"""
    
    def __init__(self, ml_weight: float = 0.7, arima_weight: float = 0.3):
        """
        Args:
            ml_weight: Weight for ML predictions (default 70%)
            arima_weight: Weight for ARIMA predictions (default 30%)
        """
        self.ml_weight = ml_weight
        self.arima_weight = arima_weight
        self.arima_predictor = None
        self.ml_predictor = None
    
    def fit_arima(self, returns_series: pd.Series) -> bool:
        """Fit ARIMA component"""
        self.arima_predictor = ARIMAPredictor(verbose=False)
        return self.arima_predictor.fit(returns_series)
    
    def blend_predictions(self, ml_pred: float, arima_pred: Optional[float] = None) -> float:
        """
        Blend ML and ARIMA predictions
        
        Args:
            ml_pred: ML model prediction
            arima_pred: ARIMA prediction (if available)
            
        Returns:
            Blended prediction
        """
        if arima_pred is None or np.isnan(arima_pred):
            return ml_pred
        
        # Normalize weights
        total_weight = self.ml_weight + self.arima_weight
        ml_w = self.ml_weight / total_weight
        arima_w = self.arima_weight / total_weight
        
        return ml_w * ml_pred + arima_w * arima_pred
    
    def get_weights(self) -> dict:
        """Get ensemble weights"""
        total = self.ml_weight + self.arima_weight
        return {
            "ml": self.ml_weight / total,
            "arima": self.arima_weight / total
        }


def add_arima_features(hist: pd.DataFrame, target_col: str = "ret_1d", 
                       arima_horizons: list = [1, 5, 20]) -> pd.DataFrame:
    """
    Add ARIMA predictions as features
    
    Args:
        hist: DataFrame with price/return data
        target_col: Column to use for ARIMA training
        arima_horizons: List of horizons to predict (1d, 5d, 20d)
        
    Returns:
        DataFrame with ARIMA predictions added as features
    """
    hist = hist.copy()
    
    if target_col not in hist.columns:
        print(f"⚠️  Column '{target_col}' not found, skipping ARIMA features")
        return hist
    
    returns = hist[target_col]
    
    try:
        # Fit ARIMA
        predictor = ARIMAPredictor(max_p=3, max_d=1, max_q=3, verbose=False)
        
        if not predictor.fit(returns):
            print("⚠️  Could not fit ARIMA, skipping ARIMA features")
            return hist
        
        # Generate predictions for different horizons
        for horizon in arima_horizons:
            pred = predictor.predict(steps=horizon)
            if pred is not None:
                # Add to features (lagged so no look-ahead)
                hist[f"arima_pred_{horizon}d"] = np.nan
                hist.iloc[:-horizon, hist.columns.get_loc(f"arima_pred_{horizon}d")] = pred[:-horizon] if len(pred) > horizon else pred
                hist[f"arima_pred_{horizon}d"] = hist[f"arima_pred_{horizon}d"].shift(1)
        
        print(f"✅ Added ARIMA predictions (order={predictor.get_fitted_order()})")
        
    except Exception as e:
        print(f"⚠️  Could not add ARIMA features: {e}")
    
    return hist
