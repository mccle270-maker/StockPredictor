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
            # pmdarima's ARIMA uses .predict() with n_periods parameter
            forecast = self.model.predict(n_periods=steps)
            return np.array(forecast)
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


# =============================================================================
# VOLATILITY FORECASTING (ARIMA on realized vol - actually useful!)
# =============================================================================

class VolatilityForecaster:
    """
    ARIMA-based volatility forecaster.
    
    Unlike returns, volatility exhibits strong autocorrelation (volatility clustering),
    making ARIMA actually useful for predicting future volatility levels.
    
    This is valuable for:
    - Options strategies: Compare forecast vol to IV to find rich/cheap options
    - Position sizing: Reduce size when vol expected to spike
    - Risk management: Anticipate drawdown periods
    """
    
    def __init__(self, lookback: int = 60, vol_window: int = 20, verbose: bool = False):
        """
        Args:
            lookback: Days of vol history to use for ARIMA fitting
            vol_window: Window for calculating realized volatility (default 20d)
            verbose: Print fitting details
        """
        self.lookback = lookback
        self.vol_window = vol_window
        self.verbose = verbose
        self.arima = None
        self.last_vol = None
        self.vol_mean = None
        self.vol_std = None
    
    def fit_and_forecast(self, price_series: pd.Series, horizon: int = 5) -> dict:
        """
        Fit ARIMA on realized volatility and forecast.
        
        Args:
            price_series: Series of closing prices
            horizon: Days ahead to forecast (default 5)
            
        Returns:
            dict with vol_forecast, vol_direction, vol_zscore, etc.
        """
        result = {
            "vol_forecast": None,
            "vol_current": None,
            "vol_direction": "neutral",  # up, down, neutral
            "vol_change_pct": None,
            "vol_zscore": None,
            "vol_regime": "normal",  # low, normal, high, extreme
            "arima_order": None,
            "success": False
        }
        
        if price_series is None or len(price_series) < self.lookback + self.vol_window:
            return result
        
        try:
            # Calculate realized volatility series
            returns = price_series.pct_change().dropna()
            vol_series = returns.rolling(window=self.vol_window).std() * np.sqrt(252)  # Annualized
            vol_series = vol_series.dropna()
            
            if len(vol_series) < self.lookback:
                return result
            
            # Use last N days for fitting
            vol_to_fit = vol_series.tail(self.lookback)
            self.last_vol = float(vol_to_fit.iloc[-1])
            self.vol_mean = float(vol_to_fit.mean())
            self.vol_std = float(vol_to_fit.std())
            
            # Fit ARIMA on volatility
            self.arima = ARIMAPredictor(max_p=3, max_d=1, max_q=3, verbose=self.verbose)
            
            if not self.arima.fit(vol_to_fit):
                # Fallback: use simple mean reversion
                result["vol_forecast"] = self.vol_mean
                result["vol_current"] = self.last_vol
                result["success"] = True
                return result
            
            # Forecast volatility
            forecast = self.arima.predict(steps=horizon)
            if forecast is not None and len(forecast) > 0:
                # Average forecast over horizon
                vol_forecast = float(np.mean(forecast))
                
                result["vol_forecast"] = vol_forecast
                result["vol_current"] = self.last_vol
                result["vol_change_pct"] = (vol_forecast - self.last_vol) / self.last_vol * 100
                result["arima_order"] = self.arima.get_fitted_order()
                result["success"] = True
                
                # Determine direction
                change = vol_forecast - self.last_vol
                if change > self.vol_std * 0.5:
                    result["vol_direction"] = "up"
                elif change < -self.vol_std * 0.5:
                    result["vol_direction"] = "down"
                else:
                    result["vol_direction"] = "neutral"
                
                # Calculate z-score of current vol vs history
                if self.vol_std > 0:
                    result["vol_zscore"] = (self.last_vol - self.vol_mean) / self.vol_std
                
                # Determine regime
                zscore = result["vol_zscore"] or 0
                if zscore < -1:
                    result["vol_regime"] = "low"
                elif zscore > 2:
                    result["vol_regime"] = "extreme"
                elif zscore > 1:
                    result["vol_regime"] = "high"
                else:
                    result["vol_regime"] = "normal"
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️ VolatilityForecaster error: {e}")
        
        return result


# =============================================================================
# TREND STRUCTURE DETECTION (ARIMA on smoothed momentum)
# =============================================================================

class TrendStructureDetector:
    """
    Detect structural trends using ARIMA on smoothed momentum.
    
    Uses ARIMA to analyze if there's persistent structure in:
    - Price momentum (are trends continuing or mean-reverting?)
    - Z-score trajectory (are prediction signals strengthening?)
    
    This helps confirm or question ML predictions:
    - If ML says BUY but trend structure is DOWN → lower confidence
    - If ML says BUY and trend structure is UP → higher confidence
    """
    
    def __init__(self, momentum_window: int = 10, smooth_window: int = 5, verbose: bool = False):
        """
        Args:
            momentum_window: Window for momentum calculation
            smooth_window: Window for smoothing momentum
            verbose: Print details
        """
        self.momentum_window = momentum_window
        self.smooth_window = smooth_window
        self.verbose = verbose
    
    def analyze_trend(self, price_series: pd.Series, horizon: int = 5) -> dict:
        """
        Analyze trend structure in price momentum.
        
        Args:
            price_series: Series of closing prices
            horizon: Forecast horizon
            
        Returns:
            dict with trend_direction, trend_strength, structure_detected, etc.
        """
        result = {
            "trend_direction": "neutral",  # up, down, neutral
            "trend_strength": 0.0,  # 0-1 scale
            "structure_detected": False,  # True if ARIMA found non-trivial order
            "momentum_forecast": None,
            "arima_order": None,
            "momentum_current": None,
            "success": False
        }
        
        if price_series is None or len(price_series) < 60:
            return result
        
        try:
            # Calculate momentum (rate of change)
            momentum = price_series.pct_change(self.momentum_window)
            
            # Smooth the momentum to reduce noise
            smooth_momentum = momentum.rolling(window=self.smooth_window).mean()
            smooth_momentum = smooth_momentum.dropna()
            
            if len(smooth_momentum) < 30:
                return result
            
            # Use last 60 days of smoothed momentum
            mom_to_fit = smooth_momentum.tail(60)
            current_momentum = float(mom_to_fit.iloc[-1])
            result["momentum_current"] = current_momentum
            
            # Fit ARIMA on smoothed momentum
            arima = ARIMAPredictor(max_p=3, max_d=1, max_q=3, verbose=self.verbose)
            
            if arima.fit(mom_to_fit):
                order = arima.get_fitted_order()
                result["arima_order"] = order
                
                # Check if structure was detected (not 0,0,0)
                if order != (0, 0, 0):
                    result["structure_detected"] = True
                
                # Forecast momentum
                forecast = arima.predict(steps=horizon)
                if forecast is not None and len(forecast) > 0:
                    mom_forecast = float(np.mean(forecast))
                    result["momentum_forecast"] = mom_forecast
                    result["success"] = True
                    
                    # Determine trend direction based on forecast
                    mom_std = float(mom_to_fit.std())
                    if mom_forecast > mom_std * 0.5:
                        result["trend_direction"] = "up"
                    elif mom_forecast < -mom_std * 0.5:
                        result["trend_direction"] = "down"
                    else:
                        result["trend_direction"] = "neutral"
                    
                    # Trend strength (0-1 based on magnitude relative to history)
                    if mom_std > 0:
                        result["trend_strength"] = min(1.0, abs(mom_forecast) / (2 * mom_std))
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ TrendStructureDetector error: {e}")
        
        return result
    
    def analyze_zscore_trajectory(self, zscore_series: pd.Series, horizon: int = 5) -> dict:
        """
        Analyze if z-scores are trending (getting stronger/weaker).
        
        Args:
            zscore_series: Series of prediction z-scores over time
            horizon: Forecast horizon
            
        Returns:
            dict with zscore_direction, strengthening, etc.
        """
        result = {
            "zscore_direction": "neutral",  # strengthening, weakening, neutral
            "zscore_forecast": None,
            "structure_detected": False,
            "arima_order": None,
            "success": False
        }
        
        if zscore_series is None or len(zscore_series) < 20:
            return result
        
        try:
            clean_zscores = zscore_series.dropna().tail(60)
            
            if len(clean_zscores) < 20:
                return result
            
            arima = ARIMAPredictor(max_p=2, max_d=1, max_q=2, verbose=self.verbose)
            
            if arima.fit(clean_zscores):
                order = arima.get_fitted_order()
                result["arima_order"] = order
                
                if order != (0, 0, 0):
                    result["structure_detected"] = True
                
                forecast = arima.predict(steps=horizon)
                if forecast is not None and len(forecast) > 0:
                    zscore_forecast = float(np.mean(forecast))
                    current_zscore = float(clean_zscores.iloc[-1])
                    result["zscore_forecast"] = zscore_forecast
                    result["success"] = True
                    
                    # Determine if signal is strengthening or weakening
                    change = zscore_forecast - current_zscore
                    if abs(zscore_forecast) > abs(current_zscore) and change * current_zscore > 0:
                        result["zscore_direction"] = "strengthening"
                    elif abs(zscore_forecast) < abs(current_zscore):
                        result["zscore_direction"] = "weakening"
                    else:
                        result["zscore_direction"] = "neutral"
                        
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Z-score trajectory analysis error: {e}")
        
        return result


# =============================================================================
# UNIFIED ARIMA ANALYSIS FOR OPTIONS STRATEGIES
# =============================================================================

def get_arima_signals(price_series: pd.Series, horizon: int = 5, verbose: bool = False) -> dict:
    """
    Get comprehensive ARIMA-based signals for options and trading.
    
    Combines:
    - Volatility forecast (for IV vs RV decisions)
    - Trend structure (for directional confirmation)
    
    Args:
        price_series: Series of closing prices
        horizon: Forecast horizon in days
        verbose: Print details
        
    Returns:
        dict with all ARIMA signals for options strategies
    """
    result = {
        # Volatility signals
        "vol_forecast": None,
        "vol_current": None,
        "vol_direction": "neutral",
        "vol_change_pct": None,
        "vol_regime": "normal",
        
        # Trend signals  
        "trend_direction": "neutral",
        "trend_strength": 0.0,
        "trend_structure_detected": False,
        
        # Options recommendations
        "iv_recommendation": "neutral",  # sell_premium, buy_premium, neutral
        "direction_confirmation": "none",  # confirms, contradicts, none
        
        # Metadata
        "success": False
    }
    
    if price_series is None or len(price_series) < 60:
        return result
    
    # Get volatility forecast
    vol_forecaster = VolatilityForecaster(verbose=verbose)
    vol_result = vol_forecaster.fit_and_forecast(price_series, horizon)
    
    if vol_result["success"]:
        result["vol_forecast"] = vol_result["vol_forecast"]
        result["vol_current"] = vol_result["vol_current"]
        result["vol_direction"] = vol_result["vol_direction"]
        result["vol_change_pct"] = vol_result["vol_change_pct"]
        result["vol_regime"] = vol_result["vol_regime"]
        
        # IV recommendation based on vol forecast
        if vol_result["vol_direction"] == "down" and vol_result["vol_regime"] in ["high", "extreme"]:
            result["iv_recommendation"] = "sell_premium"  # Vol expected to drop from high levels
        elif vol_result["vol_direction"] == "up" and vol_result["vol_regime"] == "low":
            result["iv_recommendation"] = "buy_premium"  # Vol expected to rise from low levels
        else:
            result["iv_recommendation"] = "neutral"
    
    # Get trend structure
    trend_detector = TrendStructureDetector(verbose=verbose)
    trend_result = trend_detector.analyze_trend(price_series, horizon)
    
    if trend_result["success"]:
        result["trend_direction"] = trend_result["trend_direction"]
        result["trend_strength"] = trend_result["trend_strength"]
        result["trend_structure_detected"] = trend_result["structure_detected"]
        result["success"] = True
    
    return result


def check_direction_confirmation(ml_direction: str, arima_signals: dict) -> dict:
    """
    Check if ARIMA trend confirms or contradicts ML prediction.
    
    Args:
        ml_direction: "bullish", "bearish", or "neutral" from ML model
        arima_signals: Output from get_arima_signals()
        
    Returns:
        dict with confirmation status and adjusted confidence
    """
    result = {
        "confirmation": "none",  # confirms, contradicts, none
        "confidence_adjustment": 0.0,  # -0.2 to +0.2
        "reasoning": ""
    }
    
    trend_dir = arima_signals.get("trend_direction", "neutral")
    trend_strength = arima_signals.get("trend_strength", 0.0)
    structure_detected = arima_signals.get("trend_structure_detected", False)
    
    # No structure detected = no adjustment
    if not structure_detected or trend_dir == "neutral":
        result["reasoning"] = "No clear trend structure detected"
        return result
    
    ml_dir_normalized = ml_direction.lower()
    
    # Check for confirmation
    if ml_dir_normalized == "bullish" and trend_dir == "up":
        result["confirmation"] = "confirms"
        result["confidence_adjustment"] = 0.1 + (0.1 * trend_strength)
        result["reasoning"] = f"ARIMA confirms upward trend (strength: {trend_strength:.0%})"
    elif ml_dir_normalized == "bearish" and trend_dir == "down":
        result["confirmation"] = "confirms"
        result["confidence_adjustment"] = 0.1 + (0.1 * trend_strength)
        result["reasoning"] = f"ARIMA confirms downward trend (strength: {trend_strength:.0%})"
    
    # Check for contradiction
    elif ml_dir_normalized == "bullish" and trend_dir == "down":
        result["confirmation"] = "contradicts"
        result["confidence_adjustment"] = -0.1 - (0.1 * trend_strength)
        result["reasoning"] = f"ARIMA shows downward trend, contradicting bullish ML signal"
    elif ml_dir_normalized == "bearish" and trend_dir == "up":
        result["confirmation"] = "contradicts"
        result["confidence_adjustment"] = -0.1 - (0.1 * trend_strength)
        result["reasoning"] = f"ARIMA shows upward trend, contradicting bearish ML signal"
    
    return result
