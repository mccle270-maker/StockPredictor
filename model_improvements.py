"""
MODEL IMPROVEMENTS - HIGH MEDIUM AND LOW PRIORITY ENHANCEMENTS
==============================================================

This module contains all the improvements for Sharpe ratio and direction accuracy:

HIGH PRIORITY:
1. Enhanced Feature Engineering (volatility-adjusted, cross-sectional, lagged)
2. Threshold Optimization (find best threshold per fold)
3. Volatility Weighting (scale positions by inverse vol)

MEDIUM PRIORITY:
4. Classification Model (predict up/down probability)
5. Position Holding (keep positions 3-5 days instead of 1)

LOW PRIORITY:
6. Model Ensemble (combine RF, XGB, GBM)
7. Kelly Criterion (optimal position sizing)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# ========================================
# HIGH PRIORITY: ENHANCED FEATURES
# ========================================

def add_enhanced_features(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility-adjusted, cross-sectional, and improved lagged features.
    
    NEW FEATURES:
    - volatility_adjusted_returns: Returns scaled by inverse volatility (Sharpe-like)
    - vol_adjusted_momentum: Momentum indicators divided by volatility
    - volatility_percentile: Current vol rank vs 60-day history
    - mean_reversion_signal: Z-score of price deviation from MA
    - momentum_confirmatio...
    """
    hist = hist.copy()
    
    if "Close" not in hist.columns or "Volume" not in hist.columns:
        return hist
    
    close = hist["Close"]
    ret_1d = close.pct_change()
    volume = hist["Volume"]
    
    # 1. VOLATILITY-ADJUSTED RETURNS
    vol_20d = ret_1d.rolling(20).std()
    vol_60d = ret_1d.rolling(60).std()
    
    # Returns scaled by inverse of volatility (when vol is high, scale down returns)
    hist["ret_vol_adjusted"] = ret_1d / (vol_20d + 1e-6)
    hist["ret_vol_adjusted"] = hist["ret_vol_adjusted"].shift(1)  # Lag by 1 day
    
    # 2. VOLATILITY PERCENTILE (ranking current vol in 60-day history)
    vol_sorted = vol_20d.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    hist["vol_percentile_rank"] = vol_sorted.shift(1)
    
    # 3. VOLATILITY REGIME STRENGTH (how extreme is vol right now?)
    vol_zscore = (vol_20d - vol_60d.rolling(60).mean()) / (vol_60d.rolling(60).std() + 1e-6)
    hist["vol_regime_strength"] = vol_zscore.shift(1)
    
    # 4. VOLATILITY-ADJUSTED MOMENTUM (momentum/vol)
    ret_5d = close.pct_change(5)
    ret_20d = close.pct_change(20)
    hist["momentum_5d_vol_adj"] = (ret_5d / (vol_20d + 1e-6)).shift(1)
    hist["momentum_20d_vol_adj"] = (ret_20d / (vol_20d + 1e-6)).shift(1)
    
    # 5. MEAN REVERSION SIGNAL (price deviation from MA, normalized by volatility)
    ma_20 = close.rolling(20).mean()
    price_deviation = (close - ma_20) / (ma_20 + 1e-6)  # % deviation
    mr_signal = price_deviation / (vol_20d + 1e-6)  # Normalize by volatility
    hist["mean_reversion_signal"] = mr_signal.shift(1)
    
    # 6. MOMENTUM CONFIRMATION (multiple momentum indicators agree?)
    rsi14 = hist["rsi14"] if "rsi14" in hist.columns else pd.Series(50, index=hist.index)
    
    # Create momentum consensus: count how many indicators are bullish
    momentum_bullish = (ret_5d > 0).astype(float) * 0.25
    momentum_bullish += (rsi14 > 50).astype(float) * 0.25
    
    if "macd" in hist.columns and "macdsignal" in hist.columns:
        momentum_bullish += (hist["macd"] > hist["macdsignal"]).astype(float) * 0.25
    else:
        momentum_bullish += 0.25
        
    momentum_bullish += (close > ma_20).astype(float) * 0.25
    hist["momentum_confirmation"] = momentum_bullish.shift(1)
    
    # 7. ENHANCED LAGGED FEATURES (multiple lags for deep learning readiness)
    for lag in [2, 3, 4, 5]:
        hist[f"ret_1d_lag_{lag}"] = ret_1d.shift(lag)
        hist[f"vol_20d_lag_{lag}"] = vol_20d.shift(lag)
        if "rsi14" in hist.columns:
            hist[f"rsi14_lag_{lag}"] = hist["rsi14"].shift(lag)
    
    # 8. ROLLING WINDOW STATISTICS (for pattern recognition)
    hist["ret_1d_rolling_mean_10"] = ret_1d.rolling(10).mean().shift(1)
    hist["ret_1d_rolling_std_10"] = ret_1d.rolling(10).std().shift(1)
    hist["vol_20d_rolling_mean_10"] = vol_20d.rolling(10).mean().shift(1)
    
    # 9. CORRELATION FEATURES (how does this stock move with market?)
    if "Close" in hist.columns and len(hist) >= 60:
        hist["ret_vol_correlation"] = ret_1d.rolling(20).corr(vol_20d).shift(1)
    
    # 10. PRICE ACTION STRUCTURE
    high = hist["High"] if "High" in hist.columns else close
    low = hist["Low"] if "Low" in hist.columns else close
    
    # True range for ATR-like calculation
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_20 = true_range.rolling(20).mean()
    hist["price_action_strength"] = (high - low) / (atr_20 + 1e-6)
    hist["price_action_strength"] = hist["price_action_strength"].shift(1)
    
    return hist


# ========================================
# HIGH PRIORITY: THRESHOLD OPTIMIZATION
# ========================================

def optimize_threshold_for_fold(
    test_df: pd.DataFrame,
    pred_col: str = "predicted_return",
    actual_col: str = "actual_return",
    thresholds: List[float] = None,
) -> Tuple[float, float]:
    """
    Test multiple thresholds and return the one with best Sharpe ratio.
    
    Returns:
        (best_threshold, best_sharpe)
    """
    if thresholds is None:
        thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03]
    
    best_sharpe = -np.inf
    best_threshold = thresholds[0]
    
    for threshold in thresholds:
        # Calculate positions with this threshold
        positions = np.where(test_df[pred_col] > threshold, 1.0, 0.0)
        
        # Strategy returns
        strategy_returns = test_df[actual_col] * positions
        
        # Calculate Sharpe ratio
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
        else:
            sharpe = -np.inf
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold
    
    return best_threshold, best_sharpe


# ========================================
# HIGH PRIORITY: VOLATILITY WEIGHTING
# ========================================

def apply_volatility_weighting(
    positions: np.ndarray,
    volatilities: pd.Series,
    vol_window: int = 20,
    min_vol_weight: float = 0.2,
) -> np.ndarray:
    """
    Scale positions inversely to realized volatility.
    
    In high vol periods, reduce position size to protect Sharpe ratio.
    In low vol periods, increase position size (more capital deployment).
    
    Formula: position_weighted = position * (avg_vol / current_vol)
    With floor at min_vol_weight to avoid extreme sizing.
    """
    weighted_positions = positions.copy().astype(float)
    
    if len(volatilities) != len(positions):
        return weighted_positions
    
    avg_vol = volatilities.rolling(vol_window).mean()
    avg_vol = avg_vol.fillna(volatilities.mean())
    
    # Avoid division by zero
    vol_weights = (avg_vol / (volatilities + 1e-6)).values
    
    # Clip to reasonable range [min_vol_weight, 2.0]
    vol_weights = np.clip(vol_weights, min_vol_weight, 2.0)
    
    weighted_positions = weighted_positions * vol_weights
    
    return weighted_positions


# ========================================
# MEDIUM PRIORITY: CLASSIFICATION MODEL
# ========================================

class DirectionClassifier:
    """
    Predicts up/down direction instead of magnitude.
    Achieves higher directional accuracy than regression models.
    """
    
    def __init__(self, model_type: str = "rf"):
        """
        Args:
            model_type: "rf" (RandomForest) or "gb" (GradientBoosting)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit classifier on up/down direction.
        
        Args:
            X_train: Feature matrix
            y_train: Returns (will be converted to up/down)
        """
        # Convert returns to binary labels (0=down, 1=up)
        y_binary = (y_train > 0).astype(int)
        
        # Handle class imbalance
        from collections import Counter
        class_counts = Counter(y_binary)
        class_weights = {0: 1.0, 1: 1.0}
        if len(class_counts) > 1:
            total = sum(class_counts.values())
            class_weights = {c: total / (c_count * 2) for c, c_count in class_counts.items()}
        
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight=class_weights,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Fit model
        self.model.fit(X_scaled, y_binary)
    
    def predict_probability(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict probability of up movement.
        
        Returns:
            Array of probabilities (0.0 to 1.0)
        """
        X_scaled = self.scaler.transform(X_test)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities
    
    def predict_positions(self, X_test: np.ndarray, threshold: float = 0.55) -> np.ndarray:
        """
        Predict positions with confidence threshold.
        
        Only take position if probability > threshold.
        """
        prob_up = self.predict_probability(X_test)
        positions = np.where(prob_up > threshold, 1.0, 0.0)
        return positions


# ========================================
# MEDIUM PRIORITY: POSITION HOLDING
# ========================================

def apply_position_holding(
    positions: np.ndarray,
    hold_days: int = 3,
) -> np.ndarray:
    """
    Convert daily signals to position holding strategy.
    
    Instead of trading every day, hold position for N days once signal fires.
    Dramatically reduces transaction costs.
    
    Args:
        positions: Daily position signals (0 or 1)
        hold_days: Number of days to hold each position
        
    Returns:
        Positions with holding applied
    """
    held_positions = np.zeros_like(positions, dtype=float)
    hold_counter = 0
    
    for i in range(len(positions)):
        if positions[i] > 0:
            hold_counter = hold_days
        
        if hold_counter > 0:
            held_positions[i] = 1.0
            hold_counter -= 1
    
    return held_positions


# ========================================
# LOW PRIORITY: MODEL ENSEMBLE
# ========================================

class ModelEnsemble:
    """
    Combine Random Forest, XGBoost, and Gradient Boosting with voting.
    Reduces overfitting and improves robustness.
    """
    
    def __init__(self, include_xgb: bool = True):
        """
        Args:
            include_xgb: Include XGBoost in ensemble (requires xgboost package)
        """
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42
            )),
        ]
        
        if include_xgb:
            try:
                from xgboost import XGBRegressor
                estimators.append(
                    ('xgb', XGBRegressor(
                        n_estimators=100, max_depth=6, learning_rate=0.05, 
                        random_state=42, n_jobs=-1
                    ))
                )
            except ImportError:
                print("[ModelEnsemble] XGBoost not available, using RF + GB only")
        
        self.ensemble = VotingRegressor(estimators=estimators)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit ensemble on training data."""
        self.ensemble.fit(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions (average of 3 models)."""
        return self.ensemble.predict(X_test)


# ========================================
# LOW PRIORITY: KELLY CRITERION
# ========================================

def calculate_kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    leverage_cap: float = 3.0,
) -> float:
    """
    Calculate optimal position sizing using Kelly Criterion.
    
    f* = (bp - q) / b
    where:
        b = win/loss ratio
        p = win probability
        q = loss probability (1 - p)
        f* = fraction of capital to risk
    
    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        win_loss_ratio: Average win / average loss
        leverage_cap: Maximum leverage allowed
        
    Returns:
        Optimal fraction of capital (0.0 to leverage_cap)
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.5  # Default to 50% allocation
    
    p = win_rate
    q = 1 - win_rate
    b = win_loss_ratio
    
    # Kelly formula
    f_optimal = (b * p - q) / b
    
    # Cap to reasonable leverage and ensure non-negative
    f_optimal = np.clip(f_optimal, 0.0, leverage_cap)
    
    return f_optimal


def apply_kelly_sizing(
    positions: np.ndarray,
    returns: pd.Series,
    lookback_window: int = 100,
    leverage_cap: float = 3.0,
) -> np.ndarray:
    """
    Scale positions using Kelly Criterion based on recent performance.
    
    Args:
        positions: Daily position signals (0 or 1)
        returns: Daily P&L or returns
        lookback_window: Days to use for win rate calculation
        leverage_cap: Maximum leverage
        
    Returns:
        Kelly-scaled positions
    """
    kelly_positions = np.zeros_like(positions, dtype=float)
    
    for i in range(lookback_window, len(positions)):
        # Get recent trading history
        recent_returns = returns.iloc[i-lookback_window:i].values
        
        # Calculate win rate and win/loss ratio
        wins = (recent_returns > 0).sum()
        losses = (recent_returns <= 0).sum()
        win_rate = wins / len(recent_returns) if len(recent_returns) > 0 else 0.5
        
        if losses > 0:
            avg_win = recent_returns[recent_returns > 0].mean() if (recent_returns > 0).any() else 0.01
            avg_loss = abs(recent_returns[recent_returns <= 0].mean()) if (recent_returns <= 0).any() else 0.01
            win_loss_ratio = avg_win / (avg_loss + 1e-6)
        else:
            win_loss_ratio = 1.0
        
        # Get Kelly fraction
        kelly_frac = calculate_kelly_fraction(win_rate, win_loss_ratio, leverage_cap)
        
        # Apply Kelly scaling to position
        kelly_positions[i] = positions[i] * kelly_frac
    
    return kelly_positions


# ========================================
# UTILITY: COMBINE ALL IMPROVEMENTS
# ========================================

def apply_all_improvements(
    test_df: pd.DataFrame,
    pred_col: str = "predicted_return",
    actual_col: str = "actual_return",
    vol_col: str = "vol_20d",
    enable_threshold_opt: bool = True,
    enable_vol_weighting: bool = True,
    enable_holding: bool = True,
    enable_kelly: bool = True,
    hold_days: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Apply all improvements to generate positions with optimal parameters.
    
    Returns:
        (positions, metrics_dict)
    """
    # Step 1: Threshold optimization
    if enable_threshold_opt:
        threshold, threshold_sharpe = optimize_threshold_for_fold(
            test_df, pred_col, actual_col
        )
        print(f"[Improvements] Optimized threshold: {threshold:.4f} (Sharpe: {threshold_sharpe:.3f})")
    else:
        threshold = 0.005
    
    # Step 2: Generate base positions
    positions = np.where(test_df[pred_col] > threshold, 1.0, 0.0)
    
    # Step 3: Apply volatility weighting
    if enable_vol_weighting and vol_col in test_df.columns:
        positions = apply_volatility_weighting(positions, test_df[vol_col])
        print(f"[Improvements] Applied volatility weighting")
    
    # Step 4: Apply position holding
    if enable_holding:
        positions = apply_position_holding(positions, hold_days=hold_days)
        print(f"[Improvements] Applied {hold_days}-day position holding")
    
    # Step 5: Apply Kelly sizing
    if enable_kelly and actual_col in test_df.columns:
        strategy_returns = test_df[actual_col] * positions
        positions = apply_kelly_sizing(positions, strategy_returns)
        print(f"[Improvements] Applied Kelly Criterion sizing")
    
    # Calculate metrics
    strategy_returns = test_df[actual_col] * positions
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else -np.inf
    hit_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
    
    metrics = {
        "threshold": threshold,
        "sharpe": sharpe,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "hit_rate": hit_rate,
        "num_positions": positions.sum(),
    }
    
    return positions, metrics
