"""
Option Strategy Generator
=========================

Generate option trading strategies based on predictions and Greeks.
This module provides rule-based strategy recommendations.
"""

from __future__ import annotations

import pandas as pd
from typing import Union, List, Dict, Any, Optional


def classify_delta_strike(delta: float, is_call: bool = True) -> str:
    """
    Classify strike type based on delta value.
    
    Args:
        delta: Option delta value
        is_call: True for calls, False for puts
    
    Returns:
        Strike classification: "ATM", "OTM", or "ITM"
    """
    abs_delta = abs(delta) if delta is not None else 0.5
    
    if is_call:
        # Calls: 0.4–0.6 = ATM, 0.2–0.4 = OTM, >0.6 = ITM
        if 0.4 <= abs_delta <= 0.6:
            return "ATM"
        elif 0.2 <= abs_delta < 0.4:
            return "OTM"
        elif abs_delta > 0.6:
            return "ITM"
        else:
            return "Deep OTM"
    else:
        # Puts: -0.4 to -0.6 = ATM, -0.2 to -0.4 = OTM, <-0.6 = ITM
        if 0.4 <= abs_delta <= 0.6:
            return "ATM"
        elif 0.2 <= abs_delta < 0.4:
            return "OTM"
        elif abs_delta > 0.6:
            return "ITM"
        else:
            return "Deep OTM"


def determine_strategy(
    pred_return: float,
    confidence: float,
    iv_realized_diff: float,
    delta: Optional[float] = None,
) -> tuple[str, str, str]:
    """
    Determine the recommended option strategy based on prediction metrics.
    
    Args:
        pred_return: Predicted return (e.g., 0.03 = 3%)
        confidence: Confidence score (0-1)
        iv_realized_diff: IV minus realized vol (positive = IV rich)
        delta: Option delta for strike selection
    
    Returns:
        Tuple of (recommended_action, directional_bias, suggested_strike)
    """
    # Default values
    suggested_strike = "ATM"
    
    # High confidence directional trades
    if confidence > 0.5:
        if pred_return > 0.03:
            # Strong bullish signal
            if iv_realized_diff > 0.2:
                # IV is rich, use spreads to reduce vega exposure
                action = "Bull Call Spread"
            else:
                action = "Buy Call"
            bias = "Bullish"
            suggested_strike = classify_delta_strike(delta if delta else 0.5, is_call=True)
            
        elif pred_return < -0.03:
            # Strong bearish signal
            if iv_realized_diff > 0.2:
                # IV is rich, use spreads
                action = "Bear Put Spread"
            else:
                action = "Buy Put"
            bias = "Bearish"
            suggested_strike = classify_delta_strike(delta if delta else -0.5, is_call=False)
            
        else:
            # Weak directional signal but high confidence
            if iv_realized_diff > 0.2:
                # IV rich, sell premium
                action = "Iron Condor"
                bias = "Neutral"
            else:
                # Low IV, consider calendar or hold
                action = "Hold / No Trade"
                bias = "Neutral"
    
    else:
        # Low confidence - prefer neutral/credit strategies
        if iv_realized_diff > 0.2:
            # IV is rich, harvest premium
            if pred_return > 0.01:
                action = "Bull Put Spread (Credit)"
                bias = "Slightly Bullish"
            elif pred_return < -0.01:
                action = "Bear Call Spread (Credit)"
                bias = "Slightly Bearish"
            else:
                action = "Iron Condor"
                bias = "Neutral"
        else:
            # Low confidence, low IV - no edge
            action = "Hold / No Trade"
            bias = "Neutral"
    
    return action, bias, suggested_strike


def generate_option_strategy(
    predictions: Union[pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Generate option trading strategies from predictions.
    
    Args:
        predictions: DataFrame or list of dicts containing:
            - symbol/ticker: Stock symbol
            - pred_return: Predicted return
            - p_up: Probability of up move (optional)
            - confidence: Confidence score (0-1)
            - delta, gamma, theta, vega, rho: Greeks (optional)
            - iv: Implied volatility
            - put_call_oi: Put/call open interest ratio (optional)
            - iv_realized_diff: IV minus realized vol
            - theo_price: Theoretical option price
    
    Returns:
        DataFrame with columns:
            - symbol
            - recommended_action
            - suggested_strike
            - directional_bias
            - estimated_cost
            - confidence
            - IV_richness
    """
    # Convert to DataFrame if needed
    if isinstance(predictions, list):
        df = pd.DataFrame(predictions)
    else:
        df = predictions.copy()
    
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol", "recommended_action", "suggested_strike",
            "directional_bias", "estimated_cost", "confidence", "IV_richness"
        ])
    
    # Normalize column names (handle variations)
    column_mapping = {
        "ticker": "symbol",
        "pred_next_ret": "pred_return",
        "prob_up": "p_up",
        "confidence_score": "confidence",
        "atm_iv": "iv",
        "put_call_oi_ratio": "put_call_oi",
        "theo_atm_call_price": "theo_price",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    results = []
    
    def safe_float(val, default=0.0):
        """Safely convert a value to float, handling Series and None."""
        if val is None:
            return default
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else default
        try:
            result = float(val)
            return result if pd.notna(result) else default
        except (ValueError, TypeError):
            return default
    
    def safe_str(val, default="UNKNOWN"):
        """Safely convert a value to string, handling Series."""
        if val is None:
            return default
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else default
        return str(val) if val is not None else default
    
    for _, row in df.iterrows():
        # Extract values with defaults using safe conversions
        symbol = safe_str(row.get("symbol", row.get("ticker", "UNKNOWN"))).upper()
        pred_return = safe_float(row.get("pred_return", 0.0), 0.0)
        confidence = safe_float(row.get("confidence", 0.5), 0.5)
        
        # Greeks
        delta = row.get("delta")
        if delta is not None:
            try:
                delta = safe_float(delta, None)
            except (ValueError, TypeError):
                delta = None
        
        # IV metrics
        iv = safe_float(row.get("iv", 0.0), 0.0)
        iv_realized_diff = safe_float(row.get("iv_realized_diff", 0.0), 0.0)
        
        # If iv_realized_diff not provided, estimate from iv
        if iv_realized_diff == 0.0 and iv > 0:
            # Assume realized vol is roughly 80% of IV on average
            iv_realized_diff = iv - (iv * 0.8)
        
        # Theo price / estimated cost - use safe conversion
        estimated_cost = safe_float(row.get("theo_price"), None)
        
        # Determine strategy
        action, bias, strike = determine_strategy(
            pred_return=pred_return,
            confidence=confidence,
            iv_realized_diff=iv_realized_diff,
            delta=delta,
        )
        
        results.append({
            "symbol": symbol,
            "recommended_action": action,
            "suggested_strike": strike,
            "directional_bias": bias,
            "estimated_cost": estimated_cost,
            "confidence": round(confidence, 4) if confidence is not None else 0.0,
            "IV_richness": round(iv_realized_diff, 4) if iv_realized_diff is not None else 0.0,
        })
    
    return pd.DataFrame(results)


def get_strategy_summary(strategy_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of generated strategies.
    
    Args:
        strategy_df: Output from generate_option_strategy()
    
    Returns:
        Dict with summary statistics
    """
    if strategy_df.empty:
        return {
            "total_strategies": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "avg_confidence": 0.0,
            "total_estimated_cost": 0.0,
            "strategy_breakdown": {},
        }
    
    bias_counts = strategy_df["directional_bias"].value_counts().to_dict()
    
    actionable = strategy_df[
        ~strategy_df["recommended_action"].str.contains("Hold|No Trade", case=False, na=False)
    ]
    
    # Calculate average confidence
    avg_conf = strategy_df["confidence"].mean() if "confidence" in strategy_df.columns else 0.5
    
    # Calculate total estimated cost (sum of non-null estimated_cost values)
    total_cost = 0.0
    if "estimated_cost" in strategy_df.columns:
        costs = strategy_df["estimated_cost"].dropna()
        total_cost = costs.sum() * 100  # Multiply by 100 for contract sizing
    
    # Get strategy breakdown (only actionable strategies)
    strategy_breakdown = actionable["recommended_action"].value_counts().to_dict()
    
    return {
        "total_strategies": len(strategy_df),
        "bullish_count": bias_counts.get("Bullish", 0) + bias_counts.get("Slightly Bullish", 0),
        "bearish_count": bias_counts.get("Bearish", 0) + bias_counts.get("Slightly Bearish", 0),
        "neutral_count": bias_counts.get("Neutral", 0),
        "avg_confidence": float(avg_conf),
        "total_estimated_cost": float(total_cost),
        "strategy_breakdown": strategy_breakdown,
        # Legacy keys for backward compatibility
        "total_signals": len(strategy_df),
        "actionable": len(actionable),
    }
