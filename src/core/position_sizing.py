"""
Position Sizing Module
======================

Volatility-scaled position sizing for consistent risk exposure.
Targets ~1% daily volatility per position, with leverage caps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from ..config import POSITION_SIZING_CONFIG


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    ticker: str
    base_qty: int                    # Original quantity before scaling
    scaled_qty: int                  # Volatility-adjusted quantity
    scale_factor: float              # Multiplier applied (capped at max_leverage)
    daily_vol: float                 # Ticker's rolling daily volatility
    target_vol: float                # Target daily volatility
    leverage: float                  # Effective leverage (scaled_qty / base_qty)
    capped: bool                     # True if leverage was capped
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "base_qty": self.base_qty,
            "scaled_qty": self.scaled_qty,
            "scale_factor": round(self.scale_factor, 3),
            "daily_vol": round(self.daily_vol, 4),
            "target_vol": round(self.target_vol, 4),
            "leverage": round(self.leverage, 2),
            "capped": self.capped,
        }


def compute_rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    use_atr: bool = False,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    atr_period: int = 14,
) -> float:
    """
    Compute rolling daily volatility for a price series.
    
    Args:
        prices: Close prices
        window: Rolling window for std dev calculation
        use_atr: If True, use ATR instead of return std dev
        high: High prices (required if use_atr=True)
        low: Low prices (required if use_atr=True)
        atr_period: ATR period
        
    Returns:
        Daily volatility as decimal (e.g., 0.02 = 2% daily vol)
    """
    if len(prices) < window:
        # Not enough data - return a conservative default
        return 0.02  # 2% daily vol assumption
    
    if use_atr and high is not None and low is not None:
        # ATR-based volatility
        tr1 = high - low
        tr2 = abs(high - prices.shift(1))
        tr3 = abs(low - prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean().iloc[-1]
        # Convert ATR to daily vol percentage
        current_price = prices.iloc[-1]
        daily_vol = atr / current_price if current_price > 0 else 0.02
    else:
        # Standard return-based volatility
        returns = prices.pct_change().dropna()
        if len(returns) < window:
            return 0.02
        daily_vol = returns.tail(window).std()
    
    # Sanity check
    if np.isnan(daily_vol) or daily_vol <= 0:
        return 0.02
    
    return float(daily_vol)


def compute_position_scale(
    daily_vol: float,
    target_vol: float = None,
    max_leverage: float = None,
    min_position_pct: float = None,
) -> tuple[float, bool]:
    """
    Compute position scale factor based on volatility.
    
    Args:
        daily_vol: Ticker's daily volatility
        target_vol: Target daily volatility (default from config)
        max_leverage: Maximum allowed leverage (default from config)
        min_position_pct: Minimum position as % of base (default from config)
        
    Returns:
        (scale_factor, was_capped) tuple
    """
    config = POSITION_SIZING_CONFIG
    target_vol = target_vol or config["target_daily_vol"]
    max_leverage = max_leverage or config["max_leverage"]
    min_position_pct = min_position_pct or config["min_position_pct"]
    
    if daily_vol <= 0:
        return 1.0, False
    
    # Scale factor = target_vol / actual_vol
    # If stock is 2x as volatile as target, scale = 0.5 (half position)
    # If stock is 0.5x as volatile, scale = 2.0 (double position, but capped)
    raw_scale = target_vol / daily_vol
    
    # Apply caps
    was_capped = False
    
    if raw_scale > max_leverage:
        raw_scale = max_leverage
        was_capped = True
    
    if raw_scale < min_position_pct:
        raw_scale = min_position_pct
        was_capped = True
    
    return float(raw_scale), was_capped


def scale_position_size(
    ticker: str,
    base_qty: int,
    prices: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
) -> PositionSize:
    """
    Scale a position size based on volatility.
    
    Args:
        ticker: Stock ticker
        base_qty: Original quantity before scaling
        prices: Close price series
        high: High price series (optional, for ATR)
        low: Low price series (optional, for ATR)
        
    Returns:
        PositionSize with scaled quantity and metadata
    """
    config = POSITION_SIZING_CONFIG
    
    # Compute volatility
    daily_vol = compute_rolling_volatility(
        prices=prices,
        window=config["vol_lookback_days"],
        use_atr=config["use_atr"],
        high=high,
        low=low,
        atr_period=config["atr_period"],
    )
    
    # Compute scale factor
    scale_factor, was_capped = compute_position_scale(daily_vol)
    
    # Apply to quantity
    scaled_qty = max(1, int(round(base_qty * scale_factor)))
    
    # Compute effective leverage
    leverage = scaled_qty / base_qty if base_qty > 0 else 1.0
    
    return PositionSize(
        ticker=ticker,
        base_qty=base_qty,
        scaled_qty=scaled_qty,
        scale_factor=scale_factor,
        daily_vol=daily_vol,
        target_vol=config["target_daily_vol"],
        leverage=leverage,
        capped=was_capped,
    )


def scale_signals_by_volatility(
    signals: dict[str, dict],
    price_data: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Scale all signal quantities by volatility.
    
    Args:
        signals: Dict mapping ticker -> signal dict (must have 'qty' key)
        price_data: Dict mapping ticker -> DataFrame with 'Close' column
        
    Returns:
        Updated signals dict with scaled quantities and sizing metadata
    """
    scaled_signals = {}
    scaling_log = []  # Structured log for transparency

    config = POSITION_SIZING_CONFIG
    window = config.get("vol_lookback_days", 20)
    use_atr = config.get("use_atr", False)
    atr_period = config.get("atr_period", 14)

    for ticker, signal in signals.items():
        base_qty = signal.get("qty", 1)
        df = price_data.get(ticker)
        # Default values if no price data
        if df is None or df.empty or "Close" not in df.columns:
            daily_vol = None
            scale_factor = 1.0
            scaled_qty = base_qty
            capped = False
            error = "no_price_data"
        else:
            high = df.get("High") if "High" in df.columns else None
            low = df.get("Low") if "Low" in df.columns else None
            # Compute rolling volatility (always use config)
            daily_vol = compute_rolling_volatility(
                prices=df["Close"],
                window=window,
                use_atr=use_atr,
                high=high,
                low=low,
                atr_period=atr_period,
            )
            scale_factor, capped = compute_position_scale(
                daily_vol,
                target_vol=config["target_daily_vol"],
                max_leverage=config["max_leverage"],
                min_position_pct=config["min_position_pct"],
            )
            scaled_qty = max(1, int(round(base_qty * scale_factor)))
            error = None

        # Always keep ticker in pipeline
        new_signal = signal.copy()
        new_signal["qty"] = scaled_qty
        new_signal["position_sizing"] = {
            "base_qty": base_qty,
            "scaled_qty": scaled_qty,
            "scale_factor": round(scale_factor, 3),
            "daily_vol": round(daily_vol, 4) if daily_vol is not None else None,
            "target_vol": config["target_daily_vol"],
            "max_leverage": config["max_leverage"],
            "min_position_pct": config["min_position_pct"],
            "capped": capped,
            "error": error,
        }
        scaled_signals[ticker] = new_signal

        # Structured log entry
        scaling_log.append({
            "ticker": ticker,
            "raw_qty": base_qty,
            "volatility": round(daily_vol, 4) if daily_vol is not None else None,
            "scale_factor": round(scale_factor, 3),
            "scaled_qty": scaled_qty,
            "capped": capped,
            "error": error,
        })

    # Print structured log for transparency
    print("\n[Position Sizing Log]")
    for entry in scaling_log:
        msg = (f"{entry['ticker']}: qty {entry['raw_qty']}→{entry['scaled_qty']} "
               f"(vol={entry['volatility'] if entry['volatility'] is not None else 'NA'} "
               f"scale={entry['scale_factor']:.2f}x)" + (" [CAPPED]" if entry['capped'] else ""))
        if entry['error']:
            msg += f" [ERROR: {entry['error']}]"
        print(msg)

    return scaled_signals


def get_position_sizing_summary(signals: dict[str, dict]) -> dict:
    """
    Get summary of position sizing across all signals.
    
    Args:
        signals: Dict of signals with position_sizing metadata
        
    Returns:
        Summary dict with aggregate metrics
    """
    total_base = 0
    total_scaled = 0
    vol_weighted_sum = 0
    capped_count = 0
    
    for ticker, signal in signals.items():
        sizing = signal.get("position_sizing", {})
        base = sizing.get("base_qty", signal.get("qty", 1))
        scaled = sizing.get("scaled_qty", signal.get("qty", 1))
        vol = sizing.get("daily_vol", 0) or 0
        
        total_base += base
        total_scaled += scaled
        vol_weighted_sum += vol * scaled
        if sizing.get("capped", False):
            capped_count += 1
    
    avg_vol = vol_weighted_sum / total_scaled if total_scaled > 0 else 0
    
    return {
        "total_base_qty": total_base,
        "total_scaled_qty": total_scaled,
        "overall_scale": total_scaled / total_base if total_base > 0 else 1.0,
        "avg_weighted_vol": avg_vol,
        "capped_positions": capped_count,
        "position_count": len(signals),
    }
