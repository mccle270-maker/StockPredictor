"""
Signals Service
===============

Generate trading signals from predictions for auto-trading execution.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

from ..config import ExecutionModel
from ..core.pricing import suggest_options_strategy, normalize_strategy


def is_us_tradeable_symbol(ticker: str) -> bool:
    """
    Check if symbol is tradeable on Alpaca (US stocks only).
    Non-US symbols contain periods (e.g., LYC.AX for Australian stocks).
    """
    ticker_clean = str(ticker).upper().strip()
    
    if "." in ticker_clean:
        non_us_markers = [".AX", ".L", ".TO", ".V", ".NZ", ".AS", ".KL", ".SG", ".HK"]
        if any(ticker_clean.endswith(marker) for marker in non_us_markers):
            return False
    return True


def build_signals_from_pred_df(
    pred_df: pd.DataFrame,
    *,
    prediction_horizon: int,
    trade_mode: str,
    prefer_spreads: bool,
    dte_min: int,
    dte_max: int,
    max_strike: float,
    max_premium: float,
    width_pct: float,
    exec_model: ExecutionModel,
) -> dict[str, dict[str, Any]]:
    """
    Convert prediction DataFrame to trading signals dict.
    
    Args:
        pred_df: DataFrame with ticker, pred_next_ret, last_close, atm_iv, put_call_oi_ratio
        prediction_horizon: Days ahead for prediction (1-5)
        trade_mode: "Stock only", "Options only", or "Options if suggested"
        prefer_spreads: Whether to prefer spread strategies over single-leg
        dte_min, dte_max: Days to expiration range for options
        max_strike: Maximum strike price
        max_premium: Maximum option premium
        width_pct: Spread width as percentage
        exec_model: ExecutionModel with friction parameters
    
    Returns:
        Dict mapping ticker -> signal dict with asset type, action/strategy, qty, etc.
    """
    signals: dict[str, dict[str, Any]] = {}
    
    if pred_df is None or pred_df.empty:
        return signals
    
    for _, row in pred_df.iterrows():
        tk = str(row.get("ticker", "")).upper().strip()
        if not tk:
            continue
        
        # Filter non-US symbols
        if not is_us_tradeable_symbol(tk):
            print(f"{tk}: Non-US market symbol, skipping (not supported by Alpaca)")
            continue
        
        pred = float(row.get("pred_next_ret") or 0.0)
        
        # Position size based on confidence
        pred_abs = abs(pred)
        if pred_abs >= 0.02:
            qty_contracts = 3
        elif pred_abs >= 0.01:
            qty_contracts = 2
        else:
            qty_contracts = 1
        
        # Stock action
        if pred >= 0.005:
            stock_action = "BUY"
        elif pred <= -0.01:
            stock_action = "SHORT"
        else:
            stock_action = "HOLD"
        
        # Options strategy suggestion
        strat_text, _bias = suggest_options_strategy(
            pred_ret=pred,
            put_call_ratio=row.get("put_call_oi_ratio"),
            atm_iv=row.get("atm_iv"),
            horizon=int(prediction_horizon),
        )
        strategy = normalize_strategy(strat_text, prefer_spreads=prefer_spreads)
        
        use_options = (trade_mode == "Options only") or \
                      (trade_mode == "Options if suggested" and strategy is not None)
        
        if use_options and strategy is not None:
            signals[tk] = {
                "asset": "option",
                "strategy": strategy,
                "dte_min": int(dte_min),
                "dte_max": int(dte_max),
                "max_strike": float(max_strike),
                "max_premium": float(max_premium),
                "width_pct": float(width_pct),
                "qty": qty_contracts,
                "raw_strategy_text": str(strat_text),
                "pred_next_ret": float(pred),
                "last_close": float(row.get("last_close")) if row.get("last_close") is not None else None,
                "execution": {
                    "delay_days": int(exec_model.delay_days),
                    "half_spread_bps": float(exec_model.half_spread_bps),
                    "slippage_bps": float(exec_model.slippage_bps),
                    "fee_bps": float(exec_model.fee_bps),
                },
            }
        else:
            signals[tk] = {
                "asset": "stock",
                "action": stock_action,
                "qty": qty_contracts,
                "pred_next_ret": float(pred),
                "execution": {
                    "delay_days": int(exec_model.delay_days),
                    "half_spread_bps": float(exec_model.half_spread_bps),
                    "slippage_bps": float(exec_model.slippage_bps),
                    "fee_bps": float(exec_model.fee_bps),
                },
            }
        
        print(f"{tk}: qty={qty_contracts} {stock_action if signals[tk]['asset'] == 'stock' else strategy}")
    
    return signals


def build_signals_from_backtest(
    results_df: pd.DataFrame,
    universe_text: str,
) -> dict[str, dict[str, Any]]:
    """
    Convert portfolio backtest results to simple BUY/SELL/HOLD signals.
    
    Args:
        results_df: DataFrame with 'sharpe' column
        universe_text: Comma-separated tickers string
    
    Returns:
        Dict mapping ticker -> signal dict
    """
    if results_df is None or results_df.empty:
        return {}
    
    recent_sharpe = results_df["sharpe"].tail(3).mean()
    
    if recent_sharpe > 1.0:
        action = "BUY"
    elif recent_sharpe < -0.5:
        action = "SELL"
    else:
        action = "HOLD"
    
    tickers = [t.strip().upper() for t in universe_text.split(",") if t.strip()]
    
    return {t: {"asset": "stock", "action": action, "qty": 1} for t in tickers}
