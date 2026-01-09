"""
Signals Service
===============

Generate trading signals from predictions for auto-trading execution.

Key Features:
- Rolling z-score filtering (soft or hard)
- Trade limiting per ticker per period
- Eligibility checking (optional)
- Volatility-scaled position sizing
- Weak signal logging for analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from ..config import (
    ExecutionModel, 
    is_ticker_eligible, 
    log_ticker_eligibility,
    ZSCORE_GATING_CONFIG,
    POSITION_SIZING_CONFIG,
    TRADE_LIMIT_CONFIG,
    REGIME_FILTER_CONFIG,
    get_zscore_threshold,
    is_zscore_hard_filter_enabled,
    get_zscore_log_path,
    is_trade_limiting_enabled,
    get_trade_limit,
    is_regime_filter_enabled,
)
from ..core.pricing import suggest_options_strategy, normalize_strategy
from ..core.position_sizing import scale_signals_by_volatility, get_position_sizing_summary
from ..core.zscore_filter import (
    ZScoreFilter,
    ZScoreResult,
    get_zscore_filter,
    compute_prediction_zscore,
)
from ..core.trade_limiter import (
    TradeLimiter,
    SkippedSignal,
    get_trade_limiter,
    apply_trade_limits,
    rank_signals_by_conviction,
)
from ..core.regime_filter import (
    RegimeFilter,
    BlockedTrade,
    get_regime_filter,
    apply_regime_filter,
    get_current_regime,
)


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
    enforce_eligibility: bool = False,  # Changed default: don't hard-exclude by eligibility
    enforce_zscore_gating: bool = False,  # Changed: use soft filtering by default
    prediction_history: dict[str, list[float]] | None = None,
    min_zscore: float | None = None,  # None = use config default
    apply_volatility_scaling: bool = True,
    price_data: dict[str, pd.DataFrame] | None = None,
    log_weak_signals: bool = True,  # Log weak signals for analysis
    apply_trade_limits: bool = True,  # NEW: apply per-ticker trade limits
    include_skipped_signals: bool = True,  # NEW: include skipped signals with flag
    apply_regime_filter: bool = True,  # NEW: filter by market regime
) -> dict[str, dict[str, Any]]:
    """
    Convert prediction DataFrame to trading signals dict.
    
    Now uses SOFT FILTERING by default:
    - All tickers get signals (unless non-US)
    - Weak signals are TAGGED with z_score_passes=False
    - Trade limits are applied per ticker per period
    - Set enforce_zscore_gating=True for hard filtering (old behavior)
    
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
        enforce_eligibility: If True, hard-filter tickers that don't meet eligibility thresholds
        enforce_zscore_gating: If True, hard-filter when |z-score| < min_zscore (False = soft tag only)
        prediction_history: Dict mapping ticker -> list of recent predictions
        min_zscore: Minimum |z-score| threshold (None = use ZSCORE_GATING_CONFIG default)
        apply_volatility_scaling: If True, scale positions by volatility
        price_data: Dict mapping ticker -> DataFrame with Close/High/Low for vol scaling
        log_weak_signals: If True, log weak signals to file for analysis
        apply_trade_limits: If True, limit trades per ticker per period
        include_skipped_signals: If True, include skipped signals with trade_allowed=False
        apply_regime_filter: If True, filter by market regime (SPY 200DMA, VIX)
    
    Returns:
        Dict mapping ticker -> signal dict with asset type, action/strategy, qty, z-score tags, etc.
    """
    signals: dict[str, dict[str, Any]] = {}
    disabled_tickers: list[str] = []
    weak_signals: list[dict] = []  # For logging weak z-scores
    candidate_signals: list[dict] = []  # For z-score ranking
    
    if pred_df is None or pred_df.empty:
        return signals
    
    # Get z-score filter
    zscore_filter = get_zscore_filter()
    
    # Use provided threshold or config default
    if min_zscore is None:
        min_zscore = ZSCORE_GATING_CONFIG.get("min_zscore", 1.0)
    
    # Initialize prediction history if not provided
    if prediction_history is None:
        prediction_history = {}
    
    for _, row in pred_df.iterrows():
        tk = str(row.get("ticker", "")).upper().strip()
        if not tk:
            continue
        
        # Filter non-US symbols (always hard-filter these)
        if not is_us_tradeable_symbol(tk):
            print(f"⚠️ {tk}: Non-US market symbol, skipping (not supported by Alpaca)")
            continue
        
        # Check eligibility (soft or hard based on flag)
        eligibility_status = "eligible"
        eligibility_reason = "meets_thresholds"
        eligible, reason = is_ticker_eligible(tk)
        if not eligible:
            eligibility_status = "ineligible"
            eligibility_reason = reason
            if enforce_eligibility:
                # Hard filter
                disabled_tickers.append(tk)
                print(f"🚫 {log_ticker_eligibility(tk)}")
                continue
            else:
                # Soft tag only
                print(f"⚠️ {log_ticker_eligibility(tk)} (soft-tagged, not excluded)")
        
        pred = float(row.get("pred_next_ret") or 0.0)
        
        # Get per-ticker z-score threshold
        ticker_threshold = get_zscore_threshold(tk)
        
        # Calculate z-score properly
        # Priority 1: Use pred_zscore from prediction if it's a real value (not 0.0, which indicates not calculated)
        # Priority 2: Use ZScoreFilter with prediction_history
        # Priority 3: Compute from prediction distribution (mean/std of all predictions in this batch)
        
        pred_zscore_from_model = row.get("pred_zscore")
        has_valid_model_zscore = (
            pred_zscore_from_model is not None 
            and not pd.isna(pred_zscore_from_model) 
            and abs(float(pred_zscore_from_model)) > 1e-9  # Not exactly 0.0
        )
        
        ticker_history = prediction_history.get(tk, []) if prediction_history else []
        has_prediction_history = len(ticker_history) >= 10
        
        if has_valid_model_zscore:
            # Use the pre-computed z-score from the prediction model
            z_score = float(pred_zscore_from_model)
            zscore_result = type('ZScoreResult', (), {
                'z_score': z_score,
                'signal_strength': 'very_strong' if abs(z_score) >= 2.0 else 'strong' if abs(z_score) >= 1.5 else 'moderate' if abs(z_score) >= 1.0 else 'weak',
                'history_length': -1  # Indicates from model
            })()
        elif has_prediction_history:
            # Fall back to ZScoreFilter with prediction_history
            zscore_result = zscore_filter.evaluate(tk, pred, ticker_history)
            z_score = zscore_result.z_score
        else:
            # Compute z-score from the current batch of predictions
            # This compares this prediction to all other predictions in this run
            all_preds = pred_df["pred_next_ret"].dropna().values
            if len(all_preds) >= 3:
                batch_mean = float(np.mean(all_preds))
                batch_std = float(np.std(all_preds, ddof=1))
                if batch_std > 1e-9:
                    z_score = float((pred - batch_mean) / batch_std)
                else:
                    z_score = 0.0
            else:
                z_score = 0.0
            
            zscore_result = type('ZScoreResult', (), {
                'z_score': z_score,
                'signal_strength': 'very_strong' if abs(z_score) >= 2.0 else 'strong' if abs(z_score) >= 1.5 else 'moderate' if abs(z_score) >= 1.0 else 'weak',
                'history_length': len(all_preds)  # From batch
            })()
        
        # Determine if signal passes z-score threshold
        passes_zscore = abs(z_score) >= ticker_threshold
        
        # Log weak signals
        if not passes_zscore:
            weak_signal_info = {
                "ticker": tk,
                "pred": pred,
                "z_score": z_score,
                "threshold": ticker_threshold,
                "strength": zscore_result.signal_strength,
                "history_len": zscore_result.history_length,
            }
            weak_signals.append(weak_signal_info)
            
            if enforce_zscore_gating:
                # Hard filter
                print(f"📉 {tk}: z-score={z_score:.2f} < {ticker_threshold} (pred={pred*100:.2f}%), FILTERED OUT")
                continue
            else:
                # Soft tag only
                print(f"📊 {tk}: z-score={z_score:.2f} < {ticker_threshold} (pred={pred*100:.2f}%), tagged as weak")
        
        # Position size based on confidence AND z-score magnitude
        pred_abs = abs(pred)
        zscore_abs = abs(z_score)
        
        # Base quantity from prediction magnitude
        if pred_abs >= 0.02:
            base_qty = 3
        elif pred_abs >= 0.01:
            base_qty = 2
        else:
            base_qty = 1
        
        # Boost quantity for high z-scores (more conviction)
        boost_threshold = ZSCORE_GATING_CONFIG.get("boost_threshold", 2.0)
        if passes_zscore and zscore_abs >= boost_threshold:
            qty_contracts = min(base_qty + 1, 5)  # Cap at 5
        else:
            qty_contracts = base_qty
        
        # Reduce quantity for weak signals
        if not passes_zscore:
            qty_contracts = max(1, qty_contracts - 1)  # Reduce but keep at least 1
        
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
        
        # Get RSI for signal filtering (from prediction output)
        rsi_value = row.get("rsi14")
        if rsi_value is not None and not pd.isna(rsi_value):
            rsi_value = float(rsi_value)
        else:
            rsi_value = None
        
        # Build signal dict with z-score tagging
        zscore_tags = {
            "z_score": float(z_score),
            "z_score_passes": passes_zscore,
            "z_score_threshold": ticker_threshold,
            "z_score_strength": zscore_result.signal_strength,
            "z_score_history_len": zscore_result.history_length,
            "rsi14": rsi_value,  # For signal filtering
            "eligibility_status": eligibility_status,
            "eligibility_reason": eligibility_reason,
        }
        
        if use_options and strategy is not None:
            signal_dict = {
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
                **zscore_tags,
            }
        else:
            signal_dict = {
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
                **zscore_tags,
            }
        
        candidate_signals.append({
            "ticker": tk,
            "signal": signal_dict,
            "z_score_abs": abs(z_score),
            "passes_zscore": passes_zscore,
            "stock_action": stock_action,
            "strategy": strategy,
        })
    
    # Rank signals by |z-score| (highest conviction first)
    candidate_signals.sort(key=lambda x: x["z_score_abs"], reverse=True)
    
    # Build final signals dict
    strong_count = 0
    weak_count = 0
    for cand in candidate_signals:
        tk = cand["ticker"]
        signals[tk] = cand["signal"]
        z = cand["signal"]["z_score"]
        passes = cand["passes_zscore"]
        action_str = cand["stock_action"] if cand["signal"]["asset"] == "stock" else cand["strategy"]
        
        if passes:
            strong_count += 1
            icon = "✅"
        else:
            weak_count += 1
            icon = "⚠️"
        print(f"{icon} {tk}: z={z:+.2f} qty={cand['signal']['qty']} {action_str} {'[STRONG]' if passes else '[WEAK]'}")
    
    # Apply volatility-scaled position sizing
    if apply_volatility_scaling and price_data is not None and signals:
        print(f"\n📏 Applying volatility-scaled position sizing...")
        signals = scale_signals_by_volatility(signals, price_data)
        sizing_summary = get_position_sizing_summary(signals)
        print(f"   Overall scale: {sizing_summary['overall_scale']:.2f}x, "
              f"Capped: {sizing_summary['capped_positions']}/{sizing_summary['position_count']}")
    
    # Apply trade limits per ticker
    skipped_signals: list[SkippedSignal] = []
    if apply_trade_limits and is_trade_limiting_enabled() and signals:
        print(f"\n🔒 Applying trade limits...")
        limiter = get_trade_limiter()
        limiter.reset_session()  # Reset for new prediction run
        
        signals, skipped_signals = limiter.apply_limits(
            signals,
            include_skipped=include_skipped_signals,
        )
        
        # Count allowed vs skipped
        allowed_count = sum(1 for s in signals.values() if s.get("trade_allowed", True))
        skipped_count = len(skipped_signals)
        
        print(f"   Trade limit: {TRADE_LIMIT_CONFIG['max_trades_per_ticker']} per {TRADE_LIMIT_CONFIG['period']}")
        print(f"   Ranking method: {TRADE_LIMIT_CONFIG['ranking_method']}")
        print(f"   Allowed: {allowed_count}, Skipped: {skipped_count}")
        
        if skipped_signals:
            for sk in skipped_signals[:5]:
                print(f"   ⏭️ Skipped: {sk.ticker} (rank={sk.rank}, {sk.ranking_method}={sk.ranking_value:.4f})")
            if len(skipped_signals) > 5:
                print(f"   ... +{len(skipped_signals) - 5} more skipped")
    
    # Apply regime filter (SPY 200DMA, VIX)
    blocked_trades: list[BlockedTrade] = []
    if apply_regime_filter and is_regime_filter_enabled() and signals:
        print(f"\n📈 Applying market regime filter...")
        regime_filter = get_regime_filter()
        regime_state = regime_filter.get_regime_state()
        
        print(f"   Regime: {regime_state.regime.value.upper()}")
        print(f"   SPY: ${regime_state.spy_price:.2f} vs 200DMA: ${regime_state.spy_200dma:.2f} ({regime_state.spy_vs_200dma_pct:+.1f}%)")
        print(f"   VIX: {regime_state.vix_level:.1f} | RSI: {regime_state.spy_rsi:.1f}")
        print(f"   Longs allowed: {regime_state.longs_allowed} | Shorts allowed: {regime_state.shorts_allowed}")
        
        signals, blocked_trades = regime_filter.filter_signals(signals)
        
        blocked_count = len(blocked_trades)
        allowed_count = sum(1 for s in signals.values() if not s.get("regime_blocked", False))
        print(f"   Allowed: {allowed_count}, Blocked: {blocked_count}")
        
        if blocked_trades:
            for bt in blocked_trades[:5]:
                print(f"   🚫 Blocked: {bt.ticker} {bt.direction.upper()} | {bt.reason}")
            if len(blocked_trades) > 5:
                print(f"   ... +{len(blocked_trades) - 5} more blocked")

    # Log summary
    print(f"\n📊 Signal Summary:")
    if disabled_tickers and enforce_eligibility:
        print(f"   Hard-filtered (eligibility): {len(disabled_tickers)} ({', '.join(disabled_tickers)})")
    if weak_signals:
        weak_tks = [f"{s['ticker']}(z={s['z_score']:.1f})" for s in weak_signals[:5]]
        more = f" +{len(weak_signals)-5} more" if len(weak_signals) > 5 else ""
        if enforce_zscore_gating:
            print(f"   Hard-filtered (z-score): {len(weak_signals)} ({', '.join(weak_tks)}{more})")
        else:
            print(f"   Weak signals (tagged): {len(weak_signals)} ({', '.join(weak_tks)}{more})")
    print(f"   Strong signals: {strong_count}")
    print(f"   Weak signals included: {weak_count}")
    if skipped_signals:
        print(f"   Skipped (trade limit): {len(skipped_signals)}")
    print(f"   Total signals: {len(signals)} (ranked by |z-score|)")
    if blocked_trades:
        print(f"   Regime blocked: {len(blocked_trades)}")
    
    # Log weak signals to file if enabled
    if log_weak_signals and weak_signals and ZSCORE_GATING_CONFIG.get("log_weak_signals", True):
        try:
            import json
            log_path = get_zscore_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                for ws in weak_signals:
                    ws["timestamp"] = datetime.now().isoformat() if "datetime" in dir() else ""
                    f.write(json.dumps(ws) + "\n")
            print(f"   📝 Logged {len(weak_signals)} weak signals to {log_path.name}")
        except Exception as e:
            print(f"   ⚠️ Failed to log weak signals: {e}")
    
    return signals


# Import datetime for logging
from datetime import datetime


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
