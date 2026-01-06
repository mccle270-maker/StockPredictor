import pandas as pd

def suggest_option_strategies(
    model_outputs: dict,
    iv_metrics: dict,
    min_prob_up: float = 0.55,
    min_pred_ret: float = 0.01,
    iv_rich_threshold: float = 0.7,
    spread_min_iv_rank: float = 0.6,
) -> pd.DataFrame:
    """
    Suggest option strategies (calls, puts, vertical spreads) based on model outputs and IV metrics.
    Args:
        model_outputs: dict of {ticker: {pred_next_ret, prob_up, prob_down, ...}}
        iv_metrics: dict of {ticker: {iv_rank, iv_percentile, iv, ...}}
        min_prob_up: minimum probability up for call/vertical
        min_pred_ret: minimum predicted return for bullish strategies
        iv_rich_threshold: IV rank above which spreads are preferred
        spread_min_iv_rank: minimum IV rank for verticals
    Returns:
        DataFrame with columns: ticker, strategy, reason, pred_ret, prob_up, iv_rank, iv, details
    """
    rows = []
    for tk, out in model_outputs.items():
        pred_ret = out.get('pred_next_ret', 0)
        prob_up = out.get('prob_up', 0.5)
        prob_down = out.get('prob_down', 0.5)
        ivm = iv_metrics.get(tk, {})
        iv_rank = ivm.get('iv_rank', 0.5)
        iv = ivm.get('iv', None)
        details = {}
        # Bullish
        if pred_ret > min_pred_ret and prob_up > min_prob_up:
            if iv_rank >= iv_rich_threshold:
                strategy = 'bull_call_spread'
                reason = f"IV rich (rank {iv_rank:.2f}), bullish model"
            else:
                strategy = 'long_call'
                reason = f"Bullish model, IV not rich (rank {iv_rank:.2f})"
            details = {'pred_ret': pred_ret, 'prob_up': prob_up, 'iv_rank': iv_rank, 'iv': iv}
            rows.append({'ticker': tk, 'strategy': strategy, 'reason': reason, **details})
        # Bearish
        elif pred_ret < -min_pred_ret and prob_down > min_prob_up:
            if iv_rank >= iv_rich_threshold:
                strategy = 'bear_put_spread'
                reason = f"IV rich (rank {iv_rank:.2f}), bearish model"
            else:
                strategy = 'long_put'
                reason = f"Bearish model, IV not rich (rank {iv_rank:.2f})"
            details = {'pred_ret': pred_ret, 'prob_down': prob_down, 'iv_rank': iv_rank, 'iv': iv}
            rows.append({'ticker': tk, 'strategy': strategy, 'reason': reason, **details})
        # Neutral/high IV: consider iron condor/straddle (optional)
        elif iv_rank >= 0.9:
            strategy = 'iron_condor'
            reason = f"IV very rich (rank {iv_rank:.2f}), neutral model"
            details = {'pred_ret': pred_ret, 'prob_up': prob_up, 'prob_down': prob_down, 'iv_rank': iv_rank, 'iv': iv}
            rows.append({'ticker': tk, 'strategy': strategy, 'reason': reason, **details})
    df = pd.DataFrame(rows)
    return df
