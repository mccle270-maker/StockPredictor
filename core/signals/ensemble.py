import numpy as np

# Example config (should be imported from your config system)
DEFAULT_MODEL_WEIGHTS = {
    'rf': 1.0,
    'xgb': 1.0,
    'gbrt': 1.0,
}

def ensemble_predictions(
    model_preds: dict,
    method: str = 'weighted_average',
    model_weights: dict = None,
    vote_threshold: float = 0.0,
):
    """
    Combine predictions from multiple models.
    Args:
        model_preds: dict of {model_type: {'pred_next_ret': float, 'prob_up': float, ...}}
        method: 'weighted_average' or 'majority_vote'
        model_weights: dict of {model_type: weight}
        vote_threshold: threshold for majority vote (default 0.0)
    Returns:
        dict with 'ensemble_pred', 'ensemble_confidence', and per-model details
    """
    if model_weights is None:
        model_weights = DEFAULT_MODEL_WEIGHTS.copy()
    # Only use models present in input
    weights = np.array([model_weights.get(m, 1.0) for m in model_preds.keys()])
    preds = np.array([model_preds[m]['pred_next_ret'] for m in model_preds.keys()])
    prob_ups = np.array([
        model_preds[m].get('prob_up', None) for m in model_preds.keys()
    ])
    model_list = list(model_preds.keys())

    if method == 'weighted_average':
        wsum = np.sum(weights)
        if wsum == 0:
            weights = np.ones_like(weights)
            wsum = np.sum(weights)
        ensemble_pred = float(np.dot(preds, weights) / wsum)
        # Confidence: weighted stddev or mean abs deviation
        confidence = float(np.sum(weights * np.abs(preds - ensemble_pred)) / wsum)
        # Optionally, combine prob_up if available
        if np.all([p is not None for p in prob_ups]):
            ensemble_prob_up = float(np.dot(prob_ups, weights) / wsum)
        else:
            ensemble_prob_up = None
        return {
            'ensemble_pred': ensemble_pred,
            'ensemble_confidence': 1.0 - confidence,  # Lower disagreement = higher confidence
            'ensemble_prob_up': ensemble_prob_up,
            'model_preds': model_preds,
            'model_weights': dict(zip(model_list, weights)),
            'method': method,
        }
    elif method == 'majority_vote':
        # Each model votes up/down based on threshold
        votes = np.array([int(p > vote_threshold) for p in preds])
        vote_sum = np.dot(votes, weights)
        wsum = np.sum(weights)
        # Majority: 1 if >50% weighted vote, else 0
        ensemble_pred = 1 if vote_sum > (wsum / 2) else 0
        # Confidence: fraction of weighted votes for majority
        confidence = float(max(vote_sum, wsum - vote_sum) / wsum)
        return {
            'ensemble_pred': ensemble_pred,
            'ensemble_confidence': confidence,
            'model_preds': model_preds,
            'model_weights': dict(zip(model_list, weights)),
            'method': method,
        }
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
