"""
Risk and performance metrics.
Pure functions for calculating Sharpe, drawdown, VaR, etc.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Dict, Any, Tuple


# ============================================================================
# RETURNS METRICS
# ============================================================================

def compute_sharpe(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Optional[float]:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Daily returns series or numpy array
        risk_free: Annual risk-free rate (as decimal)
        periods_per_year: Trading days per year
    
    Returns:
        Annualized Sharpe ratio or None if insufficient data
    """
    # Handle both pandas Series and numpy arrays
    if isinstance(returns, np.ndarray):
        r = returns[~np.isnan(returns)]
    else:
        r = returns.dropna().values
    
    if len(r) < 2 or np.std(r) == 0:
        return None
    
    daily_rf = risk_free / periods_per_year
    excess = r - daily_rf
    return float((np.mean(excess) / np.std(excess)) * np.sqrt(periods_per_year))


def compute_sortino(
    returns: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> Optional[float]:
    """
    Compute annualized Sortino ratio (uses downside deviation).
    """
    r = returns.dropna()
    if len(r) < 2:
        return None
    
    daily_rf = risk_free / periods_per_year
    excess = r - daily_rf
    downside = r.where(r < 0, 0)
    downside_std = downside.std()
    
    if downside_std == 0:
        return None
    
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def compute_calmar(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> Optional[float]:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    """
    r = returns.dropna()
    if len(r) < 2:
        return None
    
    ann_return = r.mean() * periods_per_year
    max_dd = compute_max_drawdown(r)
    
    if max_dd is None or max_dd >= 0:
        return None
    
    return ann_return / abs(max_dd)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    risk_free: float = 0.0,
) -> Optional[float]:
    """
    Compute deflated Sharpe ratio (adjusts for multiple testing).
    
    Args:
        returns: Daily returns series
        n_trials: Number of strategies/parameters tried
        risk_free: Annual risk-free rate
    
    Returns:
        Deflated Sharpe ratio as a probability (0-1)
    """
    r = returns.dropna()
    if n_trials <= 0 or len(r) < 5 or r.std() == 0:
        return None
    
    excess = r - risk_free / 252
    mu, sigma, T = excess.mean(), excess.std(), len(r)
    sharpe_daily = mu / sigma
    z_strat = sharpe_daily * np.sqrt(T)
    
    if n_trials == 1:
        return float(norm.cdf(z_strat))
    
    z_alpha = norm.ppf(1.0 - 1.0 / n_trials)
    z_deflated = z_strat - z_alpha
    return float(norm.cdf(z_deflated))


# ============================================================================
# DRAWDOWN
# ============================================================================

def compute_drawdown(returns: pd.Series) -> pd.DataFrame:
    """
    Compute equity curve, peaks, drawdowns, and duration.
    
    Returns:
        DataFrame with columns: equity, peak, drawdown, duration
    """
    r = returns.fillna(0.0)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    
    # Duration calculation
    duration = (drawdown != 0).astype(int)
    duration = duration.groupby((duration == 0).cumsum()).cumsum()
    
    return pd.DataFrame({
        "equity": equity,
        "peak": peak,
        "drawdown": drawdown,
        "duration": duration,
    })


def compute_max_drawdown(returns: pd.Series) -> Optional[float]:
    """
    Compute maximum drawdown from returns series.
    
    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.15 for 15% drawdown)
    """
    r = returns.dropna()
    if len(r) == 0:
        return None
    
    dd_df = compute_drawdown(r)
    return float(dd_df["drawdown"].min())


def compute_max_drawdown_duration(returns: pd.Series) -> Optional[int]:
    """
    Compute maximum drawdown duration in periods.
    """
    r = returns.dropna()
    if len(r) == 0:
        return None
    
    dd_df = compute_drawdown(r)
    return int(dd_df["duration"].max())


# ============================================================================
# ROLLING METRICS
# ============================================================================

def rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    r = returns.dropna()
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std()
    return (mu / (sigma + 1e-9)) * np.sqrt(periods_per_year)


def rolling_sortino(
    returns: pd.Series,
    window: int = 60,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling Sortino ratio."""
    r = returns.dropna()
    downside = r.where(r < 0, 0)
    mu = r.rolling(window).mean()
    downside_std = downside.rolling(window).std()
    return (mu / (downside_std + 1e-9)) * np.sqrt(periods_per_year)


def rolling_vol(
    returns: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling annualized volatility."""
    r = returns.dropna()
    return r.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_var_es(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.95,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling Value-at-Risk and Expected Shortfall.
    
    Returns:
        Tuple of (VaR series, ES series)
    """
    r = returns.dropna()
    
    def calc_var_es(x):
        if len(x) == 0:
            return np.nan, np.nan
        var = np.quantile(x, 1 - alpha)
        es = x[x <= var].mean() if np.isfinite(var) else np.nan
        return var, es
    
    var_series = r.rolling(window).apply(lambda x: calc_var_es(x)[0], raw=False)
    es_series = r.rolling(window).apply(lambda x: calc_var_es(x)[1], raw=False)
    
    return var_series, es_series


# ============================================================================
# SUMMARY METRICS
# ============================================================================

def summarize_risk(returns: pd.Series) -> Dict[str, Any]:
    """
    Produce a simple risk label and guidance.
    
    Returns:
        Dict with label, score, summary, and individual metrics
    """
    r = returns.dropna()
    if len(r) == 0:
        return {
            "label": "unknown",
            "score": None,
            "summary": "No returns to evaluate risk.",
            "check": "Add data",
        }
    
    dd_df = compute_drawdown(r)
    max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
    
    sharpe_60 = rolling_sharpe(r, window=60).iloc[-1] if len(r) >= 60 else np.nan
    vol_20 = rolling_vol(r, window=20).iloc[-1] if len(r) >= 20 else np.nan
    var_series, es_series = rolling_var_es(r, window=250, alpha=0.95)
    var_latest = var_series.iloc[-1] if len(var_series.dropna()) else np.nan
    es_latest = es_series.iloc[-1] if len(es_series.dropna()) else np.nan
    
    # Heuristic scoring (lower is better)
    score = 0
    if max_dd < -0.2:
        score += 2
    elif max_dd < -0.1:
        score += 1
    
    if pd.notna(sharpe_60) and sharpe_60 < 0.5:
        score += 1
    if pd.notna(vol_20) and vol_20 > 0.5:
        score += 1
    if pd.notna(es_latest) and es_latest < -0.04:
        score += 1
    
    if score <= 1:
        label = "low"
        summary = "Risk looks contained (shallow drawdowns / decent Sharpe)."
        check = "Scan drawdowns briefly."
    elif score == 2:
        label = "medium"
        summary = "Mixed risk: watch tails/vol spikes."
        check = "Check tail risk (VaR/ES) and vol chart."
    else:
        label = "high"
        summary = "Elevated risk: deep DD or weak Sharpe/vol spikes."
        check = "Inspect drawdown + tail risk charts before trading."
    
    return {
        "label": label,
        "score": score,
        "summary": summary,
        "check": check,
        "max_dd": max_dd,
        "sharpe_60": sharpe_60,
        "vol_20": vol_20,
        "var95": var_latest,
        "es95": es_latest,
    }


def prepare_risk_timeseries(returns: pd.Series) -> Dict[str, Any]:
    """
    Bundle of common risk time series for charting.
    
    Returns:
        Dict with drawdown DataFrame and rolling metrics series
    """
    dd_df = compute_drawdown(returns)
    sharpe_60 = rolling_sharpe(returns, window=60)
    sortino_60 = rolling_sortino(returns, window=60)
    vol_20 = rolling_vol(returns, window=20)
    var95, es95 = rolling_var_es(returns, window=250, alpha=0.95)
    
    return {
        "drawdown": dd_df,
        "sharpe_60": sharpe_60,
        "sortino_60": sortino_60,
        "vol_20": vol_20,
        "var95": var95,
        "es95": es95,
    }


# ============================================================================
# BACKTEST METRICS
# ============================================================================

def compute_backtest_metrics(
    returns: pd.Series,
    positions: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive backtest metrics.
    
    Args:
        returns: Strategy returns series
        positions: Position series (for turnover calculation)
    
    Returns:
        Dict with all metrics
    """
    r = returns.dropna()
    
    metrics = {
        "total_return": (1 + r).prod() - 1 if len(r) > 0 else 0,
        "ann_return": r.mean() * 252 if len(r) > 0 else 0,
        "ann_vol": r.std() * np.sqrt(252) if len(r) > 0 else 0,
        "sharpe": compute_sharpe(r),
        "sortino": compute_sortino(r),
        "max_dd": compute_max_drawdown(r),
        "calmar": compute_calmar(r),
        "hit_rate": (r > 0).mean() if len(r) > 0 else 0,
        "n_periods": len(r),
    }
    
    # Win/loss ratio
    wins = r[r > 0]
    losses = r[r < 0]
    if len(losses) > 0 and losses.mean() != 0:
        metrics["win_loss_ratio"] = abs(wins.mean() / losses.mean()) if len(wins) > 0 else 0
    else:
        metrics["win_loss_ratio"] = np.inf if len(wins) > 0 else 0
    
    # Turnover (if positions provided)
    if positions is not None:
        pos = positions.fillna(0)
        metrics["turnover"] = pos.diff().abs().sum() / 2
    
    return metrics


def compute_direction_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Compute directional accuracy (what % of predictions got the sign right).
    """
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if len(aligned) == 0:
        return 0.0
    
    return (np.sign(aligned["actual"]) == np.sign(aligned["predicted"])).mean()
