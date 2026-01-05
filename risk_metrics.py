import pandas as pd
import numpy as np


def compute_drawdown(returns: pd.Series) -> pd.DataFrame:
    """Compute equity curve, peaks, drawdowns, and duration."""
    r = returns.fillna(0.0)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak) - 1.0
    duration = (drawdown != 0).astype(int)
    duration = duration.groupby((duration == 0).cumsum()).cumsum()
    return pd.DataFrame({
        "equity": equity,
        "peak": peak,
        "drawdown": drawdown,
        "duration": duration,
    })


def rolling_sharpe(returns: pd.Series, window: int = 60, periods_per_year: int = 252) -> pd.Series:
    r = returns.dropna()
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std()
    sharpe = (mu / (sigma + 1e-9)) * np.sqrt(periods_per_year)
    return sharpe


def rolling_sortino(returns: pd.Series, window: int = 60, periods_per_year: int = 252) -> pd.Series:
    r = returns.dropna()
    downside = r.where(r < 0, 0)
    mu = r.rolling(window).mean()
    downside_std = downside.rolling(window).std()
    sortino = (mu / (downside_std + 1e-9)) * np.sqrt(periods_per_year)
    return sortino


def rolling_vol(returns: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    r = returns.dropna()
    vol = r.rolling(window).std() * np.sqrt(periods_per_year)
    return vol


def rolling_var_es(returns: pd.Series, window: int = 250, alpha: float = 0.95):
    r = returns.dropna()
    def _calc(x: pd.Series):
        if x.empty:
            return np.nan, np.nan
        var = np.quantile(x, 1 - alpha)
        es = x[x <= var].mean() if np.isfinite(var) else np.nan
        return var, es
    out = r.rolling(window).apply(lambda x: _calc(pd.Series(x))[0], raw=False)
    es = r.rolling(window).apply(lambda x: _calc(pd.Series(x))[1], raw=False)
    return out, es


def prepare_risk_timeseries(returns: pd.Series):
    """Convenience bundle for common risk series."""
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


def summarize_risk(returns: pd.Series) -> dict:
    """Produce a simple risk label and quick guidance on what to inspect."""
    r = returns.dropna()
    if r.empty:
        return {
            "label": "unknown",
            "score": None,
            "summary": "No returns to evaluate risk.",
            "check": "Add data",
        }

    dd = compute_drawdown(r)
    max_dd = float(dd["drawdown"].min()) if not dd.empty else 0.0

    sharpe_60 = rolling_sharpe(r, window=60).iloc[-1] if len(r) >= 60 else np.nan
    vol_20 = rolling_vol(r, window=20).iloc[-1] if len(r) >= 20 else np.nan
    var95, es95 = rolling_var_es(r, window=250, alpha=0.95)
    var_latest = var95.iloc[-1] if len(var95.dropna()) else np.nan
    es_latest = es95.iloc[-1] if len(es95.dropna()) else np.nan

    # Heuristic scoring (lower is better risk)
    score = 0
    if max_dd < -0.2:
        score += 2
    elif max_dd < -0.1:
        score += 1

    if pd.notna(sharpe_60) and sharpe_60 < 0.5:
        score += 1
    if pd.notna(vol_20) and vol_20 > 0.5:  # 50% ann vol threshold
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
