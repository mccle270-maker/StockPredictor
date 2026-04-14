"""
Quick diagnostic for data/provider health and feature availability.

Usage:
    python data_pipeline_diagnostic.py AAPL --period 2y
    python data_pipeline_diagnostic.py AAPL MSFT --period 5y --json-out results/data_diag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.aggregator import get_aggregator
from src.data.market import get_price_history, get_spx
from src.data.macro import get_macro_df, get_vix
from src.data.fundamentals import get_fundamental_features
from src.data.options import get_option_snapshot_features
from src.data.news import get_news_for_ticker
from src.core.features import build_all_features
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS


def _safe_len(obj: Any) -> int:
    try:
        return len(obj)
    except Exception:
        return 0


def diagnose_ticker(ticker: str, period: str) -> dict[str, Any]:
    result: dict[str, Any] = {"ticker": ticker, "period": period}

    hist = get_price_history(ticker, period=period, interval="1d")
    result["price_rows"] = _safe_len(hist)
    if hist is None or hist.empty:
        result["error"] = "no_price_history"
        return result

    spx_df = None
    vix_series = None
    try:
        spx_df = get_spx(hist.index.min(), hist.index.max())
    except Exception as exc:
        result["spx_error"] = str(exc)[:120]
    try:
        vix_series = get_vix(period=period)
    except Exception as exc:
        result["vix_error"] = str(exc)[:120]

    feat_df = build_all_features(hist.copy(), spx_df=spx_df, vix_series=vix_series)
    spx_returns = spx_df["Close"].pct_change() if spx_df is not None and not spx_df.empty and "Close" in spx_df else None
    macro_df = get_macro_df(period=period, spx_returns=spx_returns)
    fundamentals = get_fundamental_features(ticker)
    options = get_option_snapshot_features(ticker)
    news = get_news_for_ticker(ticker, limit=5)

    feature_cols = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
    macro_cols = [c for c in MACRO_COLUMNS if c in macro_df.columns]

    combined = feat_df.join(macro_df, how="left")
    for key, value in fundamentals.items():
        combined[key] = value

    nan_rates = combined[feature_cols + macro_cols].isna().mean().sort_values(ascending=False)
    constant_cols = [
        col for col in feature_cols + macro_cols
        if col in combined.columns and combined[col].nunique(dropna=False) <= 1
    ]

    result.update({
        "spx_rows": _safe_len(spx_df),
        "vix_rows": _safe_len(vix_series),
        "macro_rows": _safe_len(macro_df),
        "feature_count": len(feature_cols),
        "macro_feature_count": len(macro_cols),
        "model_ready_rows": int(combined[feature_cols + macro_cols].ffill().bfill().fillna(0).dropna().shape[0]),
        "constant_features": constant_cols[:25],
        "top_nan_features": {k: float(v) for k, v in nan_rates.head(15).items()},
        "fundamentals_nonzero": {k: float(v) for k, v in fundamentals.items() if isinstance(v, (int, float)) and v},
        "options_snapshot": options,
        "news_count": len(news),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--period", default="2y")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    aggregator = get_aggregator()
    report = {
        "provider_health": aggregator.get_provider_health(),
        "tickers": [diagnose_ticker(ticker.upper(), args.period) for ticker in args.tickers],
    }

    print(json.dumps(report, indent=2, default=str))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved diagnostic to {out_path}")


if __name__ == "__main__":
    main()
