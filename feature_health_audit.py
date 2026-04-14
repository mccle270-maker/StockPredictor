#!/usr/bin/env python3
"""
Quick feature health audit for one or more tickers.

Usage:
    python feature_health_audit.py AAPL MSFT GOOGL --period 10y
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import pandas as pd

from src.config import FEATURE_COLUMNS, MACRO_COLUMNS, FUNDAMENTAL_COLUMNS
from src.core.features import build_all_features
from src.data.market import get_price_history, get_spx
from src.data.macro import get_macro_df, get_vix, align_macro_to_index
from src.data.fundamentals import get_fundamental_features


def audit_ticker(ticker: str, period: str) -> None:
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        print(f"\n[{ticker}] no price data available")
        return

    spx_df = get_spx(hist.index.min(), hist.index.max())
    vix_series = get_vix(period=period)
    macro_df = align_macro_to_index(get_macro_df(period=period), hist.index)
    fundamentals = get_fundamental_features(ticker)

    feat_df = build_all_features(
        hist.copy(),
        spx_df=spx_df,
        vix_series=vix_series,
        macro_df=macro_df,
        fundamentals=fundamentals,
    )

    available = [c for c in FEATURE_COLUMNS if c in feat_df.columns]
    nan_rates = feat_df[available].isna().mean().sort_values(ascending=False)
    constant_cols = [c for c in available if feat_df[c].nunique(dropna=False) <= 1]
    mostly_nan = nan_rates[nan_rates > 0.30]

    print(f"\n[{ticker}] rows={len(feat_df)} available_features={len(available)}")
    print(f"macro_features_present={sum(c in feat_df.columns for c in MACRO_COLUMNS)}")
    print(f"fundamentals={fundamentals}")
    print(f"constant_features={constant_cols[:12]}")
    print("worst_nan_rates:")
    print(nan_rates.head(12).to_string())

    if len(mostly_nan):
        print("mostly_nan_over_30pct:")
        print(mostly_nan.to_string())

    ready = feat_df[available].ffill().bfill().fillna(0)
    print(f"model_ready_rows={len(ready.dropna())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit feature health for tickers.")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to inspect")
    parser.add_argument("--period", default="10y", help="History period to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ticker in args.tickers:
        audit_ticker(ticker.upper(), args.period)


if __name__ == "__main__":
    main()
