#!/usr/bin/env python3
"""
Screen triple-barrier labeling versus simple direction labels.

Compares:
- simple_binary: up/down on forward return
- triple_barrier: 3-class label based on pt/sl hit first within horizon

Evaluates live/current walk-forward Sharpe with transaction costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.backtest import _prepare_features
from src.data.market import get_price_history
from src.core.metrics import compute_calmar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN", "GOOGL"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-years", type=float, default=0.5)
    parser.add_argument("--test-years", type=float, default=0.125)
    parser.add_argument("--step-days", type=int, default=31)
    parser.add_argument("--purge-gap-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--cost-per-trade", type=float, default=0.0005)
    parser.add_argument("--pt-mult", type=float, default=1.5)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--output", default="results/triple_barrier_screen_live.json")
    return parser.parse_args()


def generate_splits(n_rows: int, train_years: float, test_years: float, step_days: int,
                    purge_gap_days: int, embargo_days: int) -> list[tuple[int, int, int, int]]:
    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    start = 0
    splits = []
    while True:
        train_start = start
        train_end = train_start + train_days
        test_start = train_end + max(0, purge_gap_days)
        test_end = test_start + test_days
        if test_end > n_rows:
            break
        splits.append((train_start, train_end, test_start, test_end))
        start += step_days + max(0, embargo_days)
    return splits


def build_barrier_labels(hist: pd.DataFrame, horizon: int, pt_mult: float, sl_mult: float) -> pd.Series:
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float) if "High" in hist.columns else close
    low = hist["Low"].astype(float) if "Low" in hist.columns else close
    vol = close.pct_change().rolling(20).std().clip(lower=1e-4)
    labels = pd.Series(index=hist.index, dtype=float)

    for i in range(len(hist) - horizon):
        entry = close.iloc[i]
        sigma = float(vol.iloc[i]) if pd.notna(vol.iloc[i]) else 0.01
        pt = entry * (1 + pt_mult * sigma)
        sl = entry * (1 - sl_mult * sigma)
        path_high = high.iloc[i + 1 : i + 1 + horizon]
        path_low = low.iloc[i + 1 : i + 1 + horizon]
        assigned = 1
        if ((path_high >= pt) & (path_low <= sl)).any():
            first_pt = np.where(path_high.values >= pt)[0]
            first_sl = np.where(path_low.values <= sl)[0]
            first_pt_idx = first_pt[0] if len(first_pt) else np.inf
            first_sl_idx = first_sl[0] if len(first_sl) else np.inf
            if first_pt_idx < first_sl_idx:
                assigned = 2  # up
            elif first_sl_idx < first_pt_idx:
                assigned = 0  # down
            else:
                assigned = 1  # neutral
        elif (path_high >= pt).any():
            assigned = 2
        elif (path_low <= sl).any():
            assigned = 0
        else:
            assigned = 1
        labels.iloc[i] = assigned

    return labels


def simulate(class_preds: np.ndarray, actual_returns: np.ndarray, cost_per_trade: float) -> pd.Series:
    positions = np.where(class_preds == 2, 1.0, np.where(class_preds == 0, -1.0, 0.0))
    pnl = []
    prev_pos = 0.0
    for pos, ret in zip(positions, actual_returns):
        trade = abs(pos - prev_pos)
        pnl.append(pos * ret - cost_per_trade * trade)
        prev_pos = pos
    return pd.Series(pnl)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"folds": 0}
    df = pd.DataFrame(rows)
    return {
        "folds": int(len(df)),
        "mean_sharpe": float(df["sharpe"].mean()),
        "median_sharpe": float(df["sharpe"].median()),
        "mean_calmar": float(df["calmar"].dropna().mean()) if df["calmar"].notna().any() else None,
        "mean_accuracy": float(df["accuracy"].mean()),
        "positive_sharpe_pct": float((df["sharpe"] > 0).mean()),
        "mean_trades": float(df["num_trades"].mean()),
    }


def run_ticker(ticker: str, args: argparse.Namespace) -> dict[str, Any]:
    hist = get_price_history(ticker, period=args.period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No history for {ticker}")
    df, feat_cols = _prepare_features(hist, ticker, args.period, args.horizon)
    aligned_hist = hist.loc[df.index].copy()
    df["simple_label"] = np.where(df["ftarget_ret_horizon_ahead"] > 0, 2, 0)
    df["tb_label"] = build_barrier_labels(aligned_hist, args.horizon, args.pt_mult, args.sl_mult).reindex(df.index)
    df = df.dropna(subset=["tb_label"]).copy()
    df["tb_label"] = df["tb_label"].astype(int)
    splits = generate_splits(len(df), args.train_years, args.test_years, args.step_days, args.purge_gap_days, args.embargo_days)
    if len(splits) < 4:
        raise ValueError(f"Not enough folds for {ticker}")

    results = {"simple_binary": [], "triple_barrier": []}

    for train_start, train_end, test_start, test_end in splits:
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        X_train = train_df[feat_cols].values
        X_test = test_df[feat_cols].values
        y_test_ret = test_df["ftarget_ret_horizon_ahead"].values

        for name, label_col in [("simple_binary", "simple_label"), ("triple_barrier", "tb_label")]:
            y_train = train_df[label_col].values
            if name == "simple_binary":
                y_train = (y_train == 2).astype(int)
                clf = XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    min_child_weight=20,
                    reg_alpha=0.5,
                    reg_lambda=12.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                )
            else:
                clf = XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    min_child_weight=20,
                    reg_alpha=0.5,
                    reg_lambda=12.0,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=42,
                )
            clf.fit(X_train, y_train)
            class_preds = clf.predict(X_test)
            if name == "simple_binary":
                class_preds = np.where(class_preds > 0.5, 2, 0)
            pnl_series = simulate(class_preds, y_test_ret, args.cost_per_trade)
            std_daily = float(pnl_series.std(ddof=1)) if len(pnl_series) > 1 else 0.0
            sharpe = float(np.sqrt(252) * pnl_series.mean() / std_daily) if std_daily > 0 else 0.0
            calmar = compute_calmar(pnl_series)
            y_true_dir = np.where(y_test_ret > 0, 2, 0)
            pred_dir = np.where(class_preds == 1, y_true_dir, class_preds)

            results[name].append(
                {
                    "sharpe": sharpe,
                    "calmar": float(calmar) if calmar is not None else None,
                    "accuracy": float((pred_dir == y_true_dir).mean()),
                    "num_trades": int(np.count_nonzero(np.diff(np.concatenate([[0.0], np.where(class_preds == 2, 1.0, np.where(class_preds == 0, -1.0, 0.0))])))),
                }
            )

    return {
        "prepared_rows": int(len(df)),
        "num_features": int(len(feat_cols)),
        "variants": {name: summarize(rows) for name, rows in results.items()},
    }


def main() -> None:
    args = parse_args()
    report = {"config": vars(args), "tickers": {}}
    for ticker in [t.upper() for t in args.tickers]:
        report["tickers"][ticker] = run_ticker(ticker, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
