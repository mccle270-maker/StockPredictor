#!/usr/bin/env python3
"""
Screen calibrated probability sizing before wiring it into the app.

Variants:
- binary_uncalibrated: classifier probabilities, fixed +/-1 positions
- binary_calibrated: isotonic-calibrated probabilities, fixed +/-1 positions
- confidence_sized_calibrated: isotonic-calibrated probabilities, size by confidence

Uses walk-forward validation with purge/embargo and transaction costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-years", type=float, default=0.5)
    parser.add_argument("--test-years", type=float, default=0.125)
    parser.add_argument("--step-days", type=int, default=31)
    parser.add_argument("--purge-gap-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--cost-per-trade", type=float, default=0.0005)
    parser.add_argument("--output", default="results/probability_sizing_screen_live.json")
    return parser.parse_args()


def generate_splits(
    n_rows: int,
    train_years: float,
    test_years: float,
    step_days: int,
    purge_gap_days: int,
    embargo_days: int,
) -> list[tuple[int, int, int, int]]:
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


def get_classifier(calibrated: bool) -> Any:
    base = XGBClassifier(
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
    if calibrated:
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    return base


def binary_positions(prob_up: np.ndarray, edge: float = 0.05) -> np.ndarray:
    return np.where(prob_up > 0.5 + edge, 1.0, np.where(prob_up < 0.5 - edge, -1.0, 0.0))


def confidence_positions(prob_up: np.ndarray, edge: float = 0.03, min_size: float = 0.25) -> np.ndarray:
    margin = prob_up - 0.5
    signed = np.sign(margin)
    strength = np.clip(np.abs(margin) / 0.5, 0.0, 1.0)
    active = np.abs(margin) >= edge
    size = np.where(active, np.maximum(strength, min_size), 0.0)
    return signed * size


def simulate_positions(positions: np.ndarray, actual_returns: np.ndarray, cost_per_trade: float) -> pd.Series:
    pnl = []
    prev_pos = 0.0
    for pos, ret in zip(positions, actual_returns):
        trade = abs(pos - prev_pos)
        pnl.append(pos * ret - cost_per_trade * trade)
        prev_pos = pos
    return pd.Series(pnl)


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def run_ticker(
    ticker: str,
    period: str,
    horizon: int,
    train_years: float,
    test_years: float,
    step_days: int,
    purge_gap_days: int,
    embargo_days: int,
    cost_per_trade: float,
) -> dict[str, Any]:
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No history for {ticker}")
    df, feat_cols = _prepare_features(hist, ticker, period, horizon)
    splits = generate_splits(len(df), train_years, test_years, step_days, purge_gap_days, embargo_days)
    if len(splits) < 4:
        raise ValueError(f"Not enough folds for {ticker}")

    variants = {
        "binary_uncalibrated": [],
        "binary_calibrated": [],
        "confidence_sized_calibrated": [],
    }

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        X_train = train_df[feat_cols].values
        y_train = (train_df["ftarget_ret_horizon_ahead"].values > 0).astype(int)
        X_test = test_df[feat_cols].values
        y_test_dir = (test_df["ftarget_ret_horizon_ahead"].values > 0).astype(int)
        y_test_ret = test_df["ftarget_ret_horizon_ahead"].values

        for name, calibrated in [
            ("binary_uncalibrated", False),
            ("binary_calibrated", True),
            ("confidence_sized_calibrated", True),
        ]:
            clf = get_classifier(calibrated=calibrated)
            clf.fit(X_train, y_train)
            prob_up = clf.predict_proba(X_test)[:, 1]

            if name == "confidence_sized_calibrated":
                positions = confidence_positions(prob_up)
            else:
                positions = binary_positions(prob_up)

            pnl_series = simulate_positions(positions, y_test_ret, cost_per_trade)
            std_daily = float(pnl_series.std(ddof=1)) if len(pnl_series) > 1 else 0.0
            sharpe = float(np.sqrt(252) * pnl_series.mean() / std_daily) if std_daily > 0 else 0.0
            calmar = compute_calmar(pnl_series)
            pred_dir = (prob_up > 0.5).astype(int)
            accuracy = float((pred_dir == y_test_dir).mean())

            variants[name].append(
                {
                    "fold_id": fold_id,
                    "sharpe": sharpe,
                    "calmar": float(calmar) if calmar is not None else None,
                    "accuracy": accuracy,
                    "num_trades": int(np.count_nonzero(np.diff(np.concatenate([[0.0], positions])))),
                }
            )

    return {
        "prepared_rows": int(len(df)),
        "num_features": int(len(feat_cols)),
        "variants": {name: summarize_variant(rows) for name, rows in variants.items()},
    }


def main() -> None:
    args = parse_args()
    report = {"config": vars(args), "tickers": {}}
    for ticker in [t.upper() for t in args.tickers]:
        report["tickers"][ticker] = run_ticker(
            ticker=ticker,
            period=args.period,
            horizon=args.horizon,
            train_years=args.train_years,
            test_years=args.test_years,
            step_days=args.step_days,
            purge_gap_days=args.purge_gap_days,
            embargo_days=args.embargo_days,
            cost_per_trade=args.cost_per_trade,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
