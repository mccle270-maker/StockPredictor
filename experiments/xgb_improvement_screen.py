#!/usr/bin/env python3
"""
Screen XGBoost improvements before wiring them into the app.

What this tests:
- Baseline optimized XGBoost
- A few more regularized XGBoost parameter combinations
- Volatility-scaled target
- SHAP stability feature filter using earlier folds only, evaluated on later folds

This is intentionally conservative:
- Walk-forward only
- Purge/embargo enabled
- Transaction costs included
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.backtest import _prepare_features
from src.data.market import get_price_history
from src.core.models import make_model
from src.core.metrics import compute_calmar


@dataclass
class Variant:
    name: str
    model_kwargs: dict[str, Any]
    use_vol_scaled_target: bool = False
    stable_features: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-years", type=float, default=0.5)
    parser.add_argument("--test-years", type=float, default=0.125)
    parser.add_argument("--step-days", type=int, default=31)
    parser.add_argument("--purge-gap-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.002)
    parser.add_argument("--cost-per-trade", type=float, default=0.0005)
    parser.add_argument("--output", default="results/xgb_improvement_screen.json")
    return parser.parse_args()


def _build_df(ticker: str, period: str, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No history for {ticker}")
    return _prepare_features(hist, ticker, period, horizon)


def _generate_splits(
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


def _fit_predict(
    variant: Variant,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    threshold: float,
    cost_per_trade: float,
) -> tuple[dict[str, Any], dict[str, float] | None]:
    current_feats = [f for f in feat_cols if variant.stable_features is None or f in variant.stable_features]
    if len(current_feats) < 8:
        return {"error": "too_few_features"}, None

    X_train = train_df[current_feats].values
    X_test = test_df[current_feats].values
    y_train = train_df["ftarget_ret_horizon_ahead"].values.copy()
    y_test = test_df["ftarget_ret_horizon_ahead"].values.copy()

    test_scale = None
    if variant.use_vol_scaled_target and "vol_20d" in current_feats:
        train_scale = np.clip(train_df["vol_20d"].values, 1e-4, None)
        test_scale = np.clip(test_df["vol_20d"].values, 1e-4, None)
        y_train = y_train / train_scale

    model = make_model(model_type="xgb", random_state=42, task="reg", **variant.model_kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if test_scale is not None:
        y_pred = y_pred * test_scale

    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    pnl = []
    prev_pos = 0
    for pos, ret in zip(positions, y_test):
        trade = abs(pos - prev_pos)
        pnl.append(pos * ret - cost_per_trade * trade)
        prev_pos = pos
    pnl = np.asarray(pnl)
    pnl_series = pd.Series(pnl, index=test_df.index)

    std_daily = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    sharpe = float(np.sqrt(252) * pnl.mean() / std_daily) if std_daily > 0 else 0.0
    accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
    calmar = compute_calmar(pnl_series)

    contribs = None
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        dtest = xgb.DMatrix(X_test, feature_names=current_feats)
        raw_contribs = booster.predict(dtest, pred_contribs=True)
        contrib_df = pd.DataFrame(raw_contribs[:, :-1], columns=current_feats)
        contribs = contrib_df.abs().mean().to_dict()

    return {
        "sharpe": sharpe,
        "calmar": float(calmar) if calmar is not None else None,
        "accuracy": accuracy,
        "num_features": len(current_feats),
        "num_trades": int(np.count_nonzero(np.diff(np.concatenate([[0], positions])))),
        "total_pnl": float(pnl.sum()),
    }, contribs


def _stable_features_from_first_half(
    df: pd.DataFrame,
    feat_cols: list[str],
    splits: list[tuple[int, int, int, int]],
    threshold: float,
    cost_per_trade: float,
) -> list[str]:
    first_half = splits[: max(2, len(splits) // 2)]
    rows: list[dict[str, Any]] = []
    baseline = Variant(name="baseline", model_kwargs={})

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(first_half):
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        metrics, contribs = _fit_predict(baseline, train_df, test_df, feat_cols, threshold, cost_per_trade)
        if contribs is None:
            continue
        for feature, value in contribs.items():
            rows.append({"fold_id": fold_id, "feature": feature, "mean_abs_shap": value})

    if not rows:
        return []

    shap_df = pd.DataFrame(rows)
    summary = (
        shap_df.groupby("feature")["mean_abs_shap"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_shap", "std": "std_abs_shap"})
    )
    summary["stability_cv"] = summary["std_abs_shap"] / summary["mean_abs_shap"].replace(0, np.nan)
    summary = summary[(summary["count"] >= 2) & (summary["mean_abs_shap"] > summary["mean_abs_shap"].median())]
    summary = summary[(summary["stability_cv"].fillna(0) <= 1.0)]
    summary = summary.sort_values(["mean_abs_shap", "stability_cv"], ascending=[False, True])
    return summary.head(20)["feature"].tolist()


def run_variant_on_splits(
    df: pd.DataFrame,
    feat_cols: list[str],
    splits: list[tuple[int, int, int, int]],
    variant: Variant,
    threshold: float,
    cost_per_trade: float,
) -> dict[str, Any]:
    fold_metrics: list[dict[str, Any]] = []
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        metrics, _ = _fit_predict(variant, train_df, test_df, feat_cols, threshold, cost_per_trade)
        if "error" in metrics:
            continue
        metrics["fold_id"] = fold_id
        fold_metrics.append(metrics)

    if not fold_metrics:
        return {"variant": variant.name, "folds": 0}

    folds_df = pd.DataFrame(fold_metrics)
    return {
        "variant": variant.name,
        "folds": int(len(folds_df)),
        "mean_sharpe": float(folds_df["sharpe"].mean()),
        "median_sharpe": float(folds_df["sharpe"].median()),
        "mean_calmar": float(folds_df["calmar"].dropna().mean()) if folds_df["calmar"].notna().any() else None,
        "mean_accuracy": float(folds_df["accuracy"].mean()),
        "mean_num_features": float(folds_df["num_features"].mean()),
        "positive_sharpe_pct": float((folds_df["sharpe"] > 0).mean()),
    }


def main() -> None:
    args = parse_args()

    variants = [
        Variant("baseline_optimized", {}),
        Variant(
            "xgb_regularized_shallow",
            {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "min_child_weight": 20,
                "reg_alpha": 0.5,
                "reg_lambda": 12.0,
            },
        ),
        Variant(
            "xgb_medium_regularized",
            {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.65,
                "min_child_weight": 12,
                "reg_alpha": 0.15,
                "reg_lambda": 8.0,
            },
        ),
        Variant("xgb_vol_scaled_target", {}, use_vol_scaled_target=True),
    ]

    report: dict[str, Any] = {
        "tickers": {},
        "config": vars(args),
    }

    for ticker in [t.upper() for t in args.tickers]:
        df, feat_cols = _build_df(ticker, args.period, args.horizon)
        splits = _generate_splits(
            len(df),
            args.train_years,
            args.test_years,
            args.step_days,
            args.purge_gap_days,
            args.embargo_days,
        )
        if len(splits) < 4:
            raise ValueError(f"Not enough walk-forward folds for {ticker}: {len(splits)}")

        stable_features = _stable_features_from_first_half(
            df,
            feat_cols,
            splits,
            args.threshold,
            args.cost_per_trade,
        )
        later_half = splits[len(splits) // 2 :]
        later_baseline = run_variant_on_splits(
            df,
            feat_cols,
            later_half,
            Variant("baseline_later_half", {}),
            args.threshold,
            args.cost_per_trade,
        )
        later_stable = run_variant_on_splits(
            df,
            feat_cols,
            later_half,
            Variant("xgb_shap_stable_later_half", {}, stable_features=stable_features),
            args.threshold,
            args.cost_per_trade,
        )

        variant_results = [
            run_variant_on_splits(df, feat_cols, splits, variant, args.threshold, args.cost_per_trade)
            for variant in variants
        ]
        variant_results.extend([later_baseline, later_stable])

        report["tickers"][ticker] = {
            "prepared_rows": int(len(df)),
            "num_features": int(len(feat_cols)),
            "stable_features_from_first_half": stable_features,
            "variants": variant_results,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
