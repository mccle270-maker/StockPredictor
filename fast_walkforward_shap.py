#!/usr/bin/env python3
"""
Fast walk-forward XGBoost backtest with transaction costs and fold-wise SHAP analysis.

This script is designed to:
1. Run a faster walk-forward backtest on the current best model family (XGBoost)
2. Calculate SHAP-style contribution values on each out-of-sample fold separately
3. Aggregate feature importance stability across time periods

Implementation note:
- Uses XGBoost's built-in `pred_contribs=True`, which returns tree-SHAP contributions
  without requiring the external `shap` package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.services.backtest import _prepare_features
from src.data.market import get_price_history
from src.core.models import make_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast walk-forward XGB + SHAP analysis")
    parser.add_argument("--ticker", default="auto", help="Ticker symbol or 'auto'")
    parser.add_argument("--period", default="10y", help="History period")
    parser.add_argument("--train-years", type=float, default=2.0, help="Training years per fold")
    parser.add_argument("--test-years", type=float, default=0.5, help="Test years per fold")
    parser.add_argument("--step-days", type=int, default=126, help="Step days between folds")
    parser.add_argument("--threshold", type=float, default=0.002, help="Signal threshold")
    parser.add_argument("--cost-per-trade", type=float, default=0.0005, help="Transaction cost")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--output-prefix", default="fast_walkforward_shap", help="Output file prefix")
    return parser.parse_args()


def _auto_select_ticker(period: str) -> Optional[str]:
    """Pick a cached ticker with the deepest available history for the requested period."""
    price_cache_dir = Path(".cache/data/price")
    if not price_cache_dir.exists():
        return None

    best_ticker = None
    best_len = -1
    for cache_file in price_cache_dir.glob("*.cache"):
        try:
            import pickle

            with open(cache_file, "rb") as f:
                entry = pickle.load(f)
            params = entry.get("params", {})
            if params.get("period") != period or params.get("interval") != "1d":
                continue
            ticker = entry.get("identifier")
            data = entry.get("data")
            if ticker and data is not None and len(data) > best_len:
                best_ticker = ticker
                best_len = len(data)
        except Exception:
            continue
    return best_ticker


def _compute_fold_contribs(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute mean absolute tree-SHAP contribution per feature for a fold."""
    if not hasattr(model, "get_booster"):
        return {}

    booster = model.get_booster()
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    contribs = booster.predict(dtest, pred_contribs=True)
    # Last column is the bias term.
    contrib_df = pd.DataFrame(contribs[:, :-1], columns=feature_names)
    return contrib_df.abs().mean().to_dict()


def main() -> None:
    args = _parse_args()

    ticker = args.ticker.upper() if args.ticker.lower() != "auto" else None
    if ticker is None:
        ticker = _auto_select_ticker(args.period)
        if ticker is None:
            raise SystemExit("No cached ticker found for auto selection.")

    hist = get_price_history(ticker, period=args.period, interval="1d")
    if hist is None or hist.empty:
        raise SystemExit(f"No history available for {ticker} ({args.period})")

    df, feat_cols = _prepare_features(hist, ticker, args.period, args.horizon)
    if len(df) < 200:
        raise SystemExit(f"Insufficient prepared rows for {ticker}: {len(df)}")

    train_days = int(252 * args.train_years)
    test_days = int(252 * args.test_years)

    fold_rows: list[dict] = []
    shap_rows: list[dict] = []
    start = 0
    fold_id = 0

    while True:
        train_start = start
        train_end = train_start + train_days
        test_start_idx = train_end
        test_end_idx = test_start_idx + test_days

        if test_end_idx > len(df):
            break

        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start_idx:test_end_idx]

        if len(train_df) < 80 or len(test_df) < 20:
            start += args.step_days
            continue

        X_train = train_df[feat_cols].values
        y_train = train_df["ftarget_ret_horizon_ahead"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["ftarget_ret_horizon_ahead"].values

        model = make_model(model_type="xgb", random_state=42, task="reg")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        positions = np.where(y_pred > args.threshold, 1, np.where(y_pred < -args.threshold, -1, 0))

        pnl = []
        prev_pos = 0
        for pos, ret in zip(positions, y_test):
            trade = abs(pos - prev_pos)
            pnl.append(pos * ret - args.cost_per_trade * trade)
            prev_pos = pos
        pnl = np.array(pnl)

        std_daily = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(np.sqrt(252) * pnl.mean() / std_daily) if std_daily > 0 else 0.0
        accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean())
        num_trades = int(np.count_nonzero(np.diff(np.concatenate([[0], positions]))))

        fold_rows.append(
            {
                "fold_id": fold_id,
                "ticker": ticker,
                "train_start": str(train_df.index[0].date()),
                "train_end": str(train_df.index[-1].date()),
                "test_start": str(test_df.index[0].date()),
                "test_end": str(test_df.index[-1].date()),
                "sharpe": sharpe,
                "accuracy": accuracy,
                "total_pnl": float(pnl.sum()),
                "avg_daily_pnl": float(pnl.mean()),
                "num_trades": num_trades,
            }
        )

        contribs = _compute_fold_contribs(model, X_test, feat_cols)
        for feature, mean_abs_shap in contribs.items():
            shap_rows.append(
                {
                    "fold_id": fold_id,
                    "ticker": ticker,
                    "feature": feature,
                    "mean_abs_shap": float(mean_abs_shap),
                    "test_start": str(test_df.index[0].date()),
                    "test_end": str(test_df.index[-1].date()),
                }
            )

        fold_id += 1
        start += args.step_days

    if not fold_rows:
        raise SystemExit("No walk-forward folds generated. Try a shorter period or smaller training window.")

    folds_df = pd.DataFrame(fold_rows)
    shap_df = pd.DataFrame(shap_rows)

    shap_summary = (
        shap_df.groupby("feature")["mean_abs_shap"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_abs_shap", "std": "std_abs_shap", "median": "median_abs_shap"})
    )
    shap_summary["stability_cv"] = shap_summary["std_abs_shap"] / shap_summary["mean_abs_shap"].replace(0, np.nan)
    shap_summary["importance_stability_score"] = shap_summary["mean_abs_shap"] / shap_summary["std_abs_shap"].replace(0, np.nan)
    shap_summary = shap_summary.sort_values(["mean_abs_shap", "importance_stability_score"], ascending=[False, False])

    summary = {
        "ticker": ticker,
        "model": "xgb",
        "period": args.period,
        "train_years": args.train_years,
        "test_years": args.test_years,
        "step_days": args.step_days,
        "threshold": args.threshold,
        "cost_per_trade": args.cost_per_trade,
        "folds": int(len(folds_df)),
        "mean_sharpe": float(folds_df["sharpe"].mean()),
        "median_sharpe": float(folds_df["sharpe"].median()),
        "mean_accuracy": float(folds_df["accuracy"].mean()),
        "positive_sharpe_pct": float((folds_df["sharpe"] > 0).mean()),
        "mean_total_pnl": float(folds_df["total_pnl"].mean()),
        "top_stable_features": shap_summary.head(15).to_dict(orient="records"),
    }

    prefix = Path(args.output_prefix)
    folds_path = prefix.with_name(f"{prefix.name}_folds.csv")
    shap_path = prefix.with_name(f"{prefix.name}_shap_folds.csv")
    shap_summary_path = prefix.with_name(f"{prefix.name}_shap_summary.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")

    folds_df.to_csv(folds_path, index=False)
    shap_df.to_csv(shap_path, index=False)
    shap_summary.to_csv(shap_summary_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Ticker: {ticker}")
    print(f"Folds: {len(folds_df)}")
    print(f"Mean Sharpe (with costs): {summary['mean_sharpe']:+.4f}")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Positive Sharpe %: {summary['positive_sharpe_pct']:.2%}")
    print("\nTop SHAP-stable features:")
    print(shap_summary.head(10).to_string(index=False))
    print(f"\nSaved: {folds_path}, {shap_path}, {shap_summary_path}, {summary_path}")


if __name__ == "__main__":
    main()
