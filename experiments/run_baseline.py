#!/usr/bin/env python3
"""
BASELINE EXPERIMENT
====================
Experiment ID: BASELINE_001
Date: 2026-01-07
Purpose: Establish frozen reference point for all future experiments

Configuration:
- Dataset: Last 25% of data (strictly out-of-sample)
- Model: Ensemble (average of RF, XGB, GBRT)
- z_score_threshold: 0.0 (disabled)
- regime_filter: OFF
- trade_limits: OFF
- volatility_scaling: ON (only active gating)

DO NOT MODIFY THIS FILE AFTER BASELINE IS ESTABLISHED.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')

from src.data.market import get_price_history
from src.core.features import build_all_features, add_gbm_features
from src.core.models import make_model
from src.core.metrics import compute_sharpe, compute_drawdown
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS, TRADING_DAYS_PER_YEAR


# =============================================================================
# BASELINE CONFIGURATION (FROZEN - DO NOT MODIFY)
# =============================================================================
BASELINE_CONFIG = {
    "experiment_id": "BASELINE_001",
    "created": "2026-01-07",
    "description": "Frozen baseline for quantitative experiments",
    
    # Dataset split
    "test_fraction": 0.25,  # Last 25% is out-of-sample
    "min_train_rows": 500,
    
    # Model settings
    "models": ["rf", "xgb", "gbrt"],  # Ensemble = average of these
    "ensemble_method": "mean",  # Average predictions
    
    # Gating settings (all disabled except vol scaling)
    "z_score_threshold": 0.0,  # Disabled
    "regime_filter": False,    # OFF
    "trade_limits": False,     # OFF
    "volatility_scaling": True,  # ON
    
    # Trading parameters
    "trade_threshold": 0.002,  # Min predicted return to trade
    "horizon": 1,              # 1-day prediction
    
    # Universe
    "tickers": ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"],
}


def prepare_features(ticker: str, period: str = "10y") -> tuple[pd.DataFrame, list]:
    """
    Build feature matrix for a ticker.
    Returns (df with features and target, list of feature columns).
    """
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Build all features
    hist = build_all_features(hist)
    
    # Add GBM features if not present
    if "gbm_prob_up_1d" not in hist.columns:
        hist = add_gbm_features(hist, horizons=(1,))
    
    # Target: forward 1-day return
    hist["target"] = hist["Close"].pct_change(1).shift(-1)
    
    # Available feature columns
    feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    # Drop rows with NaN target
    df = hist[feat_cols + ["target", "Close"]].dropna().copy()
    
    return df, feat_cols


def run_ensemble_backtest(
    ticker: str,
    config: dict,
) -> dict:
    """
    Run ensemble backtest on a single ticker.
    Uses last test_fraction of data as out-of-sample.
    """
    try:
        df, feat_cols = prepare_features(ticker, period="10y")
        
        if len(df) < config["min_train_rows"] + 100:
            return {"ticker": ticker, "error": "Insufficient data"}
        
        # Split: train on first (1 - test_fraction), test on last test_fraction
        n = len(df)
        split_idx = int(n * (1 - config["test_fraction"]))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        X_train = train_df[feat_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feat_cols].values
        y_test = test_df["target"].values
        
        # Train ensemble models
        predictions = []
        for model_type in config["models"]:
            model = make_model(model_type=model_type, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # Ensemble: average predictions
        if config["ensemble_method"] == "mean":
            y_pred = np.mean(predictions, axis=0)
        else:
            y_pred = np.median(predictions, axis=0)
        
        # Volatility scaling (only active gating)
        if config["volatility_scaling"]:
            vol = train_df["Close"].pct_change().rolling(20).std().iloc[-1]
            if vol > 0:
                # Scale positions by inverse volatility (target 1% daily vol)
                target_vol = 0.01
                vol_scale = np.clip(target_vol / vol, 0.25, 2.0)
            else:
                vol_scale = 1.0
        else:
            vol_scale = 1.0
        
        # Trading simulation
        threshold = config["trade_threshold"]
        positions = np.where(y_pred > threshold, 1, 
                    np.where(y_pred < -threshold, -1, 0))
        
        # Scale by volatility
        positions = positions * vol_scale
        
        # Calculate returns
        strategy_returns = positions * y_test
        
        # Metrics
        sharpe = compute_sharpe(strategy_returns)
        dd_df = compute_drawdown(pd.Series(strategy_returns))
        max_dd = float(dd_df["drawdown"].min()) if not dd_df.empty else 0.0
        
        # Trade statistics
        trade_count = int(np.count_nonzero(np.diff(positions)))
        winning_trades = np.sum((positions[:-1] != 0) & (strategy_returns[:-1] > 0))
        total_trades = np.sum(positions[:-1] != 0)
        win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0
        
        # Cumulative return and CAGR
        cum_return = float(np.sum(strategy_returns))
        test_days = len(test_df)
        years = test_days / TRADING_DAYS_PER_YEAR
        cagr = float((1 + cum_return) ** (1 / years) - 1) if years > 0 and cum_return > -1 else 0.0
        
        # Average return per trade
        avg_return_per_trade = float(cum_return / trade_count) if trade_count > 0 else 0.0
        
        return {
            "ticker": ticker,
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "cagr": round(cagr, 4),
            "total_return": round(cum_return, 4),
            "avg_return_per_trade": round(avg_return_per_trade, 6),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "test_days": test_days,
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
        }
        
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def run_baseline():
    """
    Run the baseline experiment across all tickers.
    """
    print("=" * 70)
    print("BASELINE EXPERIMENT: BASELINE_001")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Test Fraction: {BASELINE_CONFIG['test_fraction']*100:.0f}% (out-of-sample)")
    print(f"Ensemble: {BASELINE_CONFIG['models']}")
    print(f"Z-Score Threshold: {BASELINE_CONFIG['z_score_threshold']} (disabled)")
    print(f"Regime Filter: {BASELINE_CONFIG['regime_filter']}")
    print(f"Trade Limits: {BASELINE_CONFIG['trade_limits']}")
    print(f"Volatility Scaling: {BASELINE_CONFIG['volatility_scaling']}")
    print("=" * 70)
    print()
    
    results = []
    
    for ticker in BASELINE_CONFIG["tickers"]:
        print(f"Processing {ticker}...", end=" ")
        result = run_ensemble_backtest(ticker, BASELINE_CONFIG)
        results.append(result)
        
        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Sharpe={result['sharpe']:.2f}, MaxDD={result['max_drawdown']*100:.1f}%, Trades={result['trade_count']}")
    
    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("\nERROR: No valid results!")
        return
    
    aggregate = {
        "avg_sharpe": round(np.mean([r["sharpe"] for r in valid_results]), 4),
        "avg_max_drawdown": round(np.mean([r["max_drawdown"] for r in valid_results]), 4),
        "avg_cagr": round(np.mean([r["cagr"] for r in valid_results]), 4),
        "avg_win_rate": round(np.mean([r["win_rate"] for r in valid_results]), 4),
        "total_trades": sum(r["trade_count"] for r in valid_results),
        "avg_return_per_trade": round(np.mean([r["avg_return_per_trade"] for r in valid_results]), 6),
    }
    
    # Build final output
    output = {
        "experiment_id": BASELINE_CONFIG["experiment_id"],
        "created": datetime.now().isoformat(),
        "config": BASELINE_CONFIG,
        "aggregate_metrics": aggregate,
        "per_ticker_results": results,
    }
    
    # Print summary
    print()
    print("=" * 70)
    print("BASELINE RESULTS (AGGREGATE)")
    print("=" * 70)
    print(f"  Avg Sharpe Ratio:      {aggregate['avg_sharpe']:.4f}")
    print(f"  Avg Max Drawdown:      {aggregate['avg_max_drawdown']*100:.2f}%")
    print(f"  Avg CAGR:              {aggregate['avg_cagr']*100:.2f}%")
    print(f"  Avg Win Rate:          {aggregate['avg_win_rate']*100:.1f}%")
    print(f"  Total Trades:          {aggregate['total_trades']}")
    print(f"  Avg Return/Trade:      {aggregate['avg_return_per_trade']*100:.4f}%")
    print("=" * 70)
    
    # Save results
    output_path = PROJECT_ROOT / "experiments" / "baseline_metrics.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Baseline saved to: {output_path}")
    print("\nThis baseline is now FROZEN. Do not modify.")
    print("All future experiments will be compared against these metrics.")
    
    return output


if __name__ == "__main__":
    run_baseline()
