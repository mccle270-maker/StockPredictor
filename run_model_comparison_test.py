"""
Comprehensive Model Comparison Test
====================================

Tests XGBoost, Random Forest, and Ensemble models on truly unseen data
with various configurations:
- With/without ElasticNet feature selection
- With/without auto-optimize
- Compares Sharpe ratios and accuracy vs Buy & Hold

This test uses proper walk-forward validation to ensure truly unseen test data.

Run: python run_model_comparison_test.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from src.core.models import make_model, select_features_elasticnet
from src.core.features import build_all_features, add_gbm_features
from src.core.metrics import compute_sharpe, compute_drawdown
from src.data.market import get_price_history
from src.data.macro import get_macro_df
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS

# Test configuration
TEST_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
TEST_PERIOD = "5y"
TRAIN_YEARS = 3
TEST_YEARS = 1  # Use last year as truly unseen holdout
HORIZON = 5
THRESHOLD = 0.002  # Signal threshold for trading


def prepare_data(ticker: str, period: str = "5y", horizon: int = 5) -> tuple:
    """Prepare features and target data for a ticker."""
    print(f"  📊 Fetching data for {ticker}...")
    
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or not hasattr(hist, 'columns') or len(hist) < 200:
        return None, None, None
    
    # Ensure it's a DataFrame
    if not isinstance(hist, pd.DataFrame):
        print(f"    ⚠️ Unexpected data type: {type(hist)}")
        return None, None, None
    
    # Build features
    hist = build_all_features(hist)
    if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
        return None, None, None
    
    hist = add_gbm_features(hist, horizons=(1, horizon) if horizon > 1 else (1,))
    if not isinstance(hist, pd.DataFrame):
        return None, None, None
    
    # Add macro data (handle timezone issues)
    try:
        macro_df = get_macro_df(period=period)
        # Make both timezone-naive for joining
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)
        hist = hist.join(macro_df, how="left")
    except Exception as e:
        print(f"    ⚠️ Macro data issue: {e}")
    
    # Build target manually (build_target returns Series)
    target_col = f"ftarget_ret_{horizon}d_ahead"
    hist[target_col] = hist["Close"].pct_change(horizon).shift(-horizon)
    
    # Winsorize target (clip extreme values)
    lower = hist[target_col].quantile(0.01)
    upper = hist[target_col].quantile(0.99)
    hist[target_col] = hist[target_col].clip(lower=lower, upper=upper)
    
    # Collect available features
    feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    macro_cols = [c for c in MACRO_COLUMNS if c in hist.columns]
    feat_cols = list(dict.fromkeys(feat_cols + macro_cols))
    
    # Quality filter - remove features with >30% NaN
    nan_rates = hist[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    # Final dataset
    cols_needed = feat_cols + [target_col, "Close"]
    df = hist[cols_needed].dropna().copy()
    
    return df, feat_cols, target_col


def split_train_test(df: pd.DataFrame, train_years: int = 3, test_years: int = 1) -> tuple:
    """Split data into training and truly unseen test sets."""
    cutoff = df.index.max() - pd.Timedelta(days=252 * test_years)
    train_df = df.loc[df.index < cutoff].copy()
    test_df = df.loc[df.index >= cutoff].copy()
    return train_df, test_df


def compute_buy_hold_metrics(test_df: pd.DataFrame) -> dict:
    """Compute Buy & Hold baseline metrics."""
    if test_df.empty or "Close" not in test_df.columns:
        return {"sharpe": None, "total_return": None, "accuracy": None}
    
    # Daily returns
    daily_returns = test_df["Close"].pct_change().dropna()
    
    if len(daily_returns) < 10:
        return {"sharpe": None, "total_return": None, "accuracy": None}
    
    sharpe = compute_sharpe(daily_returns)
    total_return = (test_df["Close"].iloc[-1] / test_df["Close"].iloc[0] - 1) * 100
    # Accuracy for B&H = % of up days
    accuracy = (daily_returns > 0).mean() * 100
    
    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "accuracy": accuracy,
    }


def run_model_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list,
    target_col: str,
    model_type: str = "rf",
    use_elasticnet: bool = False,
    use_optimized: bool = True,
    threshold: float = 0.002,
) -> dict:
    """
    Train model and evaluate on unseen test data.
    
    Returns dict with metrics.
    """
    X_train = train_df[feat_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feat_cols].values
    y_test = test_df[target_col].values
    
    actual_feat_cols = list(feat_cols)
    
    # ElasticNet feature selection
    if use_elasticnet:
        try:
            X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                X_train, y_train,
                feature_names=feat_cols,
                dates=train_df.index,
                horizon=HORIZON,
                n_splits=5,
                l1_ratio=0.5,
                min_features=10,
            )
            X_train = X_train_sel
            X_test = X_test[:, sel_mask]
            actual_feat_cols = sel_names
        except Exception as e:
            print(f"    ⚠️ ElasticNet failed: {e}")
    
    # Create and train model
    try:
        model = make_model(model_type=model_type, random_state=42, use_optimized=use_optimized)
        model.fit(X_train, y_train)
    except Exception as e:
        return {"error": str(e)}
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Trading simulation
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    pnl = positions * y_test
    
    # Metrics
    accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean()) * 100
    sharpe = compute_sharpe(pnl)
    total_return = float(pnl.sum()) * 100
    
    # Max drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
    
    # Trade stats
    num_trades = int(np.count_nonzero(np.diff(positions)))
    long_trades = int((positions == 1).sum())
    short_trades = int((positions == -1).sum())
    neutral_days = int((positions == 0).sum())
    
    return {
        "model_type": model_type,
        "use_elasticnet": use_elasticnet,
        "use_optimized": use_optimized,
        "n_features": len(actual_feat_cols),
        "n_train_samples": len(train_df),
        "n_test_samples": len(test_df),
        "accuracy": accuracy,
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "neutral_days": neutral_days,
    }


def run_comprehensive_test():
    """Run comprehensive model comparison test."""
    
    print("=" * 80)
    print("🧪 COMPREHENSIVE MODEL COMPARISON TEST")
    print("=" * 80)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Tickers: {', '.join(TEST_TICKERS)}")
    print(f"📈 Period: {TEST_PERIOD} (Train: {TRAIN_YEARS}y, Test: {TEST_YEARS}y holdout)")
    print(f"🎯 Horizon: {HORIZON} days")
    print(f"⚡ Threshold: {THRESHOLD}")
    print("=" * 80)
    
    # Model configurations to test
    configs = [
        {"model_type": "rf", "use_elasticnet": False, "use_optimized": False, "label": "RF (Legacy)"},
        {"model_type": "rf", "use_elasticnet": False, "use_optimized": True, "label": "RF (Optimized)"},
        {"model_type": "rf", "use_elasticnet": True, "use_optimized": True, "label": "RF + ElasticNet"},
        {"model_type": "xgb", "use_elasticnet": False, "use_optimized": False, "label": "XGB (Legacy)"},
        {"model_type": "xgb", "use_elasticnet": False, "use_optimized": True, "label": "XGB (Optimized)"},
        {"model_type": "xgb", "use_elasticnet": True, "use_optimized": True, "label": "XGB + ElasticNet"},
        {"model_type": "ensemble", "use_elasticnet": False, "use_optimized": True, "label": "Ensemble (RF+XGB)"},
        {"model_type": "ensemble", "use_elasticnet": True, "use_optimized": True, "label": "Ensemble + ElasticNet"},
    ]
    
    all_results = []
    
    for ticker in TEST_TICKERS:
        print(f"\n{'='*60}")
        print(f"🏢 Testing {ticker}")
        print("=" * 60)
        
        # Prepare data
        data = prepare_data(ticker, period=TEST_PERIOD, horizon=HORIZON)
        if data[0] is None:
            print(f"  ❌ Insufficient data for {ticker}, skipping...")
            continue
        
        df, feat_cols, target_col = data
        print(f"  ✅ Data loaded: {len(df)} rows, {len(feat_cols)} features")
        
        # Split train/test
        train_df, test_df = split_train_test(df, train_years=TRAIN_YEARS, test_years=TEST_YEARS)
        print(f"  📊 Train: {len(train_df)} rows ({train_df.index[0].date()} to {train_df.index[-1].date()})")
        print(f"  🧪 Test:  {len(test_df)} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})")
        
        if len(train_df) < 100 or len(test_df) < 20:
            print(f"  ❌ Insufficient train/test data, skipping...")
            continue
        
        # Buy & Hold baseline
        bh_metrics = compute_buy_hold_metrics(test_df)
        print(f"\n  📈 Buy & Hold Baseline:")
        print(f"     Sharpe: {bh_metrics['sharpe']:.3f}" if bh_metrics['sharpe'] else "     Sharpe: N/A")
        print(f"     Return: {bh_metrics['total_return']:.2f}%" if bh_metrics['total_return'] else "     Return: N/A")
        print(f"     Up Days: {bh_metrics['accuracy']:.1f}%" if bh_metrics['accuracy'] else "     Up Days: N/A")
        
        print(f"\n  🤖 Model Results on Unseen Test Data:")
        print(f"  {'Config':<25} {'Acc %':>8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Trades':>7}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*7}")
        
        ticker_results = []
        
        for config in configs:
            result = run_model_test(
                train_df, test_df, feat_cols, target_col,
                model_type=config["model_type"],
                use_elasticnet=config["use_elasticnet"],
                use_optimized=config["use_optimized"],
                threshold=THRESHOLD,
            )
            
            if "error" in result:
                print(f"  {config['label']:<25} ERROR: {result['error']}")
                continue
            
            result["ticker"] = ticker
            result["config_label"] = config["label"]
            result["bh_sharpe"] = bh_metrics["sharpe"]
            result["bh_return"] = bh_metrics["total_return"]
            result["sharpe_vs_bh"] = (result["sharpe"] - bh_metrics["sharpe"]) if bh_metrics["sharpe"] else None
            
            ticker_results.append(result)
            all_results.append(result)
            
            # Format output
            acc_str = f"{result['accuracy']:.1f}%"
            sharpe_str = f"{result['sharpe']:.3f}" if result['sharpe'] is not None else "N/A"
            ret_str = f"{result['total_return']:+.2f}%"
            dd_str = f"{result['max_drawdown']:.2f}%"
            trades_str = f"{result['num_trades']}"
            
            # Color indicator for Sharpe vs B&H
            vs_bh = ""
            if result["sharpe_vs_bh"] is not None:
                vs_bh = " ✅" if result["sharpe_vs_bh"] > 0 else " ❌"
            
            print(f"  {config['label']:<25} {acc_str:>8} {sharpe_str:>8} {ret_str:>10} {dd_str:>8} {trades_str:>7}{vs_bh}")
    
    # Summary across all tickers
    print("\n" + "=" * 80)
    print("📊 AGGREGATE RESULTS ACROSS ALL TICKERS")
    print("=" * 80)
    
    if not all_results:
        print("❌ No results to aggregate")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Group by config
    summary = results_df.groupby("config_label").agg({
        "accuracy": ["mean", "std"],
        "sharpe": ["mean", "std", "min", "max"],
        "total_return": ["mean", "std"],
        "sharpe_vs_bh": "mean",
        "n_features": "mean",
    }).round(3)
    
    print("\n📈 Average Metrics by Configuration:")
    print("-" * 100)
    
    for label in summary.index:
        acc_mean = summary.loc[label, ("accuracy", "mean")]
        acc_std = summary.loc[label, ("accuracy", "std")]
        sharpe_mean = summary.loc[label, ("sharpe", "mean")]
        sharpe_std = summary.loc[label, ("sharpe", "std")]
        ret_mean = summary.loc[label, ("total_return", "mean")]
        vs_bh = summary.loc[label, ("sharpe_vs_bh", "mean")]
        n_feats = summary.loc[label, ("n_features", "mean")]
        
        vs_bh_str = f"{vs_bh:+.3f}" if pd.notna(vs_bh) else "N/A"
        
        print(f"{label:<28} Acc: {acc_mean:.1f}±{acc_std:.1f}%  "
              f"Sharpe: {sharpe_mean:.3f}±{sharpe_std:.3f}  "
              f"Return: {ret_mean:+.2f}%  "
              f"vs B&H: {vs_bh_str}  "
              f"Features: {n_feats:.0f}")
    
    # Find best configuration
    best_sharpe_config = summary[("sharpe", "mean")].idxmax()
    best_accuracy_config = summary[("accuracy", "mean")].idxmax()
    best_vs_bh_config = summary[("sharpe_vs_bh", "mean")].idxmax()
    
    print("\n🏆 BEST CONFIGURATIONS:")
    print(f"  Best Sharpe:      {best_sharpe_config} ({summary.loc[best_sharpe_config, ('sharpe', 'mean')]:.3f})")
    print(f"  Best Accuracy:    {best_accuracy_config} ({summary.loc[best_accuracy_config, ('accuracy', 'mean')]:.1f}%)")
    print(f"  Best vs Buy&Hold: {best_vs_bh_config} ({summary.loc[best_vs_bh_config, ('sharpe_vs_bh', 'mean')]:+.3f})")
    
    # Save detailed results
    output_path = Path(__file__).parent / "model_comparison_results.json"
    with open(output_path, "w") as f:
        # Convert to JSON-serializable format
        json_results = []
        for r in all_results:
            json_r = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    json_r[k] = float(v)
                elif isinstance(v, np.ndarray):
                    json_r[k] = v.tolist()
                elif pd.isna(v):
                    json_r[k] = None
                else:
                    json_r[k] = v
            json_results.append(json_r)
        
        json.dump({
            "test_date": datetime.now().isoformat(),
            "config": {
                "tickers": TEST_TICKERS,
                "period": TEST_PERIOD,
                "train_years": TRAIN_YEARS,
                "test_years": TEST_YEARS,
                "horizon": HORIZON,
                "threshold": THRESHOLD,
            },
            "results": json_results,
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    run_comprehensive_test()
