"""
Test Model Performance on 2025 Data
====================================

Train on 2020-2024, test on 2025 (out-of-sample).
Compare all three trading modes AND legacy models (RF, XGB).

This is a proper holdout test - NO data leakage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.data.market import get_price_history
from src.core.features import build_all_features, add_gbm_features
from src.config import FEATURE_COLUMNS


# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "JPM", "V"]
TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

HORIZON = 5  # 5-day prediction
THRESHOLD_UP = 0.015   # +1.5% = UP
THRESHOLD_DOWN = -0.015  # -1.5% = DOWN

# Trading mode thresholds
MODES = {
    "conservative": {"long_conf": 0.45, "short_conf": 0.70},
    "balanced": {"long_conf": 0.42, "short_conf": 0.55},
    "aggressive": {"long_conf": 0.38, "short_conf": 0.45},
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(ticker: str) -> tuple:
    """Prepare train and test data for a ticker."""
    
    # Get 6 years of data (2019-2025) for warmup
    hist = get_price_history(ticker, period="6y", interval="1d")
    if hist is None or hist.empty:
        return None, None, None, None
    
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    # Build features
    hist = build_all_features(hist)
    if hist is None:
        return None, None, None, None
    
    hist = add_gbm_features(hist, horizons=(1, HORIZON))
    
    # Add regime features
    hist["ma_50"] = hist["Close"].rolling(50).mean()
    hist["ma_200"] = hist["Close"].rolling(200).mean()
    hist["price_vs_ma50"] = (hist["Close"] / hist["ma_50"] - 1).shift(1)
    hist["price_vs_ma200"] = (hist["Close"] / hist["ma_200"] - 1).shift(1)
    hist["ma50_vs_ma200"] = (hist["ma_50"] / hist["ma_200"] - 1).shift(1)
    hist["golden_cross"] = (hist["ma_50"] > hist["ma_200"]).astype(int).shift(1)
    
    if "vol_20d" in hist.columns:
        hist["vol_percentile"] = hist["vol_20d"].rolling(252).rank(pct=True).shift(1)
    
    rolling_max = hist["Close"].rolling(252, min_periods=1).max()
    hist["drawdown_pct"] = ((hist["Close"] / rolling_max) - 1).shift(1)
    hist["in_correction"] = (hist["drawdown_pct"] < -0.10).astype(int)
    hist["in_bear_market"] = (hist["drawdown_pct"] < -0.20).astype(int)
    
    ret_20d = hist["Close"].pct_change(20)
    hist["momentum_20d_zscore"] = (
        (ret_20d - ret_20d.rolling(252).mean()) / 
        ret_20d.rolling(252).std()
    ).shift(1)
    
    # Create targets
    future_ret = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    hist["target_class"] = np.where(
        future_ret > THRESHOLD_UP, 2,
        np.where(future_ret < THRESHOLD_DOWN, 0, 1)
    )
    hist["target_return"] = future_ret
    
    # Feature columns
    base_feat_cols = [c for c in FEATURE_COLUMNS if c in hist.columns]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross",
        "vol_percentile", "drawdown_pct", "in_correction", "in_bear_market", 
        "momentum_20d_zscore",
    ]
    feat_cols = [c for c in base_feat_cols + regime_cols if c in hist.columns]
    
    # Filter high NaN features
    nan_rates = hist[feat_cols].isna().mean()
    feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
    
    # Fill NaNs
    hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
    
    # Split by date
    train_mask = (hist.index >= TRAIN_START) & (hist.index <= TRAIN_END)
    test_mask = (hist.index >= TEST_START) & (hist.index <= TEST_END)
    
    train_df = hist[train_mask].dropna(subset=["target_class", "target_return"])
    test_df = hist[test_mask].dropna(subset=["target_class", "target_return"])
    
    if len(train_df) < 100 or len(test_df) < 20:
        return None, None, None, None
    
    return train_df, test_df, feat_cols, hist


# =============================================================================
# CLASSIFICATION MODEL (Adaptive)
# =============================================================================

def train_classifier(train_df, feat_cols):
    """Train the adaptive classifier."""
    X = train_df[feat_cols].values
    y = train_df["target_class"].values.astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Class weights
    class_counts = np.bincount(y)
    total = len(y)
    weights = {
        0: 1.5 * total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,
        1: 1.0 * total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,
        2: 1.0 * total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,
    }
    sample_weights = np.array([weights[c] for c in y])
    
    # XGBoost classifier
    base_clf = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42,
        eval_metric="mlogloss",
    )
    base_clf.fit(X_scaled, y, sample_weight=sample_weights)
    
    # Calibrate
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_scaled, y)
    
    return clf, scaler


def evaluate_classifier(clf, scaler, test_df, feat_cols, mode_config):
    """Evaluate classifier on test data with a specific trading mode."""
    X_test = test_df[feat_cols].values
    X_scaled = scaler.transform(X_test)
    
    # Predict probabilities
    proba = clf.predict_proba(X_scaled)
    
    results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        down_prob, neutral_prob, up_prob = proba[i][0], proba[i][1], proba[i][2]
        actual_ret = row["target_return"]
        
        # Determine signal based on mode
        if up_prob >= mode_config["long_conf"] and up_prob > down_prob:
            signal = 1  # BUY
            confidence = up_prob
        elif down_prob >= mode_config["short_conf"] and down_prob > up_prob:
            signal = -1  # SELL
            confidence = down_prob
        else:
            signal = 0  # HOLD
            confidence = max(up_prob, down_prob, neutral_prob)
        
        results.append({
            "date": idx,
            "signal": signal,
            "actual_return": actual_ret,
            "up_prob": up_prob,
            "down_prob": down_prob,
            "confidence": confidence,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# REGRESSION MODELS (Legacy - RF and XGB)
# =============================================================================

def train_regressor(train_df, feat_cols, model_type="xgb"):
    """Train a regression model."""
    X = train_df[feat_cols].values
    y = train_df["target_return"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == "xgb":
        model = XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42,
        )
    else:  # rf
        model = RandomForestRegressor(
            n_estimators=100, max_depth=None, min_samples_leaf=4,
            max_features=0.7, random_state=42,
        )
    
    model.fit(X_scaled, y)
    return model, scaler


def evaluate_regressor(model, scaler, test_df, feat_cols, threshold=0.005):
    """Evaluate regressor on test data."""
    X_test = test_df[feat_cols].values
    X_scaled = scaler.transform(X_test)
    
    predictions = model.predict(X_scaled)
    
    results = []
    for i, (idx, row) in enumerate(test_df.iterrows()):
        pred_ret = predictions[i]
        actual_ret = row["target_return"]
        
        # Signal based on predicted return
        if pred_ret > threshold:
            signal = 1  # BUY
        elif pred_ret < -threshold:
            signal = -1  # SELL
        else:
            signal = 0  # HOLD
        
        results.append({
            "date": idx,
            "signal": signal,
            "pred_return": pred_ret,
            "actual_return": actual_ret,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(results_df):
    """Calculate trading metrics from results."""
    # Strategy returns (signal * actual_return)
    results_df = results_df.copy()
    results_df["strategy_return"] = results_df["signal"] * results_df["actual_return"]
    
    # Only count days we traded
    traded = results_df[results_df["signal"] != 0]
    
    if len(traded) == 0:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "accuracy": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    
    # Calculate metrics
    total_ret = results_df["strategy_return"].sum()
    daily_rets = results_df["strategy_return"]
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    
    # Accuracy: did we predict direction correctly?
    traded["correct"] = (traded["signal"] * traded["actual_return"]) > 0
    accuracy = traded["correct"].mean()
    
    # Win rate
    wins = traded[traded["strategy_return"] > 0]
    losses = traded[traded["strategy_return"] < 0]
    win_rate = len(wins) / len(traded) if len(traded) > 0 else 0
    avg_win = wins["strategy_return"].mean() if len(wins) > 0 else 0
    avg_loss = losses["strategy_return"].mean() if len(losses) > 0 else 0
    
    return {
        "total_return": total_ret,
        "sharpe": sharpe,
        "accuracy": accuracy,
        "num_trades": len(traded),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "long_trades": len(traded[traded["signal"] == 1]),
        "short_trades": len(traded[traded["signal"] == -1]),
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 70)
    print("TESTING MODEL PERFORMANCE ON 2025 DATA")
    print("=" * 70)
    print(f"\nTrain period: {TRAIN_START} to {TRAIN_END}")
    print(f"Test period:  {TEST_START} to {TEST_END}")
    print(f"Horizon: {HORIZON} days")
    print(f"\nTickers: {', '.join(TICKERS)}")
    print()
    
    # Store results for each approach
    all_results = {
        "conservative": [],
        "balanced": [],
        "aggressive": [],
        "xgb_regressor": [],
        "rf_regressor": [],
    }
    
    # Buy & Hold for comparison
    bh_returns = []
    
    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"Processing {ticker}...")
        print("=" * 50)
        
        # Prepare data
        train_df, test_df, feat_cols, full_hist = prepare_data(ticker)
        
        if train_df is None:
            print(f"  ⚠️ Skipping {ticker} - insufficient data")
            continue
        
        print(f"  Train: {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
        print(f"  Test:  {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")
        print(f"  Features: {len(feat_cols)}")
        
        # Buy & Hold return for test period
        test_start_price = test_df["Close"].iloc[0]
        test_end_price = test_df["Close"].iloc[-1]
        bh_ret = (test_end_price / test_start_price - 1)
        bh_returns.append(bh_ret)
        
        # === ADAPTIVE CLASSIFIER ===
        print("\n  Training adaptive classifier...")
        clf, clf_scaler = train_classifier(train_df, feat_cols)
        
        for mode_name, mode_config in MODES.items():
            results = evaluate_classifier(clf, clf_scaler, test_df, feat_cols, mode_config)
            metrics = calculate_metrics(results)
            all_results[mode_name].append({
                "ticker": ticker,
                **metrics,
                "bh_return": bh_ret,
            })
        
        # === XGB REGRESSOR ===
        print("  Training XGB regressor...")
        xgb_model, xgb_scaler = train_regressor(train_df, feat_cols, "xgb")
        xgb_results = evaluate_regressor(xgb_model, xgb_scaler, test_df, feat_cols)
        xgb_metrics = calculate_metrics(xgb_results)
        all_results["xgb_regressor"].append({
            "ticker": ticker,
            **xgb_metrics,
            "bh_return": bh_ret,
        })
        
        # === RF REGRESSOR ===
        print("  Training RF regressor...")
        rf_model, rf_scaler = train_regressor(train_df, feat_cols, "rf")
        rf_results = evaluate_regressor(rf_model, rf_scaler, test_df, feat_cols)
        rf_metrics = calculate_metrics(rf_results)
        all_results["rf_regressor"].append({
            "ticker": ticker,
            **rf_metrics,
            "bh_return": bh_ret,
        })
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: 2025 PERFORMANCE")
    print("=" * 70)
    
    print(f"\nBuy & Hold (baseline): {np.mean(bh_returns):.2%} avg return")
    
    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'Sharpe':>10} {'Accuracy':>10} {'Return':>10} {'Trades':>8} {'Win Rate':>10}")
    print("-" * 70)
    
    for model_name in ["conservative", "balanced", "aggressive", "xgb_regressor", "rf_regressor"]:
        results = all_results[model_name]
        if not results:
            continue
        
        df = pd.DataFrame(results)
        avg_sharpe = df["sharpe"].mean()
        avg_accuracy = df["accuracy"].mean()
        avg_return = df["total_return"].mean()
        total_trades = df["num_trades"].sum()
        avg_win_rate = df["win_rate"].mean()
        
        display_name = model_name.replace("_", " ").title()
        print(f"{display_name:<20} {avg_sharpe:>10.2f} {avg_accuracy:>10.1%} {avg_return:>10.2%} {total_trades:>8} {avg_win_rate:>10.1%}")
    
    # Detailed breakdown per ticker
    print("\n" + "=" * 70)
    print("DETAILED RESULTS BY TICKER (BALANCED MODE)")
    print("=" * 70)
    
    balanced_df = pd.DataFrame(all_results["balanced"])
    if not balanced_df.empty:
        print(f"\n{'Ticker':<8} {'Sharpe':>10} {'Accuracy':>10} {'Return':>10} {'B&H':>10} {'Trades':>8} {'Win%':>8}")
        print("-" * 66)
        
        for _, row in balanced_df.iterrows():
            beat_bh = "✅" if row["total_return"] > row["bh_return"] else "❌"
            print(f"{row['ticker']:<8} {row['sharpe']:>10.2f} {row['accuracy']:>10.1%} {row['total_return']:>10.2%} {row['bh_return']:>10.2%} {row['num_trades']:>8} {row['win_rate']:>7.1%} {beat_bh}")
    
    # Trade breakdown
    print("\n" + "=" * 70)
    print("TRADE BREAKDOWN BY MODE")
    print("=" * 70)
    
    for mode_name in ["conservative", "balanced", "aggressive"]:
        results = all_results[mode_name]
        if not results:
            continue
        
        df = pd.DataFrame(results)
        total_long = df["long_trades"].sum()
        total_short = df["short_trades"].sum()
        total = df["num_trades"].sum()
        
        print(f"\n{mode_name.title():}")
        print(f"  Total trades: {total}")
        print(f"  Long trades:  {total_long} ({total_long/total*100:.1f}%)" if total > 0 else "  Long trades:  0")
        print(f"  Short trades: {total_short} ({total_short/total*100:.1f}%)" if total > 0 else "  Short trades: 0")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Find best model
    best_sharpe = -999
    best_model = None
    for model_name, results in all_results.items():
        if results:
            avg = np.mean([r["sharpe"] for r in results])
            if avg > best_sharpe:
                best_sharpe = avg
                best_model = model_name
    
    print(f"\nBest performing model: {best_model.replace('_', ' ').title()} (Sharpe: {best_sharpe:.2f})")
    
    balanced_results = all_results.get("balanced", [])
    if balanced_results:
        balanced_df = pd.DataFrame(balanced_results)
        beat_bh = (balanced_df["total_return"] > balanced_df["bh_return"]).sum()
        print(f"Balanced mode beat B&H on {beat_bh}/{len(balanced_df)} tickers")
    
    return all_results


if __name__ == "__main__":
    main()
