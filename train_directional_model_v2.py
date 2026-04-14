"""
Directional Classification Model V2 - PROPER TRAIN/TEST SPLIT
==============================================================

Fixed data leakage: 
- Train on 2018-2021 (includes 2018 correction, 2020 COVID crash, 2021 bull)
- Test on 2022 bear market (truly out-of-sample)
- Test on 2025 bull market (truly out-of-sample)

Run: python train_directional_model_v2.py
"""

import os
import sys
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.core.features import build_all_features, add_gbm_features
from src.core.metrics import compute_sharpe
from src.data.market import get_price_history
from src.config import FEATURE_COLUMNS

# Config
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "JPM", "V"]
HORIZON = 5
THRESHOLD_UP = 0.015    # +1.5% = UP (lowered for more signals)
THRESHOLD_DOWN = -0.015 # -1.5% = DOWN

COST_BPS = 10
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime-aware features."""
    if "Close" not in df.columns:
        return df
        
    # Moving averages
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    df["price_vs_ma50"] = (df["Close"] / df["ma_50"] - 1).shift(1)
    df["price_vs_ma200"] = (df["Close"] / df["ma_200"] - 1).shift(1)
    df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"] - 1).shift(1)
    df["golden_cross"] = (df["ma_50"] > df["ma_200"]).astype(int).shift(1)
    df["death_cross"] = (df["ma_50"] < df["ma_200"]).astype(int).shift(1)
    
    # Trend strength
    df["days_below_ma200"] = (df["Close"] < df["ma_200"]).rolling(60).sum().shift(1)
    df["days_below_ma50"] = (df["Close"] < df["ma_50"]).rolling(20).sum().shift(1)
    
    # Volatility regime
    if "vol_20d" in df.columns:
        df["vol_percentile"] = df["vol_20d"].rolling(252).rank(pct=True).shift(1)
        df["high_vol_regime"] = (df["vol_percentile"] > 0.8).astype(int)
        df["low_vol_regime"] = (df["vol_percentile"] < 0.2).astype(int)
    
    # Drawdown
    rolling_max = df["Close"].rolling(252, min_periods=1).max()
    df["drawdown_pct"] = ((df["Close"] / rolling_max) - 1).shift(1)
    df["in_correction"] = (df["drawdown_pct"] < -0.10).astype(int)
    df["in_bear_market"] = (df["drawdown_pct"] < -0.20).astype(int)
    
    # Momentum
    ret_20d = df["Close"].pct_change(20)
    df["momentum_20d_zscore"] = (
        (ret_20d - ret_20d.rolling(252).mean()) / 
        ret_20d.rolling(252).std()
    ).shift(1)
    
    # RSI extremes
    if "rsi14" in df.columns:
        df["rsi_oversold"] = (df["rsi14"] < 30).astype(int).shift(1)
        df["rsi_overbought"] = (df["rsi14"] > 70).astype(int).shift(1)
    
    return df


def get_ticker_data(ticker: str) -> pd.DataFrame:
    """Get and prepare data for a ticker."""
    hist = get_price_history(ticker, period="max", interval="1d")
    if hist is None or not isinstance(hist, pd.DataFrame):
        return None
    
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    hist = build_all_features(hist)
    if hist is None:
        return None
    
    hist = add_gbm_features(hist, horizons=(1, HORIZON))
    hist = add_regime_features(hist)
    
    # Create TARGET as classification labels
    future_ret = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    hist["target_class"] = np.where(
        future_ret > THRESHOLD_UP, 2,      # UP
        np.where(future_ret < THRESHOLD_DOWN, 0,  # DOWN
        1)  # NEUTRAL
    )
    hist["target_return"] = future_ret
    
    return hist


def prepare_classification_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Prepare data for classification."""
    
    base_feat_cols = [c for c in FEATURE_COLUMNS]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross", "death_cross",
        "days_below_ma200", "days_below_ma50",
        "vol_percentile", "high_vol_regime", "low_vol_regime",
        "drawdown_pct", "in_correction", "in_bear_market", 
        "momentum_20d_zscore", "rsi_oversold", "rsi_overbought",
    ]
    
    all_X = []
    all_y = []
    all_returns = []
    
    for ticker in tickers:
        hist = get_ticker_data(ticker)
        if hist is None:
            continue
        
        # Filter date range
        mask = (hist.index >= start_date) & (hist.index <= end_date)
        df = hist.loc[mask].copy()
        
        # Get features
        feat_cols = [c for c in base_feat_cols + regime_cols if c in df.columns]
        
        # Quality filter
        nan_rates = df[feat_cols].isna().mean()
        feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
        
        df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0)
        
        # Drop NaN targets
        df = df.dropna(subset=["target_class", "target_return"])
        
        if len(df) < 50:
            continue
        
        X = df[feat_cols].values
        y = df["target_class"].values.astype(int)
        returns = df["target_return"].values
        
        all_X.append(X)
        all_y.append(y)
        all_returns.append(returns)
    
    if not all_X:
        return None, None, None, []
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    returns = np.concatenate(all_returns)
    
    return X, y, returns, feat_cols


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Train classification model with strong class balancing."""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # AGGRESSIVE class weights to favor DOWN predictions
    # The market naturally goes up more, so we need to over-weight DOWN
    class_counts = np.bincount(y_train)
    total = len(y_train)
    
    # Aggressive weighting: 2x weight on DOWN class
    class_weights = {
        0: 2.0 * total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,  # DOWN - 2x weight
        1: total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,  # NEUTRAL
        2: total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,  # UP
    }
    
    print(f"   Class distribution:")
    print(f"      DOWN (0): {class_counts[0]} ({class_counts[0]/total*100:.1f}%)")
    print(f"      NEUTRAL (1): {class_counts[1]} ({class_counts[1]/total*100:.1f}%)")
    print(f"      UP (2): {class_counts[2]} ({class_counts[2]/total*100:.1f}%)")
    print(f"   Class weights (2x DOWN): {class_weights}")
    
    # Train XGBoost classifier
    if HAS_XGB:
        sample_weights = np.array([class_weights[c] for c in y_train])
        
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,  # Regularization
            reg_alpha=0.1,       # L1
            reg_lambda=1.0,      # L2
            random_state=42,
            eval_metric="mlogloss",
        )
        clf.fit(X_scaled, y_train, sample_weight=sample_weights)
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_scaled, y_train)
    
    return clf, scaler


def evaluate_trading(
    clf,
    scaler,
    feat_cols: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    returns_test: np.ndarray,
    period_name: str,
) -> dict:
    """Evaluate trading performance with proper position sizing."""
    
    X_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_scaled)
    
    # Get probabilities for confidence weighting
    y_proba = clf.predict_proba(X_scaled)
    
    # Classification metrics
    accuracy = (y_pred == y_test).mean() * 100
    
    # CONFIDENCE-WEIGHTED positions
    # Only trade when confident (> 50% probability for predicted class)
    positions = np.zeros(len(y_pred))
    for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
        conf = proba[pred]
        if pred == 2 and conf > 0.45:  # UP
            positions[i] = 1
        elif pred == 0 and conf > 0.45:  # DOWN
            positions[i] = -1
        # else: stay FLAT
    
    # PnL
    pnl = positions * returns_test
    
    # Apply costs
    cost_per_trade = COST_BPS / 10000
    pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = pos_changes * cost_per_trade
    pnl_net = pnl - costs
    
    # Buy & hold
    bh_pnl = returns_test
    
    # Metrics
    sharpe_strategy = compute_sharpe(pnl_net)
    sharpe_bh = compute_sharpe(bh_pnl)
    total_return_strat = pnl_net.sum() * 100
    total_return_bh = bh_pnl.sum() * 100
    
    # Position breakdown
    n_long = (positions == 1).sum()
    n_short = (positions == -1).sum()
    n_flat = (positions == 0).sum()
    
    # Direction accuracy (for non-neutral predictions & non-neutral reality)
    traded = positions != 0
    if traded.sum() > 0:
        # Direction match: long & up, or short & down
        direction_correct = ((positions == 1) & (returns_test > 0)) | \
                           ((positions == -1) & (returns_test < 0))
        direction_acc = direction_correct[traded].mean() * 100
        win_rate = ((positions * returns_test) > 0)[traded].mean() * 100
    else:
        direction_acc = 0
        win_rate = 0
    
    return {
        "accuracy": accuracy,
        "direction_accuracy": direction_acc,
        "win_rate": win_rate,
        "sharpe_strategy": sharpe_strategy,
        "sharpe_bh": sharpe_bh,
        "return_strategy": total_return_strat,
        "return_bh": total_return_bh,
        "n_long": int(n_long),
        "n_short": int(n_short),
        "n_flat": int(n_flat),
        "n_trades": int(np.count_nonzero(np.diff(positions))),
    }


def main():
    print("="*80)
    print("🎯 DIRECTIONAL CLASSIFICATION MODEL V2 - PROPER TRAIN/TEST SPLIT")
    print("="*80)
    print(f"📊 UP threshold: > +{THRESHOLD_UP*100:.1f}%")
    print(f"📊 DOWN threshold: < {THRESHOLD_DOWN*100:.1f}%")
    print(f"📊 NEUTRAL: between {THRESHOLD_DOWN*100:.1f}% and +{THRESHOLD_UP*100:.1f}%")
    print(f"💰 Transaction Cost: {COST_BPS} bps")
    print("="*80)
    
    # PROPER TRAIN/TEST SPLIT:
    # Train: 2018-2021 (includes 2018 correction, 2020 COVID crash, 2021 bull)
    # Test 1: 2022 bear market (OUT-OF-SAMPLE)
    # Test 2: 2024-2025 bull market (OUT-OF-SAMPLE)
    
    print("\n📊 Preparing training data (2018-2021)...")
    print("   Includes: 2018 correction, 2020 COVID crash, 2021 bull run")
    X_train, y_train, ret_train, feat_cols = prepare_classification_data(
        TRAIN_TICKERS, "2018-01-01", "2021-12-31"
    )
    
    if X_train is None:
        print("❌ Could not prepare training data")
        return
    
    print(f"   Total samples: {len(y_train)}")
    print(f"   Features: {len(feat_cols)}")
    
    # Train
    print("\n🔧 Training classifier with 2x DOWN class weight...")
    clf, scaler = train_classifier(X_train, y_train)
    
    # Feature importance
    if hasattr(clf, "feature_importances_"):
        importances = dict(zip(feat_cols, clf.feature_importances_))
        top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n   Top 10 Features:")
        for feat, imp in top_feats:
            print(f"      {feat}: {imp:.4f}")
    
    # Test on 2022 (bear market - OUT OF SAMPLE)
    print("\n" + "="*60)
    print("🐻 Testing on 2022 BEAR MARKET (OUT-OF-SAMPLE)")
    print("="*60)
    
    X_test_2022, y_test_2022, ret_test_2022, _ = prepare_classification_data(
        TRAIN_TICKERS, "2022-01-01", "2022-12-31"
    )
    
    if X_test_2022 is not None:
        results_2022 = evaluate_trading(
            clf, scaler, feat_cols, X_test_2022, y_test_2022, ret_test_2022, "2022"
        )
        
        print(f"\n📊 2022 Results (OUT-OF-SAMPLE):")
        print(f"   Classification Accuracy: {results_2022['accuracy']:.1f}%")
        print(f"   Direction Accuracy: {results_2022['direction_accuracy']:.1f}%")
        print(f"   Win Rate: {results_2022['win_rate']:.1f}%")
        print(f"   Strategy Sharpe: {results_2022['sharpe_strategy']:.3f}")
        print(f"   Buy & Hold Sharpe: {results_2022['sharpe_bh']:.3f}")
        print(f"   Strategy Return: {results_2022['return_strategy']:+.1f}%")
        print(f"   Buy & Hold Return: {results_2022['return_bh']:+.1f}%")
        print(f"   Long days: {results_2022['n_long']}")
        print(f"   Short days: {results_2022['n_short']}")
        print(f"   Flat days: {results_2022['n_flat']}")
        
        beat_bh_2022 = results_2022['sharpe_strategy'] > results_2022['sharpe_bh']
        print(f"\n   {'✅ BEATS' if beat_bh_2022 else '❌ LOSES TO'} Buy & Hold!")
    else:
        results_2022 = None
        beat_bh_2022 = False
    
    # Test on 2024-2025 (bull market - OUT OF SAMPLE)
    print("\n" + "="*60)
    print("📈 Testing on 2024-2025 BULL MARKET (OUT-OF-SAMPLE)")
    print("="*60)
    
    X_test_2025, y_test_2025, ret_test_2025, _ = prepare_classification_data(
        TRAIN_TICKERS, "2024-01-01", "2025-12-31"
    )
    
    if X_test_2025 is not None:
        results_2025 = evaluate_trading(
            clf, scaler, feat_cols, X_test_2025, y_test_2025, ret_test_2025, "2025"
        )
        
        print(f"\n📊 2024-2025 Results (OUT-OF-SAMPLE):")
        print(f"   Classification Accuracy: {results_2025['accuracy']:.1f}%")
        print(f"   Direction Accuracy: {results_2025['direction_accuracy']:.1f}%")
        print(f"   Win Rate: {results_2025['win_rate']:.1f}%")
        print(f"   Strategy Sharpe: {results_2025['sharpe_strategy']:.3f}")
        print(f"   Buy & Hold Sharpe: {results_2025['sharpe_bh']:.3f}")
        print(f"   Strategy Return: {results_2025['return_strategy']:+.1f}%")
        print(f"   Buy & Hold Return: {results_2025['return_bh']:+.1f}%")
        print(f"   Long days: {results_2025['n_long']}")
        print(f"   Short days: {results_2025['n_short']}")
        print(f"   Flat days: {results_2025['n_flat']}")
        
        beat_bh_2025 = results_2025['sharpe_strategy'] > results_2025['sharpe_bh']
        print(f"\n   {'✅ BEATS' if beat_bh_2025 else '❌ LOSES TO'} Buy & Hold!")
    else:
        results_2025 = None
        beat_bh_2025 = False
    
    # Summary
    print("\n" + "="*100)
    print("📊 FINAL COMPARISON (ALL OUT-OF-SAMPLE)")
    print("="*100)
    print(f"{'Period':<20} {'Acc':>8} {'Win%':>8} {'Sharpe':>10} {'B&H Shp':>10} "
          f"{'Long':>8} {'Short':>8} {'Flat':>8} {'Status':>10}")
    print("-"*100)
    
    if results_2022:
        print(f"{'2022 Bear':<20} {results_2022['accuracy']:>7.1f}% {results_2022['win_rate']:>7.1f}% "
              f"{results_2022['sharpe_strategy']:>10.3f} {results_2022['sharpe_bh']:>10.3f} "
              f"{results_2022['n_long']:>8} {results_2022['n_short']:>8} {results_2022['n_flat']:>8} "
              f"{'✅ BEAT' if beat_bh_2022 else '❌ LOSE':>10}")
    
    if results_2025:
        print(f"{'2024-2025 Bull':<20} {results_2025['accuracy']:>7.1f}% {results_2025['win_rate']:>7.1f}% "
              f"{results_2025['sharpe_strategy']:>10.3f} {results_2025['sharpe_bh']:>10.3f} "
              f"{results_2025['n_long']:>8} {results_2025['n_short']:>8} {results_2025['n_flat']:>8} "
              f"{'✅ BEAT' if beat_bh_2025 else '❌ LOSE':>10}")
    
    # Grade
    print("\n" + "="*80)
    print("🎯 MODEL ASSESSMENT")
    print("="*80)
    
    if beat_bh_2022 and beat_bh_2025:
        grade = "A - Works in BOTH bull and bear markets!"
        emoji = "🏆"
    elif beat_bh_2022:
        grade = "B+ - Good for bear markets, needs work for bulls"
        emoji = "📈"
    elif beat_bh_2025:
        grade = "B - Good for bull markets, needs work for bears"
        emoji = "🐂"
    else:
        grade = "C - Needs more work"
        emoji = "⚠️"
    
    print(f"\n{emoji} GRADE: {grade}")
    
    short_2022 = results_2022['n_short'] if results_2022 else 0
    short_2025 = results_2025['n_short'] if results_2025 else 0
    
    if short_2022 > short_2025:
        print("✅ Model learned regime awareness - shorts more in bear markets!")
    else:
        print("⚠️ Model not fully regime-aware")
    
    # Save model
    model_path = MODEL_DIR / "directional_classifier_v2.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "classifier": clf,
            "scaler": scaler,
            "feature_cols": feat_cols,
            "thresholds": {"up": THRESHOLD_UP, "down": THRESHOLD_DOWN},
            "trained_on": "2018-2021",
        }, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
