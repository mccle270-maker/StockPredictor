"""
Adaptive Trading Model - Multiple Configurations
=================================================

This creates three modes:
1. CONSERVATIVE (default): Capital preservation, mostly flat
2. BALANCED: Moderate longs, careful shorts 
3. AGGRESSIVE: More frequent trading, seeks higher returns

All use the same base classifier, but with different position thresholds.

Key insight from previous tests:
- Conservative model: Sharpe 0.479, 9/12 positive (75%)
- Shorting hurt in regime transitions (2023 H1)
- Best approach: Be conservative with shorts, aggressive only when highly confident

Run: python train_adaptive_model.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

from src.core.features import build_all_features, add_gbm_features
from src.core.metrics import compute_sharpe
from src.data.market import get_price_history
from src.config import FEATURE_COLUMNS

# Config
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "JPM", "V"]
HORIZON = 5
THRESHOLD_UP = 0.015
THRESHOLD_DOWN = -0.015
COST_BPS = 10

# Three trading modes
MODES = {
    "conservative": {
        "long_conf": 0.45,      # Need 45% UP prob to go long
        "short_conf": 0.70,     # Need 70% DOWN prob to short (rarely short)
        "description": "Capital preservation - rarely trades",
    },
    "balanced": {
        "long_conf": 0.42,      # Slightly lower for longs
        "short_conf": 0.55,     # More willing to short
        "description": "Balanced risk/reward",
    },
    "aggressive": {
        "long_conf": 0.38,      # More frequent longs
        "short_conf": 0.45,     # More frequent shorts
        "description": "Seeks higher returns - more risk",
    },
}

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime features."""
    if "Close" not in df.columns:
        return df
    
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    df["price_vs_ma50"] = (df["Close"] / df["ma_50"] - 1).shift(1)
    df["price_vs_ma200"] = (df["Close"] / df["ma_200"] - 1).shift(1)
    df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"] - 1).shift(1)
    df["golden_cross"] = (df["ma_50"] > df["ma_200"]).astype(int).shift(1)
    df["death_cross"] = (df["ma_50"] < df["ma_200"]).astype(int).shift(1)
    
    if "vol_20d" in df.columns:
        df["vol_percentile"] = df["vol_20d"].rolling(252).rank(pct=True).shift(1)
        df["high_vol_regime"] = (df["vol_percentile"] > 0.8).astype(int)
    
    rolling_max = df["Close"].rolling(252, min_periods=1).max()
    df["drawdown_pct"] = ((df["Close"] / rolling_max) - 1).shift(1)
    df["in_correction"] = (df["drawdown_pct"] < -0.10).astype(int)
    df["in_bear_market"] = (df["drawdown_pct"] < -0.20).astype(int)
    
    ret_20d = df["Close"].pct_change(20)
    df["momentum_20d_zscore"] = (
        (ret_20d - ret_20d.rolling(252).mean()) / 
        ret_20d.rolling(252).std()
    ).shift(1)
    
    return df


def get_ticker_data(ticker: str) -> pd.DataFrame:
    """Get and prepare data."""
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
    
    future_ret = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    hist["target_class"] = np.where(
        future_ret > THRESHOLD_UP, 2,
        np.where(future_ret < THRESHOLD_DOWN, 0, 1)
    )
    hist["target_return"] = future_ret
    
    return hist


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get valid features."""
    base_feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross", "death_cross",
        "vol_percentile", "high_vol_regime", "drawdown_pct", 
        "in_correction", "in_bear_market", "momentum_20d_zscore",
    ]
    feat_cols = [c for c in base_feat_cols + regime_cols if c in df.columns]
    nan_rates = df[feat_cols].isna().mean()
    return [c for c in feat_cols if nan_rates[c] < 0.3]


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train calibrated classifier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    class_counts = np.bincount(y)
    total = len(y)
    weights = {
        0: 1.5 * total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,
        1: 1.0 * total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,
        2: 1.0 * total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,
    }
    sample_weights = np.array([weights[c] for c in y])
    
    if HAS_XGB:
        base_clf = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42,
            eval_metric="mlogloss",
        )
        base_clf.fit(X_scaled, y, sample_weight=sample_weights)
    else:
        base_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        base_clf.fit(X_scaled, y, sample_weight=sample_weights)
    
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_scaled, y)
    
    return clf, scaler


def get_position(up_prob: float, down_prob: float, mode: str) -> int:
    """Get position based on mode thresholds."""
    config = MODES[mode]
    long_conf = config["long_conf"]
    short_conf = config["short_conf"]
    
    if up_prob >= long_conf and up_prob > down_prob:
        return 1  # LONG
    elif down_prob >= short_conf and down_prob > up_prob:
        return -1  # SHORT
    return 0  # FLAT


def walk_forward_test(tickers: list, start_date: str, end_date: str, mode: str) -> dict:
    """Run walk-forward test for a specific mode."""
    
    # Load all data
    all_data = {}
    for ticker in tickers:
        df = get_ticker_data(ticker)
        if df is not None:
            all_data[ticker] = df
    
    sample_df = list(all_data.values())[0]
    feat_cols = get_feature_cols(sample_df)
    
    # Walk-forward folds
    train_months = 24
    test_months = 6
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    folds = []
    current = start + pd.DateOffset(months=train_months)
    
    while current < end:
        train_start = current - pd.DateOffset(months=train_months)
        train_end = current - pd.DateOffset(days=1)
        test_start = current
        test_end = min(current + pd.DateOffset(months=test_months) - pd.DateOffset(days=1), end)
        
        folds.append({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })
        current += pd.DateOffset(months=test_months)
    
    fold_results = []
    
    for fold_idx, fold in enumerate(folds):
        # Prepare training data
        X_train_list, y_train_list = [], []
        
        for ticker, df in all_data.items():
            mask = (df.index >= fold["train_start"]) & (df.index <= fold["train_end"])
            train_df = df.loc[mask].copy()
            
            available_cols = [c for c in feat_cols if c in train_df.columns]
            train_df[available_cols] = train_df[available_cols].ffill().bfill().fillna(0)
            train_df = train_df.dropna(subset=["target_class", "target_return"])
            
            if len(train_df) > 30:
                X_train_list.append(train_df[available_cols].values)
                y_train_list.append(train_df["target_class"].values.astype(int))
        
        if not X_train_list:
            continue
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        clf, scaler = train_model(X_train, y_train)
        
        # Test
        fold_pnl = []
        fold_bh_pnl = []
        fold_positions = []
        
        for ticker, df in all_data.items():
            mask = (df.index >= fold["test_start"]) & (df.index <= fold["test_end"])
            test_df = df.loc[mask].copy()
            
            available_cols = [c for c in feat_cols if c in test_df.columns]
            test_df[available_cols] = test_df[available_cols].ffill().bfill().fillna(0)
            test_df = test_df.dropna(subset=["target_return"])
            
            if len(test_df) < 5:
                continue
            
            X_test = test_df[available_cols].values
            X_scaled = scaler.transform(X_test)
            y_proba = clf.predict_proba(X_scaled)
            
            for i in range(len(X_test)):
                proba = y_proba[i]
                down_prob, neutral_prob, up_prob = proba[0], proba[1], proba[2]
                position = get_position(up_prob, down_prob, mode)
                
                ret = test_df["target_return"].iloc[i]
                pnl = position * ret
                
                fold_pnl.append(pnl)
                fold_bh_pnl.append(ret)
                fold_positions.append(position)
        
        if fold_pnl:
            fold_pnl = np.array(fold_pnl)
            fold_bh_pnl = np.array(fold_bh_pnl)
            fold_positions = np.array(fold_positions)
            
            # Costs
            pos_changes = np.abs(np.diff(np.concatenate([[0], fold_positions])))
            costs = pos_changes * (COST_BPS / 10000)
            fold_pnl_net = fold_pnl - costs
            
            sharpe = compute_sharpe(fold_pnl_net)
            sharpe_bh = compute_sharpe(fold_bh_pnl)
            
            beat = sharpe > sharpe_bh if sharpe and sharpe_bh else False
            positive = sharpe > 0 if sharpe else False
            
            fold_results.append({
                "sharpe": sharpe if sharpe else 0,
                "sharpe_bh": sharpe_bh if sharpe_bh else 0,
                "return": fold_pnl_net.sum() * 100,
                "return_bh": fold_bh_pnl.sum() * 100,
                "n_long": (fold_positions == 1).sum(),
                "n_short": (fold_positions == -1).sum(),
                "n_flat": (fold_positions == 0).sum(),
                "beat": beat,
                "positive": positive,
            })
    
    if fold_results:
        df_results = pd.DataFrame(fold_results)
        return {
            "avg_sharpe": df_results['sharpe'].mean(),
            "avg_sharpe_bh": df_results['sharpe_bh'].mean(),
            "total_return": df_results['return'].sum(),
            "beat_rate": df_results['beat'].mean(),
            "positive_rate": df_results['positive'].mean(),
            "total_long": df_results['n_long'].sum(),
            "total_short": df_results['n_short'].sum(),
            "total_flat": df_results['n_flat'].sum(),
            "fold_results": fold_results,
        }
    
    return None


def main():
    print("="*80)
    print("🎛️ ADAPTIVE TRADING MODEL - THREE MODES")
    print("="*80)
    
    for mode, config in MODES.items():
        print(f"\n{mode.upper()}: {config['description']}")
        print(f"   Long @ {config['long_conf']*100:.0f}% conf, Short @ {config['short_conf']*100:.0f}% conf")
    
    print("\n" + "="*80)
    print("Running walk-forward tests for each mode...")
    print("="*80)
    
    results = {}
    
    for mode in MODES.keys():
        print(f"\n📊 Testing {mode.upper()} mode...")
        result = walk_forward_test(TICKERS, "2018-01-01", "2025-12-31", mode)
        results[mode] = result
        
        if result:
            print(f"   Avg Sharpe: {result['avg_sharpe']:.3f}")
            print(f"   Beat B&H: {result['beat_rate']*100:.0f}%")
            print(f"   Positive: {result['positive_rate']*100:.0f}%")
            print(f"   Long: {result['total_long']} | Short: {result['total_short']} | Flat: {result['total_flat']}")
    
    # Summary comparison
    print("\n" + "="*100)
    print("📊 MODE COMPARISON")
    print("="*100)
    print(f"{'Mode':<15} {'Avg Sharpe':>12} {'B&H Sharpe':>12} {'Beat %':>10} {'Positive %':>12} {'Long':>8} {'Short':>8} {'Flat':>8}")
    print("-"*95)
    
    for mode, result in results.items():
        if result:
            print(f"{mode.upper():<15} {result['avg_sharpe']:>12.3f} {result['avg_sharpe_bh']:>12.3f} "
                  f"{result['beat_rate']*100:>9.0f}% {result['positive_rate']*100:>11.0f}% "
                  f"{result['total_long']:>8} {result['total_short']:>8} {result['total_flat']:>8}")
    
    # Find best mode
    best_mode = max(results.keys(), key=lambda m: results[m]['avg_sharpe'] if results[m] else -999)
    best_result = results[best_mode]
    
    print("\n" + "="*80)
    print("🏆 RESULTS")
    print("="*80)
    print(f"\n   Best Mode: {best_mode.upper()}")
    print(f"   Avg Sharpe: {best_result['avg_sharpe']:.3f}")
    print(f"   Beat B&H: {best_result['beat_rate']*100:.0f}% of folds")
    print(f"   Positive Sharpe: {best_result['positive_rate']*100:.0f}% of folds")
    
    # Save all modes
    model_path = MODEL_DIR / "adaptive_model_config.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "modes": MODES,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "fold_results"} for k, v in results.items() if v},
            "best_mode": best_mode,
        }, f)
    print(f"\n💾 Config saved to: {model_path}")
    
    print("\n" + "="*80)
    print("✅ ADAPTIVE MODEL COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
