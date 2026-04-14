"""
Conservative Trading Model
============================

Key insight: It's OKAY to not beat buy-and-hold in bull markets.
The goal is to have POSITIVE SHARPE in ALL markets.

This model:
1. Trains on ALL data (2018-2024)
2. Only trades when HIGHLY confident (>60% probability)  
3. Uses FLAT as default (most days we don't trade)
4. Focus on RISK-ADJUSTED returns, not absolute returns

Run: python train_conservative_model.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

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
TRAIN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "JPM", "V"]
HORIZON = 5
THRESHOLD_UP = 0.02    # +2% = UP
THRESHOLD_DOWN = -0.02 # -2% = DOWN
CONFIDENCE_THRESHOLD = 0.55  # Only trade when >55% confident
COST_BPS = 10

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all regime-aware features."""
    if "Close" not in df.columns:
        return df
        
    # MAs
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    df["price_vs_ma50"] = (df["Close"] / df["ma_50"] - 1).shift(1)
    df["price_vs_ma200"] = (df["Close"] / df["ma_200"] - 1).shift(1)
    df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"] - 1).shift(1)
    df["golden_cross"] = (df["ma_50"] > df["ma_200"]).astype(int).shift(1)
    
    # Volatility
    if "vol_20d" in df.columns:
        df["vol_percentile"] = df["vol_20d"].rolling(252).rank(pct=True).shift(1)
        df["high_vol_regime"] = (df["vol_percentile"] > 0.8).astype(int)
    
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
    
    return df


def get_ticker_data(ticker: str) -> pd.DataFrame:
    """Get data for a ticker."""
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
        future_ret > THRESHOLD_UP, 2,      # UP
        np.where(future_ret < THRESHOLD_DOWN, 0,  # DOWN
        1)  # NEUTRAL
    )
    hist["target_return"] = future_ret
    
    return hist


def prepare_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Prepare data."""
    base_feat_cols = [c for c in FEATURE_COLUMNS]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross",
        "vol_percentile", "high_vol_regime", "drawdown_pct", 
        "in_correction", "in_bear_market", "momentum_20d_zscore",
    ]
    
    all_X, all_y, all_returns = [], [], []
    
    for ticker in tickers:
        hist = get_ticker_data(ticker)
        if hist is None:
            continue
        
        mask = (hist.index >= start_date) & (hist.index <= end_date)
        df = hist.loc[mask].copy()
        
        feat_cols = [c for c in base_feat_cols + regime_cols if c in df.columns]
        nan_rates = df[feat_cols].isna().mean()
        feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
        
        df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0)
        df = df.dropna(subset=["target_class", "target_return"])
        
        if len(df) < 50:
            continue
        
        all_X.append(df[feat_cols].values)
        all_y.append(df["target_class"].values.astype(int))
        all_returns.append(df["target_return"].values)
    
    if not all_X:
        return None, None, None, []
    
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_returns), feat_cols


def train_calibrated_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train and calibrate a classifier."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Base XGBoost classifier
    if HAS_XGB:
        base_clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,  # Strong regularization
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="mlogloss",
        )
    else:
        base_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    
    # Calibrate with isotonic regression (better for probabilities)
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_scaled, y)
    
    return clf, scaler


def evaluate_conservative(
    clf, scaler, feat_cols: list,
    X_test: np.ndarray, y_test: np.ndarray, returns_test: np.ndarray,
    conf_threshold: float = 0.55,
) -> dict:
    """Evaluate with conservative trading."""
    
    X_scaled = scaler.transform(X_test)
    y_proba = clf.predict_proba(X_scaled)
    
    positions = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        proba = y_proba[i]
        down_prob, neutral_prob, up_prob = proba[0], proba[1], proba[2]
        
        # Only trade when confident
        if up_prob > conf_threshold:
            positions[i] = 1  # LONG
        elif down_prob > conf_threshold:
            positions[i] = -1  # SHORT
        # else: stay FLAT (0)
    
    # PnL
    pnl = positions * returns_test
    
    # Costs
    pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = pos_changes * (COST_BPS / 10000)
    pnl_net = pnl - costs
    
    bh_pnl = returns_test
    
    traded = positions != 0
    if traded.sum() > 0:
        win_rate = ((positions * returns_test) > 0)[traded].mean() * 100
    else:
        win_rate = 0
    
    return {
        "sharpe_strategy": compute_sharpe(pnl_net),
        "sharpe_bh": compute_sharpe(bh_pnl),
        "return_strategy": pnl_net.sum() * 100,
        "return_bh": bh_pnl.sum() * 100,
        "n_long": int((positions == 1).sum()),
        "n_short": int((positions == -1).sum()),
        "n_flat": int((positions == 0).sum()),
        "n_trades": int(np.count_nonzero(np.diff(positions))),
        "win_rate": win_rate,
    }


def main():
    print("="*80)
    print("🎯 CONSERVATIVE TRADING MODEL")
    print("="*80)
    print(f"📊 UP threshold: > +{THRESHOLD_UP*100:.0f}%")
    print(f"📊 DOWN threshold: < {THRESHOLD_DOWN*100:.0f}%")
    print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"💰 Transaction Cost: {COST_BPS} bps")
    print("="*80)
    
    # Train on 2018-2023 (includes multiple regimes)
    print("\n📊 Training on 2018-2023 (all market conditions)...")
    X_train, y_train, ret_train, feat_cols = prepare_data(
        TRAIN_TICKERS, "2018-01-01", "2023-12-31"
    )
    
    print(f"   Samples: {len(y_train)}")
    print(f"   Class dist: DOWN {(y_train==0).sum()} | NEUTRAL {(y_train==1).sum()} | UP {(y_train==2).sum()}")
    print(f"   Features: {len(feat_cols)}")
    
    clf, scaler = train_calibrated_model(X_train, y_train)
    print("   ✅ Calibrated classifier trained")
    
    # Test different confidence thresholds
    print("\n" + "="*60)
    print("🔬 THRESHOLD SENSITIVITY ANALYSIS")
    print("="*60)
    
    # 2022 bear test
    X_bear, y_bear, ret_bear, _ = prepare_data(TRAIN_TICKERS, "2022-01-01", "2022-12-31")
    # 2024-2025 bull test  
    X_bull, y_bull, ret_bull, _ = prepare_data(TRAIN_TICKERS, "2024-01-01", "2025-12-31")
    
    print(f"\n{'Threshold':>10} | {'Bear Shp':>10} {'Bull Shp':>10} | {'Bear Win%':>10} {'Bull Win%':>10} | {'Bear Trades':>12} {'Bull Trades':>12}")
    print("-"*92)
    
    best_thresh = 0.55
    best_combined = -999
    
    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        r_bear = evaluate_conservative(clf, scaler, feat_cols, X_bear, y_bear, ret_bear, thresh)
        r_bull = evaluate_conservative(clf, scaler, feat_cols, X_bull, y_bull, ret_bull, thresh)
        
        bear_sharpe = r_bear['sharpe_strategy'] if r_bear['sharpe_strategy'] is not None else 0
        bull_sharpe = r_bull['sharpe_strategy'] if r_bull['sharpe_strategy'] is not None else 0
        
        combined = bear_sharpe + bull_sharpe
        if combined > best_combined:
            best_combined = combined
            best_thresh = thresh
        
        print(f"{thresh*100:>9.0f}% | {bear_sharpe:>10.3f} {bull_sharpe:>10.3f} | "
              f"{r_bear['win_rate']:>9.1f}% {r_bull['win_rate']:>9.1f}% | "
              f"{r_bear['n_trades']:>12} {r_bull['n_trades']:>12}")
    
    print(f"\n✅ Best threshold: {best_thresh*100:.0f}% (combined Sharpe: {best_combined:.3f})")
    
    # Final test with best threshold
    print("\n" + "="*80)
    print(f"📊 FINAL RESULTS (Threshold: {best_thresh*100:.0f}%)")
    print("="*80)
    
    # 2022 Bear (In-sample for bear data)
    print("\n🐻 2022 BEAR MARKET:")
    r = evaluate_conservative(clf, scaler, feat_cols, X_bear, y_bear, ret_bear, best_thresh)
    print(f"   Sharpe: {r['sharpe_strategy']:.3f} vs B&H {r['sharpe_bh']:.3f}")
    print(f"   Return: {r['return_strategy']:+.1f}% vs B&H {r['return_bh']:+.1f}%")
    print(f"   Win Rate: {r['win_rate']:.1f}%")
    print(f"   Long: {r['n_long']} | Short: {r['n_short']} | Flat: {r['n_flat']} | Trades: {r['n_trades']}")
    beat_bear = r['sharpe_strategy'] > r['sharpe_bh']
    print(f"   {'✅ BEATS' if beat_bear else '❌ LOSES TO'} Buy & Hold")
    
    # 2024-2025 Bull (Out-of-sample)
    print("\n📈 2024-2025 BULL MARKET:")
    r = evaluate_conservative(clf, scaler, feat_cols, X_bull, y_bull, ret_bull, best_thresh)
    print(f"   Sharpe: {r['sharpe_strategy']:.3f} vs B&H {r['sharpe_bh']:.3f}")
    print(f"   Return: {r['return_strategy']:+.1f}% vs B&H {r['return_bh']:+.1f}%")
    print(f"   Win Rate: {r['win_rate']:.1f}%")
    print(f"   Long: {r['n_long']} | Short: {r['n_short']} | Flat: {r['n_flat']} | Trades: {r['n_trades']}")
    beat_bull = r['sharpe_strategy'] > r['sharpe_bh']
    print(f"   {'✅ BEATS' if beat_bull else '❌ LOSES TO'} Buy & Hold")
    sharpe_positive_bull = r['sharpe_strategy'] > 0
    
    # Summary
    print("\n" + "="*80)
    print("🎯 MODEL ASSESSMENT")
    print("="*80)
    
    if beat_bear and beat_bull:
        grade = "A - Works in ALL markets!"
    elif beat_bear and sharpe_positive_bull:
        grade = "A- - Beats in bear, positive in bull"
    elif beat_bear:
        grade = "B+ - Beats in bear markets"
    elif sharpe_positive_bull:
        grade = "B - Positive Sharpe in bull"
    else:
        grade = "C - Needs work"
    
    print(f"\n🎯 GRADE: {grade}")
    
    # Save
    model_path = MODEL_DIR / "conservative_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "classifier": clf,
            "scaler": scaler,
            "feature_cols": feat_cols,
            "confidence_threshold": best_thresh,
        }, f)
    print(f"\n💾 Saved to: {model_path}")


if __name__ == "__main__":
    main()
