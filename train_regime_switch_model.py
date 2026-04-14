"""
Regime-Switching Classification Model
======================================

Instead of one model, we train:
1. BULL model - trained on 2019-2021 bull data (biased toward longs)
2. BEAR model - trained on 2020 crash + 2022 bear (biased toward shorts)

Then we detect the regime and use the appropriate model.

Run: python train_regime_switch_model.py
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

from sklearn.preprocessing import StandardScaler

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
THRESHOLD_UP = 0.015
THRESHOLD_DOWN = -0.015
COST_BPS = 10

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime-aware features."""
    if "Close" not in df.columns:
        return df
        
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_200"] = df["Close"].rolling(200).mean()
    df["price_vs_ma50"] = (df["Close"] / df["ma_50"] - 1).shift(1)
    df["price_vs_ma200"] = (df["Close"] / df["ma_200"] - 1).shift(1)
    df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"] - 1).shift(1)
    df["golden_cross"] = (df["ma_50"] > df["ma_200"]).astype(int).shift(1)
    df["death_cross"] = (df["ma_50"] < df["ma_200"]).astype(int).shift(1)
    df["days_below_ma200"] = (df["Close"] < df["ma_200"]).rolling(60).sum().shift(1)
    df["days_below_ma50"] = (df["Close"] < df["ma_50"]).rolling(20).sum().shift(1)
    
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


def detect_regime(df: pd.DataFrame) -> str:
    """Detect if we're in BULL, BEAR, or UNCERTAIN regime."""
    if len(df) < 50:
        return "UNCERTAIN"
    
    latest = df.iloc[-1]
    
    # Strong signals
    death_cross = latest.get("death_cross", 0) == 1
    golden_cross = latest.get("golden_cross", 0) == 1
    in_bear = latest.get("in_bear_market", 0) == 1
    in_correction = latest.get("in_correction", 0) == 1
    
    price_vs_ma200 = latest.get("price_vs_ma200", 0)
    drawdown = latest.get("drawdown_pct", 0)
    
    # BEAR regime
    if in_bear or (death_cross and drawdown < -0.15):
        return "BEAR"
    
    # BULL regime  
    if golden_cross and price_vs_ma200 > 0.05:
        return "BULL"
    
    # Mild correction
    if in_correction:
        return "BEAR"
    
    # Default based on trend
    if price_vs_ma200 > 0:
        return "BULL"
    else:
        return "BEAR"


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
    
    future_ret = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    hist["target_class"] = np.where(
        future_ret > THRESHOLD_UP, 2,      # UP
        np.where(future_ret < THRESHOLD_DOWN, 0,  # DOWN
        1)  # NEUTRAL
    )
    hist["target_return"] = future_ret
    
    return hist


def prepare_data(tickers: list, start_date: str, end_date: str) -> tuple:
    """Prepare classification data."""
    
    base_feat_cols = [c for c in FEATURE_COLUMNS]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross", "death_cross",
        "days_below_ma200", "days_below_ma50", "vol_percentile", "high_vol_regime",
        "drawdown_pct", "in_correction", "in_bear_market", "momentum_20d_zscore",
    ]
    
    all_X = []
    all_y = []
    all_returns = []
    
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


def train_model(X: np.ndarray, y: np.ndarray, bias: str = "neutral") -> tuple:
    """Train a classifier with optional bias."""
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    class_counts = np.bincount(y)
    total = len(y)
    
    # Adjust weights based on desired bias
    if bias == "bull":
        # Favor UP predictions
        weights = {
            0: 0.8 * total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: 1.0 * total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,
            2: 1.5 * total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,
        }
    elif bias == "bear":
        # Favor DOWN predictions
        weights = {
            0: 2.0 * total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: 1.0 * total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,
            2: 0.8 * total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,
        }
    else:
        weights = {
            0: total / (3 * class_counts[0]) if class_counts[0] > 0 else 1.0,
            1: total / (3 * class_counts[1]) if class_counts[1] > 0 else 1.0,
            2: total / (3 * class_counts[2]) if class_counts[2] > 0 else 1.0,
        }
    
    sample_weights = np.array([weights[c] for c in y])
    
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf.fit(X_scaled, y, sample_weight=sample_weights)
    
    return clf, scaler


def evaluate_regime_switch(
    bull_clf, bull_scaler,
    bear_clf, bear_scaler,
    feat_cols: list,
    tickers: list,
    start_date: str,
    end_date: str,
) -> dict:
    """Evaluate with regime switching."""
    
    all_pnl = []
    all_bh_pnl = []
    all_positions = []
    regimes_used = {"BULL": 0, "BEAR": 0, "UNCERTAIN": 0}
    
    for ticker in tickers:
        hist = get_ticker_data(ticker)
        if hist is None:
            continue
        
        mask = (hist.index >= start_date) & (hist.index <= end_date)
        df = hist.loc[mask].copy()
        
        available_cols = [c for c in feat_cols if c in df.columns]
        df[available_cols] = df[available_cols].ffill().bfill().fillna(0)
        df = df.dropna(subset=["target_return"])
        
        if len(df) < 50:
            continue
        
        positions = []
        
        for i in range(len(df)):
            # Use rolling window to detect regime
            window = df.iloc[max(0, i-60):i+1]
            regime = detect_regime(window)
            regimes_used[regime] += 1
            
            # Select model
            if regime == "BULL":
                clf, scaler = bull_clf, bull_scaler
            elif regime == "BEAR":
                clf, scaler = bear_clf, bear_scaler
            else:  # UNCERTAIN - use bull model (default to long bias in uncertainty)
                clf, scaler = bull_clf, bull_scaler
            
            # Get features for this row
            row = df.iloc[[i]]
            X = row[available_cols].values
            X_scaled = scaler.transform(X)
            
            pred = clf.predict(X_scaled)[0]
            proba = clf.predict_proba(X_scaled)[0]
            conf = proba[pred]
            
            # Position based on prediction and confidence
            if pred == 2 and conf > 0.45:  # UP
                positions.append(1)
            elif pred == 0 and conf > 0.45:  # DOWN
                positions.append(-1)
            else:
                positions.append(0)
        
        positions = np.array(positions)
        returns = df["target_return"].values
        
        # PnL with costs
        pnl = positions * returns
        pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        costs = pos_changes * (COST_BPS / 10000)
        pnl_net = pnl - costs
        
        all_pnl.append(pnl_net)
        all_bh_pnl.append(returns)
        all_positions.append(positions)
    
    # Aggregate
    pnl = np.concatenate(all_pnl)
    bh_pnl = np.concatenate(all_bh_pnl)
    positions = np.concatenate(all_positions)
    
    return {
        "sharpe_strategy": compute_sharpe(pnl),
        "sharpe_bh": compute_sharpe(bh_pnl),
        "return_strategy": pnl.sum() * 100,
        "return_bh": bh_pnl.sum() * 100,
        "n_long": int((positions == 1).sum()),
        "n_short": int((positions == -1).sum()),
        "n_flat": int((positions == 0).sum()),
        "regimes_used": regimes_used,
        "win_rate": ((positions * np.concatenate(all_bh_pnl)) > 0)[positions != 0].mean() * 100 if (positions != 0).sum() > 0 else 0,
    }


def main():
    print("="*80)
    print("🔄 REGIME-SWITCHING CLASSIFICATION MODEL")
    print("="*80)
    print("Strategy: Use BULL model in bull markets, BEAR model in bear markets")
    print("="*80)
    
    # Train BULL model on 2019, 2021 (strong bull years)
    print("\n📈 Training BULL model (2019, 2021)...")
    X_bull, y_bull, _, feat_cols = prepare_data(TRAIN_TICKERS, "2019-01-01", "2019-12-31")
    X_bull2, y_bull2, _, _ = prepare_data(TRAIN_TICKERS, "2021-01-01", "2021-12-31")
    if X_bull2 is not None:
        X_bull = np.vstack([X_bull, X_bull2]) if X_bull is not None else X_bull2
        y_bull = np.concatenate([y_bull, y_bull2]) if y_bull is not None else y_bull2
    
    print(f"   Samples: {len(y_bull)}")
    print(f"   Class dist: DOWN {(y_bull==0).sum()} | NEUTRAL {(y_bull==1).sum()} | UP {(y_bull==2).sum()}")
    bull_clf, bull_scaler = train_model(X_bull, y_bull, bias="bull")
    
    # Train BEAR model on 2020 crash + 2022 bear
    print("\n🐻 Training BEAR model (2020 crash, 2022)...")
    X_bear1, y_bear1, _, _ = prepare_data(TRAIN_TICKERS, "2020-01-01", "2020-06-30")  # COVID crash
    X_bear2, y_bear2, _, _ = prepare_data(TRAIN_TICKERS, "2022-01-01", "2022-12-31")  # 2022 bear
    X_bear = np.vstack([X_bear1, X_bear2]) if X_bear1 is not None and X_bear2 is not None else X_bear2
    y_bear = np.concatenate([y_bear1, y_bear2]) if y_bear1 is not None and y_bear2 is not None else y_bear2
    
    print(f"   Samples: {len(y_bear)}")
    print(f"   Class dist: DOWN {(y_bear==0).sum()} | NEUTRAL {(y_bear==1).sum()} | UP {(y_bear==2).sum()}")
    bear_clf, bear_scaler = train_model(X_bear, y_bear, bias="bear")
    
    # Test on 2018 (correction year - OUT OF SAMPLE)
    print("\n" + "="*60)
    print("📉 Testing on 2018 CORRECTION (OUT-OF-SAMPLE)")
    print("="*60)
    
    results_2018 = evaluate_regime_switch(
        bull_clf, bull_scaler, bear_clf, bear_scaler, feat_cols,
        TRAIN_TICKERS, "2018-01-01", "2018-12-31"
    )
    
    print(f"   Sharpe: {results_2018['sharpe_strategy']:.3f} vs B&H {results_2018['sharpe_bh']:.3f}")
    print(f"   Return: {results_2018['return_strategy']:+.1f}% vs B&H {results_2018['return_bh']:+.1f}%")
    print(f"   Long: {results_2018['n_long']} | Short: {results_2018['n_short']} | Flat: {results_2018['n_flat']}")
    print(f"   Regimes: {results_2018['regimes_used']}")
    beat_2018 = results_2018['sharpe_strategy'] > results_2018['sharpe_bh']
    print(f"   {'✅ BEATS' if beat_2018 else '❌ LOSES TO'} Buy & Hold")
    
    # Test on 2023 (recovery year - OUT OF SAMPLE for bear model)
    print("\n" + "="*60)
    print("📈 Testing on 2023 RECOVERY (OUT-OF-SAMPLE)")
    print("="*60)
    
    results_2023 = evaluate_regime_switch(
        bull_clf, bull_scaler, bear_clf, bear_scaler, feat_cols,
        TRAIN_TICKERS, "2023-01-01", "2023-12-31"
    )
    
    print(f"   Sharpe: {results_2023['sharpe_strategy']:.3f} vs B&H {results_2023['sharpe_bh']:.3f}")
    print(f"   Return: {results_2023['return_strategy']:+.1f}% vs B&H {results_2023['return_bh']:+.1f}%")
    print(f"   Long: {results_2023['n_long']} | Short: {results_2023['n_short']} | Flat: {results_2023['n_flat']}")
    print(f"   Regimes: {results_2023['regimes_used']}")
    beat_2023 = results_2023['sharpe_strategy'] > results_2023['sharpe_bh']
    print(f"   {'✅ BEATS' if beat_2023 else '❌ LOSES TO'} Buy & Hold")
    
    # Test on 2024-2025 (current bull - OUT OF SAMPLE)
    print("\n" + "="*60)
    print("🚀 Testing on 2024-2025 BULL (OUT-OF-SAMPLE)")
    print("="*60)
    
    results_2025 = evaluate_regime_switch(
        bull_clf, bull_scaler, bear_clf, bear_scaler, feat_cols,
        TRAIN_TICKERS, "2024-01-01", "2025-12-31"
    )
    
    print(f"   Sharpe: {results_2025['sharpe_strategy']:.3f} vs B&H {results_2025['sharpe_bh']:.3f}")
    print(f"   Return: {results_2025['return_strategy']:+.1f}% vs B&H {results_2025['return_bh']:+.1f}%")
    print(f"   Long: {results_2025['n_long']} | Short: {results_2025['n_short']} | Flat: {results_2025['n_flat']}")
    print(f"   Regimes: {results_2025['regimes_used']}")
    beat_2025 = results_2025['sharpe_strategy'] > results_2025['sharpe_bh']
    print(f"   {'✅ BEATS' if beat_2025 else '❌ LOSES TO'} Buy & Hold")
    
    # Summary
    print("\n" + "="*100)
    print("📊 FINAL SUMMARY")
    print("="*100)
    print(f"{'Period':<20} {'Sharpe':>10} {'B&H Shp':>10} {'Return':>12} {'B&H Ret':>12} {'Win%':>8} {'Status':>10}")
    print("-"*84)
    print(f"{'2018 Correction':<20} {results_2018['sharpe_strategy']:>10.3f} {results_2018['sharpe_bh']:>10.3f} "
          f"{results_2018['return_strategy']:>11.1f}% {results_2018['return_bh']:>11.1f}% "
          f"{results_2018['win_rate']:>7.1f}% {'✅' if beat_2018 else '❌':>10}")
    print(f"{'2023 Recovery':<20} {results_2023['sharpe_strategy']:>10.3f} {results_2023['sharpe_bh']:>10.3f} "
          f"{results_2023['return_strategy']:>11.1f}% {results_2023['return_bh']:>11.1f}% "
          f"{results_2023['win_rate']:>7.1f}% {'✅' if beat_2023 else '❌':>10}")
    print(f"{'2024-2025 Bull':<20} {results_2025['sharpe_strategy']:>10.3f} {results_2025['sharpe_bh']:>10.3f} "
          f"{results_2025['return_strategy']:>11.1f}% {results_2025['return_bh']:>11.1f}% "
          f"{results_2025['win_rate']:>7.1f}% {'✅' if beat_2025 else '❌':>10}")
    
    wins = sum([beat_2018, beat_2023, beat_2025])
    print(f"\n🎯 Beat B&H in {wins}/3 periods")
    
    if wins >= 2:
        print("✅ Regime-switching model is working!")
    else:
        print("⚠️ Model needs more tuning")
    
    # Save
    model_path = MODEL_DIR / "regime_switch_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "bull_clf": bull_clf,
            "bull_scaler": bull_scaler,
            "bear_clf": bear_clf,
            "bear_scaler": bear_scaler,
            "feature_cols": feat_cols,
        }, f)
    print(f"\n💾 Saved to: {model_path}")


if __name__ == "__main__":
    main()
