"""
All-Weather Trading Model with Shorting
========================================

Goal: Beat buy-and-hold in BOTH bull AND bear markets through intelligent shorting.

Strategy:
1. Use REGIME DETECTION to know when we're in bull vs bear
2. In BULL markets: Go LONG on high-confidence UP signals, FLAT otherwise
3. In BEAR markets: Go SHORT on high-confidence DOWN signals, FLAT otherwise
4. Key insight: Don't fight the trend - go WITH the regime

This creates an "all-weather" model that should work in any condition.

Run: python train_all_weather_model.py
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

# Confidence thresholds - KEY PARAMETERS
BULL_LONG_CONF = 0.40    # Lower threshold in bull (trend is our friend)
BULL_SHORT_CONF = 0.65   # Higher threshold to short in bull (against trend)
BEAR_LONG_CONF = 0.65    # Higher threshold to go long in bear (against trend)
BEAR_SHORT_CONF = 0.40   # Lower threshold to short in bear (trend is our friend)
NEUTRAL_CONF = 0.55      # Middle ground for neutral regime

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def detect_regime(df: pd.DataFrame, lookback: int = 60) -> str:
    """Detect market regime from price data."""
    if len(df) < 200:
        return "NEUTRAL"
    
    close = df["Close"].iloc[-lookback:]
    ma_50 = df["Close"].rolling(50).mean().iloc[-1]
    ma_200 = df["Close"].rolling(200).mean().iloc[-1]
    
    # Trend direction
    golden_cross = ma_50 > ma_200
    
    # Momentum
    ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
    
    # Drawdown
    rolling_max = df["Close"].rolling(252, min_periods=1).max().iloc[-1]
    current_dd = (close.iloc[-1] / rolling_max) - 1
    
    # Strong BEAR signals
    if current_dd < -0.15 or (not golden_cross and ret_20d < -0.05):
        return "BEAR"
    
    # Strong BULL signals
    if golden_cross and ret_20d > 0.02:
        return "BULL"
    
    return "NEUTRAL"


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
    
    # Trend strength
    df["trend_strength"] = abs(df["ma50_vs_ma200"])
    
    return df


def get_ticker_data(ticker: str) -> pd.DataFrame:
    """Get and prepare ticker data."""
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
    """Get valid feature columns."""
    base_feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    regime_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross", "death_cross",
        "vol_percentile", "high_vol_regime", "drawdown_pct", 
        "in_correction", "in_bear_market", "momentum_20d_zscore", "trend_strength",
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


def regime_aware_position(
    up_prob: float, 
    down_prob: float, 
    regime: str,
) -> int:
    """Determine position based on regime and probabilities."""
    
    if regime == "BULL":
        # In bull market: favor longs, be cautious with shorts
        if up_prob >= BULL_LONG_CONF and up_prob > down_prob:
            return 1  # LONG
        elif down_prob >= BULL_SHORT_CONF and down_prob > up_prob:
            return -1  # SHORT (only if very confident)
        return 0  # FLAT
        
    elif regime == "BEAR":
        # In bear market: favor shorts, be cautious with longs
        if down_prob >= BEAR_SHORT_CONF and down_prob > up_prob:
            return -1  # SHORT
        elif up_prob >= BEAR_LONG_CONF and up_prob > down_prob:
            return 1  # LONG (only if very confident)
        return 0  # FLAT
        
    else:  # NEUTRAL
        # Balanced approach
        if up_prob >= NEUTRAL_CONF and up_prob > down_prob:
            return 1
        elif down_prob >= NEUTRAL_CONF and down_prob > up_prob:
            return -1
        return 0


def walk_forward_test(tickers: list, start_date: str, end_date: str) -> dict:
    """Walk-forward test with regime-aware positioning."""
    
    print(f"\n📊 All-Weather Walk-Forward Test: {start_date} to {end_date}")
    print("="*80)
    
    # Load all data
    all_data = {}
    for ticker in tickers:
        df = get_ticker_data(ticker)
        if df is not None:
            all_data[ticker] = df
    
    sample_df = list(all_data.values())[0]
    feat_cols = get_feature_cols(sample_df)
    
    # Walk-forward folds (2-year train, 6-month test)
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
    
    print(f"   Folds: {len(folds)}")
    
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
        
        # Test with regime-aware positioning
        fold_pnl = []
        fold_bh_pnl = []
        fold_positions = []
        regime_counts = {"BULL": 0, "BEAR": 0, "NEUTRAL": 0}
        
        for ticker, df in all_data.items():
            mask = (df.index >= fold["test_start"]) & (df.index <= fold["test_end"])
            test_df = df.loc[mask].copy()
            
            available_cols = [c for c in feat_cols if c in test_df.columns]
            test_df[available_cols] = test_df[available_cols].ffill().bfill().fillna(0)
            test_df = test_df.dropna(subset=["target_return"])
            
            if len(test_df) < 5:
                continue
            
            # For each day, detect regime and make position decision
            for i in range(len(test_df)):
                # Use historical window for regime detection
                hist_end_idx = df.index.get_loc(test_df.index[i])
                hist_window = df.iloc[max(0, hist_end_idx-252):hist_end_idx+1]
                regime = detect_regime(hist_window)
                regime_counts[regime] += 1
                
                # Get prediction
                X_row = test_df[available_cols].iloc[[i]].values
                X_scaled = scaler.transform(X_row)
                proba = clf.predict_proba(X_scaled)[0]
                down_prob, neutral_prob, up_prob = proba[0], proba[1], proba[2]
                
                # Regime-aware position
                position = regime_aware_position(up_prob, down_prob, regime)
                
                ret = test_df["target_return"].iloc[i]
                pnl = position * ret
                
                fold_pnl.append(pnl)
                fold_bh_pnl.append(ret)
                fold_positions.append(position)
        
        if fold_pnl:
            fold_pnl = np.array(fold_pnl)
            fold_bh_pnl = np.array(fold_bh_pnl)
            fold_positions = np.array(fold_positions)
            
            # Apply costs
            pos_changes = np.abs(np.diff(np.concatenate([[0], fold_positions])))
            costs = pos_changes * (COST_BPS / 10000)
            fold_pnl_net = fold_pnl - costs
            
            sharpe = compute_sharpe(fold_pnl_net)
            sharpe_bh = compute_sharpe(fold_bh_pnl)
            ret = fold_pnl_net.sum() * 100
            ret_bh = fold_bh_pnl.sum() * 100
            n_long = (fold_positions == 1).sum()
            n_short = (fold_positions == -1).sum()
            n_flat = (fold_positions == 0).sum()
            
            traded = fold_positions != 0
            win_rate = ((fold_positions * fold_bh_pnl) > 0)[traded].mean() * 100 if traded.sum() > 0 else 0
            
            beat = sharpe > sharpe_bh if sharpe and sharpe_bh else False
            positive = sharpe > 0 if sharpe else False
            
            period = f"{fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}"
            status = "✅ BEAT" if beat else ("🟡 +VE" if positive else "❌ LOSE")
            
            print(f"\n📅 Fold {fold_idx+1}: {period}")
            print(f"   Regime mix: BULL={regime_counts['BULL']} BEAR={regime_counts['BEAR']} NEUT={regime_counts['NEUTRAL']}")
            print(f"   Sharpe: {sharpe:.3f} vs B&H {sharpe_bh:.3f} {status}")
            print(f"   Return: {ret:+.1f}% vs B&H {ret_bh:+.1f}%")
            print(f"   Long: {n_long} | Short: {n_short} | Flat: {n_flat} | Win%: {win_rate:.1f}%")
            
            fold_results.append({
                "period": period,
                "sharpe": sharpe if sharpe else 0,
                "sharpe_bh": sharpe_bh if sharpe_bh else 0,
                "return": ret,
                "return_bh": ret_bh,
                "n_long": n_long,
                "n_short": n_short,
                "n_flat": n_flat,
                "win_rate": win_rate,
                "beat": beat,
                "positive": positive,
                "regimes": regime_counts.copy(),
            })
        
        # Reset regime counts for next fold
        regime_counts = {"BULL": 0, "BEAR": 0, "NEUTRAL": 0}
    
    # Summary
    if fold_results:
        df_results = pd.DataFrame(fold_results)
        
        print("\n" + "="*100)
        print("📊 ALL-WEATHER MODEL SUMMARY")
        print("="*100)
        
        avg_sharpe = df_results['sharpe'].mean()
        avg_sharpe_bh = df_results['sharpe_bh'].mean()
        total_ret = df_results['return'].sum()
        total_ret_bh = df_results['return_bh'].sum()
        beats = df_results['beat'].sum()
        positives = df_results['positive'].sum()
        total_shorts = df_results['n_short'].sum()
        
        print(f"\n   Average Sharpe: {avg_sharpe:.3f} (B&H: {avg_sharpe_bh:.3f})")
        print(f"   Total Return: {total_ret:.1f}% (B&H: {total_ret_bh:.1f}%)")
        print(f"   Beat B&H: {beats}/{len(df_results)} folds ({beats/len(df_results)*100:.0f}%)")
        print(f"   Positive: {positives}/{len(df_results)} folds ({positives/len(df_results)*100:.0f}%)")
        print(f"   Total Short Positions: {total_shorts}")
        
        # Compare to previous conservative model
        print("\n" + "="*80)
        print("📈 COMPARISON TO CONSERVATIVE MODEL")
        print("="*80)
        print(f"   Conservative: Avg Sharpe 0.479, Beat 3/12 (25%), Positive 9/12 (75%)")
        print(f"   All-Weather:  Avg Sharpe {avg_sharpe:.3f}, Beat {beats}/{len(df_results)} ({beats/len(df_results)*100:.0f}%), Positive {positives}/{len(df_results)} ({positives/len(df_results)*100:.0f}%)")
        
        improvement = avg_sharpe - 0.479
        print(f"\n   Sharpe improvement: {improvement:+.3f}")
        
        if avg_sharpe > 0.479 and beats > 3:
            print("   🎯 ALL-WEATHER MODEL IS BETTER!")
        elif avg_sharpe > 0.479:
            print("   🟡 Better Sharpe but fewer beats")
        else:
            print("   ⚠️ Conservative model is still better")
        
        return {
            "fold_results": fold_results,
            "avg_sharpe": avg_sharpe,
            "avg_sharpe_bh": avg_sharpe_bh,
            "beat_rate": beats / len(df_results),
            "positive_rate": positives / len(df_results),
            "total_shorts": total_shorts,
        }
    
    return None


def main():
    print("="*80)
    print("🌤️ ALL-WEATHER TRADING MODEL")
    print("="*80)
    print("Strategy: Regime-aware positioning with intelligent shorting")
    print(f"   BULL: Long @ {BULL_LONG_CONF*100:.0f}% conf, Short @ {BULL_SHORT_CONF*100:.0f}% conf")
    print(f"   BEAR: Short @ {BEAR_SHORT_CONF*100:.0f}% conf, Long @ {BEAR_LONG_CONF*100:.0f}% conf")
    print(f"   NEUTRAL: Both @ {NEUTRAL_CONF*100:.0f}% conf")
    print("="*80)
    
    results = walk_forward_test(TICKERS, "2018-01-01", "2025-12-31")
    
    if results:
        print("\n" + "="*80)
        print("✅ ALL-WEATHER TEST COMPLETE")
        print("="*80)


if __name__ == "__main__":
    main()
