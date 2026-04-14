"""
Comprehensive Backtest - Production Model Validation
=====================================================

Run a rigorous backtest with:
1. Walk-forward validation (retrain every 6 months)
2. 10 tickers across different sectors
3. Full period 2018-2025
4. Transaction costs (10 bps)
5. Realistic position sizing

This will determine if the production model is truly robust.

Run: python run_comprehensive_backtest.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta

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
TEST_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "JPM", "V"]
HORIZON = 5
THRESHOLD_UP = 0.015
THRESHOLD_DOWN = -0.015
LONG_CONF = 0.45
SHORT_CONF = 0.50
COST_BPS = 10
TRAIN_WINDOW_MONTHS = 24  # 2 years of training
TEST_WINDOW_MONTHS = 6    # 6 months per fold


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
        "in_correction", "in_bear_market", "momentum_20d_zscore",
    ]
    feat_cols = [c for c in base_feat_cols + regime_cols if c in df.columns]
    nan_rates = df[feat_cols].isna().mean()
    return [c for c in feat_cols if nan_rates[c] < 0.3]


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train classifier."""
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


def walk_forward_backtest(tickers: list, start_date: str, end_date: str) -> dict:
    """Run walk-forward validation."""
    
    print(f"\n📊 Walk-Forward Backtest: {start_date} to {end_date}")
    print(f"   Train window: {TRAIN_WINDOW_MONTHS} months | Test window: {TEST_WINDOW_MONTHS} months")
    print("="*80)
    
    # Load all data
    all_data = {}
    for ticker in tickers:
        df = get_ticker_data(ticker)
        if df is not None:
            all_data[ticker] = df
            print(f"   ✓ {ticker}: {len(df)} rows")
    
    if not all_data:
        print("   ❌ No data loaded")
        return None
    
    # Get common feature columns
    sample_df = list(all_data.values())[0]
    feat_cols = get_feature_cols(sample_df)
    print(f"\n   Features: {len(feat_cols)}")
    
    # Generate fold dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    folds = []
    current = start + pd.DateOffset(months=TRAIN_WINDOW_MONTHS)
    
    while current < end:
        train_start = current - pd.DateOffset(months=TRAIN_WINDOW_MONTHS)
        train_end = current - pd.DateOffset(days=1)
        test_start = current
        test_end = min(current + pd.DateOffset(months=TEST_WINDOW_MONTHS) - pd.DateOffset(days=1), end)
        
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        
        current += pd.DateOffset(months=TEST_WINDOW_MONTHS)
    
    print(f"\n   Folds: {len(folds)}")
    
    # Run walk-forward
    all_results = []
    fold_summaries = []
    
    for fold_idx, fold in enumerate(folds):
        print(f"\n📅 Fold {fold_idx + 1}/{len(folds)}: Test {fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}")
        
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
            print(f"   ⚠️ Skipping fold - no training data")
            continue
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        print(f"   Training on {len(y_train)} samples...")
        clf, scaler = train_model(X_train, y_train)
        
        # Test on each ticker
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
            
            returns = test_df["target_return"].values
            positions = np.zeros(len(X_test))
            
            for i in range(len(X_test)):
                proba = y_proba[i]
                down_prob, neutral_prob, up_prob = proba[0], proba[1], proba[2]
                
                if up_prob >= LONG_CONF and up_prob > down_prob:
                    positions[i] = 1
                elif down_prob >= SHORT_CONF and down_prob > up_prob:
                    positions[i] = -1
            
            # PnL with costs
            pnl = positions * returns
            pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
            costs = pos_changes * (COST_BPS / 10000)
            pnl_net = pnl - costs
            
            fold_pnl.extend(pnl_net)
            fold_bh_pnl.extend(returns)
            fold_positions.extend(positions)
        
        if fold_pnl:
            fold_pnl = np.array(fold_pnl)
            fold_bh_pnl = np.array(fold_bh_pnl)
            fold_positions = np.array(fold_positions)
            
            sharpe = compute_sharpe(fold_pnl)
            sharpe_bh = compute_sharpe(fold_bh_pnl)
            ret = fold_pnl.sum() * 100
            ret_bh = fold_bh_pnl.sum() * 100
            n_long = (fold_positions == 1).sum()
            n_short = (fold_positions == -1).sum()
            n_flat = (fold_positions == 0).sum()
            
            traded = fold_positions != 0
            win_rate = ((fold_positions * fold_bh_pnl) > 0)[traded].mean() * 100 if traded.sum() > 0 else 0
            
            status = "✅" if sharpe > sharpe_bh else ("🟡" if sharpe > 0 else "❌")
            
            print(f"   Sharpe: {sharpe:.3f} vs B&H {sharpe_bh:.3f} {status}")
            print(f"   Return: {ret:+.1f}% vs B&H {ret_bh:+.1f}%")
            print(f"   Win%: {win_rate:.1f}% | Long: {n_long} | Short: {n_short} | Flat: {n_flat}")
            
            fold_summaries.append({
                "fold": fold_idx + 1,
                "period": f"{fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}",
                "sharpe": sharpe if sharpe else 0,
                "sharpe_bh": sharpe_bh if sharpe_bh else 0,
                "return": ret,
                "return_bh": ret_bh,
                "win_rate": win_rate,
                "n_long": n_long,
                "n_short": n_short,
                "n_flat": n_flat,
                "beat_bh": sharpe > sharpe_bh if sharpe and sharpe_bh else False,
                "positive": sharpe > 0 if sharpe else False,
            })
            
            all_results.extend(fold_pnl)
    
    # Overall summary
    print("\n" + "="*100)
    print("📊 WALK-FORWARD BACKTEST SUMMARY")
    print("="*100)
    
    if fold_summaries:
        df_summary = pd.DataFrame(fold_summaries)
        
        print(f"\n{'Fold':<6} {'Period':<22} {'Sharpe':>8} {'B&H':>8} {'Return':>10} {'Win%':>7} {'L/S/F':>12} {'Status':<6}")
        print("-"*90)
        
        for _, row in df_summary.iterrows():
            status = "✅ BEAT" if row['beat_bh'] else ("🟡 +VE" if row['positive'] else "❌ LOSE")
            lsf = f"{row['n_long']}/{row['n_short']}/{row['n_flat']}"
            print(f"{row['fold']:<6} {row['period']:<22} {row['sharpe']:>8.3f} {row['sharpe_bh']:>8.3f} "
                  f"{row['return']:>9.1f}% {row['win_rate']:>6.1f}% {lsf:>12} {status:<6}")
        
        # Aggregate stats
        avg_sharpe = df_summary['sharpe'].mean()
        avg_sharpe_bh = df_summary['sharpe_bh'].mean()
        total_return = df_summary['return'].sum()
        total_return_bh = df_summary['return_bh'].sum()
        beats = df_summary['beat_bh'].sum()
        positives = df_summary['positive'].sum()
        avg_win = df_summary['win_rate'].mean()
        
        print("\n" + "-"*90)
        print(f"{'TOTAL':<6} {'':<22} {avg_sharpe:>8.3f} {avg_sharpe_bh:>8.3f} "
              f"{total_return:>9.1f}% {avg_win:>6.1f}%")
        
        print("\n" + "="*80)
        print("📈 FINAL STATISTICS")
        print("="*80)
        print(f"   Average Sharpe: {avg_sharpe:.3f} (B&H: {avg_sharpe_bh:.3f})")
        print(f"   Total Return: {total_return:.1f}% (B&H: {total_return_bh:.1f}%)")
        print(f"   Average Win Rate: {avg_win:.1f}%")
        print(f"   Beat B&H: {beats}/{len(df_summary)} folds ({beats/len(df_summary)*100:.0f}%)")
        print(f"   Positive Sharpe: {positives}/{len(df_summary)} folds ({positives/len(df_summary)*100:.0f}%)")
        
        # Calculate overall metrics from all returns
        all_pnl = np.array(all_results)
        overall_sharpe = compute_sharpe(all_pnl) if len(all_pnl) > 0 else 0
        cumsum = np.cumsum(all_pnl)
        running_max = np.maximum.accumulate(cumsum)
        max_dd = (running_max - cumsum).max() * 100
        
        print(f"\n   Overall Sharpe (all trades): {overall_sharpe:.3f}")
        print(f"   Max Drawdown: {max_dd:.1f}%")
        
        # Grade
        print("\n" + "="*80)
        print("🎯 MODEL GRADE")
        print("="*80)
        
        beat_pct = beats / len(df_summary)
        positive_pct = positives / len(df_summary)
        
        if beat_pct >= 0.7 and positive_pct == 1.0 and avg_sharpe > 1.0:
            grade = "A+ - Exceptional performance"
        elif beat_pct >= 0.6 and positive_pct >= 0.9 and avg_sharpe > 0.5:
            grade = "A - Strong performance"
        elif beat_pct >= 0.5 and positive_pct >= 0.8:
            grade = "B+ - Good performance"
        elif positive_pct >= 0.7:
            grade = "B - Positive in most conditions"
        elif positive_pct >= 0.5:
            grade = "C+ - Mixed results"
        else:
            grade = "C - Needs improvement"
        
        print(f"\n🎯 GRADE: {grade}")
        
        if avg_sharpe > avg_sharpe_bh:
            print("✅ OUTPERFORMS Buy & Hold on average!")
        if overall_sharpe > 0.5:
            print("✅ Overall Sharpe > 0.5 - Production Ready!")
        if positive_pct >= 0.8:
            print("✅ Positive in 80%+ of folds - Consistent!")
        
        return {
            "fold_summaries": fold_summaries,
            "avg_sharpe": avg_sharpe,
            "avg_sharpe_bh": avg_sharpe_bh,
            "total_return": total_return,
            "total_return_bh": total_return_bh,
            "beat_rate": beat_pct,
            "positive_rate": positive_pct,
            "overall_sharpe": overall_sharpe,
            "max_drawdown": max_dd,
            "grade": grade,
        }
    
    return None


def main():
    print("="*80)
    print("🔬 COMPREHENSIVE BACKTEST - PRODUCTION MODEL VALIDATION")
    print("="*80)
    print(f"📅 Period: 2018-01-01 to 2025-12-31")
    print(f"📊 Tickers: {', '.join(TEST_TICKERS)}")
    print(f"💰 Transaction Cost: {COST_BPS} bps")
    print(f"🎯 Walk-Forward: Train {TRAIN_WINDOW_MONTHS}mo, Test {TEST_WINDOW_MONTHS}mo")
    print("="*80)
    
    results = walk_forward_backtest(TEST_TICKERS, "2018-01-01", "2025-12-31")
    
    if results:
        # Save results
        import json
        results_path = Path(__file__).parent / "backtest_results_production.json"
        with open(results_path, "w") as f:
            json.dump({k: v for k, v in results.items() if k != "fold_summaries"}, f, indent=2)
        print(f"\n💾 Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
