"""
Regime-Balanced Model Training
==============================

Creates a model that performs well in BOTH bull and bear markets by:
1. Training on balanced data (includes 2022 bear + 2023-2024 bull)
2. Adding regime indicators as features
3. Teaching the model when to go flat or short

Run: python train_balanced_model.py
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
warnings.filterwarnings("ignore")

from src.core.models import make_model, select_features_elasticnet
from src.core.features import build_all_features, add_gbm_features
from src.core.metrics import compute_sharpe, compute_drawdown
from src.data.market import get_price_history
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS

# Training configuration
TRAIN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "JPM", "V",
]

# Include BOTH bull and bear market periods
TRAIN_PERIODS = [
    # Bear market 2022
    ("2022-01-01", "2022-12-31", "bear"),
    # Recovery/Bull 2023
    ("2023-01-01", "2023-12-31", "bull"),
    # Bull 2024
    ("2024-01-01", "2024-12-31", "bull"),
]

# Test on recent data (2025)
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

HORIZON = 5
THRESHOLD = 0.002
COST_BPS = 10

# Output paths
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that help the model identify market regime.
    These help the model learn WHEN to go long vs short vs flat.
    """
    # Price vs Moving Averages (trend indicators)
    if "Close" in df.columns:
        # 50-day and 200-day MA crossover signals
        df["ma_50"] = df["Close"].rolling(50).mean()
        df["ma_200"] = df["Close"].rolling(200).mean()
        df["price_vs_ma50"] = (df["Close"] / df["ma_50"] - 1).shift(1)
        df["price_vs_ma200"] = (df["Close"] / df["ma_200"] - 1).shift(1)
        df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"] - 1).shift(1)
        
        # Golden cross / Death cross indicator
        df["golden_cross"] = (df["ma_50"] > df["ma_200"]).astype(int).shift(1)
        
        # Trend strength (consecutive days above/below MA)
        above_ma200 = df["Close"] > df["ma_200"]
        df["days_above_ma200"] = above_ma200.groupby((~above_ma200).cumsum()).cumcount().shift(1)
        df["days_below_ma200"] = (~above_ma200).groupby(above_ma200.cumsum()).cumcount().shift(1)
        
    # Volatility regime
    if "vol_20d" in df.columns:
        df["vol_percentile"] = df["vol_20d"].rolling(252).rank(pct=True).shift(1)
        df["high_vol_regime"] = (df["vol_percentile"] > 0.8).astype(int).shift(1)
        
    # Drawdown indicator (are we in a correction?)
    if "Close" in df.columns:
        rolling_max = df["Close"].rolling(252, min_periods=1).max()
        df["drawdown_pct"] = ((df["Close"] / rolling_max) - 1).shift(1)
        df["in_correction"] = (df["drawdown_pct"] < -0.10).astype(int)
        df["in_bear_market"] = (df["drawdown_pct"] < -0.20).astype(int)
        
    # Recent momentum
    if "ret_20d" in df.columns or "Close" in df.columns:
        if "ret_20d" not in df.columns:
            df["ret_20d"] = df["Close"].pct_change(20)
        df["momentum_20d_zscore"] = (
            (df["ret_20d"] - df["ret_20d"].rolling(252).mean()) / 
            df["ret_20d"].rolling(252).std()
        ).shift(1)
        
    # Market breadth proxy (if available from macro)
    # VIX level as fear indicator
    if "vix" in df.columns:
        df["vix_high"] = (df["vix"] > 25).astype(int).shift(1)
        df["vix_extreme"] = (df["vix"] > 35).astype(int).shift(1)
    
    return df


def get_all_data_for_ticker(ticker: str) -> pd.DataFrame:
    """Fetch all historical data for a ticker."""
    hist = get_price_history(ticker, period="max", interval="1d")
    if hist is None or not isinstance(hist, pd.DataFrame):
        return None
    
    # Make timezone naive
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    
    # Build all features
    hist = build_all_features(hist)
    if hist is None or not isinstance(hist, pd.DataFrame):
        return None
    
    hist = add_gbm_features(hist, horizons=(1, HORIZON))
    
    # Add regime-specific features
    hist = add_regime_features(hist)
    
    # Build target
    target_col = f"ftarget_ret_{HORIZON}d_ahead"
    hist[target_col] = hist["Close"].pct_change(HORIZON).shift(-HORIZON)
    
    # Winsorize
    lower = hist[target_col].quantile(0.01)
    upper = hist[target_col].quantile(0.99)
    hist[target_col] = hist[target_col].clip(lower=lower, upper=upper)
    
    return hist


def prepare_balanced_training_data(tickers: list) -> tuple:
    """
    Prepare training data that includes both bull and bear market periods.
    Returns combined X, y with balanced representation.
    """
    print("📊 Preparing balanced training data...")
    print(f"   Tickers: {len(tickers)}")
    print(f"   Periods: {len(TRAIN_PERIODS)}")
    
    all_X = []
    all_y = []
    all_dates = []
    all_tickers = []
    all_regimes = []
    
    # Collect features
    base_feat_cols = [c for c in FEATURE_COLUMNS]
    regime_feat_cols = [
        "price_vs_ma50", "price_vs_ma200", "ma50_vs_ma200", "golden_cross",
        "days_above_ma200", "days_below_ma200", "vol_percentile", "high_vol_regime",
        "drawdown_pct", "in_correction", "in_bear_market", "momentum_20d_zscore",
    ]
    
    for ticker in tickers:
        print(f"   Loading {ticker}...")
        
        hist = get_all_data_for_ticker(ticker)
        if hist is None:
            print(f"      ❌ No data for {ticker}")
            continue
        
        # Determine available features
        feat_cols = [c for c in base_feat_cols + regime_feat_cols if c in hist.columns]
        target_col = f"ftarget_ret_{HORIZON}d_ahead"
        
        # Filter for quality
        nan_rates = hist[feat_cols].isna().mean()
        feat_cols = [c for c in feat_cols if nan_rates[c] < 0.3]
        
        # Fill NaNs
        hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
        
        # Extract data for each training period
        for period_start, period_end, regime_label in TRAIN_PERIODS:
            mask = (hist.index >= period_start) & (hist.index <= period_end)
            period_df = hist.loc[mask].dropna(subset=[target_col])
            
            if len(period_df) < 50:
                continue
            
            X = period_df[feat_cols].values
            y = period_df[target_col].values
            
            all_X.append(X)
            all_y.append(y)
            all_dates.extend(period_df.index.tolist())
            all_tickers.extend([ticker] * len(y))
            all_regimes.extend([regime_label] * len(y))
    
    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    # Report balance
    regimes_arr = np.array(all_regimes)
    n_bear = (regimes_arr == "bear").sum()
    n_bull = (regimes_arr == "bull").sum()
    
    print(f"\n📊 Training Data Summary:")
    print(f"   Total samples: {len(y_combined)}")
    print(f"   Bear market samples: {n_bear} ({n_bear/len(y_combined)*100:.1f}%)")
    print(f"   Bull market samples: {n_bull} ({n_bull/len(y_combined)*100:.1f}%)")
    print(f"   Features: {len(feat_cols)}")
    
    # Report target distribution
    pct_positive = (y_combined > 0).mean() * 100
    pct_negative = (y_combined < 0).mean() * 100
    print(f"   Target positive: {pct_positive:.1f}%")
    print(f"   Target negative: {pct_negative:.1f}%")
    
    return X_combined, y_combined, feat_cols, regimes_arr


def apply_transaction_costs(pnl: np.ndarray, positions: np.ndarray, cost_bps: float = 10) -> np.ndarray:
    """Apply transaction costs."""
    cost_per_trade = cost_bps / 10000
    pos_changes = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = pos_changes * cost_per_trade
    return pnl - costs


def train_and_evaluate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feat_cols: list,
    regimes_train: np.ndarray,
    model_type: str = "xgb",
    use_elasticnet: bool = True,
) -> dict:
    """Train model and return it with metadata."""
    
    print(f"\n🔧 Training {model_type.upper()} model...")
    
    n_features_orig = len(feat_cols)
    sel_mask = None
    
    # Feature selection with ElasticNet
    if use_elasticnet:
        try:
            print("   Running ElasticNet feature selection...")
            # Create dummy dates for elasticnet
            dummy_dates = pd.date_range("2020-01-01", periods=len(y_train), freq="D")
            
            X_train_sel, sel_names, sel_mask = select_features_elasticnet(
                X_train, y_train,
                feature_names=feat_cols,
                dates=dummy_dates,
                horizon=HORIZON,
                n_splits=5,
                l1_ratio=0.5,
                min_features=15,  # Keep more features for regime detection
            )
            X_train = X_train_sel
            feat_cols_selected = sel_names
            print(f"   Selected {len(sel_names)}/{n_features_orig} features")
        except Exception as e:
            print(f"   ⚠️ ElasticNet failed: {e}")
            feat_cols_selected = feat_cols
    else:
        feat_cols_selected = feat_cols
    
    # Train model
    model = make_model(model_type=model_type, random_state=42, use_optimized=True)
    model.fit(X_train, y_train)
    
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feat_cols_selected, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n   Top 10 Features:")
        for feat, imp in top_features:
            print(f"      {feat}: {imp:.4f}")
    
    return {
        "model": model,
        "feature_cols": feat_cols_selected,
        "feature_mask": sel_mask,
        "model_type": model_type,
        "n_features": len(feat_cols_selected),
        "trained_on": datetime.now().isoformat(),
        "train_samples": len(y_train),
        "train_regimes": {
            "bear": int((regimes_train == "bear").sum()),
            "bull": int((regimes_train == "bull").sum()),
        },
    }


def test_on_period(
    model_info: dict,
    tickers: list,
    test_start: str,
    test_end: str,
    period_name: str = "Test",
) -> list:
    """Test model on a specific period."""
    
    print(f"\n📊 Testing on {period_name} ({test_start} to {test_end})...")
    
    model = model_info["model"]
    feat_cols = model_info["feature_cols"]
    sel_mask = model_info["feature_mask"]
    
    results = []
    
    for ticker in tickers:
        hist = get_all_data_for_ticker(ticker)
        if hist is None:
            continue
        
        # Get available features
        available_feat_cols = [c for c in feat_cols if c in hist.columns]
        if len(available_feat_cols) < len(feat_cols) * 0.8:
            print(f"   ⚠️ {ticker}: Missing too many features")
            continue
        
        # Fill missing features with 0
        for c in feat_cols:
            if c not in hist.columns:
                hist[c] = 0
        
        hist[feat_cols] = hist[feat_cols].ffill().bfill().fillna(0)
        
        target_col = f"ftarget_ret_{HORIZON}d_ahead"
        
        # Filter to test period
        mask = (hist.index >= test_start) & (hist.index <= test_end)
        test_df = hist.loc[mask].dropna(subset=[target_col])
        
        if len(test_df) < 20:
            continue
        
        X_test = test_df[feat_cols].values
        y_test = test_df[target_col].values
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Trading simulation with SYMMETRIC threshold
        # Go LONG if pred > threshold
        # Go SHORT if pred < -threshold
        # Stay FLAT otherwise
        positions = np.where(
            y_pred > THRESHOLD, 1,
            np.where(y_pred < -THRESHOLD, -1, 0)
        )
        
        # PnL
        pnl_raw = positions * y_test
        pnl_net = apply_transaction_costs(pnl_raw, positions, COST_BPS)
        
        # Buy & hold
        bh_returns = test_df["Close"].pct_change().fillna(0).values
        
        # Metrics
        accuracy = float((np.sign(y_pred) == np.sign(y_test)).mean()) * 100
        sharpe_net = compute_sharpe(pnl_net)
        sharpe_bh = compute_sharpe(bh_returns)
        return_net = float(pnl_net.sum()) * 100
        return_bh = float((test_df["Close"].iloc[-1] / test_df["Close"].iloc[0] - 1)) * 100
        
        # Position stats
        n_long = int((positions == 1).sum())
        n_short = int((positions == -1).sum())
        n_flat = int((positions == 0).sum())
        n_trades = int(np.count_nonzero(np.diff(positions)))
        
        results.append({
            "ticker": ticker,
            "period": period_name,
            "accuracy": accuracy,
            "sharpe_net": sharpe_net,
            "sharpe_bh": sharpe_bh,
            "sharpe_vs_bh": (sharpe_net - sharpe_bh) if sharpe_bh else None,
            "return_net": return_net,
            "return_bh": return_bh,
            "n_long": n_long,
            "n_short": n_short,
            "n_flat": n_flat,
            "n_trades": n_trades,
            "beat_bh": (sharpe_net > sharpe_bh) if sharpe_bh else False,
        })
    
    return results


def print_results_table(results: list, title: str):
    """Print results in a nice table."""
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print("="*80)
    print(f"{'Ticker':<8} {'Acc':>7} {'Sharpe':>8} {'B&H':>8} {'Return':>10} {'Long':>6} {'Short':>6} {'Flat':>6} {'Status':>8}")
    print("-"*80)
    
    for r in results:
        status = "✅ BEAT" if r["beat_bh"] else "❌ LOSE"
        print(f"{r['ticker']:<8} {r['accuracy']:>6.1f}% {r['sharpe_net']:>8.3f} {r['sharpe_bh']:>8.3f} "
              f"{r['return_net']:>+9.2f}% {r['n_long']:>6} {r['n_short']:>6} {r['n_flat']:>6} {status:>8}")
    
    # Averages
    if results:
        avg_acc = np.mean([r["accuracy"] for r in results])
        avg_sharpe = np.mean([r["sharpe_net"] for r in results])
        avg_sharpe_bh = np.mean([r["sharpe_bh"] for r in results])
        avg_ret = np.mean([r["return_net"] for r in results])
        win_rate = np.mean([r["beat_bh"] for r in results]) * 100
        avg_short = np.mean([r["n_short"] for r in results])
        
        print("-"*80)
        print(f"{'AVERAGE':<8} {avg_acc:>6.1f}% {avg_sharpe:>8.3f} {avg_sharpe_bh:>8.3f} "
              f"{avg_ret:>+9.2f}% {'':>6} {avg_short:>6.0f} {'':>6} {win_rate:.0f}% win")


def main():
    print("="*80)
    print("🎯 REGIME-BALANCED MODEL TRAINING")
    print("="*80)
    print(f"📅 Training includes: 2022 (bear) + 2023-2024 (bull)")
    print(f"📊 Tickers: {', '.join(TRAIN_TICKERS)}")
    print(f"💰 Transaction Cost: {COST_BPS} bps")
    print("="*80)
    
    # Step 1: Prepare balanced training data
    X_train, y_train, feat_cols, regimes = prepare_balanced_training_data(TRAIN_TICKERS)
    
    # Step 2: Train model
    model_info = train_and_evaluate_model(
        X_train, y_train, feat_cols, regimes,
        model_type="xgb",
        use_elasticnet=True,
    )
    
    # Step 3: Test on 2022 (bear market - in-sample check)
    results_2022 = test_on_period(
        model_info, TRAIN_TICKERS,
        "2022-01-01", "2022-12-31",
        "2022 Bear (In-Sample)"
    )
    print_results_table(results_2022, "2022 BEAR MARKET (In-Sample)")
    
    # Step 4: Test on 2023 (recovery - in-sample check)  
    results_2023 = test_on_period(
        model_info, TRAIN_TICKERS,
        "2023-01-01", "2023-12-31",
        "2023 Recovery (In-Sample)"
    )
    print_results_table(results_2023, "2023 RECOVERY (In-Sample)")
    
    # Step 5: Test on 2025 (truly out-of-sample)
    results_2025 = test_on_period(
        model_info, TRAIN_TICKERS,
        "2025-01-01", "2025-12-31",
        "2025 Bull (Out-of-Sample)"
    )
    print_results_table(results_2025, "2025 BULL MARKET (Out-of-Sample)")
    
    # Summary comparison
    print("\n" + "="*80)
    print("📊 SUMMARY COMPARISON")
    print("="*80)
    
    def summarize(results, name):
        if not results:
            return None
        return {
            "period": name,
            "avg_accuracy": np.mean([r["accuracy"] for r in results]),
            "avg_sharpe": np.mean([r["sharpe_net"] for r in results]),
            "avg_sharpe_bh": np.mean([r["sharpe_bh"] for r in results]),
            "win_rate": np.mean([r["beat_bh"] for r in results]) * 100,
            "avg_short_days": np.mean([r["n_short"] for r in results]),
        }
    
    summaries = [
        summarize(results_2022, "2022 Bear"),
        summarize(results_2023, "2023 Recovery"),
        summarize(results_2025, "2025 Bull"),
    ]
    
    print(f"{'Period':<20} {'Acc':>8} {'Sharpe':>10} {'B&H Shp':>10} {'Beat B&H':>10} {'Avg Short':>10}")
    print("-"*70)
    for s in summaries:
        if s:
            print(f"{s['period']:<20} {s['avg_accuracy']:>7.1f}% {s['avg_sharpe']:>10.3f} "
                  f"{s['avg_sharpe_bh']:>10.3f} {s['win_rate']:>9.0f}% {s['avg_short_days']:>10.1f}d")
    
    # Grade the model
    print("\n" + "="*80)
    print("🎯 MODEL ASSESSMENT")
    print("="*80)
    
    # Check if model learned to short
    avg_short_2022 = np.mean([r["n_short"] for r in results_2022]) if results_2022 else 0
    beat_2022 = np.mean([r["beat_bh"] for r in results_2022]) * 100 if results_2022 else 0
    beat_2025 = np.mean([r["beat_bh"] for r in results_2025]) * 100 if results_2025 else 0
    
    print(f"\n📈 Key Metrics:")
    print(f"   Avg short days in 2022: {avg_short_2022:.1f}")
    print(f"   Beat B&H in 2022 (bear): {beat_2022:.0f}%")
    print(f"   Beat B&H in 2025 (bull): {beat_2025:.0f}%")
    
    if avg_short_2022 > 30 and beat_2022 > 50 and beat_2025 > 50:
        grade = "A - Excellent regime-aware model"
    elif avg_short_2022 > 10 and beat_2022 > 30:
        grade = "B - Good improvement, learning to short"
    elif avg_short_2022 > 0:
        grade = "C - Some progress, still long-biased"
    else:
        grade = "D - Model still not shorting"
    
    print(f"\n🎯 GRADE: {grade}")
    
    # Save the model
    model_path = MODEL_DIR / "balanced_xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save results
    results_path = MODEL_DIR / "balanced_training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "trained_on": datetime.now().isoformat(),
            "train_periods": TRAIN_PERIODS,
            "train_samples": int(len(y_train)),
            "feature_cols": model_info["feature_cols"],
            "results_2022": results_2022,
            "results_2023": results_2023,
            "results_2025": results_2025,
            "summaries": summaries,
        }, f, indent=2, default=str)
    print(f"💾 Results saved to: {results_path}")
    
    print("\n" + "="*80)
    print("✅ BALANCED TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
