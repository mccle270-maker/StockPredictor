#!/usr/bin/env python3
"""
Feature Validation Script v2
============================
Builds features directly (bypassing the numpy conversion) to inspect
every column with NaN rates, stats, and flag issues.
"""
import signal, sys, time, warnings
warnings.filterwarnings("ignore")

def _timeout(signum, frame):
    print("\n\n❌ TIMEOUT after 180s")
    sys.exit(1)
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(180)

import pandas as pd
import numpy as np

t0 = time.time()
print("=" * 80)
print("FEATURE VALIDATION v2 — AAPL 2y (pre-numpy, inspecting DataFrame)")
print("=" * 80)

# Import
print("\n⏳ Importing...")
from prediction_model import (
    FEATURE_COLUMNS, MACRO_COLUMNS, FEAT_GROUP_ORDER,
    add_price_features, add_regime_features, add_energy_features,
    add_advanced_features, get_price_history, get_macro_df, get_fundamental_features,
)
try:
    from model_improvements import add_enhanced_features
    HAS_IMPROVEMENTS = True
except ImportError:
    HAS_IMPROVEMENTS = False

print(f"   Import took {time.time()-t0:.1f}s")

ticker = "AAPL"
period = "2y"

# Step 1: Price data
print(f"\n⏳ Fetching price history for {ticker} ({period})...")
t1 = time.time()
hist = get_price_history(ticker, period=period, interval="1d")
if hist is None or hist.empty:
    print("❌ No price data!"); sys.exit(1)
print(f"   Got {len(hist)} rows in {time.time()-t1:.1f}s")
print(f"   Columns: {list(hist.columns)}")

# Step 2: Add features
print("\n⏳ Adding price features...")
hist = add_price_features(hist)
print(f"   Columns now: {len(hist.columns)}")

print("⏳ Adding regime features...")
hist = add_regime_features(hist)
print(f"   Columns now: {len(hist.columns)}")

print("⏳ Adding energy features...")
try:
    hist = add_energy_features(hist, ticker=ticker, period=period)
except Exception as e:
    print(f"   ⚠️ Energy features failed: {e}")
print(f"   Columns now: {len(hist.columns)}")

if HAS_IMPROVEMENTS:
    print("⏳ Adding enhanced features...")
    hist = add_enhanced_features(hist)
    print(f"   Columns now: {len(hist.columns)}")

print("⏳ Adding advanced features...")
try:
    hist = add_advanced_features(hist)
except Exception as e:
    print(f"   ⚠️ Advanced features failed: {e}")
print(f"   Columns now: {len(hist.columns)}")

# TA-Lib
try:
    from talib_integration import add_talib_indicators
    print("⏳ Adding TA-Lib indicators...")
    hist = add_talib_indicators(hist)
    print(f"   Columns now: {len(hist.columns)}")
except Exception as e:
    print(f"   ⚠️ TA-Lib: {e}")

# Pandas-TA
try:
    from pandas_ta_integration import add_pandas_ta_indicators
    print("⏳ Adding Pandas-TA indicators...")
    hist = add_pandas_ta_indicators(hist, categories=["momentum", "trend", "volatility", "volume"])
    print(f"   Columns now: {len(hist.columns)}")
except Exception as e:
    print(f"   ⚠️ Pandas-TA: {e}")

# Macro
print("⏳ Adding macro data...")
try:
    macro_df = get_macro_df(symbol="^GSPC", period=period)
    hist = hist.join(macro_df, how="left")
    print(f"   Macro joined. Columns now: {len(hist.columns)}")
except Exception as e:
    print(f"   ⚠️ Macro failed: {e}")
    for c in MACRO_COLUMNS:
        if c not in hist.columns:
            hist[c] = 0.0

# Fill macro
for c in MACRO_COLUMNS:
    if c not in hist.columns:
        hist[c] = np.nan
macro_present = [c for c in MACRO_COLUMNS if c in hist.columns]
if macro_present:
    hist[macro_present] = hist[macro_present].ffill().bfill()

# Fundamentals
print("⏳ Adding fundamental data...")
try:
    fund_feats = get_fundamental_features(ticker)
    for k, v in fund_feats.items():
        hist[k] = v
    print(f"   Fundamentals: {fund_feats}")
except Exception as e:
    print(f"   ⚠️ Fundamentals failed: {e}")

# News sentiment
print("⏳ Adding news sentiment...")
try:
    from data_fetch import get_news_sentiment
    sentiment_data = get_news_sentiment(ticker, lookback_days=7)
    hist["news_sentiment"] = sentiment_data.get("sentiment_score", 0.0)
    hist["news_count"] = sentiment_data.get("article_count", 0)
    hist["news_sentiment"] = hist["news_sentiment"].ffill().bfill().fillna(0.0)
    hist["news_count"] = hist["news_count"].ffill().bfill().fillna(0)
except Exception as e:
    print(f"   ⚠️ News: {e}")
    hist["news_sentiment"] = 0.0
    hist["news_count"] = 0

print(f"\n   Total columns in hist: {len(hist.columns)}")

# ============================================================================
# Now validate every expected feature
# ============================================================================
all_expected = FEATURE_COLUMNS + MACRO_COLUMNS
print(f"   Expected features: {len(all_expected)}")

present = [c for c in all_expected if c in hist.columns]
missing = [c for c in all_expected if c not in hist.columns]
print(f"   Present: {len(present)}, Missing: {len(missing)}")
if missing:
    print(f"\n🚫 MISSING features: {missing}")

print("\n" + "=" * 100)
print(f"{'FEATURE':<40} {'STATUS':>8} {'NaN%':>7} {'MEAN':>12} {'STD':>12} {'MIN':>12} {'MAX':>12}")
print("-" * 100)

issues = []
ok_count = 0
warn_count = 0
fail_count = 0
missing_count = 0

for col in all_expected:
    if col not in hist.columns:
        print(f"{col:<40} {'MISSING':>8}")
        issues.append((col, "MISSING", "Not in DataFrame"))
        missing_count += 1
        continue

    series = hist[col].dropna() if hist[col].dtype != 'object' else hist[col]
    nan_rate = hist[col].isna().mean()
    
    try:
        std_val = float(series.std()) if len(series) > 0 else 0.0
        mean_val = float(series.mean()) if len(series) > 0 else 0.0
        min_val = float(series.min()) if len(series) > 0 else 0.0
        max_val = float(series.max()) if len(series) > 0 else 0.0
    except:
        std_val = mean_val = min_val = max_val = 0.0

    all_zero = (hist[col] == 0).all() if nan_rate < 1.0 else False

    status = "✅ OK"
    if nan_rate > 0.10:
        status = "❌ FAIL"
        issues.append((col, "HIGH_NAN", f"NaN={nan_rate:.1%}"))
        fail_count += 1
    elif nan_rate > 0.05:
        status = "⚠️ WARN"
        issues.append((col, "NAN", f"NaN={nan_rate:.1%}"))
        warn_count += 1
    elif all_zero:
        status = "⚠️ ZERO"
        issues.append((col, "ALL_ZERO", "All values=0"))
        warn_count += 1
    elif std_val < 1e-12:
        status = "⚠️ CONST"
        issues.append((col, "CONSTANT", f"std={std_val:.2e}"))
        warn_count += 1
    else:
        ok_count += 1

    nan_str = f"{nan_rate:.1%}" if nan_rate > 0 else "0%"
    print(f"{col:<40} {status:>8} {nan_str:>7} {mean_val:>12.4f} {std_val:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"  ✅ OK:       {ok_count}")
print(f"  ⚠️  Warnings: {warn_count}")
print(f"  ❌ Failures:  {fail_count}")
print(f"  🚫 Missing:   {missing_count}")
print(f"  Total rows:  {len(hist)}")

if issues:
    print(f"\n{'ISSUE DETAILS':}")
    print("-" * 70)
    for col, itype, detail in issues:
        print(f"  {col:<40} [{itype:<10}] {detail}")

total = ok_count + warn_count + fail_count + missing_count
pct = ok_count / total * 100 if total > 0 else 0
print(f"\n🎯 Feature health: {ok_count}/{total} ({pct:.0f}%) features are clean")
print(f"⏱️  Total time: {time.time()-t0:.1f}s")

signal.alarm(0)
