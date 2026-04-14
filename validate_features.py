#!/usr/bin/env python3
"""
Feature Validation Script
=========================
Runs build_features_and_target on AAPL 2y, then reports:
- Every feature: present/missing, NaN rate, mean, std, min, max, sample values
- Flags features with >5% NaN, zero-variance, or all-zero
- Summary table at the end
"""
import signal, sys, time, warnings
warnings.filterwarnings("ignore")

# Global 120s timeout
def _timeout(signum, frame):
    print("\n\n❌ TIMEOUT after 120s")
    sys.exit(1)
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(120)

import pandas as pd
import numpy as np

t0 = time.time()
print("=" * 80)
print("FEATURE VALIDATION — AAPL 2y")
print("=" * 80)

# Import
print("\n⏳ Importing prediction_model...")
from prediction_model import build_features_and_target, FEATURE_COLUMNS, MACRO_COLUMNS
print(f"   Import took {time.time()-t0:.1f}s")

# Build features
print("\n⏳ Building features for AAPL (period=2y)...")
t1 = time.time()
result = build_features_and_target("AAPL", period="2y", horizon=1)
print(f"   Build took {time.time()-t1:.1f}s")

if result is None:
    print("❌ build_features_and_target returned None!")
    sys.exit(1)

X, y, feat_names = result[0], result[1], result[2]
print(f"   X shape: {X.shape}, y shape: {y.shape}")
print(f"   Feature names: {len(feat_names)}")

# Convert X to DataFrame for easier analysis
if isinstance(X, np.ndarray):
    df = pd.DataFrame(X, columns=feat_names)
elif isinstance(X, pd.DataFrame):
    df = X
    feat_names = list(df.columns)
else:
    print(f"❌ Unexpected X type: {type(X)}")
    sys.exit(1)

all_expected = FEATURE_COLUMNS + MACRO_COLUMNS
print(f"\n   Expected features: {len(all_expected)}")
print(f"   Actually present:  {len(feat_names)}")

# ============================================================================
# Feature-by-feature analysis
# ============================================================================
print("\n" + "=" * 80)
print(f"{'FEATURE':<40} {'STATUS':>8} {'NaN%':>6} {'MEAN':>10} {'STD':>10} {'MIN':>10} {'MAX':>10}")
print("-" * 80)

issues = []
ok_count = 0
warn_count = 0
fail_count = 0
missing_count = 0

for col in all_expected:
    if col not in feat_names:
        print(f"{col:<40} {'MISSING':>8}")
        issues.append((col, "MISSING", "Not present in output"))
        missing_count += 1
        continue

    series = df[col]
    nan_rate = series.isna().mean()
    std_val = series.std()
    mean_val = series.mean()
    min_val = series.min()
    max_val = series.max()
    all_zero = (series == 0).all()

    # Determine status
    status = "✅ OK"
    if nan_rate > 0.10:
        status = "❌ FAIL"
        issues.append((col, "HIGH_NAN", f"NaN rate = {nan_rate:.1%}"))
        fail_count += 1
    elif nan_rate > 0.05:
        status = "⚠️ WARN"
        issues.append((col, "NAN", f"NaN rate = {nan_rate:.1%}"))
        warn_count += 1
    elif all_zero:
        status = "⚠️ ZERO"
        issues.append((col, "ALL_ZERO", "All values are 0"))
        warn_count += 1
    elif std_val == 0 or (std_val is not None and pd.notna(std_val) and std_val < 1e-12):
        status = "⚠️ CONST"
        issues.append((col, "CONSTANT", f"std={std_val}"))
        warn_count += 1
    else:
        ok_count += 1

    nan_str = f"{nan_rate:.1%}" if nan_rate > 0 else "0%"
    print(f"{col:<40} {status:>8} {nan_str:>6} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

# Check for extra features not in expected list
extras = [c for c in feat_names if c not in all_expected]
if extras:
    print(f"\n📎 Extra features ({len(extras)}): {extras[:20]}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  ✅ OK:       {ok_count}")
print(f"  ⚠️  Warnings: {warn_count}")
print(f"  ❌ Failures:  {fail_count}")
print(f"  🚫 Missing:   {missing_count}")
print(f"  📎 Extra:     {len(extras)}")
print(f"  Total rows:  {len(df)}")

if issues:
    print(f"\n{'ISSUE DETAILS':}")
    print("-" * 60)
    for col, itype, detail in issues:
        print(f"  {col:<35} [{itype}] {detail}")

total = ok_count + warn_count + fail_count + missing_count
pct = ok_count / total * 100 if total > 0 else 0
print(f"\n🎯 Feature health: {ok_count}/{total} ({pct:.0f}%) features are clean")
print(f"⏱️  Total time: {time.time()-t0:.1f}s")

signal.alarm(0)
