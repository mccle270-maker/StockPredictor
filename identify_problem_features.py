#!/usr/bin/env python3
"""
Identify Problem Features Script
================================

Analyzes which features have >5% NaN rate across all tickers and categorizes
the root causes for fixes.

Usage:
    python identify_problem_features.py

Output:
    experiments/problem_features_analysis.json
    PROBLEM_FEATURES.md
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable ElasticNet for full feature analysis
os.environ["USE_ELASTICNET_SELECT"] = "0"

# Import from src/ structure
from src.data.market import get_price_history
from src.data.macro import get_macro_df
from src.data.fundamentals import get_fundamental_features
from src.core.features import build_all_features
from src.config import FEATURE_COLUMNS, MACRO_COLUMNS, REGIME_COLUMNS

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "XOM", "JNJ"]
DATA_PERIOD = "2y"
NAN_THRESHOLD = 0.05  # 5%
OUTPUT_DIR = PROJECT_ROOT / "experiments"

# Feature categorization based on source
TECHNICAL_FEATURES = [
    "rsi14", "macd", "macdsignal", "macdhist", "mfi14", 
    "atr_14", "adx_14", "willr_14", "cci_14", "roc_14",
    "obv", "obv_sma20", "bb_upper", "bb_middle", "bb_lower", "bb_pctb",
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
    "stoch_k", "stoch_d", "ultosc", "trix", "dpo",
]

MACRO_FEATURES = [
    "mkt_ret_1d", "vix", "t10y", "term_spread", "unrate", 
    "cpi", "oas", "fed_funds",
]

FUNDAMENTAL_FEATURES = [
    "fund_pe_trailing", "fund_pb", "fund_marketcap",
]

REGIME_FEATURES = [
    "regime_bull", "regime_bear", "regime_vix_low", "regime_vix_medium", 
    "regime_vix_high", "regime_covid", "regime_high_corr", "regime_low_corr",
    "bull_streak", "bear_streak",
]

GBM_FEATURES = [
    "gbm_mu", "gbm_sig", "gbm_prob_up_1d", "gbm_prob_up_5d",
    "gbm_exp_ret_1d", "gbm_exp_ret_5d", "gbm_p05_1d", "gbm_p95_1d",
]

ROLLING_FEATURES = [
    "vol_10d", "vol_20d", "vol_60d", "cumret_5d", "cumret_20d",
    "ret_1d", "ret_5d", "ret_20d",
]

RELATIVE_STRENGTH_FEATURES = [
    "rel_strength_1d", "rel_momentum_5d", "beta_60d",
]


def get_feature_category(feature: str) -> str:
    """Determine which category a feature belongs to."""
    if feature in MACRO_FEATURES:
        return "MACRO"
    if feature in FUNDAMENTAL_FEATURES:
        return "FUNDAMENTAL"
    if feature in REGIME_FEATURES:
        return "REGIME"
    if feature in GBM_FEATURES:
        return "GBM"
    if feature in TECHNICAL_FEATURES:
        return "TECHNICAL"
    if feature in ROLLING_FEATURES:
        return "ROLLING"
    if feature in RELATIVE_STRENGTH_FEATURES:
        return "RELATIVE_STRENGTH"
    # Check by prefix
    if feature.startswith("gbm_"):
        return "GBM"
    if feature.startswith("regime_"):
        return "REGIME"
    if feature.startswith("fund_"):
        return "FUNDAMENTAL"
    if feature.startswith("rel_"):
        return "RELATIVE_STRENGTH"
    if feature.startswith("vol_") or feature.startswith("ret_") or feature.startswith("cumret_"):
        return "ROLLING"
    return "TECHNICAL"


def infer_likely_cause(feature: str, avg_nan_pct: float, category: str) -> Tuple[str, str]:
    """
    Infer the likely cause and fix category for a problem feature.
    
    Returns: (likely_cause, fix_category)
    """
    # 100% NaN = external API not working
    if avg_nan_pct >= 0.99:
        if category == "MACRO":
            return "FRED API not returning data", "B"
        if category == "FUNDAMENTAL":
            return "FMP API not returning data", "B"
        if category == "RELATIVE_STRENGTH":
            return "SPX data not loading or timezone mismatch", "B"
        return "Data source completely unavailable", "B"
    
    # ~12% NaN = rolling window warmup (60 days / 502 days ≈ 12%)
    if 0.10 <= avg_nan_pct <= 0.15:
        if category in ["GBM", "ROLLING"]:
            return "60-day rolling window warmup period", "A"
        if category == "TECHNICAL":
            return "Technical indicator warmup period", "A"
        if category == "RELATIVE_STRENGTH":
            return "60-day beta/momentum calculation warmup", "A"
    
    # ~4-10% NaN = shorter warmup
    if 0.04 <= avg_nan_pct < 0.10:
        if category == "TECHNICAL":
            return "14-20 day indicator warmup", "A"
        return "Short rolling window warmup", "A"
    
    # ~20%+ NaN = likely needs 200 days (SMA_200)
    if avg_nan_pct >= 0.20:
        if "200" in feature:
            return "200-day SMA warmup period", "A"
        return "Long lookback period required", "A"
    
    # Sporadic NaN = calculation issue or data gaps
    if category == "TECHNICAL":
        return "Possible calculation issue or data gaps", "C"
    
    return "Unknown cause - needs investigation", "C"


def get_fix_description(category: str) -> str:
    """Get description for fix category."""
    descriptions = {
        "A": "Needs more historical data (increase lookback or drop warmup rows)",
        "B": "External API issue (add caching/fallback values)",
        "C": "Calculation bug (fix formula or handling)",
        "D": "Feature not applicable to all tickers (make optional)",
    }
    return descriptions.get(category, "Unknown")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_features() -> Dict[str, Any]:
    """Main analysis function."""
    print("=" * 70)
    print("  PROBLEM FEATURES ANALYSIS")
    print("=" * 70)
    print()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tickers_analyzed": TICKERS,
        "data_period": DATA_PERIOD,
        "nan_threshold": NAN_THRESHOLD,
        "features_checked": len(FEATURE_COLUMNS),
        "per_feature": {},
        "per_ticker": {},
        "problem_features": [],
        "by_category": defaultdict(list),
        "by_fix_type": defaultdict(list),
    }
    
    # Collect NaN rates per feature per ticker
    feature_nan_rates = defaultdict(list)  # feature -> [nan_rate per ticker]
    feature_worst_ticker = {}  # feature -> ticker with highest NaN
    ticker_data = {}  # ticker -> DataFrame with features
    
    # Fetch macro data once (shared across all tickers)
    print("  → Fetching macro data (FRED)...")
    try:
        macro_df = get_macro_df(period=DATA_PERIOD)
        print(f"    Got {len(macro_df)} rows, columns: {list(macro_df.columns)}")
    except Exception as e:
        print(f"    ⚠ Macro data failed: {e}")
        macro_df = None
    
    for ticker in TICKERS:
        print(f"  → Fetching {ticker}...")
        
        try:
            hist = get_price_history(ticker, period=DATA_PERIOD, interval="1d")
            if hist is None or hist.empty:
                print(f"    ⚠ No data for {ticker}")
                continue
            
            print(f"    Got {len(hist)} rows, building features...")
            hist_with_features = build_all_features(hist.copy())
            
            # Join macro data (like the prediction service does)
            if macro_df is not None and not macro_df.empty:
                hist_with_features = hist_with_features.join(macro_df, how="left")
                # Fill NaN in macro columns
                for col in macro_df.columns:
                    if col in hist_with_features.columns:
                        hist_with_features[col] = hist_with_features[col].ffill().bfill().fillna(0)
            
            # Fetch fundamentals (with rate limiting awareness)
            try:
                fund_feats = get_fundamental_features(ticker)
                for k, v in fund_feats.items():
                    hist_with_features[k] = v
            except Exception as e:
                print(f"    ⚠ Fundamentals unavailable: {str(e)[:50]}")
            
            ticker_data[ticker] = hist_with_features
            
            # Calculate NaN rate for each feature
            ticker_feature_nans = {}
            for feature in FEATURE_COLUMNS:
                if feature in hist_with_features.columns:
                    nan_rate = hist_with_features[feature].isna().mean()
                else:
                    nan_rate = 1.0  # Missing entirely = 100% NaN
                
                feature_nan_rates[feature].append(nan_rate)
                ticker_feature_nans[feature] = nan_rate
                
                # Track worst ticker
                if feature not in feature_worst_ticker or nan_rate > feature_worst_ticker[feature][1]:
                    feature_worst_ticker[feature] = (ticker, nan_rate)
            
            results["per_ticker"][ticker] = {
                "rows": len(hist_with_features),
                "features_with_nan": sum(1 for v in ticker_feature_nans.values() if v > 0),
                "problem_features": sum(1 for v in ticker_feature_nans.values() if v >= NAN_THRESHOLD),
            }
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results["per_ticker"][ticker] = {"error": str(e)}
    
    print()
    print("  Analyzing feature NaN patterns...")
    print()
    
    # Analyze each feature
    problem_features = []
    
    for feature in FEATURE_COLUMNS:
        nan_rates = feature_nan_rates.get(feature, [1.0])
        avg_nan = np.mean(nan_rates)
        max_nan = max(nan_rates)
        min_nan = min(nan_rates)
        worst_ticker, _ = feature_worst_ticker.get(feature, ("N/A", 0))
        
        category = get_feature_category(feature)
        likely_cause, fix_category = infer_likely_cause(feature, avg_nan, category)
        
        feature_info = {
            "feature": feature,
            "avg_nan_pct": round(avg_nan * 100, 2),
            "max_nan_pct": round(max_nan * 100, 2),
            "min_nan_pct": round(min_nan * 100, 2),
            "worst_ticker": worst_ticker,
            "category": category,
            "likely_cause": likely_cause,
            "fix_category": fix_category,
            "is_problem": avg_nan >= NAN_THRESHOLD,
        }
        
        results["per_feature"][feature] = feature_info
        
        if avg_nan >= NAN_THRESHOLD:
            problem_features.append(feature_info)
            results["by_category"][category].append(feature)
            results["by_fix_type"][fix_category].append(feature)
    
    # Sort by severity
    problem_features.sort(key=lambda x: x["avg_nan_pct"], reverse=True)
    results["problem_features"] = problem_features
    results["problem_count"] = len(problem_features)
    
    # Convert defaultdicts to regular dicts for JSON
    results["by_category"] = dict(results["by_category"])
    results["by_fix_type"] = dict(results["by_fix_type"])
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print summary to console."""
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Tickers analyzed: {len(results['per_ticker'])}")
    print(f"  Features checked: {results['features_checked']}")
    print(f"  Problem features (>{results['nan_threshold']*100}% NaN): {results['problem_count']}")
    print()
    
    print("  TOP PROBLEM FEATURES:")
    print("  " + "-" * 66)
    print(f"  {'Feature':<25} {'Avg NaN%':>10} {'Max NaN%':>10} {'Category':<15} {'Fix'}")
    print("  " + "-" * 66)
    
    for feat in results["problem_features"][:20]:
        print(f"  {feat['feature']:<25} {feat['avg_nan_pct']:>9.1f}% {feat['max_nan_pct']:>9.1f}% {feat['category']:<15} {feat['fix_category']}")
    
    print()
    print("  BY FIX CATEGORY:")
    for cat, features in sorted(results["by_fix_type"].items()):
        desc = get_fix_description(cat)
        print(f"    Category {cat}: {len(features)} features")
        print(f"      → {desc}")
    print()


def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate detailed Markdown report."""
    lines = [
        "# Problem Features Analysis",
        "",
        f"**Generated:** {results['timestamp']}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Tickers analyzed:** {len(results['per_ticker'])}",
        f"- **Features checked:** {results['features_checked']}",
        f"- **Problem features (>{results['nan_threshold']*100}% NaN):** {results['problem_count']}",
        "",
        "---",
        "",
        "## Problem Features Table",
        "",
        "| # | Feature | Avg NaN% | Max NaN% | Worst Ticker | Category | Likely Cause | Fix |",
        "|---|---------|----------|----------|--------------|----------|--------------|-----|",
    ]
    
    for i, feat in enumerate(results["problem_features"], 1):
        lines.append(
            f"| {i} | `{feat['feature']}` | {feat['avg_nan_pct']:.1f}% | "
            f"{feat['max_nan_pct']:.1f}% | {feat['worst_ticker']} | "
            f"{feat['category']} | {feat['likely_cause']} | {feat['fix_category']} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Fix Categories",
        "",
        "| Category | Description | Features Affected |",
        "|----------|-------------|-------------------|",
    ])
    
    for cat in ["A", "B", "C", "D"]:
        features = results["by_fix_type"].get(cat, [])
        desc = get_fix_description(cat)
        lines.append(f"| **{cat}** | {desc} | {len(features)} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Analysis by Category",
        "",
    ])
    
    # Category A: Warmup issues
    cat_a = results["by_fix_type"].get("A", [])
    if cat_a:
        lines.extend([
            "### Category A: Warmup Period Issues",
            "",
            "These features have NaN at the beginning of the dataset due to rolling window calculations.",
            "",
            "**Solution:** Drop the first N rows where N = max lookback period (typically 60-62 days).",
            "",
            "**Features:**",
            "",
        ])
        for f in cat_a:
            info = results["per_feature"].get(f, {})
            lines.append(f"- `{f}`: {info.get('avg_nan_pct', 0):.1f}% NaN — {info.get('likely_cause', 'warmup')}")
        lines.append("")
    
    # Category B: API issues
    cat_b = results["by_fix_type"].get("B", [])
    if cat_b:
        lines.extend([
            "### Category B: External API Issues",
            "",
            "These features have 100% NaN because external data sources aren't returning data.",
            "",
            "**Possible causes:**",
            "- API key not configured",
            "- API rate limit exceeded",
            "- Timezone mismatch in date index",
            "- Network/connectivity issue",
            "",
            "**Solution:** ",
            "1. Verify API keys are set (`FRED_API_KEY`, `FMP_API_KEY`)",
            "2. Add caching layer for successful API calls",
            "3. Implement fallback to last known values",
            "",
            "**Features:**",
            "",
        ])
        for f in cat_b:
            info = results["per_feature"].get(f, {})
            lines.append(f"- `{f}`: {info.get('avg_nan_pct', 0):.1f}% NaN — {info.get('likely_cause', 'API issue')}")
        lines.append("")
    
    # Category C: Calculation issues
    cat_c = results["by_fix_type"].get("C", [])
    if cat_c:
        lines.extend([
            "### Category C: Calculation Issues",
            "",
            "These features have unexpected NaN patterns suggesting a bug in the calculation.",
            "",
            "**Solution:** Review and fix the calculation formula in the relevant function.",
            "",
            "**Features:**",
            "",
        ])
        for f in cat_c:
            info = results["per_feature"].get(f, {})
            lines.append(f"- `{f}`: {info.get('avg_nan_pct', 0):.1f}% NaN — {info.get('likely_cause', 'calculation issue')}")
        lines.append("")
    
    # Category D: Optional features
    cat_d = results["by_fix_type"].get("D", [])
    if cat_d:
        lines.extend([
            "### Category D: Optional Features",
            "",
            "These features may not be applicable to all tickers.",
            "",
            "**Solution:** Make these features optional or exclude from certain ticker types.",
            "",
            "**Features:**",
            "",
        ])
        for f in cat_d:
            info = results["per_feature"].get(f, {})
            lines.append(f"- `{f}`: {info.get('avg_nan_pct', 0):.1f}% NaN")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## By Source Category",
        "",
    ])
    
    for category, features in sorted(results["by_category"].items()):
        lines.append(f"### {category} ({len(features)} features)")
        lines.append("")
        for f in features:
            info = results["per_feature"].get(f, {})
            lines.append(f"- `{f}`: {info.get('avg_nan_pct', 0):.1f}% NaN")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Recommended Fix Order",
        "",
        "1. **Category A (Warmup)** — Already handled by `validate_features()` dropping first 62 rows",
        "2. **Category B (API)** — Highest priority: fix FRED/FMP API connections or add fallbacks",
        "3. **Category C (Bugs)** — Fix any calculation errors",
        "4. **Category D (Optional)** — Low priority: make features conditional",
        "",
        "---",
        "",
        "## Per-Ticker Summary",
        "",
        "| Ticker | Rows | Features with NaN | Problem Features |",
        "|--------|------|-------------------|------------------|",
    ])
    
    for ticker, info in results["per_ticker"].items():
        if "error" in info:
            lines.append(f"| {ticker} | ERROR | {info['error'][:30]} | - |")
        else:
            lines.append(f"| {ticker} | {info['rows']} | {info['features_with_nan']} | {info['problem_features']} |")
    
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"  ✓ Markdown report saved to: {output_path}")


def main():
    """Main entry point."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run analysis
    results = analyze_features()
    
    # Print summary
    print_summary(results)
    
    # Save JSON results
    json_path = OUTPUT_DIR / f"problem_features_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ JSON results saved to: {json_path}")
    
    # Generate markdown report
    md_path = PROJECT_ROOT / "PROBLEM_FEATURES.md"
    generate_markdown_report(results, md_path)
    
    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
