#!/usr/bin/env python3
"""
Test the multi-source data pipeline.

Verifies:
1. All providers are properly configured
2. Fallback chains work correctly
3. Cache is functioning
4. Data quality is acceptable
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime


def test_providers():
    """Test individual provider availability."""
    print("=" * 60)
    print("TESTING DATA PROVIDERS")
    print("=" * 60)
    
    from src.data.providers import (
        YFinanceProvider,
        TiingoProvider,
        FinnhubProvider,
        SECEdgarProvider,
        AlphaVantageProvider,
    )
    
    providers = [
        ("YFinance", YFinanceProvider()),
        ("Tiingo", TiingoProvider()),
        ("Finnhub", FinnhubProvider()),
        ("SEC EDGAR", SECEdgarProvider()),
        ("Alpha Vantage", AlphaVantageProvider()),
    ]
    
    results = {}
    for name, provider in providers:
        available = provider.is_available()
        results[name] = available
        status = "✅ AVAILABLE" if available else "❌ UNAVAILABLE (no API key)"
        print(f"{name:15} | {status}")
    
    print()
    return results


def test_price_fallback():
    """Test price data retrieval with fallback."""
    print("=" * 60)
    print("TESTING PRICE DATA FALLBACK")
    print("=" * 60)
    
    from src.data.aggregator import get_aggregator
    
    aggregator = get_aggregator()
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        df = aggregator.get_price_history(ticker, period="1mo", use_cache=False)
        
        if df is not None:
            print(f"  ✅ Got {len(df)} rows")
            print(f"  📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"  📊 Columns: {list(df.columns)}")
            
            # Check for required columns
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  ⚠️ Missing columns: {missing}")
        else:
            print(f"  ❌ FAILED - No data returned")
    
    print()


def test_fundamentals_fallback():
    """Test fundamentals retrieval with fallback."""
    print("=" * 60)
    print("TESTING FUNDAMENTALS FALLBACK")
    print("=" * 60)
    
    from src.data.aggregator import get_aggregator
    
    aggregator = get_aggregator()
    
    test_tickers = ["AAPL", "MSFT"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        data = aggregator.get_fundamentals(ticker, use_cache=False)
        
        if data:
            print(f"  ✅ Got {len(data)} fields")
            
            # Show key metrics
            key_metrics = ["fund_pe_trailing", "fund_pb", "fund_marketcap"]
            for m in key_metrics:
                val = data.get(m)
                status = "✓" if val else "✗"
                print(f"  {status} {m}: {val}")
            
            # Show sources used
            sources = data.get("_sources", [])
            print(f"  📡 Sources: {sources}")
        else:
            print(f"  ❌ FAILED - No data returned")
    
    print()


def test_cache():
    """Test cache functionality."""
    print("=" * 60)
    print("TESTING CACHE FUNCTIONALITY")
    print("=" * 60)
    
    from src.data.cache_manager import get_cache
    
    cache = get_cache()
    
    # Test set/get
    test_data = {"test": "value", "number": 42}
    cache.set("test", "test_key", test_data)
    
    retrieved = cache.get("test", "test_key")
    if retrieved == test_data:
        print("✅ Cache set/get working")
    else:
        print("❌ Cache set/get FAILED")
    
    # Test invalidation
    cache.invalidate("test", "test_key")
    retrieved = cache.get("test", "test_key")
    if retrieved is None:
        print("✅ Cache invalidation working")
    else:
        print("❌ Cache invalidation FAILED")
    
    # Test stats
    stats = cache.get_stats()
    print(f"📊 Cache stats: {stats['memory_entries']} memory, {stats['file_entries']} file entries")
    
    print()


def test_aggregator_health():
    """Test aggregator provider health."""
    print("=" * 60)
    print("PROVIDER HEALTH STATUS")
    print("=" * 60)
    
    from src.data.aggregator import get_aggregator
    
    aggregator = get_aggregator()
    health = aggregator.get_provider_health()
    
    for name, status in health.items():
        avail = "✅" if status["available"] else "❌"
        key_req = "🔑" if status["requires_key"] else "🆓"
        success = status["success_count"]
        failure = status["failure_count"]
        
        print(f"{avail} {name:15} | {key_req} | Success: {success}, Failures: {failure}")
    
    print()


def test_data_quality():
    """Test data quality from aggregator."""
    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)
    
    from src.data.aggregator import fetch_prices, fetch_fundamentals
    
    ticker = "AAPL"
    
    # Price data quality
    print(f"\n{ticker} Price Data:")
    df = fetch_prices(ticker, period="2y")
    if df is not None:
        nan_pct = df.isna().sum() / len(df) * 100
        print(f"  Rows: {len(df)}")
        print(f"  NaN % per column:")
        for col, pct in nan_pct.items():
            status = "✅" if pct < 1 else "⚠️"
            print(f"    {status} {col}: {pct:.2f}%")
    
    # Fundamentals quality
    print(f"\n{ticker} Fundamentals:")
    funds = fetch_fundamentals(ticker)
    if funds:
        filled = sum(1 for v in funds.values() if v is not None)
        total = len(funds)
        print(f"  Fields with data: {filled}/{total}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" MULTI-SOURCE DATA PIPELINE TEST")
    print("=" * 60 + "\n")
    
    test_providers()
    test_cache()
    test_price_fallback()
    test_fundamentals_fallback()
    test_aggregator_health()
    test_data_quality()
    
    print("=" * 60)
    print(" TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
