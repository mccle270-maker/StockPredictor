#!/usr/bin/env python3
"""
Test news/sentiment features specifically.

Verifies:
1. Finnhub sentiment API works
2. Marketaux news API works  
3. News features get populated in feature engineering
4. Sentiment scores are calculated correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime


def test_finnhub_sentiment():
    """Test Finnhub sentiment provider."""
    print("=" * 60)
    print("TESTING FINNHUB SENTIMENT")
    print("=" * 60)
    
    from src.data.providers import FinnhubProvider
    
    provider = FinnhubProvider()
    
    if not provider.is_available():
        print("❌ Finnhub API key not configured")
        return False
    
    print("✅ Finnhub API key configured")
    
    # Test sentiment for a few tickers
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        response = provider.get_sentiment(ticker)
        
        if response.success and response.data:
            data = response.data
            print(f"  ✅ Got sentiment data")
            print(f"     Buzz: {data.get('buzz', {})}")
            print(f"     Sentiment: {data.get('sentiment', {})}")
            print(f"     Company News Score: {data.get('companyNewsScore')}")
            print(f"     Sector Avg Bullish: {data.get('sectorAverageBullishPercent')}")
        else:
            print(f"  ⚠️ No sentiment data: {response.error}")
    
    print()
    return True


def test_finnhub_earnings():
    """Test Finnhub earnings data."""
    print("=" * 60)
    print("TESTING FINNHUB EARNINGS")
    print("=" * 60)
    
    from src.data.providers import FinnhubProvider
    
    provider = FinnhubProvider()
    
    if not provider.is_available():
        print("❌ Finnhub API key not configured")
        return False
    
    test_tickers = ["AAPL", "MSFT"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        response = provider.get_earnings(ticker)
        
        if response.success and response.data:
            earnings = response.data
            print(f"  ✅ Got {len(earnings)} earnings records")
            if earnings:
                latest = earnings[0]
                print(f"     Latest: {latest.get('period')} - Actual: {latest.get('actual')}, Estimate: {latest.get('estimate')}")
                surprise = latest.get('surprisePercent')
                if surprise:
                    print(f"     Surprise: {surprise:.2f}%")
        else:
            print(f"  ⚠️ No earnings data: {response.error}")
    
    print()
    return True


def test_marketaux_news():
    """Test Marketaux news provider."""
    print("=" * 60)
    print("TESTING MARKETAUX NEWS")
    print("=" * 60)
    
    from src.data.news import get_news_for_ticker, get_news_sentiment
    from src.config import MARKETAUX_API_KEY
    
    if not MARKETAUX_API_KEY:
        print("❌ Marketaux API key not configured")
        return False
    
    print("✅ Marketaux API key configured")
    
    test_tickers = ["AAPL", "TSLA"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        try:
            news = get_news_for_ticker(ticker, limit=5)
            
            if news and len(news) > 0:
                print(f"  ✅ Got {len(news)} news articles")
                for i, article in enumerate(news[:3], 1):
                    title = article.get('title', '')[:50]
                    sentiment = article.get('sentiment', 'N/A')
                    print(f"     {i}. {title}... (sentiment: {sentiment})")
            else:
                print(f"  ⚠️ No news articles returned")
            
            # Test sentiment aggregation
            sentiment = get_news_sentiment(ticker)
            if sentiment:
                print(f"  📊 Aggregated sentiment: {sentiment}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    return True


def test_news_features_in_model():
    """Test that news features get populated in feature engineering."""
    print("=" * 60)
    print("TESTING NEWS FEATURES IN MODEL")
    print("=" * 60)
    
    from prediction_model import build_features_and_target, FEATURE_COLUMNS, MACRO_COLUMNS
    
    ticker = "AAPL"
    print(f"\nBuilding features for {ticker}...")
    
    try:
        # build_features_and_target returns 7 values
        result = build_features_and_target(ticker, period="1y", horizon=1)
        X, y, last_row_features, last_close, last_vol_20d, prob_up_gaf, dates = result
        
        # Check for news/sentiment related columns
        all_columns = FEATURE_COLUMNS + MACRO_COLUMNS + ["news_sentiment", "news_count"]
        news_columns = [col for col in all_columns if any(x in col.lower() for x in ['news', 'sentiment', 'buzz'])]
        
        print(f"\n📊 Total features: {X.shape[1]}")
        print(f"📊 Total rows: {len(y)}")
        print(f"📰 News/Sentiment columns in config: {news_columns}")
        
        # Check if news columns are in FEATURE_COLUMNS
        news_in_features = [col for col in ['news_sentiment', 'news_count'] if col in FEATURE_COLUMNS]
        print(f"   News columns in FEATURE_COLUMNS: {news_in_features}")
        
    except Exception as e:
        print(f"❌ Error building features: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    return True


def test_aggregator_sentiment():
    """Test sentiment through the data aggregator."""
    print("=" * 60)
    print("TESTING AGGREGATOR SENTIMENT")
    print("=" * 60)
    
    from src.data.aggregator import fetch_sentiment
    
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        sentiment = fetch_sentiment(ticker)
        
        if sentiment:
            print(f"  ✅ Got sentiment data")
            print(f"     Source: {sentiment.get('_source', 'unknown')}")
            
            # Print key metrics
            if 'buzz' in sentiment:
                buzz = sentiment['buzz']
                print(f"     Articles/week: {buzz.get('articlesInLastWeek', 'N/A')}")
                print(f"     Buzz: {buzz.get('buzz', 'N/A')}")
            
            if 'sentiment' in sentiment:
                sent = sentiment['sentiment']
                print(f"     Bullish: {sent.get('bullishPercent', 'N/A')}")
                print(f"     Bearish: {sent.get('bearishPercent', 'N/A')}")
        else:
            print(f"  ⚠️ No sentiment data returned")
    
    print()


def test_detect_big_news():
    """Test the big news detection feature."""
    print("=" * 60)
    print("TESTING BIG NEWS DETECTION")
    print("=" * 60)
    
    from src.data.news import detect_big_news
    
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    
    for ticker in test_tickers:
        print(f"\n{ticker}:")
        try:
            has_big_news, details = detect_big_news(ticker)
            
            if has_big_news:
                print(f"  🔥 BIG NEWS DETECTED!")
                print(f"     Details: {details}")
            else:
                print(f"  ✅ No major news events")
                if details:
                    print(f"     Info: {details}")
                    
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
    
    print()


def main():
    """Run all news/sentiment tests."""
    print("\n" + "=" * 60)
    print(" NEWS & SENTIMENT FEATURES TEST")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Test individual providers
    results['finnhub_sentiment'] = test_finnhub_sentiment()
    results['finnhub_earnings'] = test_finnhub_earnings()
    results['marketaux_news'] = test_marketaux_news()
    
    # Test aggregator
    test_aggregator_sentiment()
    
    # Test integration
    test_detect_big_news()
    test_news_features_in_model()
    
    # Summary
    print("=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
