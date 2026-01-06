"""
News and sentiment data from Marketaux and Alpha Vantage.
"""
import os
import requests
from typing import List, Dict, Optional


def _get_api_key(name: str) -> str:
    """Get API key dynamically (works with Streamlit secrets at runtime)."""
    try:
        import streamlit as st
        key = st.secrets.get(name, "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get(name, "")


# ============================================================================
# MARKETAUX
# ============================================================================

def get_news_marketaux(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Fetch recent news from Marketaux API.
    
    Returns:
        List of {title, source, url, published_at, sentiment}
    """
    api_key = _get_api_key("MARKETAUX_API_KEY")
    if not api_key:
        return []
    
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": ticker,
        "language": "en",
        "filter_entities": "true",
        "api_token": api_key,
        "limit": limit,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        articles = []
        for item in data.get("data", []):
            article = {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "sentiment": None,
            }
            
            # Extract sentiment for this entity
            for entity in item.get("entities", []):
                if entity.get("symbol", "").upper() == ticker.upper():
                    article["sentiment"] = entity.get("sentiment_score")
                    break
            
            articles.append(article)
        
        return articles
    
    except Exception as e:
        print(f"[get_news_marketaux] Failed for {ticker}: {e}")
        return []


# ============================================================================
# ALPHA VANTAGE
# ============================================================================

def get_news_alphavantage(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Fetch news from Alpha Vantage News API.
    
    Returns:
        List of {title, source, url, published_at, sentiment}
    """
    api_key = _get_api_key("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return []
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key,
        "limit": limit,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        articles = []
        for item in data.get("feed", [])[:limit]:
            article = {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published_at": item.get("time_published"),
                "sentiment": None,
            }
            
            # Extract sentiment for this ticker
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == ticker.upper():
                    try:
                        article["sentiment"] = float(ts.get("ticker_sentiment_score", 0))
                    except (ValueError, TypeError):
                        pass
                    break
            
            articles.append(article)
        
        return articles
    
    except Exception as e:
        print(f"[get_news_alphavantage] Failed for {ticker}: {e}")
        return []


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def get_news(ticker: str, limit: int = 5) -> List[Dict]:
    """
    Get news from available sources, trying Marketaux first.
    
    Returns:
        List of news articles with sentiment
    """
    # Try Marketaux first
    articles = get_news_marketaux(ticker, limit)
    
    # Fallback to Alpha Vantage
    if not articles:
        articles = get_news_alphavantage(ticker, limit)
    
    return articles


def get_sentiment_score(ticker: str, lookback_days: int = 7) -> Optional[float]:
    """
    Get average sentiment score from recent news.
    
    Returns:
        Average sentiment (-1 to 1) or None if no data
    """
    articles = get_news(ticker, limit=10)
    
    if not articles:
        return None
    
    sentiments = [a["sentiment"] for a in articles if a.get("sentiment") is not None]
    
    if not sentiments:
        return None
    
    return sum(sentiments) / len(sentiments)


def detect_significant_news(articles: List[Dict], sentiment_threshold: float = 0.5) -> bool:
    """
    Detect if there's significant news (earnings, M&A, etc.).
    
    Returns:
        True if significant news detected
    """
    if not articles:
        return False
    
    keywords = [
        "earnings", "guidance", "downgrade", "upgrade",
        "lawsuit", "investigation", "merger", "acquisition",
        "bankruptcy", "sec charges", "fraud", "buyback",
    ]
    
    for article in articles:
        title = (article.get("title") or "").lower()
        sentiment = article.get("sentiment")
        
        # Check for significant keywords
        if any(k in title for k in keywords):
            return True
        
        # Check for extreme sentiment
        if sentiment is not None and abs(sentiment) >= sentiment_threshold:
            return True
    
    return False


# Aliases for backward compatibility
def get_news_for_ticker(ticker: str, limit: int = 5) -> List[Dict]:
    """Alias for get_news."""
    return get_news(ticker, limit)


def get_news_sentiment(ticker: str, lookback_days: int = 7) -> Optional[float]:
    """Alias for get_sentiment_score."""
    return get_sentiment_score(ticker, lookback_days)


def detect_big_news(articles: List[Dict], sentiment_threshold: float = 0.5) -> bool:
    """Alias for detect_significant_news."""
    return detect_significant_news(articles, sentiment_threshold)
