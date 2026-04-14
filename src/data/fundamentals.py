"""
Fundamentals data fetching from FMP API.
"""
import os
import requests
from typing import Dict, Optional

from ..config import FMP_API_KEY
from .cache_manager import get_cache

_FMP_SESSION = requests.Session()
_FUND_CACHE = get_cache()
_FUND_MEMO: Dict[str, Dict] = {}


# ============================================================================
# FMP API
# ============================================================================

def get_fundamentals(ticker: str, api_key: Optional[str] = None) -> Dict:
    """
    Fetch key fundamentals from Financial Modeling Prep.
    
    Returns:
        Dict with fund_pe_trailing, fund_pb, fund_marketcap
    """
    cached = _FUND_CACHE.get("fundamentals", ticker)
    if cached is not None:
        _FUND_MEMO[ticker] = cached.copy() if hasattr(cached, "copy") else cached
        return cached.copy() if hasattr(cached, "copy") else cached

    api_key = api_key or FMP_API_KEY
    
    if not api_key:
        return {}
    
    base_url = "https://financialmodelingprep.com/stable"
    params = {"symbol": ticker, "apikey": api_key}
    
    out = {}
    
    try:
        # Fetch ratios
        ratios_resp = _FMP_SESSION.get(f"{base_url}/ratios", params={**params, "period": "annual", "limit": 1}, timeout=15)
        if ratios_resp.status_code == 200:
            ratios = ratios_resp.json()
            if isinstance(ratios, list) and ratios:
                r0 = ratios[0]
                # FMP has changed field names over time — check all variants
                out["fund_pe_trailing"] = (
                    r0.get("peRatio")
                    or r0.get("priceEarningsRatio")
                    or r0.get("peRatioTTM")
                )
                out["fund_pb"] = (
                    r0.get("priceToBookRatio")
                    or r0.get("pbRatio")
                    or r0.get("priceToBookRatioTTM")
                )
        
        # Fetch key metrics
        metrics_resp = _FMP_SESSION.get(f"{base_url}/key-metrics", params={**params, "period": "annual", "limit": 1}, timeout=15)
        if metrics_resp.status_code == 200:
            metrics = metrics_resp.json()
            if isinstance(metrics, list) and metrics:
                m0 = metrics[0]
                out["fund_marketcap"] = m0.get("marketCap")
                # Fill PE/PB from metrics if ratios endpoint missed them
                if not out.get("fund_pe_trailing"):
                    out["fund_pe_trailing"] = m0.get("peRatio") or m0.get("peRatioTTM")
                if not out.get("fund_pb"):
                    out["fund_pb"] = m0.get("pbRatio") or m0.get("priceToBookRatio")
    
    except Exception as e:
        print(f"[get_fundamentals] FMP failed for {ticker}: {e}")
    
    if out:
        _FUND_CACHE.set("fundamentals", ticker, out)
        _FUND_MEMO[ticker] = out.copy()

    return out


def get_pe_ratio(ticker: str) -> Optional[float]:
    """Get trailing P/E ratio."""
    data = get_fundamentals(ticker)
    return data.get("fund_pe_trailing")


def get_pb_ratio(ticker: str) -> Optional[float]:
    """Get P/B ratio."""
    data = get_fundamentals(ticker)
    return data.get("fund_pb")


def get_market_cap(ticker: str) -> Optional[float]:
    """Get market cap."""
    data = get_fundamentals(ticker)
    return data.get("fund_marketcap")


# ============================================================================
# FALLBACK TO YFINANCE
# ============================================================================

def get_fundamentals_yf(ticker: str) -> Dict:
    """
    Fallback: Get fundamentals from yfinance.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        trailing_eps = (
            info.get("trailingEps")
            or info.get("epsTrailingTwelveMonths")
        )
        book_value = info.get("bookValue")
        shares_outstanding = info.get("sharesOutstanding")

        trailing_pe = info.get("trailingPE")
        if trailing_pe in (None, 0) and current_price and trailing_eps:
            try:
                trailing_pe = float(current_price) / float(trailing_eps)
            except Exception:
                trailing_pe = None

        price_to_book = info.get("priceToBook")
        if price_to_book in (None, 0) and current_price and book_value:
            try:
                price_to_book = float(current_price) / float(book_value)
            except Exception:
                price_to_book = None

        market_cap = info.get("marketCap")
        if market_cap in (None, 0) and current_price and shares_outstanding:
            try:
                market_cap = float(current_price) * float(shares_outstanding)
            except Exception:
                market_cap = None
        
        return {
            "fund_pe_trailing": trailing_pe,
            "fund_pb": price_to_book,
            "fund_marketcap": market_cap,
        }
    except Exception:
        return {}


def get_fundamentals_with_fallback(ticker: str) -> Dict:
    """
    Get fundamentals, trying FMP first then yfinance.
    """
    data = get_fundamentals(ticker)
    
    # If FMP returned empty, try yfinance
    if not data or all(v is None for v in data.values()):
        data = get_fundamentals_yf(ticker)
    
    # Supplement any missing key metrics from yfinance
    key_metrics = ("fund_pe_trailing", "fund_pb", "fund_marketcap")
    if not data:
        data = {}
    if any(data.get(metric) in (None, 0, 0.0) for metric in key_metrics):
        yf_data = get_fundamentals_yf(ticker)
        for metric in key_metrics:
            if data.get(metric) in (None, 0, 0.0) and yf_data.get(metric) not in (None, 0, 0.0):
                data[metric] = yf_data[metric]
    
    # Last-resort provider chain with its own caching/fallbacks
    if any(data.get(metric) in (None, 0, 0.0) for metric in key_metrics):
        try:
            from .aggregator import fetch_fundamentals
            agg_data = fetch_fundamentals(ticker)
            for metric in key_metrics:
                if data.get(metric) in (None, 0, 0.0) and agg_data.get(metric) not in (None, 0, 0.0):
                    data[metric] = agg_data[metric]
        except Exception:
            pass
    
    normalized = {
        "fund_pe_trailing": data.get("fund_pe_trailing") or 0.0,
        "fund_pb": data.get("fund_pb") or 0.0,
        "fund_marketcap": data.get("fund_marketcap") or 0.0,
    }
    
    if any(normalized.values()):
        _FUND_CACHE.set("fundamentals", ticker, normalized)
        _FUND_MEMO[ticker] = normalized.copy()
    
    return normalized


# Alias for backward compatibility
def get_fundamental_features(ticker: str) -> Dict:
    """Alias for get_fundamentals_with_fallback."""
    return get_fundamentals_with_fallback(ticker)
