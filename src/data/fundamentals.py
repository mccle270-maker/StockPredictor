"""
Fundamentals data fetching from FMP API.
"""
import os
import requests
from typing import Dict, Optional

from ..config import FMP_API_KEY


# ============================================================================
# FMP API
# ============================================================================

def get_fundamentals(ticker: str, api_key: Optional[str] = None) -> Dict:
    """
    Fetch key fundamentals from Financial Modeling Prep.
    
    Returns:
        Dict with fund_pe_trailing, fund_pb, fund_marketcap
    """
    api_key = api_key or FMP_API_KEY
    
    if not api_key:
        return {}
    
    base_url = "https://financialmodelingprep.com/stable"
    params = {"symbol": ticker, "apikey": api_key}
    
    out = {}
    
    try:
        # Fetch ratios
        ratios_resp = requests.get(f"{base_url}/ratios", params=params, timeout=5)
        if ratios_resp.status_code == 200:
            ratios = ratios_resp.json()
            if isinstance(ratios, list) and ratios:
                r0 = ratios[0]
                out["fund_pe_trailing"] = r0.get("priceEarningsRatio")
                out["fund_pb"] = r0.get("priceToBookRatio")
        
        # Fetch key metrics
        metrics_resp = requests.get(f"{base_url}/key-metrics", params=params, timeout=5)
        if metrics_resp.status_code == 200:
            metrics = metrics_resp.json()
            if isinstance(metrics, list) and metrics:
                m0 = metrics[0]
                out["fund_marketcap"] = m0.get("marketCap")
    
    except Exception as e:
        print(f"[get_fundamentals] FMP failed for {ticker}: {e}")
    
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
        
        return {
            "fund_pe_trailing": info.get("trailingPE"),
            "fund_pb": info.get("priceToBook"),
            "fund_marketcap": info.get("marketCap"),
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
    
    # Convert None to 0 for consistency
    return {
        "fund_pe_trailing": data.get("fund_pe_trailing") or 0.0,
        "fund_pb": data.get("fund_pb") or 0.0,
        "fund_marketcap": data.get("fund_marketcap") or 0.0,
    }


# Alias for backward compatibility
def get_fundamental_features(ticker: str) -> Dict:
    """Alias for get_fundamentals_with_fallback."""
    return get_fundamentals_with_fallback(ticker)
