# data_fetch.py
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests
from yfinance.exceptions import YFRateLimitError

@st.cache_data(ttl=900)  # cache for 15 minutes
def get_news_for_ticker(ticker, limit=5):
    """
    Fetch recent finance news headlines for a ticker using Marketaux (or a similar API).
    Requires MARKETAUX_API_KEY in st.secrets.
    Returns a list of article dicts with title, source, url, published_at.
    """
    api_key = st.secrets.get("MARKETAUX_API_KEY")
    if not api_key:
        # No key configured; fail gracefully
        return []

    base_url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": ticker,
        "language": "en",
        "filter_entities": "true",
        "api_token": api_key,
        "limit": limit,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"News fetch error for {ticker}: {e}")
        return []

    articles = []
    for item in data.get("data", []):
        articles.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "sentiment": item.get("sentiment"),  # Marketaux often provides this
            }
        )
    return articles

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_history_cached(ticker, period="1y", interval="1d"):
    return get_history(ticker, period, interval)


def get_history(ticker, period="1y", interval="1d"):
    """
    Download OHLCV history for a ticker.
    For charts we handle rate limits by returning an empty DataFrame
    instead of crashing the whole app.
    """
    t = yf.Ticker(ticker)
    try:
        hist = t.history(period=period, interval=interval)
    except YFRateLimitError:
        st.warning(
            f"Yahoo Finance rate limited price history for {ticker}. "
            "Chart will be empty; try again later."
        )
        return pd.DataFrame()
    return hist


def get_option_chain(ticker, expiration=None, calls_only=True):
    """
    Get an options chain for a given ticker and expiration date.
    If expiration is None, use the nearest available expiration.
    """
    t = yf.Ticker(ticker)

    # Let YFRateLimitError propagate so app.py can stop the loop
    expirations = t.options
    if not expirations:
        raise ValueError(f"No options available for {ticker}")

    if expiration is None:
        expiration = expirations[0]

    chain = t.option_chain(expiration)
    if calls_only:
        df = chain.calls.copy()
    else:
        df = pd.concat([chain.calls, chain.puts], ignore_index=True)

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["expiration"] = expiration
    return df


def get_option_snapshot_features(ticker, moneyness_window=0.05):
    """
    Pull a single-snapshot options feature set for a ticker:
    - nearest expiration
    - approximate ATM implied vol (average IV near current price)
    - put/call open interest ratio

    Designed for LIVE use (today's snapshot), not historical backtesting.
    """
    t = yf.Ticker(ticker)

    # Let YFRateLimitError bubble up so app.py can catch and break
    expirations = t.options
    if not expirations:
        return {
            "opt_exp": None,
            "atm_iv": None,
            "put_call_oi_ratio": None,
        }

    expiration = expirations[0]

    try:
        chain = t.option_chain(expiration)
        calls, puts = chain.calls.copy(), chain.puts.copy()
    except YFRateLimitError:
        # propagate to app.py
        raise
    except Exception:
        return {
            "opt_exp": expiration,
            "atm_iv": None,
            "put_call_oi_ratio": None,
        }

    # Approximate underlying price from recent close
    try:
        hist = t.history(period="5d", interval="1d")
    except YFRateLimitError:
        # propagate so app knows we're being throttled
        raise

    underlying = None
    if hist is not None and not hist.empty:
        underlying = float(hist["Close"].iloc[-1])

    if underlying is not None:
        low = underlying * (1 - moneyness_window)
        high = underlying * (1 + moneyness_window)
        calls_atm = calls[(calls["strike"] >= low) & (calls["strike"] <= high)]
        puts_atm = puts[(puts["strike"] >= low) & (puts["strike"] <= high)]
    else:
        calls_atm = calls
        puts_atm = puts

    # ATM implied vol estimate from nearby strikes
    atm_iv = None
    ivs = []
    if "impliedVolatility" in calls_atm.columns:
        ivs.extend(calls_atm["impliedVolatility"].dropna().tolist())
    if "impliedVolatility" in puts_atm.columns:
        ivs.extend(puts_atm["impliedVolatility"].dropna().tolist())
    if ivs:
        atm_iv = float(np.mean(ivs))

    # Put/Call open interest ratio (all strikes)
    put_call_oi_ratio = None
    if "openInterest" in calls.columns and "openInterest" in puts.columns:
        call_oi = float(calls["openInterest"].fillna(0).sum())
        put_oi = float(puts["openInterest"].fillna(0).sum())
        if call_oi + put_oi > 0:
            put_call_oi_ratio = put_oi / (call_oi + 1e-9)

    return {
        "opt_exp": expiration,
        "atm_iv": atm_iv,
        "put_call_oi_ratio": put_call_oi_ratio,
    }
