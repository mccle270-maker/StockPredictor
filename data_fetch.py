# data_fetch.py
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from scipy.stats import norm
from yfinance.exceptions import YFRateLimitError


# -------- News API keys (Marketaux + Alpha Vantage) --------

MARKETAUX_API_KEY = st.secrets.get("MARKETAUX_API_KEY", None)
ALPHAVANTAGE_API_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY", None)

# Optional: env fallback for local python3 runs
if MARKETAUX_API_KEY is None:
    MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY")
if ALPHAVANTAGE_API_KEY is None:
    ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")


def get_news_from_marketaux(ticker, limit=5):
    """
    Fetch recent finance news headlines for a ticker using Marketaux.
    Returns a list of {title, source, url, published_at, sentiment}.
    """
    if not MARKETAUX_API_KEY:
        return []

    base_url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": ticker,
        "language": "en",
        "filter_entities": "true",
        "api_token": MARKETAUX_API_KEY,
        "limit": limit,
    }

    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Marketaux news fetch error for {ticker}: {e}")
        return []

    articles = []
    for item in data.get("data", []):
        articles.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published_at": item.get("published_at"),
                "sentiment": item.get("sentiment"),
            }
        )
    return articles


def get_news_from_alphavantage(ticker: str, limit: int = 3):
    """
    Fetch recent news for a ticker from Alpha Vantage's NEWS_SENTIMENT endpoint.
    Returns a list of {title, source, url, published_at, sentiment}.
    """
    if not ALPHAVANTAGE_API_KEY:
        return []

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "sort": "LATEST",
        "limit": limit,
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Alpha Vantage news fetch error for {ticker}: {e}")
        return []

    if "feed" not in data:
        return []

    articles = []
    for item in data["feed"][:limit]:
        articles.append(
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "url": item.get("url"),
                "published_at": item.get("time_published"),
                "sentiment": item.get("overall_sentiment_score"),
            }
        )
    return articles


@st.cache_data(ttl=900)  # cache for 15 minutes
def get_news_for_ticker(ticker, limit=5):
    """
    Unified news fetcher with fallback:
    1) Try Marketaux
    2) If that fails/empty, try Alpha Vantage
    3) If both fail, return []
    """
    # 1) Marketaux
    arts = get_news_from_marketaux(ticker, limit=limit)
    if arts:
        return arts

    # 2) Alpha Vantage fallback
    arts = get_news_from_alphavantage(ticker, limit=limit)
    if arts:
        return arts

    # 3) Nothing worked
    return []


# -------- Price history (with caching) --------

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


# -------- Options snapshot features --------

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


# ---------- Black-Scholes Greeks helpers ----------

def _bs_greeks(flag, S, K, T, r, sigma):
    """
    Basic Black-Scholes Greeks for a European option.
    flag: 'c' for call, 'p' for put
    S: spot price
    K: strike
    T: time to expiry in years
    r: risk-free rate (e.g. 0.04 for 4%)
    sigma: implied volatility (e.g. 0.35 for 35%)
    Returns dict with delta, gamma, vega, theta.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": None, "gamma": None, "vega": None, "theta": None}

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if flag == "c":
        delta = norm.cdf(d1)
        theta = (
            - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        )
    else:  # put
        delta = norm.cdf(d1) - 1.0
        theta = (
            - (S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        )

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def get_atm_greeks(ticker, risk_free_rate=0.04):
    """
    Compute approximate ATM call & put Greeks for the nearest option expiry
    using Yahoo Finance option chain + Black-Scholes.
    """
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        return None

    # nearest expiry
    expiry_str = expiries[0]
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except Exception:
        return None

    today = datetime.utcnow().date()
    days_to_expiry = (expiry_date - today).days
    if days_to_expiry <= 0:
        return None

    T = days_to_expiry / 365.0

    try:
        chain = t.option_chain(expiry_str)
    except YFRateLimitError:
        # let caller decide how to handle rate limit
        raise
    except Exception:
        return None

    calls = chain.calls.copy()
    puts = chain.puts.copy()
    if calls.empty or puts.empty:
        return None

    # spot from most recent close
    hist = t.history(period="1d", interval="1d")
    if hist is None or hist.empty:
        return None
    spot = float(hist["Close"].iloc[-1])

    # find strike closest to spot
    calls["dist"] = (calls["strike"] - spot).abs()
    puts["dist"] = (puts["strike"] - spot).abs()
    call_row = calls.sort_values("dist").iloc[0]
    put_row = puts.sort_values("dist").iloc[0]

    call_iv = float(call_row.get("impliedVolatility", np.nan))
    put_iv = float(put_row.get("impliedVolatility", np.nan))
    if not np.isfinite(call_iv) or not np.isfinite(put_iv):
        return None

    K_call = float(call_row["strike"])
    K_put = float(put_row["strike"])

    call_greeks = _bs_greeks("c", spot, K_call, T, risk_free_rate, call_iv)
    put_greeks = _bs_greeks("p", spot, K_put, T, risk_free_rate, put_iv)

    return {
        "expiry": expiry_str,
        "spot": spot,
        "K_call": K_call,
        "K_put": K_put,
        "call_iv": call_iv,
        "put_iv": put_iv,
        "call_greeks": call_greeks,
        "put_greeks": put_greeks,
    }
