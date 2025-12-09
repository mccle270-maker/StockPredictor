# data_fetch.py
import numpy as np
import pandas as pd
import yfinance as yf


def get_history(ticker, period="1y", interval="1d"):
    """
    Download OHLCV history for a ticker.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval)
    return hist


def get_option_chain(ticker, expiration=None, calls_only=True):
    """
    Get an options chain for a given ticker and expiration date.
    If expiration is None, use the nearest available expiration.
    """
    t = yf.Ticker(ticker)

    expirations = t.options  # list of strings returned by yfinance [web:42][web:45]
    if not expirations:
        raise ValueError(f"No options available for {ticker}")

    if expiration is None:
        expiration = expirations[0]

    chain = t.option_chain(expiration)  # returns (calls, puts) DataFrames [web:42][web:45]
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

    This is designed for LIVE use (today's snapshot), not historical backtesting. [web:42][web:45]
    """
    t = yf.Ticker(ticker)

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
    except Exception:
        return {
            "opt_exp": expiration,
            "atm_iv": None,
            "put_call_oi_ratio": None,
        }

    # Approximate underlying price from recent close
    hist = t.history(period="5d", interval="1d")
    underlying = None
    if not hist.empty:
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
