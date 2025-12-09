# stock_screener.py
import yfinance as yf
import pandas as pd
from data_fetch import get_history


def compute_basic_signals(ticker, lookback_days=30):
    """
    Compute simple signals for a single ticker:
    - recent return over lookback_days
    - volume spike vs average
    - 20-day realized volatility
    """
    hist = get_history(ticker, period=f"{lookback_days}d", interval="1d")
    if hist.empty or len(hist) < 5:
        return None

    hist["ret"] = hist["Close"].pct_change()
    recent_ret = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
    vol_spike = hist["Volume"].iloc[-1] / hist["Volume"].mean()
    vol_20d = hist["ret"].rolling(20).std().iloc[-1]

    t = yf.Ticker(ticker)
    try:
        cal = t.calendar  # small earnings/calendar DataFrame from Yahoo [web:86]
        next_earnings = (
            cal.loc["Earnings Date"].iloc[0]
            if "Earnings Date" in cal.index
            else None
        )
    except Exception:
        next_earnings = None

    return {
        "ticker": ticker,
        "recent_return": recent_ret,
        "volume_spike": vol_spike,
        "vol_20d": vol_20d,
        "next_earnings": next_earnings,
    }


def screen_stocks(tickers, ret_thresh=0.03, vol_spike_thresh=1.5):
    """
    Score a list of tickers and flag the ones likely to move.
    """
    rows = []
    for tk in tickers:
        sig = compute_basic_signals(tk)
        if sig is None:
            continue
        sig["flag"] = (
            abs(sig["recent_return"]) >= ret_thresh
            or sig["volume_spike"] >= vol_spike_thresh
        )
        rows.append(sig)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["flag", "volume_spike"], ascending=[False, False])
    return df


if __name__ == "__main__":
    watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY"]
    df = screen_stocks(watchlist)
    print(df.to_string(index=False))