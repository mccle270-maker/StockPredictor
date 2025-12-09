# stock_screener.py
import yfinance as yf
import pandas as pd
from data_fetch import get_history_cached


def compute_basic_signals(ticker, lookback_days=30):
    """
    Compute simple signals for a single ticker.
    """
    try:
        from data_fetch import get_history_cached
        hist = get_history_cached(ticker, period=f"{lookback_days}d", interval="1d")
        if hist.empty or len(hist) < 5:
            return None

        hist["ret"] = hist["Close"].pct_change()
        recent_ret = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
        vol_spike = hist["Volume"].iloc[-1] / hist["Volume"].mean()
        vol_20d = hist["ret"].rolling(20).std().iloc[-1]

        # Try to get earnings date (with error handling)
        next_earnings = None
        days_to_earnings = None
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is not None and 'Earnings Date' in cal.index:
                earnings_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
                days_to_earnings = (earnings_date - pd.Timestamp.now()).days
                if days_to_earnings >= 0:  # Only if earnings is in the future
                    next_earnings = earnings_date.strftime("%Y-%m-%d")
        except:
            pass  # Silently fail if earnings date not available

        return {
            "ticker": ticker,
            "recent_return": recent_ret,
            "volume_spike": vol_spike,
            "vol_20d": vol_20d,
            "next_earnings": next_earnings,
            "days_to_earnings": days_to_earnings,
        }
    except Exception as e:
        print(f"Error computing signals for {ticker}: {e}")
        return None


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