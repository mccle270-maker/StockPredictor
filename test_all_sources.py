#!/usr/bin/env python3
"""Test every data source live to see what's working and what's broken."""

import os, sys, time, json, warnings, signal as _signal
warnings.filterwarnings("ignore")

class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Test timed out after 30s")

# Set per-test timeout
_signal.signal(_signal.SIGALRM, _timeout_handler)

# Load env
from dotenv import load_dotenv
load_dotenv(".env.apis")

# Make sure src is importable
sys.path.insert(0, os.path.dirname(__file__))

results = {}

def test_source(name, fn):
    """Run a test function and record result."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        _signal.alarm(45)  # 45-second hard timeout per test
        data = fn()
        _signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - t0
        if data is None:
            results[name] = {"status": "FAIL", "reason": "returned None", "time": f"{elapsed:.1f}s"}
            print(f"  ❌ FAIL: returned None ({elapsed:.1f}s)")
        elif hasattr(data, '__len__') and len(data) == 0:
            results[name] = {"status": "FAIL", "reason": "returned empty", "time": f"{elapsed:.1f}s"}
            print(f"  ❌ FAIL: returned empty ({elapsed:.1f}s)")
        else:
            size = len(data) if hasattr(data, '__len__') else "N/A"
            results[name] = {"status": "OK", "size": size, "time": f"{elapsed:.1f}s"}
            print(f"  ✅ OK: size={size} ({elapsed:.1f}s)")
            if hasattr(data, 'head'):
                print(f"  Columns: {list(data.columns)[:10]}")
                print(f"  Shape: {data.shape}")
                print(f"  Date range: {data.index.min()} → {data.index.max()}" if hasattr(data.index, 'min') else "")
            elif isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:15]}")
            return data
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = {"status": "ERROR", "reason": str(e)[:200], "time": f"{elapsed:.1f}s"}
        print(f"  💥 ERROR: {str(e)[:200]} ({elapsed:.1f}s)")
    return None


# ============================================================
# 1. ALPACA — Price History (SKIPPED — under maintenance)
# ============================================================
print("\n" + "="*60)
print("SKIPPING: Alpaca Prices (API under maintenance)")
print("="*60)
results["Alpaca Prices"] = {"status": "SKIP", "reason": "API under maintenance", "time": "0s"}

# ============================================================
# 2. YFINANCE — Price History
# ============================================================
def test_yfinance_prices():
    from src.data.providers.yfinance_provider import YFinanceProvider
    p = YFinanceProvider()
    return p.get_price_history("AAPL", period="6mo")

test_source("yfinance Prices", test_yfinance_prices)

# ============================================================
# 3. YFINANCE — Fundamentals
# ============================================================
def test_yfinance_fundamentals():
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    info = tk.info
    if not info or info.get("trailingPE") is None:
        # Check what we actually got
        print(f"  Raw info keys: {list(info.keys())[:20]}")
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is None:
            raise RuntimeError(f"No P/E in info. Keys: {list(info.keys())[:20]}")
    return info

test_source("yfinance Fundamentals", test_yfinance_fundamentals)

# ============================================================
# 4. YFINANCE — Options
# ============================================================
def test_yfinance_options():
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    dates = tk.options
    if not dates:
        raise RuntimeError("No options dates returned")
    chain = tk.option_chain(dates[0])
    return chain.calls

test_source("yfinance Options", test_yfinance_options)

# ============================================================
# 5. FRED — Macro Data
# ============================================================
def test_fred():
    from src.data.macro import get_macro_df
    return get_macro_df()

test_source("FRED Macro", test_fred)

# ============================================================
# 6. FRED — Individual Series (VIX)
# ============================================================
def test_fred_vix():
    import requests
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={key}&file_type=json&sort_order=desc&limit=5"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError("No observations returned")
    print(f"  Latest VIX: {obs[0].get('value')} on {obs[0].get('date')}")
    return obs

test_source("FRED VIX Direct", test_fred_vix)

# ============================================================
# 7. FMP — Fundamentals
# ============================================================
def test_fmp():
    import requests
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError("FMP_API_KEY not set")
    url = f"https://financialmodelingprep.com/stable/ratios?symbol=AAPL&period=annual&limit=1&apikey={key}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) > 0:
        print(f"  P/E: {data[0].get('peRatio', data[0].get('priceEarningsRatio', 'N/A'))}")
        print(f"  P/B: {data[0].get('priceToBookRatio', 'N/A')}")
    return data

test_source("FMP Fundamentals", test_fmp)

# ============================================================
# 8. TIINGO — Prices
# ============================================================
def test_tiingo():
    from src.data.providers.tiingo_provider import TiingoProvider
    p = TiingoProvider()
    if not p.is_available():
        raise RuntimeError("Tiingo provider not available (no API key?)")
    return p.get_price_history("AAPL", period="6mo")

test_source("Tiingo Prices", test_tiingo)

# ============================================================
# 9. FINNHUB — Company News
# ============================================================
def test_finnhub_news():
    from src.data.providers.finnhub_provider import FinnhubProvider
    p = FinnhubProvider()
    if not p.is_available():
        raise RuntimeError("Finnhub provider not available (no API key?)")
    return p.get_news("AAPL")

test_source("Finnhub News", test_finnhub_news)

# ============================================================
# 10. FINNHUB — Sentiment (insider + analyst)
# ============================================================
def test_finnhub_sentiment():
    from src.data.providers.finnhub_provider import FinnhubProvider
    p = FinnhubProvider()
    if not p.is_available():
        raise RuntimeError("Finnhub provider not available (no API key?)")
    return p.get_sentiment("AAPL")

test_source("Finnhub Sentiment", test_finnhub_sentiment)

# ============================================================
# 11. FINNHUB — Earnings
# ============================================================
def test_finnhub_earnings():
    from src.data.providers.finnhub_provider import FinnhubProvider
    p = FinnhubProvider()
    if not p.is_available():
        raise RuntimeError("Finnhub provider not available (no API key?)")
    return p.get_earnings("AAPL")

test_source("Finnhub Earnings", test_finnhub_earnings)

# ============================================================
# 12. SEC EDGAR — Fundamentals
# ============================================================
def test_sec_edgar():
    from src.data.providers.sec_edgar_provider import SECEdgarProvider
    p = SECEdgarProvider()
    return p.get_fundamentals("AAPL")

test_source("SEC EDGAR Fundamentals", test_sec_edgar)

# ============================================================
# 13. ALPHA VANTAGE — Prices
# ============================================================
def test_alphavantage_prices():
    from src.data.providers.alphavantage_provider import AlphaVantageProvider
    p = AlphaVantageProvider()
    if not p.is_available():
        raise RuntimeError("Alpha Vantage provider not available (no API key?)")
    return p.get_price_history("AAPL", period="6mo")

test_source("Alpha Vantage Prices", test_alphavantage_prices)

# ============================================================
# 14. ALPHA VANTAGE — News
# ============================================================
def test_alphavantage_news():
    from src.data.news import get_alphavantage_news
    return get_alphavantage_news("AAPL")

test_source("Alpha Vantage News", test_alphavantage_news)

# ============================================================
# 15. MARKETAUX — News
# ============================================================
def test_marketaux():
    from src.data.news import get_news_articles
    return get_news_articles("AAPL")

test_source("Marketaux News", test_marketaux)

# ============================================================
# 16. STOOQ — Prices
# ============================================================
def test_stooq():
    import pandas as pd
    import io, requests
    r = requests.get("https://stooq.com/q/d/l/?s=aapl.us&i=d", timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise RuntimeError("Empty CSV from Stooq")
    return df

test_source("Stooq Prices", test_stooq)

# ============================================================
# 17. AGGREGATOR — Full Pipeline (fetch_prices)
# ============================================================
def test_aggregator_prices():
    from src.data.aggregator import get_aggregator
    agg = get_aggregator()
    return agg.fetch_prices("AAPL", period="6mo")

test_source("Aggregator fetch_prices", test_aggregator_prices)

# ============================================================
# 18. AGGREGATOR — Sentiment
# ============================================================
def test_aggregator_sentiment():
    from src.data.aggregator import fetch_sentiment
    return fetch_sentiment("AAPL")

test_source("Aggregator fetch_sentiment", test_aggregator_sentiment)

# ============================================================
# 19. AGGREGATOR — Fundamentals
# ============================================================
def test_aggregator_fundamentals():
    from src.data.aggregator import fetch_fundamentals
    return fetch_fundamentals("AAPL")

test_source("Aggregator fetch_fundamentals", test_aggregator_fundamentals)

# ============================================================
# 20. Full prediction pipeline
# ============================================================
def test_prediction():
    from prediction_model import build_features_and_target
    return build_features_and_target("AAPL", period="1y")

test_source("Full Prediction Pipeline", test_prediction)


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
ok = []
fail = []
error = []
skip = []
for name, r in results.items():
    status = r["status"]
    if status == "OK":
        ok.append(name)
        print(f"  ✅ {name}: size={r.get('size','?')} ({r['time']})")
    elif status == "SKIP":
        skip.append(name)
        print(f"  ⏭️  {name}: {r['reason']}")
    elif status == "FAIL":
        fail.append(name)
        print(f"  ❌ {name}: {r['reason']} ({r['time']})")
    else:
        error.append(name)
        print(f"  💥 {name}: {r['reason'][:100]} ({r['time']})")

print(f"\n  PASS:  {len(ok)}/{len(results)}")
print(f"  SKIP:  {len(skip)}/{len(results)}")
print(f"  FAIL:  {len(fail)}/{len(results)}")
print(f"  ERROR: {len(error)}/{len(results)}")

# Save results
with open("data_source_test_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to data_source_test_results.json")
