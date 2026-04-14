#!/usr/bin/env python3
"""Test every data source in isolated subprocesses to avoid cross-contamination."""

import subprocess, sys, json, time, os

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 60  # seconds per test

results = {}

def run_test(name, code):
    """Run a test in an isolated subprocess."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Wrap user code in a try/except that prints JSON result
    wrapped = f'''
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
os.chdir("{BASE}")
sys.path.insert(0, "{BASE}")
from dotenv import load_dotenv
load_dotenv(".env.apis")

t0 = time.time()
try:
{code}
    elapsed = time.time() - t0
    print("__RESULT__" + json.dumps({{"status": "OK", "info": info, "time": f"{{elapsed:.1f}}s"}}))
except Exception as e:
    elapsed = time.time() - t0
    print("__RESULT__" + json.dumps({{"status": "ERROR", "reason": str(e)[:300], "time": f"{{elapsed:.1f}}s"}}))
'''
    
    try:
        res = subprocess.run(
            [PYTHON, "-c", wrapped],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=BASE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = res.stdout + res.stderr
        
        # Extract result
        for line in output.split("\n"):
            if line.startswith("__RESULT__"):
                r = json.loads(line[len("__RESULT__"):])
                results[name] = r
                if r["status"] == "OK":
                    print(f"  ✅ OK ({r['time']})")
                    for k, v in r.get("info", {}).items():
                        print(f"     {k}: {v}")
                else:
                    print(f"  💥 ERROR: {r['reason'][:150]} ({r['time']})")
                return
        
        # No result line found
        results[name] = {"status": "ERROR", "reason": f"No result. stdout={res.stdout[-300:]}, stderr={res.stderr[-300:]}", "time": "?"}
        print(f"  💥 ERROR: No result line found")
        if res.stderr:
            print(f"     stderr: {res.stderr[-200:]}")
            
    except subprocess.TimeoutExpired:
        results[name] = {"status": "TIMEOUT", "reason": f"Timed out after {TIMEOUT}s", "time": f"{TIMEOUT}s"}
        print(f"  ⏰ TIMEOUT after {TIMEOUT}s")


# ============================================================
# 1. ALPACA — SKIP
# ============================================================
print(f"\n{'='*60}")
print("SKIPPING: Alpaca (under maintenance)")
print(f"{'='*60}")
results["Alpaca Prices"] = {"status": "SKIP", "reason": "Under maintenance"}

# ============================================================
# 2. YFINANCE — Prices
# ============================================================
run_test("yfinance Prices", '''
    import yfinance as yf
    df = yf.download("AAPL", period="6mo", progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty DataFrame")
    info = {"rows": len(df), "cols": list(df.columns.get_level_values(0).unique())[:6], "date_range": f"{df.index.min().date()} to {df.index.max().date()}"}
''')

# ============================================================
# 3. YFINANCE — Fundamentals
# ============================================================
run_test("yfinance Fundamentals", '''
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    d = tk.info
    if not d:
        raise RuntimeError("yfinance .info returned empty")
    pe = d.get("trailingPE") or d.get("forwardPE", "N/A")
    pb = d.get("priceToBook", "N/A")
    mc = d.get("marketCap", "N/A")
    info = {"PE": pe, "PB": pb, "marketCap": mc, "keys_count": len(d)}
''')

# ============================================================
# 4. YFINANCE — Options
# ============================================================
run_test("yfinance Options", '''
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    dates = tk.options
    if not dates:
        raise RuntimeError("No options expiration dates")
    chain = tk.option_chain(dates[0])
    info = {"expirations": len(dates), "calls": len(chain.calls), "puts": len(chain.puts), "first_expiry": dates[0]}
''')

# ============================================================
# 5. FRED — Macro (via src/data/macro.py)
# ============================================================
run_test("FRED Macro (macro.py)", '''
    from src.data.macro import get_macro_df
    df = get_macro_df()
    if df is None or df.empty:
        raise RuntimeError("get_macro_df returned empty")
    info = {"rows": len(df), "cols": list(df.columns), "date_range": f"{df.index.min().date()} to {df.index.max().date()}", "sample": {c: str(df[c].dropna().iloc[-1]) for c in df.columns[:5]}}
''')

# ============================================================
# 6. FRED — Direct VIX fetch
# ============================================================
run_test("FRED VIX Direct", '''
    import requests
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise RuntimeError("FRED_API_KEY not set")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=VIXCLS&api_key={key}&file_type=json&sort_order=desc&limit=5"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        raise RuntimeError("No observations")
    info = {"latest_vix": obs[0].get("value"), "date": obs[0].get("date"), "count": len(obs)}
''')

# ============================================================
# 7. FMP — Fundamentals
# ============================================================
run_test("FMP Fundamentals", '''
    import requests
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError("FMP_API_KEY not set")
    url = f"https://financialmodelingprep.com/stable/ratios?symbol=AAPL&period=annual&limit=1&apikey={key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("FMP returned empty")
    d = data[0] if isinstance(data, list) else data
    info = {"PE": d.get("peRatio", d.get("priceEarningsRatio", "N/A")), "PB": d.get("priceToBookRatio", "N/A")}
''')

# ============================================================
# 8. TIINGO — Prices
# ============================================================
run_test("Tiingo Prices", '''
    import requests
    key = os.environ.get("TIINGO_API_KEY", "")
    if not key:
        from src.config import get_api_key
        key = get_api_key("TIINGO_API_KEY")
    if not key:
        raise RuntimeError("TIINGO_API_KEY not set")
    headers = {"Content-Type": "application/json", "Authorization": f"Token {key}"}
    url = "https://api.tiingo.com/tiingo/daily/AAPL/prices?startDate=2025-10-01&endDate=2026-04-01"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Tiingo returned empty")
    info = {"rows": len(data), "first_date": data[0].get("date","?")[:10], "last_date": data[-1].get("date","?")[:10], "last_close": data[-1].get("close")}
''')

# ============================================================
# 9. FINNHUB — News
# ============================================================
run_test("Finnhub News", '''
    import requests
    from datetime import datetime, timedelta
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        from src.config import get_api_key
        key = get_api_key("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/company-news?symbol=AAPL&from={start}&to={end}&token={key}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    info = {"articles": len(data), "sample_headline": data[0].get("headline","")[:80] if data else "none"}
''')

# ============================================================
# 10. FINNHUB — Insider Sentiment
# ============================================================
run_test("Finnhub Insider Sentiment", '''
    import requests
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        from src.config import get_api_key
        key = get_api_key("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    url = f"https://finnhub.io/api/v1/stock/insider-sentiment?symbol=AAPL&from=2025-01-01&token={key}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    d = data.get("data", [])
    info = {"records": len(d), "latest_mspr": d[-1].get("mspr") if d else "N/A", "latest_change": d[-1].get("change") if d else "N/A"}
''')

# ============================================================
# 11. FINNHUB — Analyst Recommendations
# ============================================================
run_test("Finnhub Analyst Recs", '''
    import requests
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        from src.config import get_api_key
        key = get_api_key("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    url = f"https://finnhub.io/api/v1/stock/recommendation?symbol=AAPL&token={key}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("No recommendations returned")
    latest = data[0]
    info = {"buy": latest.get("buy"), "hold": latest.get("hold"), "sell": latest.get("sell"), "period": latest.get("period")}
''')

# ============================================================
# 12. FINNHUB — Earnings
# ============================================================
run_test("Finnhub Earnings", '''
    import requests
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        from src.config import get_api_key
        key = get_api_key("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    url = f"https://finnhub.io/api/v1/stock/earnings?symbol=AAPL&limit=4&token={key}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("No earnings returned")
    info = {"quarters": len(data), "latest_actual": data[0].get("actual"), "latest_estimate": data[0].get("estimate"), "latest_surprise_pct": data[0].get("surprisePercent")}
''')

# ============================================================
# 13. SEC EDGAR — Fundamentals
# ============================================================
run_test("SEC EDGAR Fundamentals", '''
    from src.data.providers.sec_edgar_provider import SECEdgarProvider
    p = SECEdgarProvider()
    data = p.get_fundamentals("AAPL")
    if not data:
        raise RuntimeError("SEC EDGAR returned empty")
    info = {k: str(v)[:50] for k, v in list(data.items())[:8]}
''')

# ============================================================
# 14. ALPHA VANTAGE — Prices
# ============================================================
run_test("Alpha Vantage Prices", '''
    import requests
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&outputsize=compact&apikey={key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(f"Rate limited: {data.get('Note', data.get('Information', ''))[:150]}")
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise RuntimeError(f"No time series in response. Keys: {list(data.keys())}")
    dates = sorted(ts.keys())
    info = {"days": len(ts), "first": dates[0], "last": dates[-1], "last_close": ts[dates[-1]].get("4. close")}
''')

# ============================================================
# 15. ALPHA VANTAGE — News Sentiment
# ============================================================
run_test("Alpha Vantage News", '''
    import requests
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&limit=5&apikey={key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(f"Rate limited: {data.get('Note', data.get('Information', ''))[:150]}")
    feed = data.get("feed", [])
    info = {"articles": len(feed), "sample_title": feed[0].get("title","")[:80] if feed else "none"}
''')

# ============================================================
# 16. MARKETAUX — News
# ============================================================
run_test("Marketaux News", '''
    import requests
    key = os.environ.get("MARKETAUX_API_KEY", "")
    if not key:
        raise RuntimeError("MARKETAUX_API_KEY not set")
    url = f"https://api.marketaux.com/v1/news/all?symbols=AAPL&filter_entities=true&limit=5&api_token={key}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    articles = data.get("data", [])
    info = {"articles": len(articles), "sample_title": articles[0].get("title","")[:80] if articles else "none"}
''')

# ============================================================
# 17. STOOQ — Prices
# ============================================================
run_test("Stooq Prices", '''
    import requests, io, pandas as pd
    r = requests.get("https://stooq.com/q/d/l/?s=aapl.us&i=d", timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        raise RuntimeError("Empty CSV")
    info = {"rows": len(df), "cols": list(df.columns), "last_date": str(df.iloc[-1].get("Date", "?"))}
''')

# ============================================================
# 18. AGGREGATOR — Full Pipeline
# ============================================================
run_test("Aggregator fetch_prices", '''
    from src.data.aggregator import get_aggregator
    agg = get_aggregator()
    df = agg.fetch_prices("AAPL", period="6mo")
    if df is None or (hasattr(df, "empty") and df.empty):
        raise RuntimeError("Aggregator returned empty")
    info = {"rows": len(df), "cols": list(df.columns)[:6], "provider": "auto-fallback"}
''')

# ============================================================
# 19. AGGREGATOR — Sentiment
# ============================================================
run_test("Aggregator fetch_sentiment", '''
    from src.data.aggregator import fetch_sentiment
    data = fetch_sentiment("AAPL")
    if not data:
        raise RuntimeError("fetch_sentiment returned empty dict")
    info = {k: str(v)[:50] for k, v in list(data.items())[:8]}
''')

# ============================================================
# 20. AGGREGATOR — Fundamentals
# ============================================================
run_test("Aggregator fetch_fundamentals", '''
    from src.data.aggregator import fetch_fundamentals
    data = fetch_fundamentals("AAPL")
    if not data:
        raise RuntimeError("fetch_fundamentals returned empty")
    info = {k: str(v)[:50] for k, v in list(data.items())[:8]}
''')

# ============================================================
# 21. Full Prediction Pipeline
# ============================================================
run_test("Full Prediction Pipeline", '''
    from prediction_model import build_features_and_target
    result = build_features_and_target("AAPL", period="1y")
    if result is None:
        raise RuntimeError("build_features_and_target returned None")
    X, y, df, feat_cols = result
    nan_pct = (X.isna().sum() / len(X) * 100).to_dict()
    bad_feats = {k: f"{v:.1f}%" for k, v in nan_pct.items() if v > 5}
    info = {"X_shape": list(X.shape), "y_shape": list(y.shape), "features": len(feat_cols), "high_nan_features": bad_feats if bad_feats else "none"}
''')


# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

ok, fail, error, timeout, skip = [], [], [], [], []
for name, r in results.items():
    s = r.get("status", "?")
    if s == "OK":
        ok.append(name)
        i = r.get("info", {})
        detail = ", ".join(f"{k}={v}" for k, v in list(i.items())[:3])
        print(f"  ✅ {name}: {detail} ({r.get('time','?')})")
    elif s == "SKIP":
        skip.append(name)
        print(f"  ⏭️  {name}: {r.get('reason','skipped')}")
    elif s == "TIMEOUT":
        timeout.append(name)
        print(f"  ⏰ {name}: TIMED OUT ({r.get('time','?')})")
    elif s == "ERROR":
        error.append(name)
        print(f"  💥 {name}: {r.get('reason','?')[:120]} ({r.get('time','?')})")
    else:
        fail.append(name)
        print(f"  ❌ {name}: {r.get('reason','?')[:120]}")

total = len(results)
print(f"\n  ✅ PASS:    {len(ok)}/{total}")
print(f"  ⏭️  SKIP:    {len(skip)}/{total}")
print(f"  ⏰ TIMEOUT: {len(timeout)}/{total}")
print(f"  💥 ERROR:   {len(error)}/{total}")
print(f"  ❌ FAIL:    {len(fail)}/{total}")

with open("data_source_test_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to data_source_test_results.json")
