#!/usr/bin/env python3
"""Quick isolated tests of each fixed data source."""
import subprocess, sys, os, time

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))

def run(name, code, timeout=45):
    print(f"\n{name}...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [PYTHON, "-c", f"import warnings;warnings.filterwarnings('ignore');import os,sys;sys.path.insert(0,'{BASE}');os.chdir('{BASE}');\nfrom dotenv import load_dotenv;load_dotenv('.env.apis')\n{code}"],
            capture_output=True, text=True, timeout=timeout, cwd=BASE
        )
        out = (r.stdout + r.stderr).strip()
        elapsed = time.time() - t0
        print(f"  {out[-300:]} ({elapsed:.1f}s)")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")

# 1. market.py prices (should use yfinance now, not Alpaca)
run("1. market.py prices", """
from src.data.market import get_price_history
df=get_price_history('AAPL',period='3mo')
print(f'rows={len(df) if df is not None else 0}, cols={list(df.columns)[:5] if df is not None else []}')
""")

# 2. macro (11 FRED series)
run("2. FRED macro", """
from src.data.macro import get_macro_df
m=get_macro_df(period='1y')
print(f'shape={m.shape}')
for c in m.columns:
    v=m[c].dropna().iloc[-1] if not m[c].dropna().empty else 'EMPTY'
    print(f'  {c}: {round(float(v),4) if isinstance(v,(int,float)) else v}')
""")

# 3. fundamentals (FMP + yfinance fallback)
run("3. Fundamentals", """
from src.data.fundamentals import get_fundamentals_with_fallback
f=get_fundamentals_with_fallback('AAPL')
print(f)
""")

# 4. sentiment (Finnhub)
run("4. Sentiment", """
from src.data.aggregator import fetch_sentiment
s=fetch_sentiment('AAPL')
print(f'keys={list(s.keys())[:8]}')
for k in list(s.keys())[:6]:
    print(f'  {k}: {str(s[k])[:60]}')
""")

# 5. aggregator prices (full fallback chain)
run("5. Aggregator prices", """
from src.data.aggregator import fetch_prices
df=fetch_prices('AAPL',period='3mo')
print(f'rows={len(df) if df is not None else 0}')
""")

# 6. Provider health
run("6. Provider health", """
from src.data.aggregator import get_aggregator
h=get_aggregator().get_provider_health()
for n,v in h.items():
    print(f'{n}: avail={v[\"available\"]}')
""")

print("\n\nALL DONE")
