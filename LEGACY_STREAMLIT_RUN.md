# Run the legacy Streamlit UI

This is the shortest path to launch `app.py` with accuracy populated and API keys loaded from `.env` automatically.

## Prereqs
- Python 3.12 (matching the repo venv)
- Dependencies installed in the venv (`python -m pip install -r requirements.txt`)
- `.env` file at the repo root with any keys you have (blank values are fine):
  - `FMP_API_KEY`, `FRED_API_KEY`, `MARKETAUX_API_KEY`, `ALPHAVANTAGE_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET`, etc.

## One-time note about env loading
`app.py` now calls `load_dotenv()` at startup, so any keys in `.env` are picked up automatically—no need to `export` them manually.

## Run steps
```bash
cd /Users/jakobmccleary/Desktop/Stock\ Predictor
source venv/bin/activate
streamlit run app.py
```

## Quick verification
- On launch, Streamlit prints the keys it sees in `st.secrets` and env. If you need to confirm, add `print(os.environ.get("FRED_API_KEY"))` temporarily.
- If you hit rate limits on yfinance, the app includes a fallback `YFRateLimitError` import to keep running.

## Tips
- If dependencies ever complain, re-run: `python -m pip install -r requirements.txt` (still inside the venv).
- Plotly is already pinned; if visuals fail, `python -m pip install --upgrade plotly` inside the venv usually fixes it.
