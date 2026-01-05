# Stock Predictor - Refactored Module Structure
# 
# Architecture:
#   src/
#   ├── config.py       - All constants, presets, settings
#   ├── core/           - Pure business logic (no I/O, no UI)
#   │   ├── features.py - Feature engineering (RSI, MACD, GBM, etc.)
#   │   ├── models.py   - ML model creation and training
#   │   ├── pricing.py  - Options pricing (Black-Scholes, Heston)
#   │   └── metrics.py  - Risk/performance metrics (Sharpe, drawdown)
#   ├── data/           - Data access layer (API calls, caching)
#   │   ├── market.py   - Price history (yfinance, fallbacks)
#   │   ├── macro.py    - FRED macro data
#   │   ├── fundamentals.py - FMP fundamentals
#   │   ├── news.py     - News sentiment (Marketaux, Alpha Vantage)
#   │   └── options.py  - Options chain data
#   └── services/       - Orchestration (combines core + data)
#       ├── prediction.py  - Prediction pipeline
#       ├── backtest.py    - Backtesting logic
#       ├── signals.py     - Signal generation for trading
#       └── screening.py   - Stock screening logic
#
# Usage:
#   from src.services.prediction import PredictionService
#   from src.config import DEFAULT_HORIZON, FRICTION_PRESETS
