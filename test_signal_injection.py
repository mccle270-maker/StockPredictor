import pandas as pd
from app import build_signals_from_pred_df
from types import SimpleNamespace

# Create test DataFrame for AAPL with long, short, and neutral predictions
test_pred_df = pd.DataFrame([
    {"ticker": "AAPL", "pred_next_ret": 0.02, "last_close": 190.0, "put_call_oi_ratio": 1.0, "atm_iv": 0.25},  # Long
    {"ticker": "AAPL", "pred_next_ret": -0.02, "last_close": 190.0, "put_call_oi_ratio": 1.0, "atm_iv": 0.25}, # Short
    {"ticker": "AAPL", "pred_next_ret": 0.0, "last_close": 190.0, "put_call_oi_ratio": 1.0, "atm_iv": 0.25},   # Neutral
])

# Dummy execution model
exec_model = SimpleNamespace(delay_days=1, half_spread_bps=2, slippage_bps=3, fee_bps=0)

signals = build_signals_from_pred_df(
    test_pred_df,
    prediction_horizon=1,
    trade_mode="Stock only",
    prefer_spreads=False,
    dte_min=20,
    dte_max=45,
    max_strike=250.0,
    max_premium=10.0,
    width_pct=0.1,
    exec_model=exec_model,
)

print("Injected signal sanity cases for AAPL:")
for tk, sig in signals.items():
    print(f"{tk}: {sig}")
