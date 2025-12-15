import os, json
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

WATCHLIST = ["PLTR", "SMCI", "NVDA"]
SIGNALS_PATH = "signals.json"

def load_signals() -> dict:
    if not os.path.exists(SIGNALS_PATH):
        return {}
    with open(SIGNALS_PATH, "r") as f:
        return json.load(f)

def shares_for(symbol: str) -> float:
    return 1

def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]
    trading = TradingClient(key, secret, paper=True)  # Alpaca-py supports paper trading mode [web:111]

    # Build a set of currently held symbols (open positions)
    positions = trading.get_all_positions()  # supported by alpaca-py [web:429]
    held = {p.symbol for p in positions}

    signals = load_signals()

    for symbol in WATCHLIST:
        action = signals.get(symbol, "HOLD").upper()

        if action not in {"BUY", "SELL", "HOLD"}:
            print(f"{symbol}: invalid signal '{action}', treating as HOLD")
            action = "HOLD"

        if action == "BUY" and symbol in held:
            print(f"{symbol}: BUY skipped (already holding)")
            continue
        if action == "SELL" and symbol not in held:
            print(f"{symbol}: SELL skipped (no position to close)")
            continue
        if action == "HOLD":
            print(f"{symbol}: HOLD")
            continue

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=shares_for(symbol),
            side=side,
            time_in_force=TimeInForce.DAY,
        )

        submitted = trading.submit_order(order_data=order)
        print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")

if __name__ == "__main__":
    main()
