import os, json
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


WATCHLIST = ["PLTR", "SMCI", "NVDA", "ZS", "SPY", "JPM", "MSFT", "XOM"]

BASE_DIR = Path(__file__).resolve().parent
SIGNALS_PATH = BASE_DIR / "signals.json"   # always read the same file


def load_signals() -> dict:
    if not SIGNALS_PATH.exists():
        return {}
    try:
        return json.loads(SIGNALS_PATH.read_text())
    except json.JSONDecodeError:
        print(f"ERROR: signals.json is not valid JSON: {SIGNALS_PATH}")
        return {}


def shares_for(symbol: str) -> float:
    return 1


def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    trading = TradingClient(key, secret, paper=True)

    positions = trading.get_all_positions()
    held = {p.symbol for p in positions}

    signals = load_signals()
    print("signals.json path:", str(SIGNALS_PATH))
    print("signals.json loaded:", signals)

    # OPTIONAL: only trade tickers in WATCHLIST
    for symbol, action in signals.items():
        if symbol not in WATCHLIST:
            continue

        action = str(action).upper()

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
