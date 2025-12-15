import os
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
load_dotenv()


WATCHLIST = ["PLTR", "SMCI", "NVDA"]

def get_model_signal(symbol: str) -> str:
    """
    Replace this with your real model call.
    Return: "BUY", "SELL", or "HOLD"
    """
    return "HOLD"

def shares_for(symbol: str) -> float:
    # simple sizing rule; replace later
    return 1

def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    trading = TradingClient(key, secret, paper=True)  # paper trading mode [web:111]

    for symbol in WATCHLIST:
        action = get_model_signal(symbol)

        if action == "HOLD":
            print(f"{symbol}: HOLD")
            continue

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

        order = MarketOrderRequest(
            symbol=symbol,
            qty=shares_for(symbol),
            side=side,
            time_in_force=TimeInForce.DAY
        )

        submitted = trading.submit_order(order_data=order)
        print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")

if __name__ == "__main__":
    main()
