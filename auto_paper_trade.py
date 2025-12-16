import os, json
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOptionContractsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest


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


def _get_latest_ask(option_data_client: OptionHistoricalDataClient, option_symbol: str) -> float | None:
    """
    Returns ask price (float) or None.
    """
    req = OptionLatestQuoteRequest(symbol_or_symbols=[option_symbol])
    quotes = option_data_client.get_option_latest_quote(req)

    q = quotes.get(option_symbol) if isinstance(quotes, dict) else None
    if q is None:
        return None

    # handle wrapped objects or raw dicts
    if isinstance(q, dict):
        # common keys seen in many quote payloads
        return q.get("ask_price") or q.get("ap")
    return getattr(q, "ask_price", None) or getattr(q, "ap", None)


def pick_contract_symbol(
    trade_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    underlying: str,
    right: str,                 # "CALL" or "PUT"
    spot: float | None,
    dte_max: int,
    max_premium: float,
) -> tuple[str | None, float | None]:
    """
    Returns (option_symbol, ask_price) for a contract <= dte_max and ask*100 <= max_premium.
    Chooses closest-to-spot strike first, then checks ask.
    """
    today = date.today()
    exp_lte = today + timedelta(days=int(dte_max))

    # Pull contracts from Trading API (includes expiration/strike/type) [web:265]
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
    )
    contracts = trade_client.get_option_contracts(req)

    # contracts could be a list or a response wrapper; normalize to a list
    if hasattr(contracts, "option_contracts"):
        contracts_list = contracts.option_contracts
    elif isinstance(contracts, list):
        contracts_list = contracts
    else:
        # last resort: try iterable
        try:
            contracts_list = list(contracts)
        except Exception:
            contracts_list = []

    right = right.upper().strip()
    want_call = (right == "CALL")

    candidates = []
    for c in contracts_list:
        # handle wrapped objects vs dict
        sym = getattr(c, "symbol", None) if not isinstance(c, dict) else c.get("symbol")
        exp = getattr(c, "expiration_date", None) if not isinstance(c, dict) else c.get("expiration_date")
        strike = getattr(c, "strike_price", None) if not isinstance(c, dict) else c.get("strike_price")
        ctype = getattr(c, "type", None) if not isinstance(c, dict) else c.get("type")

        if sym is None or exp is None or strike is None or ctype is None:
            continue

        # exp might be date or string "YYYY-MM-DD"
        if isinstance(exp, str):
            try:
                exp = date.fromisoformat(exp)
            except Exception:
                continue

        is_call = str(ctype).lower().startswith("c")
        if is_call != want_call:
            continue

        if not (today <= exp <= exp_lte):
            continue

        candidates.append((sym, exp, float(strike)))

    if not candidates:
        return None, None

    # Sort by (closest strike to spot) then nearest expiration
    if spot is None:
        # if no spot, just nearest expiration then mid strikes
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: (abs(x[2] - float(spot)), x[1]))

    # check premiums until we find one under cap
    for sym, exp, strike in candidates[:200]:
        ask = _get_latest_ask(option_data_client, sym)
        if ask is None:
            continue
        if ask <= 0:
            continue
        if ask * 100.0 <= float(max_premium):
            return sym, float(ask)

    return None, None


def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    trade_client = TradingClient(key, secret, paper=True)
    option_data_client = OptionHistoricalDataClient(key, secret)

    positions = trade_client.get_all_positions()
    held = {p.symbol for p in positions}

    signals = load_signals()
    print("signals.json path:", str(SIGNALS_PATH))
    print("signals.json loaded:", signals)

    for symbol, spec in signals.items():
        symbol = str(symbol).upper()

        if symbol not in WATCHLIST:
            print(f"{symbol}: not in WATCHLIST, skipping")
            continue

        # ---- NEW FORMAT: dict spec ----
        if isinstance(spec, dict):
            asset = str(spec.get("asset", "stock")).lower().strip()

            if asset == "stock":
                action = str(spec.get("action", "HOLD")).upper()
                qty = float(spec.get("qty", shares_for(symbol)))

                if action not in {"BUY", "SELL", "HOLD"}:
                    print(f"{symbol}: invalid action '{action}', treating as HOLD")
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
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
                submitted = trade_client.submit_order(order_data=order)
                print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")
                continue

            if asset == "option":
                strategy = str(spec.get("strategy", "")).upper().strip()
                dte_max = int(spec.get("dte_max", 14))
                max_premium = float(spec.get("max_premium", 200))
                qty = int(spec.get("qty", 1))
                spot = spec.get("last_close", None)
                try:
                    spot = float(spot) if spot is not None else None
                except Exception:
                    spot = None

                if strategy not in {"BUY_CALL", "BUY_PUT"}:
                    print(f"{symbol}: OPTIONS strategy '{strategy}' not supported yet -> skipping")
                    continue

                right = "CALL" if strategy == "BUY_CALL" else "PUT"

                option_symbol, ask = pick_contract_symbol(
                    trade_client=trade_client,
                    option_data_client=option_data_client,
                    underlying=symbol,
                    right=right,
                    spot=spot,
                    dte_max=dte_max,
                    max_premium=max_premium,
                )

                if option_symbol is None or ask is None:
                    print(f"{symbol}: No option contract found under ${max_premium} with <= {dte_max} DTE")
                    continue

                if option_symbol in held:
                    print(f"{symbol}: Option {option_symbol} already held -> skipping")
                    continue

                # Use LIMIT at ask to respect max premium (1 contract = 100 multiplier)
                order = LimitOrderRequest(
                    symbol=option_symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(float(ask), 2),
                )
                submitted = trade_client.submit_order(order_data=order)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} {strategy} "
                    f"-> {option_symbol} @ {ask:.2f} (limit) -> {submitted.id}"
                )
                continue

            print(f"{symbol}: unknown asset '{asset}', skipping")
            continue

        # ---- OLD FORMAT: string BUY/SELL/HOLD ----
        action = str(spec).upper()
        if action not in {"BUY", "SELL", "HOLD"}:
            print(f"{symbol}: invalid action '{action}', treating as HOLD")
            action = "HOLD"
        if action == "HOLD":
            print(f"{symbol}: HOLD")
            continue

        if action == "BUY" and symbol in held:
            print(f"{symbol}: BUY skipped (already holding)")
            continue
        if action == "SELL" and symbol not in held:
            print(f"{symbol}: SELL skipped (no position to close)")
            continue

        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=shares_for(symbol),
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        submitted = trade_client.submit_order(order_data=order)
        print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")


if __name__ == "__main__":
    main()
