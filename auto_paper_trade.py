import os, json
from pathlib import Path
from datetime import datetime, timezone, date, timedelta

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
    OptionLegRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, OrderClass

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest


# ---------------- Config ----------------
DEBUG_OPTIONS = os.environ.get("DEBUG_OPTIONS", "1").strip() not in {"0", "false", "False"}

WATCHLIST = ["PLTR", "SMCI", "NVDA", "ZS", "SPY", "JPM", "MSFT", "XOM"]

BASE_DIR = Path(__file__).resolve().parent
SIGNALS_PATH = BASE_DIR / "signals.json"


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


def _get_latest_bid_ask(option_data_client: OptionHistoricalDataClient, option_symbol: str):
    """
    Returns (bid, ask) floats or (None, None).
    """
    req = OptionLatestQuoteRequest(symbol_or_symbols=[option_symbol])
    quotes = option_data_client.get_option_latest_quote(req)

    q = quotes.get(option_symbol) if hasattr(quotes, "get") else None
    if q is None:
        if DEBUG_OPTIONS:
            print(f"DEBUG quote missing for {option_symbol} -> quotes={quotes}")
        return None, None

    if isinstance(q, dict):
        bid = q.get("bid_price") or q.get("bp")
        ask = q.get("ask_price") or q.get("ap")
    else:
        bid = getattr(q, "bid_price", None) or getattr(q, "bp", None)
        ask = getattr(q, "ask_price", None) or getattr(q, "ap", None)

    try:
        bid = float(bid) if bid is not None else None
    except Exception:
        bid = None
    try:
        ask = float(ask) if ask is not None else None
    except Exception:
        ask = None

    return bid, ask


def _list_option_contracts(
    trade_client: TradingClient,
    underlying: str,
    dte_min: int | None = None,
    dte_max: int | None = None,
    strike_lte: float | None = None,
    strike_gte: float | None = None,
):
    """
    Pull contracts from Alpaca with server-side filters (important).
    The GetOptionContractsRequest supports expiration_date_gte/lte and strike_price_gte/lte. [web:1]
    """
    kwargs = dict(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        limit=10000,  # max per docs; helps avoid pagination for most underlyings [web:1]
    )

    if dte_min is not None and dte_max is not None:
        today = date.today()
        exp_gte = today + timedelta(days=int(dte_min))
        exp_lte = today + timedelta(days=int(dte_max))
        kwargs["expiration_date_gte"] = exp_gte
        kwargs["expiration_date_lte"] = exp_lte

    # Note: these are strings in the request model docs [web:1]
    if strike_gte is not None:
        kwargs["strike_price_gte"] = str(float(strike_gte))
    if strike_lte is not None:
        kwargs["strike_price_lte"] = str(float(strike_lte))

    req = GetOptionContractsRequest(**kwargs)
    resp = trade_client.get_option_contracts(req)

    if DEBUG_OPTIONS:
        try:
            n = len(resp.option_contracts) if hasattr(resp, "option_contracts") else len(resp)
        except Exception:
            n = None
        print(f"DEBUG contracts raw type={type(resp)} count={n} for {underlying} kwargs={kwargs}")

    if hasattr(resp, "option_contracts"):
        return resp.option_contracts
    if isinstance(resp, list):
        return resp
    try:
        return list(resp)
    except Exception:
        return []


def _extract_contract_fields(c):
    """
    Returns (symbol, exp_date, strike, is_call) or (None,...).
    """
    if isinstance(c, dict):
        sym = c.get("symbol")
        exp = c.get("expiration_date")
        strike = c.get("strike_price")
        ctype = c.get("type")
    else:
        sym = getattr(c, "symbol", None)
        exp = getattr(c, "expiration_date", None)
        strike = getattr(c, "strike_price", None)
        ctype = getattr(c, "type", None)

    if sym is None or exp is None or strike is None or ctype is None:
        return None, None, None, None

    if isinstance(exp, str):
        try:
            exp = date.fromisoformat(exp)
        except Exception:
            return None, None, None, None

    is_call = str(ctype).lower().startswith("c")
    try:
        strike = float(strike)
    except Exception:
        return None, None, None, None

    return str(sym), exp, strike, is_call


def _in_dte_window(exp: date, dte_min: int, dte_max: int) -> bool:
    today = date.today()
    lo = today + timedelta(days=int(dte_min))
    hi = today + timedelta(days=int(dte_max))
    return lo <= exp <= hi


def pick_single_leg(
    trade_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    underlying: str,
    right: str,                 # "CALL" or "PUT"
    spot: float | None,
    dte_min: int,
    dte_max: int,
    max_premium: float,         # dollars (ask * 100)
    max_strike: float | None,
):
    want_call = (right.upper().strip() == "CALL")

    # Server-side constraints (DTE + strike) to avoid empty/huge results. [web:1]
    contracts = _list_option_contracts(
        trade_client,
        underlying,
        dte_min=dte_min,
        dte_max=dte_max,
        strike_lte=max_strike,
    )

    if DEBUG_OPTIONS:
        print(f"DEBUG {underlying} {right}: total contracts fetched={len(contracts)}")

    candidates = []
    for c in contracts:
        sym, exp, strike, is_call = _extract_contract_fields(c)
        if sym is None:
            continue
        if is_call != want_call:
            continue
        # still keep client-side DTE/strike just in case
        if not _in_dte_window(exp, dte_min, dte_max):
            continue
        if max_strike is not None and float(strike) > float(max_strike):
            continue
        candidates.append((sym, exp, strike))

    if not candidates:
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} {right}: no contracts after filters (DTE/strike/type)")
        return None, None

    if spot is None:
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: (abs(x[2] - float(spot)), x[1]))

    if DEBUG_OPTIONS:
        print(f"DEBUG {underlying} {right}: evaluating {min(len(candidates), 20)} of {len(candidates)} candidates")
        for sym, exp, strike in candidates[:5]:
            print(f"DEBUG candidate {sym} exp={exp} strike={strike} spot={spot}")

    for sym, exp, strike in candidates[:250]:
        bid, ask = _get_latest_bid_ask(option_data_client, sym)
        if ask is None or ask <= 0:
            if DEBUG_OPTIONS:
                print(f"DEBUG {underlying} {right}: {sym} ask missing/<=0 (bid={bid}, ask={ask})")
            continue

        premium_dollars = float(ask) * 100.0
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} {right}: {sym} bid={bid} ask={ask} premium=${premium_dollars:.2f} (cap=${max_premium})")

        if premium_dollars <= float(max_premium):
            return sym, float(ask)

    return None, None


def pick_bull_call_spread(
    trade_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    underlying: str,
    spot: float | None,
    dte_min: int,
    dte_max: int,
    max_premium: float,         # dollars (est_debit * 100)
    max_strike: float | None,
    width_pct: float = 0.05,
):
    contracts = _list_option_contracts(
        trade_client,
        underlying,
        dte_min=dte_min,
        dte_max=dte_max,
        strike_lte=max_strike,
    )

    if DEBUG_OPTIONS:
        print(f"DEBUG {underlying} BULL_CALL_SPREAD: total contracts fetched={len(contracts)} spot={spot}")

    calls = []
    for c in contracts:
        sym, exp, strike, is_call = _extract_contract_fields(c)
        if sym is None or not is_call:
            continue
        if not _in_dte_window(exp, dte_min, dte_max):
            continue
        if max_strike is not None and float(strike) > float(max_strike):
            continue
        calls.append((sym, exp, strike))

    if not calls:
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BULL_CALL_SPREAD: no calls after filters")
        return None, None, None
    if spot is None:
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BULL_CALL_SPREAD: spot is None -> can't pick ATM")
        return None, None, None

    by_exp = {}
    for sym, exp, strike in calls:
        by_exp.setdefault(exp, []).append((sym, strike))

    for exp in sorted(by_exp.keys()):
        chain = sorted(by_exp[exp], key=lambda x: x[1])

        long_sym, long_strike = min(chain, key=lambda x: abs(x[1] - float(spot)))

        target_short = long_strike * (1.0 + float(width_pct))
        higher = [(s, k) for (s, k) in chain if k > long_strike]
        if not higher:
            continue
        short_sym, short_strike = min(higher, key=lambda x: abs(x[1] - target_short))

        long_bid, long_ask = _get_latest_bid_ask(option_data_client, long_sym)
        short_bid, short_ask = _get_latest_bid_ask(option_data_client, short_sym)

        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BCS exp={exp} long={long_sym}({long_strike}) bid/ask={long_bid}/{long_ask} "
                  f"short={short_sym}({short_strike}) bid/ask={short_bid}/{short_ask}")

        if long_ask is None or short_bid is None:
            continue

        est_debit = float(long_ask) - float(short_bid)
        if est_debit <= 0:
            continue

        debit_dollars = est_debit * 100.0
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BCS exp={exp} est_debit={est_debit:.4f} => ${debit_dollars:.2f} (cap=${max_premium})")

        if debit_dollars <= float(max_premium):
            return long_sym, short_sym, est_debit

    return None, None, None


def pick_bear_put_spread(
    trade_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    underlying: str,
    spot: float | None,
    dte_min: int,
    dte_max: int,
    max_premium: float,         # dollars (est_debit * 100)
    max_strike: float | None,
    width_pct: float = 0.05,
):
    contracts = _list_option_contracts(
        trade_client,
        underlying,
        dte_min=dte_min,
        dte_max=dte_max,
        strike_lte=max_strike,
    )

    if DEBUG_OPTIONS:
        print(f"DEBUG {underlying} BEAR_PUT_SPREAD: total contracts fetched={len(contracts)} spot={spot}")

    puts = []
    for c in contracts:
        sym, exp, strike, is_call = _extract_contract_fields(c)
        if sym is None or is_call:
            continue
        if not _in_dte_window(exp, dte_min, dte_max):
            continue
        if max_strike is not None and float(strike) > float(max_strike):
            continue
        puts.append((sym, exp, strike))

    if not puts:
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BEAR_PUT_SPREAD: no puts after filters")
        return None, None, None
    if spot is None:
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BEAR_PUT_SPREAD: spot is None -> can't pick ATM")
        return None, None, None

    by_exp = {}
    for sym, exp, strike in puts:
        by_exp.setdefault(exp, []).append((sym, strike))

    for exp in sorted(by_exp.keys()):
        chain = sorted(by_exp[exp], key=lambda x: x[1])

        long_sym, long_strike = min(chain, key=lambda x: abs(x[1] - float(spot)))

        target_short = long_strike * (1.0 - float(width_pct))
        lower = [(s, k) for (s, k) in chain if k < long_strike]
        if not lower:
            continue
        short_sym, short_strike = min(lower, key=lambda x: abs(x[1] - target_short))

        long_bid, long_ask = _get_latest_bid_ask(option_data_client, long_sym)
        short_bid, short_ask = _get_latest_bid_ask(option_data_client, short_sym)

        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BPS exp={exp} long={long_sym}({long_strike}) bid/ask={long_bid}/{long_ask} "
                  f"short={short_sym}({short_strike}) bid/ask={short_bid}/{short_ask}")

        if long_ask is None or short_bid is None:
            continue

        est_debit = float(long_ask) - float(short_bid)
        if est_debit <= 0:
            continue

        debit_dollars = est_debit * 100.0
        if DEBUG_OPTIONS:
            print(f"DEBUG {underlying} BPS exp={exp} est_debit={est_debit:.4f} => ${debit_dollars:.2f} (cap=${max_premium})")

        if debit_dollars <= float(max_premium):
            return long_sym, short_sym, est_debit

    return None, None, None


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

    global WATCHLIST
    if signals:
        WATCHLIST = [str(sym).upper() for sym in signals.keys()]

    for symbol, spec in signals.items():
        symbol = str(symbol).upper()

        if symbol not in WATCHLIST:
            print(f"{symbol}: not in WATCHLIST, but signals.json requested it -> trading anyway")

        if not isinstance(spec, dict):
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
            continue

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

            dte_min = int(spec.get("dte_min", 0))
            dte_max = int(spec.get("dte_max", 45))
            max_premium = float(spec.get("max_premium", 500))  # dollars (per contract)
            qty = int(spec.get("qty", 1))

            max_strike = spec.get("max_strike", None)
            try:
                max_strike = float(max_strike) if max_strike is not None else None
            except Exception:
                max_strike = None

            width_pct = float(spec.get("width_pct", 0.05))

            if dte_min < 0:
                dte_min = 0
            if dte_max < dte_min:
                dte_max = dte_min

            spot = spec.get("last_close", None)
            try:
                spot = float(spot) if spot is not None else None
            except Exception:
                spot = None

            if DEBUG_OPTIONS:
                print(
                    f"DEBUG {symbol}: strategy={strategy} spot={spot} dte={dte_min}-{dte_max} "
                    f"max_premium=${max_premium} max_strike={max_strike} width_pct={width_pct}"
                )

            if strategy in {"BUY_CALL", "BUY_PUT"}:
                right = "CALL" if strategy == "BUY_CALL" else "PUT"
                opt_sym, ask = pick_single_leg(
                    trade_client,
                    option_data_client,
                    symbol,
                    right,
                    spot,
                    dte_min,
                    dte_max,
                    max_premium,
                    max_strike,
                )
                if opt_sym is None or ask is None:
                    strike_msg = f", strike<={max_strike}" if max_strike is not None else ""
                    print(
                        f"{symbol}: No {strategy} found with premium<=${max_premium} "
                        f"and {dte_min}-{dte_max} DTE{strike_msg}"
                    )
                    continue

                if opt_sym in held:
                    print(f"{symbol}: Option {opt_sym} already held -> skipping")
                    continue

                order = LimitOrderRequest(
                    symbol=opt_sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(float(ask), 2),
                )
                submitted = trade_client.submit_order(order_data=order)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} {strategy} "
                    f"-> {opt_sym} @ {ask:.2f} (limit) -> {submitted.id}"
                )
                continue

            if strategy == "BULL_CALL_SPREAD":
                long_call, short_call, est_debit = pick_bull_call_spread(
                    trade_client,
                    option_data_client,
                    symbol,
                    spot,
                    dte_min,
                    dte_max,
                    max_premium,
                    max_strike,
                    width_pct=width_pct,
                )
                if not long_call or not short_call or est_debit is None:
                    strike_msg = f", strike<={max_strike}" if max_strike is not None else ""
                    print(
                        f"{symbol}: No BULL_CALL_SPREAD found with est_debit<=${max_premium} "
                        f"and {dte_min}-{dte_max} DTE{strike_msg}"
                    )
                    continue

                legs = [
                    OptionLegRequest(symbol=short_call, side=OrderSide.SELL, ratio_qty=1),
                    OptionLegRequest(symbol=long_call, side=OrderSide.BUY, ratio_qty=1),
                ]
                req = MarketOrderRequest(
                    qty=qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=legs,
                )
                submitted = trade_client.submit_order(req)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} BULL_CALL_SPREAD "
                    f"-> long={long_call} short={short_call} est_debit={est_debit:.2f} -> {submitted.id}"
                )
                continue

            if strategy == "BEAR_PUT_SPREAD":
                long_put, short_put, est_debit = pick_bear_put_spread(
                    trade_client,
                    option_data_client,
                    symbol,
                    spot,
                    dte_min,
                    dte_max,
                    max_premium,
                    max_strike,
                    width_pct=width_pct,
                )
                if not long_put or not short_put or est_debit is None:
                    strike_msg = f", strike<={max_strike}" if max_strike is not None else ""
                    print(
                        f"{symbol}: No BEAR_PUT_SPREAD found with est_debit<=${max_premium} "
                        f"and {dte_min}-{dte_max} DTE{strike_msg}"
                    )
                    continue

                legs = [
                    OptionLegRequest(symbol=short_put, side=OrderSide.SELL, ratio_qty=1),
                    OptionLegRequest(symbol=long_put, side=OrderSide.BUY, ratio_qty=1),
                ]
                req = MarketOrderRequest(
                    qty=qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    legs=legs,
                )
                submitted = trade_client.submit_order(req)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} BEAR_PUT_SPREAD "
                    f"-> long={long_put} short={short_put} est_debit={est_debit:.2f} -> {submitted.id}"
                )
                continue

            print(f"{symbol}: OPTIONS strategy '{strategy}' not supported -> skipping")
            continue

        print(f"{symbol}: unknown asset '{asset}', skipping")


if __name__ == "__main__":
    main()
