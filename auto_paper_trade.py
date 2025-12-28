import os, json
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict, field

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


WATCHLIST = ["PLTR", "SMCI", "NVDA", "ZS", "SPY", "JPM", "MSFT", "XOM"]

BASE_DIR = Path(__file__).resolve().parent
SIGNALS_PATH = BASE_DIR / "signals.json"
TRADE_LOG_PATH = BASE_DIR / "trade_log.json"


# ========== TRADE MEMORY / LOGGING ==========
@dataclass
class TradeRecord:
    """Persistent record of a single trade execution."""
    trade_id: str  # Alpaca order ID
    symbol: str
    asset_type: str  # "stock", "option", "futures"
    side: str  # "BUY", "SELL"
    qty: float
    entry_price: float
    entry_date: str  # ISO format
    entry_time: str  # ISO format timestamp
    
    # Exit details (filled on close)
    exit_price: float | None = None
    exit_date: str | None = None
    exit_time: str | None = None
    
    # Performance
    pnl: float | None = None  # Realized P&L
    pnl_pct: float | None = None  # P&L %
    holding_days: int | None = None
    
    # Metadata
    strategy: str | None = None  # e.g., "BUY_CALL", "BULL_CALL_SPREAD"
    signal_strength: float | None = None  # e.g., predicted return
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    def close(self, exit_price: float, exit_date: str, exit_time: str):
        """Mark trade as closed with exit price and date."""
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.exit_time = exit_time
        
        # Calculate P&L
        if self.side.upper() == "BUY":
            self.pnl = (exit_price - self.entry_price) * self.qty
            self.pnl_pct = (exit_price / self.entry_price - 1) * 100
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.qty
            self.pnl_pct = (self.entry_price / exit_price - 1) * 100
        
        # Calculate holding days
        from datetime import datetime as dt
        try:
            entry_dt = dt.fromisoformat(self.entry_date.replace('Z', '+00:00'))
            exit_dt = dt.fromisoformat(self.exit_date.replace('Z', '+00:00'))
            self.holding_days = (exit_dt.date() - entry_dt.date()).days
        except Exception:
            self.holding_days = 0


class TradeLog:
    """Persistent trade log stored as JSON."""
    
    def __init__(self, path: Path = TRADE_LOG_PATH):
        self.path = Path(path)
        self.trades: dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        self.load()
    
    def load(self):
        """Load trades from disk."""
        if not self.path.exists():
            self.trades = {}
            return
        
        try:
            data = json.loads(self.path.read_text())
            self.trades = {tid: TradeRecord.from_dict(tr) for tid, tr in data.items()}
        except Exception as e:
            print(f"[TradeLog] Error loading trades: {e}")
            self.trades = {}
    
    def save(self):
        """Persist trades to disk."""
        try:
            data = {tid: tr.to_dict() for tid, tr in self.trades.items()}
            # Atomic write
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix="trade_log_", suffix=".json")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                import os as os_module
                os_module.replace(tmp, self.path)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
        except Exception as e:
            print(f"[TradeLog] Error saving trades: {e}")
    
    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        asset_type: str,
        side: str,
        qty: float,
        entry_price: float,
        strategy: str | None = None,
        signal_strength: float | None = None,
        notes: str = "",
    ) -> TradeRecord:
        """Record a new trade entry."""
        now = datetime.now(timezone.utc)
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            asset_type=asset_type,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_date=now.date().isoformat(),
            entry_time=now.isoformat(),
            strategy=strategy,
            signal_strength=signal_strength,
            notes=notes,
        )
        self.trades[trade_id] = record
        self.save()
        return record
    
    def close_trade(self, trade_id: str, exit_price: float):
        """Close an existing trade."""
        if trade_id not in self.trades:
            print(f"[TradeLog] Trade {trade_id} not found")
            return
        
        now = datetime.now(timezone.utc)
        self.trades[trade_id].close(
            exit_price=exit_price,
            exit_date=now.date().isoformat(),
            exit_time=now.isoformat(),
        )
        self.save()
    
    def get_stats(self) -> dict:
        """Get aggregate stats from closed trades."""
        closed = [tr for tr in self.trades.values() if tr.pnl is not None]
        if not closed:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
                "avg_holding_days": 0,
            }
        
        wins = sum(1 for tr in closed if tr.pnl > 0)
        return {
            "total_trades": len(closed),
            "win_rate": (wins / len(closed)) * 100 if closed else 0,
            "avg_pnl": sum(tr.pnl for tr in closed) / len(closed) if closed else 0,
            "total_pnl": sum(tr.pnl for tr in closed),
            "avg_holding_days": sum(tr.holding_days or 0 for tr in closed) / len(closed) if closed else 0,
        }
    
    def get_open_trades(self) -> list[TradeRecord]:
        """Get list of open (unfilled) trades."""
        return [tr for tr in self.trades.values() if tr.pnl is None]
    
    def get_closed_trades(self) -> list[TradeRecord]:
        """Get list of closed (filled) trades."""
        return [tr for tr in self.trades.values() if tr.pnl is not None]


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
    Pull contracts from Alpaca with server-side filters.
    GetOptionContractsRequest supports expiration_date_gte/lte and strike_price_gte/lte. [web:0]
    """
    kwargs = dict(
        underlying_symbols=[underlying],
        status=AssetStatus.ACTIVE,
        limit=10000,
    )

    if dte_min is not None and dte_max is not None:
        today = date.today()
        exp_gte = today + timedelta(days=int(dte_min))
        exp_lte = today + timedelta(days=int(dte_max))
        kwargs["expiration_date_gte"] = exp_gte
        kwargs["expiration_date_lte"] = exp_lte

    if strike_gte is not None:
        kwargs["strike_price_gte"] = str(float(strike_gte))
    if strike_lte is not None:
        kwargs["strike_price_lte"] = str(float(strike_lte))

    req = GetOptionContractsRequest(**kwargs)
    resp = trade_client.get_option_contracts(req)

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
    Handles ContractType enums correctly (call/put). [web:0]
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

    if hasattr(ctype, "value"):
        ctype_str = str(ctype.value).lower().strip()
    else:
        ctype_str = str(ctype).lower().strip()

    if "call" in ctype_str or ctype_str == "c":
        is_call = True
    elif "put" in ctype_str or ctype_str == "p":
        is_call = False
    else:
        return None, None, None, None

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

    contracts = _list_option_contracts(
        trade_client,
        underlying,
        dte_min=dte_min,
        dte_max=dte_max,
        strike_lte=max_strike,
    )

    candidates = []
    for c in contracts:
        sym, exp, strike, is_call = _extract_contract_fields(c)
        if sym is None:
            continue
        if is_call != want_call:
            continue
        if not _in_dte_window(exp, dte_min, dte_max):
            continue
        if max_strike is not None and float(strike) > float(max_strike):
            continue
        candidates.append((sym, exp, strike))

    if not candidates:
        return None, None

    if spot is None:
        candidates.sort(key=lambda x: x[1])
    else:
        candidates.sort(key=lambda x: (abs(x[2] - float(spot)), x[1]))

    for sym, exp, strike in candidates[:250]:
        bid, ask = _get_latest_bid_ask(option_data_client, sym)
        if ask is None or ask <= 0:
            continue

        if float(ask) * 100.0 <= float(max_premium):
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

    if not calls or spot is None:
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
        if long_ask is None or short_bid is None:
            continue

        est_debit = float(long_ask) - float(short_bid)
        if est_debit <= 0:
            continue

        if est_debit * 100.0 <= float(max_premium):
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

    if not puts or spot is None:
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
        if long_ask is None or short_bid is None:
            continue

        est_debit = float(long_ask) - float(short_bid)
        if est_debit <= 0:
            continue

        if est_debit * 100.0 <= float(max_premium):
            return long_sym, short_sym, est_debit

    return None, None, None


def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    trade_client = TradingClient(key, secret, paper=True)
    option_data_client = OptionHistoricalDataClient(key, secret)
    
    # Initialize trade log
    trade_log = TradeLog(TRADE_LOG_PATH)
    print(f"[Trade Log] Loaded {len(trade_log.trades)} trades from {TRADE_LOG_PATH}")
    stats = trade_log.get_stats()
    print(f"[Trade Log Stats] {stats['total_trades']} closed trades | Win Rate: {stats['win_rate']:.1f}% | Total P&L: ${stats['total_pnl']:.2f}")

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
            # Get last price for entry
            hist = None
            try:
                import yfinance as yf
                hist = yf.download(symbol, period="1d", progress=False)
                last_price = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else 0
            except Exception:
                last_price = 0
            
            # Log trade
            trade_log.add_trade(
                trade_id=submitted.id,
                symbol=symbol,
                asset_type="stock",
                side=action,
                qty=shares_for(symbol),
                entry_price=last_price,
                notes=f"Simple signal from signals.json",
            )
            print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")
            continue

        asset = str(spec.get("asset", "stock")).lower().strip()

        # ===== STOCKS =====
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
            
            # Get last price for entry
            hist = None
            try:
                import yfinance as yf
                hist = yf.download(symbol, period="1d", progress=False)
                last_price = float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else 0
            except Exception:
                last_price = 0
            
            # Log trade with signal strength if available
            signal_strength = spec.get("predicted_return", None)
            try:
                signal_strength = float(signal_strength) if signal_strength is not None else None
            except Exception:
                signal_strength = None
            
            trade_log.add_trade(
                trade_id=submitted.id,
                symbol=symbol,
                asset_type="stock",
                side=action,
                qty=qty,
                entry_price=last_price,
                signal_strength=signal_strength,
                notes=f"Signal from portfolio engine",
            )
            print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")
            continue

        # ===== OPTIONS =====
        if asset == "option":
            strategy = str(spec.get("strategy", "")).upper().strip()

            dte_min = int(spec.get("dte_min", 0))
            dte_max = int(spec.get("dte_max", 60))  # Extended from 45 to 60 days
            max_premium = float(spec.get("max_premium", 500))  # dollars per 1-lot / spread
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

            # --- single-leg ---
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

            # --- spreads (MLEG) as LIMIT orders ---
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

                # small buffer to improve fills, but never exceed max_premium cap
                limit_price = round(min(float(est_debit) + 0.02, float(max_premium) / 100.0), 2)

                req = LimitOrderRequest(
                    qty=qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    legs=legs,
                )
                submitted = trade_client.submit_order(order_data=req)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} BULL_CALL_SPREAD "
                    f"-> long={long_call} short={short_call} limit={limit_price:.2f} -> {submitted.id}"
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

                limit_price = round(min(float(est_debit) + 0.02, float(max_premium) / 100.0), 2)

                req = LimitOrderRequest(
                    qty=qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    legs=legs,
                )
                submitted = trade_client.submit_order(order_data=req)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} BEAR_PUT_SPREAD "
                    f"-> long={long_put} short={short_put} limit={limit_price:.2f} -> {submitted.id}"
                )
                continue

            print(f"{symbol}: OPTIONS strategy '{strategy}' not supported -> skipping")
            continue

        # ===== FUTURES =====
        if asset == "futures":
            action = str(spec.get("action", "HOLD")).upper()
            qty = float(spec.get("qty", 1))
            contract = str(spec.get("contract", "ES")).upper()  # ES, NQ, MES, MNQ, etc.
            
            # Map futures contracts to Alpaca symbols
            futures_map = {
                "ES": "ES",      # E-mini S&P 500
                "NQ": "NQ",      # E-mini Nasdaq-100
                "MES": "MES",    # Micro E-mini S&P 500
                "MNQ": "MNQ",    # Micro E-mini Nasdaq-100
                "CL": "CL",      # Crude Oil
                "GC": "GC",      # Gold
                "ZB": "ZB",      # 30-Year Treasury Bond
                "ZN": "ZN",      # 10-Year Treasury Note
            }
            
            futures_symbol = futures_map.get(contract, contract)
            
            if action not in {"BUY", "SELL", "HOLD"}:
                print(f"{symbol} (futures {contract}): invalid action '{action}', treating as HOLD")
                action = "HOLD"
            
            if action == "HOLD":
                print(f"{symbol} (futures {contract}): HOLD")
                continue
            
            side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
            order = MarketOrderRequest(
                symbol=futures_symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            try:
                submitted = trade_client.submit_order(order_data=order)
                
                # Log futures trade
                trade_log.add_trade(
                    trade_id=submitted.id,
                    symbol=futures_symbol,
                    asset_type="futures",
                    side=action,
                    qty=qty,
                    entry_price=0,  # Futures typically executed at market; set to 0 as placeholder
                    strategy=contract,
                    notes=f"Futures contract {contract}",
                )
                print(f"{datetime.now(timezone.utc).isoformat()} {symbol} (futures {contract}) {action} -> {submitted.id}")
            except Exception as e:
                print(f"{symbol} (futures {contract}): Order failed: {e}")
            continue

        print(f"{symbol}: unknown asset '{asset}', skipping")


if __name__ == "__main__":
    main()
