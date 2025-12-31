import os, json, math
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional

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
    underlying: str | None = None  # For options to aid duplicate suppression
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
        underlying: str | None = None,
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
            underlying=underlying,
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

    def close_by_symbol(self, symbol: str, exit_price: float):
        """Close the oldest open trade for a symbol using the provided exit price."""
        open_trades = [tr for tr in self.trades.values() if tr.pnl is None and tr.symbol.upper() == symbol.upper()]
        if not open_trades:
            return None
        # close the earliest entry
        tr = sorted(open_trades, key=lambda x: x.entry_time or "")[0]
        now = datetime.now(timezone.utc)
        tr.close(
            exit_price=exit_price,
            exit_date=now.date().isoformat(),
            exit_time=now.isoformat(),
        )
        self.save()
        return tr

    def has_trade_today(
        self,
        symbol: str,
        asset_type: str | None = None,
        strategy: str | None = None,
        underlying: str | None = None,
    ) -> bool:
        """Return True if a trade matching symbol/asset/strategy/underlying exists today."""
        today = date.today().isoformat()
        sym_u = symbol.upper()
        under_u = underlying.upper() if underlying else None

        for tr in self.trades.values():
            if tr.entry_date != today:
                continue
            if asset_type and (tr.asset_type or "").lower() != asset_type.lower():
                continue
            if strategy and (tr.strategy or "").upper() != strategy.upper():
                continue

            tr_under = (tr.underlying or tr.symbol).upper()
            if tr.symbol.upper() == sym_u or tr_under == sym_u or (under_u and tr_under == under_u):
                return True

        return False


def load_signals() -> dict:
    if not SIGNALS_PATH.exists():
        return {}
    try:
        return json.loads(SIGNALS_PATH.read_text())
    except json.JSONDecodeError:
        print(f"ERROR: signals.json is not valid JSON: {SIGNALS_PATH}")
        return {}


def shares_for(symbol: str) -> float:
    """
    Default share sizing. Falls back to env DEFAULT_STOCK_QTY (float) or 1.
    Per-symbol overrides supported via env STOCK_QTY_<SYMBOL> (e.g., STOCK_QTY_SPY=5).
    """
    env_key = f"STOCK_QTY_{symbol.upper()}"
    if env_key in os.environ:
        try:
            return float(os.environ[env_key])
        except Exception:
            pass

    try:
        return float(os.environ.get("DEFAULT_STOCK_QTY", 1))
    except Exception:
        return 1


def env_bool_local(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def fetch_last_close(symbol: str) -> float:
    try:
        import yfinance as yf
        hist = yf.download(symbol, period="1d", progress=False, auto_adjust=False)
        return float(hist["Close"].iloc[-1].item()) if hist is not None and not hist.empty else 0.0
    except Exception:
        return 0.0


def compute_risk_qty(buying_power: float, price: float, fallback_qty: float = 1.0) -> int:
    """
    Risk-based sizing: floor((buying_power * RISK_PER_TRADE_PCT) / price).
    Defaults: USE_RISK_SIZING=1, RISK_PER_TRADE_PCT=0.005 (0.5% of BP per trade).
    """
    use_risk = env_bool_local("USE_RISK_SIZING", True)
    risk_pct = float(os.environ.get("RISK_PER_TRADE_PCT", "0.005"))

    if not use_risk or price <= 0 or buying_power <= 0 or risk_pct <= 0:
        return int(fallback_qty)

    qty = math.floor((buying_power * risk_pct) / price)
    return max(int(qty), int(fallback_qty)) if qty > 0 else int(fallback_qty)


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


def pick_iron_condor(
    trade_client: TradingClient,
    option_data_client: OptionHistoricalDataClient,
    underlying: str,
    spot: float | None,
    dte_min: int,
    dte_max: int,
    max_premium: float,         # dollars (net credit * 100)
    max_strike: float | None,
    width_pct: float = 0.05,
):
    """
    Pick an Iron Condor (sell call spread + sell put spread for net credit).
    Returns: (short_call, long_call, short_put, long_put, est_credit)
    """
    contracts = _list_option_contracts(
        trade_client,
        underlying,
        dte_min=dte_min,
        dte_max=dte_max,
        strike_lte=max_strike,
    )

    calls = []
    puts = []
    for c in contracts:
        sym, exp, strike, is_call = _extract_contract_fields(c)
        if sym is None:
            continue
        if not _in_dte_window(exp, dte_min, dte_max):
            continue
        if max_strike is not None and float(strike) > float(max_strike):
            continue
        
        if is_call:
            calls.append((sym, exp, strike))
        else:
            puts.append((sym, exp, strike))

    if not calls or not puts or spot is None:
        return None, None, None, None, None

    by_exp_calls = {}
    for sym, exp, strike in calls:
        by_exp_calls.setdefault(exp, []).append((sym, strike))

    by_exp_puts = {}
    for sym, exp, strike in puts:
        by_exp_puts.setdefault(exp, []).append((sym, strike))

    # Try to match same expiration for both sides
    common_exps = set(by_exp_calls.keys()) & set(by_exp_puts.keys())
    if not common_exps:
        return None, None, None, None, None

    for exp in sorted(common_exps):
        call_chain = sorted(by_exp_calls[exp], key=lambda x: x[1])
        put_chain = sorted(by_exp_puts[exp], key=lambda x: x[1])

        # Short call OTM (above spot)
        short_calls = [(s, k) for (s, k) in call_chain if k > float(spot)]
        if not short_calls:
            continue
        short_call_sym, short_call_strike = short_calls[0]  # Closest OTM call

        # Long call further OTM
        long_calls = [(s, k) for (s, k) in call_chain if k > short_call_strike]
        if not long_calls:
            continue
        target_call = short_call_strike * (1.0 + float(width_pct))
        long_call_sym, long_call_strike = min(long_calls, key=lambda x: abs(x[1] - target_call))

        # Short put OTM (below spot)
        short_puts = [(s, k) for (s, k) in put_chain if k < float(spot)]
        if not short_puts:
            continue
        short_put_sym, short_put_strike = short_puts[-1]  # Closest OTM put

        # Long put further OTM
        long_puts = [(s, k) for (s, k) in put_chain if k < short_put_strike]
        if not long_puts:
            continue
        target_put = short_put_strike * (1.0 - float(width_pct))
        long_put_sym, long_put_strike = min(long_puts, key=lambda x: abs(x[1] - target_put))

        # Get bid/ask for all legs
        sc_bid, sc_ask = _get_latest_bid_ask(option_data_client, short_call_sym)
        lc_bid, lc_ask = _get_latest_bid_ask(option_data_client, long_call_sym)
        sp_bid, sp_ask = _get_latest_bid_ask(option_data_client, short_put_sym)
        lp_bid, lp_ask = _get_latest_bid_ask(option_data_client, long_put_sym)

        if None in [sc_bid, lc_ask, sp_bid, lp_ask]:
            continue

        # Net credit = (short call credit - long call cost) + (short put credit - long put cost)
        call_credit = float(sc_bid) - float(lc_ask)
        put_credit = float(sp_bid) - float(lp_ask)
        est_credit = call_credit + put_credit

        if est_credit > 0 and est_credit * 100.0 <= float(max_premium):
            return short_call_sym, long_call_sym, short_put_sym, long_put_sym, est_credit

    return None, None, None, None, None


def maybe_close_for_targets(
    trade_client: TradingClient,
    trade_log: TradeLog,
    positions: list,
    take_profit_pct: float,
    stop_loss_pct: float,
):
    """
    Scan open positions and close if P/L breaches take-profit or stop-loss thresholds.
    take_profit_pct/stop_loss_pct expressed as decimals (e.g., 0.05 = +5%, -0.03 = -3%).
    """
    if take_profit_pct is None and stop_loss_pct is None:
        return

    for p in positions:
        try:
            symbol = p.symbol
            entry_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            side = p.side.lower()  # 'long' or 'short'

            if entry_price <= 0:
                continue

            if side == "long":
                ret_pct = (current_price / entry_price) - 1.0
            else:  # short
                ret_pct = (entry_price / current_price) - 1.0

            should_take = take_profit_pct is not None and ret_pct >= take_profit_pct
            should_stop = stop_loss_pct is not None and ret_pct <= -abs(stop_loss_pct)

            if not (should_take or should_stop):
                continue

            action = "take-profit" if should_take else "stop-loss"
            side_to_close = OrderSide.SELL if side == "long" else OrderSide.BUY
            qty = float(p.qty)

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_to_close,
                time_in_force=TimeInForce.DAY,
            )
            try:
                submitted = trade_client.submit_order(order_data=order)
                # Log the close in trade log
                closed = trade_log.close_by_symbol(symbol, current_price)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} exit @ {current_price:.2f} -> {submitted.id}; ret={ret_pct:.4f}"
                )
            except Exception as e:
                print(f"{symbol}: {action} close failed: {e}")
        except Exception as e:
            print(f"[auto-exit] Error processing position {getattr(p, 'symbol', '?')}: {e}")


def maybe_close_on_pred_target(
    trade_client: TradingClient,
    trade_log: TradeLog,
    positions: list,
    enable: bool,
    min_positive: float = 0.001,
):
    """
    Close when live return >= predicted target stored in TradeLog.signal_strength.
    Only triggers for positive predicted returns above min_positive. Shorts use absolute target.
    """
    if not enable:
        return

    # Map symbol -> current price/side/qty from positions
    pos_map = {p.symbol.upper(): p for p in positions}

    for tr in trade_log.get_open_trades():
        try:
            sym = tr.symbol.upper()
            if tr.signal_strength is None:
                continue
            target = float(tr.signal_strength)
            if target <= 0:
                continue
            target = max(target, min_positive)

            p = pos_map.get(sym)
            if p is None:
                continue

            entry_price = float(tr.entry_price)
            current_price = float(p.current_price)
            side = p.side.lower()

            if entry_price <= 0:
                continue

            if side == "long":
                ret_pct = (current_price / entry_price) - 1.0
            else:  # short
                ret_pct = (entry_price / current_price) - 1.0

            if ret_pct < target:
                continue

            side_to_close = OrderSide.SELL if side == "long" else OrderSide.BUY
            qty = float(p.qty)
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side_to_close,
                time_in_force=TimeInForce.DAY,
            )
            try:
                submitted = trade_client.submit_order(order_data=order)
                closed = trade_log.close_by_symbol(sym, current_price)
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {sym} pred-target exit @ {current_price:.2f} -> {submitted.id}; ret={ret_pct:.4f} target={target:.4f}"
                )
            except Exception as e:
                print(f"{sym}: pred-target close failed: {e}")
        except Exception as e:
            print(f"[pred-target-exit] error on {getattr(tr, 'symbol', '?')}: {e}")


def main():
    key = os.environ["APCA_API_KEY_ID"]
    secret = os.environ["APCA_API_SECRET_KEY"]

    trade_client = TradingClient(key, secret, paper=True)
    option_data_client = OptionHistoricalDataClient(key, secret)

    try:
        account = trade_client.get_account()
        buying_power = float(getattr(account, "buying_power", 0) or getattr(account, "cash", 0) or 0)
    except Exception as e:
        print(f"[Trade Log] Warning: could not fetch account info for sizing: {e}")
        buying_power = 0.0
    
    # Initialize trade log
    trade_log = TradeLog(TRADE_LOG_PATH)
    print(f"[Trade Log] Loaded {len(trade_log.trades)} trades from {TRADE_LOG_PATH}")
    stats = trade_log.get_stats()
    print(f"[Trade Log Stats] {stats['total_trades']} closed trades | Win Rate: {stats['win_rate']:.1f}% | Total P&L: ${stats['total_pnl']:.2f}")

    positions = trade_client.get_all_positions()
    held = {p.symbol for p in positions}

    # Auto take-profit / stop-loss on existing positions before new entries
    try:
        take_profit_pct = float(os.environ.get("TAKE_PROFIT_PCT", "0.05")) if os.environ.get("TAKE_PROFIT_PCT") else None
        stop_loss_pct = float(os.environ.get("STOP_LOSS_PCT", None)) if os.environ.get("STOP_LOSS_PCT") else None
        maybe_close_for_targets(trade_client, trade_log, positions, take_profit_pct, stop_loss_pct)
    except Exception as e:
        print(f"[auto-exit] skipped due to config/error: {e}")

    # Auto close when live return meets predicted target from signal_strength
    try:
        use_pred_target = env_bool_local("PRED_TARGET_EXIT", True)
        min_positive = float(os.environ.get("PRED_TARGET_MIN", "0.001"))
        maybe_close_on_pred_target(trade_client, trade_log, positions, use_pred_target, min_positive=min_positive)
    except Exception as e:
        print(f"[pred-target-exit] skipped due to config/error: {e}")

    signals = load_signals()
    print("signals.json path:", str(SIGNALS_PATH))
    print("signals.json loaded:", signals)

    global WATCHLIST
    if signals:
        WATCHLIST = [str(sym).upper() for sym in signals.keys()]

    for symbol, spec in signals.items():
        symbol = str(symbol).upper()

        # ===== NEW: CONFIDENCE FILTERING =====
        # Skip low-confidence predictions to improve accuracy
        # Confidence = |predicted_return| (higher absolute value = higher confidence)
        # Thresholds tuned from diagnostics: GLD/XOM=0.001, SPY/broad=0.002
        confidence_thresholds = {
            "GLD": 0.001,
            "XOM": 0.001,  # Similar commodity behavior
            "SPY": 0.002,
            "QQQ": 0.002,
            "NVDA": 0.002,
            "MSFT": 0.002,
            "JPM": 0.002,
        }
        min_confidence = confidence_thresholds.get(symbol, 0.001)  # Default 0.001
        
        if isinstance(spec, dict):
            confidence = abs(spec.get("confidence_score", 0.0))
            if confidence < min_confidence:
                print(f"{symbol}: SKIPPED (confidence {confidence:.6f} < {min_confidence}) - Low confidence signal")
                continue
            if confidence > 0:
                print(f"{symbol}: Confidence {confidence:.6f} ✓ (threshold {min_confidence})")
        # ========================================

        if not isinstance(spec, dict):
            action = str(spec).upper()
            if action not in {"BUY", "SELL", "HOLD"}:
                print(f"{symbol}: invalid action '{action}', treating as HOLD")
                action = "HOLD"

            if trade_log.has_trade_today(symbol, asset_type="stock", underlying=symbol):
                print(f"{symbol}: skipped duplicate signal (already traded today)")
                continue

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
            last_price = fetch_last_close(symbol)
            qty_default = shares_for(symbol)
            qty = compute_risk_qty(buying_power, last_price, qty_default)

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            submitted = trade_client.submit_order(order_data=order)
            # Get last price for entry
            last_price = last_price if last_price is not None else fetch_last_close(symbol)
            
            # Log trade
            trade_log.add_trade(
                trade_id=submitted.id,
                symbol=symbol,
                asset_type="stock",
                side=action,
                qty=qty,
                entry_price=last_price,
                underlying=symbol,
                notes=f"Simple signal from signals.json",
            )
            print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} -> {submitted.id}")
            continue

        asset = str(spec.get("asset", "stock")).lower().strip()

        # ===== STOCKS =====
        if asset == "stock":
            action = str(spec.get("action", "HOLD")).upper()
            explicit_qty = spec.get("qty", None)
            qty = float(explicit_qty) if explicit_qty is not None else float(shares_for(symbol))

            if action not in {"BUY", "SHORT", "SELL", "HOLD"}:
                print(f"{symbol}: invalid action '{action}', treating as HOLD")
                action = "HOLD"

            if action == "BUY" and symbol in held:
                print(f"{symbol}: BUY skipped (already holding)")
                continue
            if action in {"SHORT", "SELL"} and symbol not in held:
                print(f"{symbol}: {action} skipped (no position to close)")
                continue
            if action == "HOLD":
                print(f"{symbol}: HOLD")
                continue

            if trade_log.has_trade_today(symbol, asset_type="stock", underlying=symbol):
                print(f"{symbol}: skipped duplicate signal (already traded today)")
                continue

            # Map actions to Alpaca order side
            if action == "BUY":
                side = OrderSide.BUY
                order_desc = f"BUY {int(qty)} shares"
            elif action == "SHORT":
                side = OrderSide.SELL  # SHORT is sell side in Alpaca
                order_desc = f"SHORT {int(qty)} shares"
            else:  # SELL
                side = OrderSide.SELL
                order_desc = f"SELL {int(qty)} shares"
            
            last_price = fetch_last_close(symbol)
            if explicit_qty is None or str(spec.get("sizing", "")).lower() == "risk":
                qty = compute_risk_qty(buying_power, last_price, qty)

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            
            # Try to submit order with error handling for non-tradeable assets
            try:
                submitted = trade_client.submit_order(order_data=order)
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg.lower():
                    print(f"{symbol}: Asset not found on Alpaca (likely non-US market) -> skipping")
                else:
                    print(f"{symbol}: Order submission failed: {e}")
                continue
            
            # Get last price for entry
            last_price = last_price if last_price is not None else fetch_last_close(symbol)
            
            # Log trade with signal strength if available
            signal_strength = spec.get("pred_next_ret", None)
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
                underlying=symbol,
                notes=f"Signal from portfolio engine ({int(qty)} qty)",
            )
            print(f"{datetime.now(timezone.utc).isoformat()} {symbol} {order_desc} -> {submitted.id}")
            continue

        # ===== OPTIONS =====
        if asset == "option":
            strategy = str(spec.get("strategy", "")).upper().strip()

            if trade_log.has_trade_today(symbol, asset_type="option", strategy=strategy, underlying=symbol):
                print(f"{symbol}: skipped duplicate option signal (already traded {strategy} today)")
                continue

            # Default: 3 DTE minimum to avoid rapid decay, 60 DTE maximum
            # This ensures options have a few days of life to allow for profitable exit
            dte_min = int(spec.get("dte_min", 3))
            dte_max = int(spec.get("dte_max", 60))  # Extended from 45 to 60 days
            max_premium = float(spec.get("max_premium", 500))  # dollars per 1-lot / spread
            qty = int(spec.get("qty", 1))

            max_strike = spec.get("max_strike", None)
            try:
                max_strike = float(max_strike) if max_strike is not None else None
            except Exception:
                max_strike = None

            width_pct = float(spec.get("width_pct", 0.05))

            signal_strength = spec.get("pred_next_ret", None)
            try:
                signal_strength = float(signal_strength) if signal_strength is not None else None
            except Exception:
                signal_strength = None

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
                try:
                    submitted = trade_client.submit_order(order_data=order)
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} {symbol} {strategy} "
                        f"-> {opt_sym} @ {ask:.2f} (limit) -> {submitted.id}"
                    )

                    trade_log.add_trade(
                        trade_id=submitted.id,
                        symbol=opt_sym,
                        asset_type="option",
                        side="BUY",
                        qty=qty,
                        entry_price=float(ask),
                        strategy=strategy,
                        signal_strength=signal_strength,
                        underlying=symbol,
                        notes=f"{strategy} on {symbol} @ {ask:.2f} (limit)",
                    )
                except Exception as e:
                    print(f"{symbol}: {strategy} order failed: {e}")
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
                try:
                    submitted = trade_client.submit_order(order_data=req)
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} {symbol} BULL_CALL_SPREAD "
                        f"-> long={long_call} short={short_call} limit={limit_price:.2f} -> {submitted.id}"
                    )

                    trade_log.add_trade(
                        trade_id=submitted.id,
                        symbol=symbol,
                        asset_type="option",
                        side="BUY",
                        qty=qty,
                        entry_price=float(limit_price),
                        strategy=strategy,
                        signal_strength=signal_strength,
                        underlying=symbol,
                        notes=(
                            f"BULL_CALL_SPREAD long={long_call} short={short_call} "
                            f"limit={limit_price:.2f} qty={qty}"
                        ),
                    )
                except Exception as e:
                    print(f"{symbol}: BULL_CALL_SPREAD order failed: {e}")
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
                try:
                    submitted = trade_client.submit_order(order_data=req)
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} {symbol} BEAR_PUT_SPREAD "
                        f"-> long={long_put} short={short_put} limit={limit_price:.2f} -> {submitted.id}"
                    )

                    trade_log.add_trade(
                        trade_id=submitted.id,
                        symbol=symbol,
                        asset_type="option",
                        side="BUY",
                        qty=qty,
                        entry_price=float(limit_price),
                        strategy=strategy,
                        signal_strength=signal_strength,
                        underlying=symbol,
                        notes=(
                            f"BEAR_PUT_SPREAD long={long_put} short={short_put} "
                            f"limit={limit_price:.2f} qty={qty}"
                        ),
                    )
                except Exception as e:
                    print(f"{symbol}: BEAR_PUT_SPREAD order failed: {e}")
                continue

            if strategy == "IRON_CONDOR":
                short_call, long_call, short_put, long_put, est_credit = pick_iron_condor(
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
                if not all([short_call, long_call, short_put, long_put]) or est_credit is None:
                    strike_msg = f", strike<={max_strike}" if max_strike is not None else ""
                    print(
                        f"{symbol}: No IRON_CONDOR found with est_credit<=${max_premium} "
                        f"and {dte_min}-{dte_max} DTE{strike_msg}"
                    )
                    continue

                # Iron Condor: 4-leg spread (sell call, buy call, sell put, buy put)
                legs = [
                    OptionLegRequest(symbol=short_call, side=OrderSide.SELL, ratio_qty=1),
                    OptionLegRequest(symbol=long_call, side=OrderSide.BUY, ratio_qty=1),
                    OptionLegRequest(symbol=short_put, side=OrderSide.SELL, ratio_qty=1),
                    OptionLegRequest(symbol=long_put, side=OrderSide.BUY, ratio_qty=1),
                ]

                # For credit spreads, we try to get at least the net credit
                limit_price = round(max(0.01, float(est_credit) - 0.05), 2)

                req = LimitOrderRequest(
                    qty=qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    legs=legs,
                )
                
                try:
                    submitted = trade_client.submit_order(order_data=req)
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} {symbol} IRON_CONDOR "
                        f"-> call spread: {short_call}/{long_call}, put spread: {short_put}/{long_put}, "
                        f"limit={limit_price:.2f} (credit) -> {submitted.id}"
                    )

                    trade_log.add_trade(
                        trade_id=submitted.id,
                        symbol=symbol,
                        asset_type="option",
                        side="SELL",  # credit strategy
                        qty=qty,
                        entry_price=float(limit_price),
                        strategy=strategy,
                        signal_strength=signal_strength,
                        underlying=symbol,
                        notes=(
                            f"IRON_CONDOR call {short_call}/{long_call}, put {short_put}/{long_put}, "
                            f"limit={limit_price:.2f} qty={qty}"
                        ),
                    )
                except Exception as e:
                    print(f"{symbol}: IRON_CONDOR order failed: {e}")
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
                    underlying=futures_symbol,
                    notes=f"Futures contract {contract}",
                )
                print(f"{datetime.now(timezone.utc).isoformat()} {symbol} (futures {contract}) {action} -> {submitted.id}")
            except Exception as e:
                print(f"{symbol} (futures {contract}): Order failed: {e}")
            continue

        print(f"{symbol}: unknown asset '{asset}', skipping")


if __name__ == "__main__":
    main()
