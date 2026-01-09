import os, json, math, logging
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Try to import streamlit for secrets access (production mode)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Performance monitoring (optional)
try:
    from src.monitoring import PerformanceMonitor, AlertLevel
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    PerformanceMonitor = None  # type: ignore
    HAS_PERFORMANCE_MONITOR = False

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
)
# OptionLegRequest was introduced in later alpaca-py releases. Older pinned versions
# (e.g., 0.32.0) don't expose it, which previously caused ImportError. We make the
# import optional so the script can still run and will skip multi-leg option orders
# if unavailable.
try:
    from alpaca.trading.requests import OptionLegRequest  # type: ignore
    HAS_OPTION_LEG = True
except ImportError:
    OptionLegRequest = None  # type: ignore
    HAS_OPTION_LEG = False
from alpaca.trading.enums import OrderSide, TimeInForce, AssetStatus, OrderClass

from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

# ========== LOGGING SETUP ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========== PRODUCTION CONFIGURATION ==========
PRODUCTION_CONFIG = {
    # Capital management
    "starting_capital": 50000.0,
    "max_portfolio_risk_pct": 0.10,  # 10% max drawdown before full halt
    
    # Position limits
    "max_open_positions": 10,
    "max_position_size_pct": 0.05,  # 5% of portfolio per position
    "max_daily_trades": 20,
    
    # Signal thresholds
    "z_score_threshold": 1.5,
    "min_confidence": 0.002,
    
    # Ticker filters
    "allowed_tickers": [],  # Empty = allow all tickers
    "excluded_tickers": ["SPY", "QQQ", "IWM", "DIA", "VTI"],  # ETFs excluded by default
    
    # Circuit breaker thresholds
    "circuit_breaker": {
        "daily_loss_limit_pct": 0.02,      # 2% daily loss limit
        "weekly_loss_limit_pct": 0.05,     # 5% weekly loss limit
        "consecutive_loss_limit": 5,        # Stop after 5 consecutive losses
        "max_drawdown_pct": 0.10,          # 10% max drawdown from peak
        "cooldown_hours": 24,              # Hours to wait after circuit break
    },
    
    # Take profit / Stop loss
    "default_take_profit_pct": 0.05,  # 5%
    "default_stop_loss_pct": 0.03,    # 3%
    
    # Options settings
    "options": {
        "dte_min": 3,
        "dte_max": 60,
        "max_premium": 500,
        "max_contracts": 5,
    },
}

# ========== API CREDENTIAL LOADING ==========
def get_alpaca_credentials() -> tuple[str, str]:
    """
    Load Alpaca API credentials from multiple sources:
    1. Streamlit secrets (production)
    2. Environment variables (development)
    3. .env file (fallback)
    
    Returns: (api_key, secret_key)
    """
    api_key = None
    secret_key = None
    
    # Try Streamlit secrets first (production deployment)
    if HAS_STREAMLIT:
        try:
            api_key = st.secrets.get("ALPACA_API_KEY") or st.secrets.get("APCA_API_KEY_ID")
            secret_key = st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("APCA_API_SECRET_KEY")
        except Exception:
            pass
    
    # Fall back to environment variables
    if not api_key:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    if not secret_key:
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "in streamlit secrets or environment variables."
        )
    
    return api_key, secret_key


WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]

BASE_DIR = Path(__file__).resolve().parent
SIGNALS_PATH = BASE_DIR / "signals.json"
TRADE_LOG_PATH = BASE_DIR / "trade_log.json"
CIRCUIT_BREAKER_PATH = BASE_DIR / "circuit_breaker_state.json"


# ========== CIRCUIT BREAKER ==========
@dataclass
class CircuitBreakerState:
    """Tracks trading circuit breaker state for risk management."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    is_tripped: bool = False
    trip_reason: str = ""
    trip_time: str = ""
    last_reset_date: str = ""
    last_week_reset: str = ""
    trades_today: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitBreakerState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CircuitBreaker:
    """
    Risk management circuit breaker that halts trading when limits are breached.
    
    Triggers on:
    - Daily P&L loss exceeds limit (e.g., -2%)
    - Weekly P&L loss exceeds limit (e.g., -5%)
    - Consecutive losses exceed limit (e.g., 5 in a row)
    - Max drawdown from peak exceeded (e.g., -10%)
    - Max daily trades exceeded
    """
    
    def __init__(self, config: dict = None, state_path: Path = CIRCUIT_BREAKER_PATH):
        self.config = config or PRODUCTION_CONFIG["circuit_breaker"]
        self.state_path = state_path
        self.state = self._load_state()
        self._check_reset()
    
    def _load_state(self) -> CircuitBreakerState:
        """Load persisted circuit breaker state."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                return CircuitBreakerState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load circuit breaker state: {e}")
        return CircuitBreakerState()
    
    def _save_state(self):
        """Persist circuit breaker state."""
        try:
            self.state_path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to save circuit breaker state: {e}")
    
    def _check_reset(self):
        """Reset daily/weekly counters at appropriate boundaries."""
        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        
        # Reset daily counters
        if self.state.last_reset_date != today:
            logger.info(f"🔄 Daily reset: {self.state.last_reset_date} → {today}")
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            self.state.last_reset_date = today
            
            # Clear trip if it was from previous day and cooldown has passed
            if self.state.is_tripped:
                try:
                    trip_time = datetime.fromisoformat(self.state.trip_time.replace('Z', '+00:00'))
                    hours_since_trip = (datetime.now(timezone.utc) - trip_time).total_seconds() / 3600
                    cooldown = self.config.get("cooldown_hours", 24)
                    if hours_since_trip >= cooldown:
                        self._clear_trip("Daily reset after cooldown")
                except Exception:
                    pass
        
        # Reset weekly counters on Monday
        if self.state.last_week_reset != week_start:
            logger.info(f"🔄 Weekly reset: {self.state.last_week_reset} → {week_start}")
            self.state.weekly_pnl = 0.0
            self.state.last_week_reset = week_start
        
        self._save_state()
    
    def _trip(self, reason: str):
        """Trip the circuit breaker."""
        self.state.is_tripped = True
        self.state.trip_reason = reason
        self.state.trip_time = datetime.now(timezone.utc).isoformat()
        self._save_state()
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
    
    def _clear_trip(self, reason: str = "Manual reset"):
        """Clear circuit breaker trip."""
        self.state.is_tripped = False
        self.state.trip_reason = ""
        self.state.trip_time = ""
        self.state.consecutive_losses = 0
        self._save_state()
        logger.info(f"✅ Circuit breaker cleared: {reason}")
    
    def update_equity(self, current_equity: float, starting_capital: float):
        """Update peak equity and drawdown tracking."""
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (self.state.peak_equity - current_equity) / self.state.peak_equity
        
        # Check max drawdown
        max_dd = self.config.get("max_drawdown_pct", 0.10)
        if self.state.current_drawdown_pct >= max_dd:
            self._trip(f"Max drawdown exceeded: {self.state.current_drawdown_pct:.1%} >= {max_dd:.1%}")
        
        self._save_state()
    
    def record_trade_result(self, pnl: float, is_win: bool, portfolio_value: float):
        """Record a trade result and check limits."""
        pnl_pct = pnl / portfolio_value if portfolio_value > 0 else 0
        
        self.state.daily_pnl += pnl_pct
        self.state.weekly_pnl += pnl_pct
        self.state.trades_today += 1
        
        if is_win:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
        
        # Check limits
        daily_limit = self.config.get("daily_loss_limit_pct", 0.02)
        weekly_limit = self.config.get("weekly_loss_limit_pct", 0.05)
        consec_limit = self.config.get("consecutive_loss_limit", 5)
        
        if self.state.daily_pnl <= -daily_limit:
            self._trip(f"Daily loss limit: {self.state.daily_pnl:.2%} <= -{daily_limit:.2%}")
        elif self.state.weekly_pnl <= -weekly_limit:
            self._trip(f"Weekly loss limit: {self.state.weekly_pnl:.2%} <= -{weekly_limit:.2%}")
        elif self.state.consecutive_losses >= consec_limit:
            self._trip(f"Consecutive losses: {self.state.consecutive_losses} >= {consec_limit}")
        
        self._save_state()
    
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        if self.state.is_tripped:
            return False, f"Circuit breaker tripped: {self.state.trip_reason}"
        
        max_daily = PRODUCTION_CONFIG.get("max_daily_trades", 20)
        if self.state.trades_today >= max_daily:
            return False, f"Max daily trades reached: {self.state.trades_today}/{max_daily}"
        
        return True, "OK"
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "is_tripped": self.state.is_tripped,
            "trip_reason": self.state.trip_reason,
            "daily_pnl": f"{self.state.daily_pnl:.2%}",
            "weekly_pnl": f"{self.state.weekly_pnl:.2%}",
            "consecutive_losses": self.state.consecutive_losses,
            "drawdown": f"{self.state.current_drawdown_pct:.2%}",
            "trades_today": self.state.trades_today,
        }
    
    def force_reset(self):
        """Manually reset circuit breaker (use with caution)."""
        self._clear_trip("Manual force reset")
        self.state.daily_pnl = 0.0
        self.state.consecutive_losses = 0
        self._save_state()
        logger.warning("⚠️ Circuit breaker force reset!")


# ========== PRE-TRADE FILTERS ==========
def is_ticker_allowed(ticker: str) -> tuple[bool, str]:
    """
    Check if ticker passes pre-trade filters.
    Returns (allowed, reason).
    """
    ticker = ticker.upper()
    
    # Check exclusion list
    excluded = PRODUCTION_CONFIG.get("excluded_tickers", [])
    if ticker in excluded:
        return False, f"Ticker in exclusion list"
    
    # Check allowlist if enabled
    allowed = PRODUCTION_CONFIG.get("allowed_tickers", [])
    if allowed and ticker not in allowed:
        return False, f"Ticker not in allowlist ({len(allowed)} tickers)"
    
    return True, "OK"


def passes_signal_thresholds(spec: dict) -> tuple[bool, str]:
    """
    Check if signal meets minimum quality thresholds.
    Returns (passed, reason).
    """
    # Z-score threshold
    z_threshold = PRODUCTION_CONFIG.get("z_score_threshold", 2.0)
    z_score = abs(spec.get("z_score", 0.0))
    if z_score < z_threshold:
        return False, f"Z-score {z_score:.2f} < {z_threshold}"
    
    # Confidence threshold
    min_confidence = PRODUCTION_CONFIG.get("min_confidence", 0.002)
    confidence = abs(spec.get("confidence_score", 0.0))
    if confidence == 0.0:
        confidence = abs(spec.get("pred_next_ret", 0.0))
    if confidence < min_confidence:
        return False, f"Confidence {confidence:.4f} < {min_confidence}"
    
    return True, "OK"


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
            # Convert UUID keys to strings for JSON serialization
            data = {str(tid): tr.to_dict() for tid, tr in self.trades.items()}
            # Atomic write
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix="trade_log_", suffix=".json")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2, default=str)
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


def _is_spread_strategy(strategy: str | None) -> bool:
    """Check if a strategy name indicates a multi-leg spread."""
    if not strategy:
        return False
    strategy_upper = strategy.upper()
    return any(x in strategy_upper for x in ["SPREAD", "CONDOR", "STRADDLE", "STRANGLE"])


def _extract_spread_legs_from_notes(notes: str) -> tuple[str | None, str | None]:
    """
    Extract long and short leg symbols from trade notes.
    Notes format: "BULL_CALL_SPREAD long=XYZ short=ABC ..."
    Returns: (long_symbol, short_symbol)
    """
    import re
    long_match = re.search(r'long=(\S+)', notes)
    short_match = re.search(r'short=(\S+)', notes)
    return (
        long_match.group(1) if long_match else None,
        short_match.group(1) if short_match else None,
    )


def _find_spread_trade_for_position(trade_log: TradeLog, symbol: str) -> TradeRecord | None:
    """
    Find a spread trade record that contains this symbol as one of its legs.
    Returns the TradeRecord if found, None otherwise.
    """
    for tid, tr in trade_log.trades.items():
        if tr.exit_price is not None:  # Already closed
            continue
        if not _is_spread_strategy(tr.strategy):
            continue
        # Check if symbol is in the notes (legs are stored there)
        if symbol in (tr.notes or ""):
            return tr
    return None


def maybe_close_for_targets(
    trade_client: TradingClient,
    trade_log: TradeLog,
    positions: list,
    take_profit_pct: float,
    stop_loss_pct: float,
    circuit_breaker: CircuitBreaker = None,
    portfolio_value: float = 0.0,
    performance_monitor = None,
):
    """
    Scan open positions and close if P/L breaches take-profit or stop-loss thresholds.
    take_profit_pct/stop_loss_pct expressed as decimals (e.g., 0.05 = +5%, -0.03 = -3%).
    
    For option spreads, closes both legs together as an MLEG order to avoid 
    uncovered option violations.
    
    Args:
        circuit_breaker: If provided, record trade results to update daily/weekly P&L tracking
        portfolio_value: Current portfolio value for P&L percentage calculation
        performance_monitor: If provided, record completed trades for performance tracking
    """
    if take_profit_pct is None and stop_loss_pct is None:
        return

    # Build a map of symbol -> position for quick lookup
    position_map = {p.symbol: p for p in positions}
    
    # Track which symbols we've already processed (to avoid double-processing spread legs)
    processed_symbols = set()
    # Track which spreads we've already closed
    closed_spreads = set()

    for p in positions:
        try:
            symbol = p.symbol
            
            # Skip if already processed as part of a spread
            if symbol in processed_symbols:
                continue
            
            entry_price = float(p.avg_entry_price)
            current_price = float(p.current_price)
            side = p.side.lower()  # 'long' or 'short'
            qty = float(p.qty)

            if entry_price <= 0:
                continue

            if side == "long":
                ret_pct = (current_price / entry_price) - 1.0
                pnl = (current_price - entry_price) * qty
            else:  # short
                ret_pct = (entry_price / current_price) - 1.0
                pnl = (entry_price - current_price) * qty

            should_take = take_profit_pct is not None and ret_pct >= take_profit_pct
            should_stop = stop_loss_pct is not None and ret_pct <= -abs(stop_loss_pct)

            if not (should_take or should_stop):
                continue

            action = "take-profit" if should_take else "stop-loss"
            
            # ========== CHECK IF THIS IS PART OF A SPREAD ==========
            spread_trade = _find_spread_trade_for_position(trade_log, symbol)
            
            if spread_trade and HAS_OPTION_LEG:
                # This position is part of a spread - close both legs together
                spread_id = str(spread_trade.trade_id)
                
                # Skip if we already closed this spread
                if spread_id in closed_spreads:
                    processed_symbols.add(symbol)
                    continue
                
                long_sym, short_sym = _extract_spread_legs_from_notes(spread_trade.notes or "")
                
                if not long_sym or not short_sym:
                    print(f"{symbol}: Could not parse spread legs from notes: {spread_trade.notes}")
                    continue
                
                # Check both legs are still in positions
                if long_sym not in position_map or short_sym not in position_map:
                    print(f"{symbol}: Spread legs missing from positions (long={long_sym}, short={short_sym})")
                    # Fall through to single-leg close as last resort
                else:
                    # Calculate combined spread P&L
                    long_pos = position_map[long_sym]
                    short_pos = position_map[short_sym]
                    
                    long_pnl = (float(long_pos.current_price) - float(long_pos.avg_entry_price)) * float(long_pos.qty)
                    short_pnl = (float(short_pos.avg_entry_price) - float(short_pos.current_price)) * float(short_pos.qty)
                    spread_pnl = long_pnl + short_pnl
                    
                    # Build MLEG close order (reverse the original spread)
                    # Original: BUY long leg, SELL short leg
                    # Close: SELL long leg, BUY short leg
                    legs = [
                        OptionLegRequest(symbol=long_sym, side=OrderSide.SELL, ratio_qty=1),
                        OptionLegRequest(symbol=short_sym, side=OrderSide.BUY, ratio_qty=1),
                    ]
                    
                    # Use limit order with current spread value
                    long_current = float(long_pos.current_price)
                    short_current = float(short_pos.current_price)
                    spread_credit = long_current - short_current  # Credit received when closing
                    limit_price = max(0.01, round(spread_credit - 0.02, 2))  # Small buffer for fill
                    
                    close_qty = int(min(float(long_pos.qty), float(short_pos.qty)))
                    
                    req = LimitOrderRequest(
                        qty=close_qty,
                        order_class=OrderClass.MLEG,
                        time_in_force=TimeInForce.DAY,
                        limit_price=limit_price,
                        legs=legs,
                    )
                    
                    try:
                        submitted = trade_client.submit_order(order_data=req)
                        
                        # Mark spread as closed
                        closed_spreads.add(spread_id)
                        processed_symbols.add(long_sym)
                        processed_symbols.add(short_sym)
                        
                        # Close the trade in log
                        exit_price = spread_credit * 100  # Per-contract value
                        trade_log.close_by_symbol(spread_trade.underlying or spread_trade.symbol, exit_price)
                        
                        # Update circuit breaker
                        if circuit_breaker is not None and portfolio_value > 0:
                            is_win = spread_pnl > 0
                            circuit_breaker.record_trade_result(spread_pnl, is_win, portfolio_value)
                            logger.info(f"📊 Circuit breaker updated (spread): P&L ${spread_pnl:+.2f}")
                        
                        # Record to performance monitor
                        if performance_monitor is not None:
                            performance_monitor.record_trade(
                                symbol=spread_trade.underlying or spread_trade.symbol,
                                side="SELL",
                                qty=close_qty,
                                entry_price=spread_trade.entry_price,
                                exit_price=exit_price,
                                pnl=spread_pnl,
                                trade_id=str(submitted.id),
                                strategy=f"{spread_trade.strategy}_{action.upper()}",
                            )
                        
                        print(
                            f"{datetime.now(timezone.utc).isoformat()} {spread_trade.underlying} "
                            f"{spread_trade.strategy} {action} close -> {submitted.id}; "
                            f"spread_pnl=${spread_pnl:.2f}"
                        )
                        continue
                        
                    except Exception as e:
                        print(f"{symbol}: Spread {action} close failed (trying individual legs): {e}")
                        # Fall through to individual close as last resort
            
            # ========== SINGLE-LEG CLOSE (stocks or single options) ==========
            side_to_close = OrderSide.SELL if side == "long" else OrderSide.BUY

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_to_close,
                time_in_force=TimeInForce.DAY,
            )
            try:
                submitted = trade_client.submit_order(order_data=order)
                processed_symbols.add(symbol)
                
                # Log the close in trade log
                closed = trade_log.close_by_symbol(symbol, current_price)
                
                # Update circuit breaker with trade result
                if circuit_breaker is not None and portfolio_value > 0:
                    is_win = pnl > 0
                    circuit_breaker.record_trade_result(pnl, is_win, portfolio_value)
                    logger.info(f"📊 Circuit breaker updated: P&L ${pnl:+.2f} ({'win' if is_win else 'loss'})")
                
                # Record to performance monitor
                if performance_monitor is not None:
                    performance_monitor.record_trade(
                        symbol=symbol,
                        side="BUY" if side == "long" else "SELL",
                        qty=qty,
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl=pnl,
                        trade_id=str(submitted.id),
                        strategy=action.upper(),
                    )
                
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {symbol} {action} exit @ {current_price:.2f} -> {submitted.id}; ret={ret_pct:.4f} pnl=${pnl:.2f}"
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
    circuit_breaker: CircuitBreaker = None,
    portfolio_value: float = 0.0,
    performance_monitor = None,
):
    """
    Close when live return >= predicted target stored in TradeLog.signal_strength.
    Only triggers for positive predicted returns above min_positive. Shorts use absolute target.
    
    For option spreads, closes both legs together as an MLEG order.
    
    Args:
        circuit_breaker: If provided, record trade results to update daily/weekly P&L tracking
        portfolio_value: Current portfolio value for P&L percentage calculation
        performance_monitor: If provided, record completed trades for performance tracking
    """
    if not enable:
        return

    # Map symbol -> position for quick lookup
    pos_map = {p.symbol.upper(): p for p in positions}
    
    # Track which spreads we've already processed
    processed_trades = set()

    for tr in trade_log.get_open_trades():
        try:
            trade_id = str(tr.trade_id)
            
            # Skip if already processed
            if trade_id in processed_trades:
                continue
            
            if tr.signal_strength is None:
                continue
            target = float(tr.signal_strength)
            if target <= 0:
                continue
            target = max(target, min_positive)

            # ========== CHECK IF THIS IS A SPREAD ==========
            if _is_spread_strategy(tr.strategy) and HAS_OPTION_LEG:
                long_sym, short_sym = _extract_spread_legs_from_notes(tr.notes or "")
                
                if not long_sym or not short_sym:
                    continue
                
                long_pos = pos_map.get(long_sym.upper())
                short_pos = pos_map.get(short_sym.upper())
                
                if not long_pos or not short_pos:
                    continue
                
                # Calculate spread return based on combined P&L
                entry_price = float(tr.entry_price)  # Original spread debit
                long_current = float(long_pos.current_price)
                short_current = float(short_pos.current_price)
                current_spread_value = long_current - short_current
                
                if entry_price <= 0:
                    continue
                
                # Spread return = (current value - entry cost) / entry cost
                ret_pct = (current_spread_value / entry_price) - 1.0
                
                if ret_pct < target:
                    continue
                
                # Calculate P&L
                qty = float(long_pos.qty)
                spread_pnl = (current_spread_value - entry_price) * qty * 100  # Per contract
                
                # Build MLEG close order
                legs = [
                    OptionLegRequest(symbol=long_sym, side=OrderSide.SELL, ratio_qty=1),
                    OptionLegRequest(symbol=short_sym, side=OrderSide.BUY, ratio_qty=1),
                ]
                
                limit_price = max(0.01, round(current_spread_value - 0.02, 2))
                close_qty = int(min(float(long_pos.qty), float(short_pos.qty)))
                
                req = LimitOrderRequest(
                    qty=close_qty,
                    order_class=OrderClass.MLEG,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price,
                    legs=legs,
                )
                
                try:
                    submitted = trade_client.submit_order(order_data=req)
                    processed_trades.add(trade_id)
                    
                    # Close in trade log
                    trade_log.close_by_symbol(tr.underlying or tr.symbol, current_spread_value * 100)
                    
                    # Update circuit breaker
                    if circuit_breaker is not None and portfolio_value > 0:
                        is_win = spread_pnl > 0
                        circuit_breaker.record_trade_result(spread_pnl, is_win, portfolio_value)
                        logger.info(f"📊 Circuit breaker updated (spread): P&L ${spread_pnl:+.2f}")
                    
                    # Record to performance monitor
                    if performance_monitor is not None:
                        performance_monitor.record_trade(
                            symbol=tr.underlying or tr.symbol,
                            side="SELL",
                            qty=close_qty,
                            entry_price=entry_price,
                            exit_price=current_spread_value,
                            pnl=spread_pnl,
                            trade_id=str(submitted.id),
                            strategy=f"{tr.strategy}_PRED_TARGET",
                        )
                    
                    print(
                        f"{datetime.now(timezone.utc).isoformat()} {tr.underlying} {tr.strategy} "
                        f"pred-target exit -> {submitted.id}; ret={ret_pct:.4f} target={target:.4f} "
                        f"pnl=${spread_pnl:.2f}"
                    )
                    continue
                    
                except Exception as e:
                    print(f"{tr.symbol}: Spread pred-target close failed: {e}")
                    continue
            
            # ========== SINGLE-LEG CLOSE ==========
            sym = tr.symbol.upper()
            p = pos_map.get(sym)
            if p is None:
                continue

            entry_price = float(tr.entry_price)
            current_price = float(p.current_price)
            side = p.side.lower()
            qty = float(p.qty)

            if entry_price <= 0:
                continue

            if side == "long":
                ret_pct = (current_price / entry_price) - 1.0
                pnl = (current_price - entry_price) * qty
            else:  # short
                ret_pct = (entry_price / current_price) - 1.0
                pnl = (entry_price - current_price) * qty

            if ret_pct < target:
                continue

            side_to_close = OrderSide.SELL if side == "long" else OrderSide.BUY
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side_to_close,
                time_in_force=TimeInForce.DAY,
            )
            try:
                submitted = trade_client.submit_order(order_data=order)
                processed_trades.add(trade_id)
                closed = trade_log.close_by_symbol(sym, current_price)
                
                # Update circuit breaker with trade result
                if circuit_breaker is not None and portfolio_value > 0:
                    is_win = pnl > 0
                    circuit_breaker.record_trade_result(pnl, is_win, portfolio_value)
                    logger.info(f"📊 Circuit breaker updated: P&L ${pnl:+.2f} ({'win' if is_win else 'loss'})")
                
                # Record to performance monitor
                if performance_monitor is not None:
                    performance_monitor.record_trade(
                        symbol=sym,
                        side="BUY" if side == "long" else "SELL",
                        qty=qty,
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl=pnl,
                        trade_id=str(submitted.id),
                        strategy="PRED_TARGET",
                    )
                
                print(
                    f"{datetime.now(timezone.utc).isoformat()} {sym} pred-target exit @ {current_price:.2f} -> {submitted.id}; ret={ret_pct:.4f} target={target:.4f} pnl=${pnl:.2f}"
                )
            except Exception as e:
                print(f"{sym}: pred-target close failed: {e}")
        except Exception as e:
            print(f"[pred-target-exit] error on {getattr(tr, 'symbol', '?')}: {e}")


def main():
    # Load credentials from secrets/environment
    try:
        key, secret = get_alpaca_credentials()
    except ValueError as e:
        logger.error(f"❌ {e}")
        return

    trade_client = TradingClient(key, secret, paper=True)
    option_data_client = OptionHistoricalDataClient(key, secret)

    # Initialize circuit breaker
    circuit_breaker = CircuitBreaker()
    cb_status = circuit_breaker.get_status()
    logger.info(f"🔌 Circuit Breaker Status: tripped={cb_status['is_tripped']}, "
                f"daily_pnl={cb_status['daily_pnl']}, trades_today={cb_status['trades_today']}")
    
    # Initialize performance monitor (if available)
    performance_monitor = None
    if HAS_PERFORMANCE_MONITOR:
        starting_capital = PRODUCTION_CONFIG.get("starting_capital", 50000.0)
        performance_monitor = PerformanceMonitor(starting_capital=starting_capital)
        logger.info(f"📊 Performance Monitor: {len(performance_monitor.trades)} historical trades loaded")
    
    # Check if trading is allowed
    can_trade, reason = circuit_breaker.can_trade()
    if not can_trade:
        logger.warning(f"⛔ Trading halted: {reason}")
        return

    try:
        account = trade_client.get_account()
        buying_power = float(getattr(account, "buying_power", 0) or getattr(account, "cash", 0) or 0)
        portfolio_value = float(getattr(account, "equity", 0) or getattr(account, "portfolio_value", 0) or 0)
        starting_capital = PRODUCTION_CONFIG.get("starting_capital", 50000.0)
        
        logger.info(f"💰 Account: equity=${portfolio_value:,.2f}, buying_power=${buying_power:,.2f}, "
                    f"starting_capital=${starting_capital:,.2f}")
        
        # Update circuit breaker with current equity
        circuit_breaker.update_equity(portfolio_value, starting_capital)
        
        # Update performance monitor with current equity
        if performance_monitor:
            performance_monitor.update_equity(portfolio_value)
        
        # Re-check after equity update (may have triggered drawdown limit)
        can_trade, reason = circuit_breaker.can_trade()
        if not can_trade:
            logger.warning(f"⛔ Trading halted after equity check: {reason}")
            return
            
    except Exception as e:
        logger.warning(f"Could not fetch account info for sizing: {e}")
        buying_power = 0.0
        portfolio_value = PRODUCTION_CONFIG.get("starting_capital", 50000.0)
    
    # Initialize trade log
    trade_log = TradeLog(TRADE_LOG_PATH)
    logger.info(f"📋 Trade Log: {len(trade_log.trades)} trades from {TRADE_LOG_PATH}")
    stats = trade_log.get_stats()
    logger.info(f"📊 Stats: {stats['total_trades']} closed | Win Rate: {stats['win_rate']:.1f}% | Total P&L: ${stats['total_pnl']:.2f}")

    positions = trade_client.get_all_positions()
    held = {p.symbol for p in positions}

    # --- ENFORCE MAX OPEN POSITIONS ---
    max_positions = PRODUCTION_CONFIG.get("max_open_positions", 10)
    open_trades = trade_log.get_open_trades()
    if len(open_trades) >= max_positions:
        logger.warning(f"🚫 Max open positions reached ({len(open_trades)}/{max_positions}). No new trades.")
        return

    # --- ENFORCE MAX RISK PER TRADE ---
    max_risk_pct = PRODUCTION_CONFIG.get("max_position_size_pct", 0.05)

    # Auto take-profit / stop-loss on existing positions before new entries
    try:
        take_profit_pct = PRODUCTION_CONFIG.get("default_take_profit_pct", 0.05)
        stop_loss_pct = PRODUCTION_CONFIG.get("default_stop_loss_pct", 0.03)
        maybe_close_for_targets(
            trade_client, trade_log, positions, take_profit_pct, stop_loss_pct,
            circuit_breaker=circuit_breaker, portfolio_value=portfolio_value,
            performance_monitor=performance_monitor
        )
    except Exception as e:
        logger.warning(f"[auto-exit] skipped: {e}")

    # Auto close when live return meets predicted target from signal_strength
    try:
        use_pred_target = env_bool_local("PRED_TARGET_EXIT", True)
        min_positive = float(os.environ.get("PRED_TARGET_MIN", "0.001"))
        maybe_close_on_pred_target(
            trade_client, trade_log, positions, use_pred_target, min_positive=min_positive,
            circuit_breaker=circuit_breaker, portfolio_value=portfolio_value,
            performance_monitor=performance_monitor
        )
    except Exception as e:
        logger.warning(f"[pred-target-exit] skipped: {e}")

    signals = load_signals()
    logger.info(f"📁 signals.json path: {SIGNALS_PATH}")
    logger.info(f"📊 Loaded {len(signals)} signals")

    global WATCHLIST
    if signals:
        WATCHLIST = [str(sym).upper() for sym in signals.keys()]

    trades_executed = 0
    trades_skipped = 0
    
    for symbol, spec in signals.items():
        symbol = str(symbol).upper()
        
        # ===== PRE-TRADE FILTER: Ticker allowlist/blocklist =====
        allowed, ticker_reason = is_ticker_allowed(symbol)
        if not allowed:
            logger.info(f"🚫 {symbol}: REJECTED - {ticker_reason}")
            trades_skipped += 1
            continue
        
        # ===== PRE-TRADE FILTER: Circuit breaker check =====
        can_trade, cb_reason = circuit_breaker.can_trade()
        if not can_trade:
            logger.warning(f"⛔ {symbol}: HALTED - {cb_reason}")
            break  # Stop processing all signals
        
        # ===== PRE-TRADE FILTER: Signal quality thresholds =====
        if isinstance(spec, dict):
            passes, threshold_reason = passes_signal_thresholds(spec)
            if not passes:
                logger.info(f"📉 {symbol}: FILTERED - {threshold_reason}")
                trades_skipped += 1
                continue

        # --- ENFORCE MAX OPEN POSITIONS (again, per symbol) ---
        open_trades = trade_log.get_open_trades()
        max_positions = PRODUCTION_CONFIG.get("max_open_positions", 10)
        if len(open_trades) >= max_positions:
            logger.warning(f"🚫 Max positions ({len(open_trades)}/{max_positions}). Skipping {symbol}.")
            trades_skipped += 1
            continue

        # --- ENFORCE MAX RISK PER TRADE ---
        # Compute max dollar risk for this trade
        stop_loss_pct = PRODUCTION_CONFIG.get("default_stop_loss_pct", 0.03)
        max_trade_risk = portfolio_value * max_risk_pct if portfolio_value > 0 else 0
        # For stocks, risk = qty * (entry_price * stop_loss_pct)
        # For options, risk = premium paid * qty
        # For now, only enforce for stocks
        if isinstance(spec, dict) and spec.get("asset", "stock") == "stock":
            last_price = fetch_last_close(symbol)
            qty_default = shares_for(symbol)
            stop_loss = stop_loss_pct if stop_loss_pct > 0 else 0.03
            max_qty = int(max_trade_risk / (last_price * stop_loss)) if last_price > 0 and stop_loss > 0 else qty_default
            if max_qty < 1:
                logger.warning(f"🚫 {symbol}: Max risk per trade too small. Skipping.")
                trades_skipped += 1
                continue
            # Override qty in spec for this trade
            spec["qty"] = min(qty_default, max_qty)

        # ===== CONFIDENCE FILTERING (legacy support) =====
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
            # Use confidence_score if available, otherwise use abs(pred_next_ret) as proxy
            confidence = abs(spec.get("confidence_score", 0.0))
            if confidence == 0.0:
                # Fallback: use predicted return magnitude as confidence proxy
                confidence = abs(spec.get("pred_next_ret", 0.0))
            if confidence < min_confidence:
                print(f"{symbol}: SKIPPED (confidence {confidence:.6f} < {min_confidence}) - Low confidence signal")
                continue
            if confidence > 0:
                print(f"{symbol}: Confidence {confidence:.6f} ✓ (threshold {min_confidence})")
        # ========================================

        # ===== Z-SCORE FILTERING (Soft) =====
        # Signals include z-score tags - log weak signals but DON'T skip by default
        # Set ZSCORE_HARD_FILTER=1 env to skip weak signals
        if isinstance(spec, dict):
            z_score = spec.get("z_score", 0.0)
            z_passes = spec.get("z_score_passes", True)
            z_strength = spec.get("z_score_strength", "unknown")
            z_threshold = spec.get("z_score_threshold", 1.0)
            
            # Log z-score info
            z_icon = "✅" if z_passes else "⚠️"
            print(f"{symbol}: Z-Score {z_score:+.2f} ({z_strength}) {z_icon} (threshold {z_threshold})")
            
            # Hard filter if enabled via env var
            if env_bool_local("ZSCORE_HARD_FILTER", False) and not z_passes:
                print(f"{symbol}: SKIPPED (z-score {z_score:.2f} < {z_threshold}) - Weak signal")
                continue
            
            # Log weak signals for analysis (even if not skipping)
            if not z_passes:
                print(f"   └─ WEAK SIGNAL: {symbol} z={z_score:.2f} pred={spec.get('pred_next_ret', 0)*100:.2f}%")
        # ====================================

        # ===== TRADE LIMIT FILTERING =====
        # Signals include trade limit info - skip if not allowed
        if isinstance(spec, dict):
            trade_allowed = spec.get("trade_allowed", True)
            trade_rank = spec.get("trade_rank", 0)
            trade_rank_value = spec.get("trade_rank_value", 0.0)
            trade_rank_method = spec.get("trade_rank_method", "zscore")
            ticker_limit = spec.get("ticker_trade_limit", 1)
            ticker_count = spec.get("ticker_trade_count", 0)
            skip_reason = spec.get("skip_reason", "")
            
            # Log trade limit info
            if trade_rank > 0:
                print(f"{symbol}: Rank #{trade_rank} by {trade_rank_method} ({trade_rank_value:.4f}) | Limit: {ticker_limit}/period")
            
            # Skip if trade not allowed
            if not trade_allowed:
                print(f"{symbol}: SKIPPED (trade limit: {skip_reason}) - Rank #{trade_rank}")
                continue
        # =================================

        # ===== REGIME FILTER CHECKING =====
        # Signals include regime info - skip if blocked
        if isinstance(spec, dict):
            regime_blocked = spec.get("regime_blocked", False)
            regime_block_reason = spec.get("regime_block_reason", "")
            regime = spec.get("regime", "neutral")
            regime_override = spec.get("regime_override", False)
            
            if regime_blocked and not regime_override:
                print(f"{symbol}: BLOCKED by regime filter ({regime}) - {regime_block_reason}")
                continue
            elif regime_override:
                print(f"{symbol}: Regime override - {spec.get('regime_note', 'high conviction')}")
        # ==================================

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
            
            # Build notes with z-score and rank info
            z_score = spec.get("z_score", 0.0)
            z_passes = spec.get("z_score_passes", True)
            z_tag = "[STRONG]" if z_passes else "[WEAK]"
            trade_rank = spec.get("trade_rank", 0)
            rank_info = f" rank#{trade_rank}" if trade_rank > 0 else ""
            notes = f"Signal from portfolio engine ({int(qty)} qty) z={z_score:+.2f} {z_tag}{rank_info}"
            
            trade_log.add_trade(
                trade_id=submitted.id,
                symbol=symbol,
                asset_type="stock",
                side=action,
                qty=qty,
                entry_price=last_price,
                signal_strength=signal_strength,
                underlying=symbol,
                notes=notes,
            )
            trades_executed += 1
            logger.info(f"✅ {symbol} {order_desc} -> {submitted.id}")
            continue

        # ===== OPTIONS =====
        if asset == "option":
            strategy = str(spec.get("strategy", "")).upper().strip()

            if trade_log.has_trade_today(symbol, asset_type="option", strategy=strategy, underlying=symbol):
                logger.info(f"{symbol}: skipped duplicate option signal (already traded {strategy} today)")
                trades_skipped += 1
                continue

            # Use options config from PRODUCTION_CONFIG
            opts_config = PRODUCTION_CONFIG.get("options", {})
            dte_min = int(spec.get("dte_min", opts_config.get("dte_min", 3)))
            dte_max = int(spec.get("dte_max", opts_config.get("dte_max", 60)))
            max_premium = float(spec.get("max_premium", opts_config.get("max_premium", 500)))
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

            # Multi-leg option orders require OptionLegRequest (not available in older alpaca-py pins)
            if strategy in {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "IRON_CONDOR"} and not HAS_OPTION_LEG:
                print(
                    f"{symbol}: Strategy {strategy} skipped because OptionLegRequest isn't available in this alpaca-py version. "
                    "Upgrade alpaca-py to enable multi-leg option orders."
                )
                continue

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
                logger.info(f"{symbol} (futures {contract}): HOLD")
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
                trades_executed += 1
                
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
                logger.info(f"✅ {symbol} (futures {contract}) {action} -> {submitted.id}")
            except Exception as e:
                logger.error(f"{symbol} (futures {contract}): Order failed: {e}")
            continue

        logger.warning(f"{symbol}: unknown asset '{asset}', skipping")
        trades_skipped += 1
    
    # ========== TRADING SUMMARY ==========
    logger.info("=" * 50)
    logger.info(f"📈 TRADING SESSION COMPLETE")
    logger.info(f"   Signals processed: {len(signals)}")
    logger.info(f"   Trades executed: {trades_executed}")
    logger.info(f"   Trades skipped/filtered: {trades_skipped}")
    logger.info(f"   Circuit breaker: {'⚠️ TRIPPED' if circuit_breaker.state.is_tripped else '✅ OK'}")
    cb_status = circuit_breaker.get_status()
    logger.info(f"   Daily P&L: {cb_status['daily_pnl']} | Weekly: {cb_status['weekly_pnl']}")
    logger.info(f"   Consecutive losses: {cb_status['consecutive_losses']}")
    
    # ========== PERFORMANCE MONITORING ==========
    if performance_monitor:
        logger.info("-" * 50)
        logger.info("📊 PERFORMANCE MONITOR")
        
        # Check for alerts
        alerts = performance_monitor.check_alerts()
        if alerts:
            for alert in alerts:
                if alert.level == AlertLevel.CRITICAL:
                    logger.critical(f"   🚨 {alert.message}")
                elif alert.level == AlertLevel.WARNING:
                    logger.warning(f"   ⚠️ {alert.message}")
        else:
            logger.info("   ✅ No alerts")
        
        # Log key metrics
        metrics = performance_monitor.metrics
        logger.info(f"   21d Sharpe: {metrics.sharpe_21d:.2f} | 63d Sharpe: {metrics.sharpe_63d:.2f}")
        logger.info(f"   Win Rate (20): {metrics.win_rate_20:.1%} | Drawdown: {metrics.current_drawdown_pct:.1%}")
        logger.info(f"   P&L Today: ${metrics.pnl_today:+,.2f} | Week: ${metrics.pnl_week:+,.2f} | Total: ${metrics.pnl_total:+,.2f}")
        
        # Generate and save daily summary
        try:
            summary = performance_monitor.generate_daily_summary()
            logger.info(f"   📄 Daily summary saved")
        except Exception as e:
            logger.warning(f"   Failed to generate daily summary: {e}")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
