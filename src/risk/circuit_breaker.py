"""
Circuit Breaker for Risk Management

A standalone, reusable circuit breaker that can be used in any trading system
to halt trading when risk limits are breached.

Triggers on:
- Daily P&L loss exceeds limit (e.g., -2%)
- Weekly P&L loss exceeds limit (e.g., -5%)
- Consecutive losses exceed limit (e.g., 5 in a row)
- Max drawdown from peak exceeded (e.g., -10%)
- Max daily trades exceeded

Usage:
    from src.risk import CircuitBreaker
    
    cb = CircuitBreaker()
    
    # Before each trade
    can_trade, reason = cb.can_trade()
    if not can_trade:
        print(f"Trading halted: {reason}")
        return
    
    # After each trade closes
    cb.record_trade_result(pnl=100.0, is_win=True, portfolio_value=50000.0)
    
    # After equity updates
    cb.update_equity(current_equity=49500.0, starting_capital=50000.0)
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker limits."""
    
    # Loss limits
    daily_loss_limit_pct: float = 0.02    # -2% daily
    weekly_loss_limit_pct: float = 0.05   # -5% weekly
    max_drawdown_pct: float = 0.10        # -10% from peak
    
    # Consecutive loss limit
    consecutive_loss_limit: int = 5
    
    # Trading limits
    max_daily_trades: int = 20
    
    # Recovery
    cooldown_hours: int = 24              # Hours before auto-reset after trip
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitBreakerConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CircuitBreakerState:
    """Persistent state for circuit breaker."""
    
    # P&L tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    
    # Trade counting
    trades_today: int = 0
    consecutive_losses: int = 0
    
    # Equity tracking
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Trip state
    is_tripped: bool = False
    trip_reason: str = ""
    trip_time: str = ""
    
    # Reset tracking
    last_reset_date: str = ""
    last_week_reset: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitBreakerState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CircuitBreaker:
    """
    Risk management circuit breaker that halts trading when limits are breached.
    
    This is a standalone implementation that can be used in any trading system.
    State is persisted to a JSON file for recovery across restarts.
    
    Args:
        config: CircuitBreakerConfig or dict with limit settings
        state_path: Path to JSON file for persisting state
    
    Example:
        >>> cb = CircuitBreaker()
        >>> can_trade, reason = cb.can_trade()
        >>> if can_trade:
        ...     # Execute trade
        ...     cb.record_trade_result(pnl=50.0, is_win=True, portfolio_value=10000.0)
    """
    
    # Default state file location
    DEFAULT_STATE_PATH = Path(__file__).parent.parent.parent / "circuit_breaker_state.json"
    
    def __init__(
        self,
        config: Optional[CircuitBreakerConfig | dict] = None,
        state_path: Optional[Path | str] = None,
    ):
        # Handle config
        if config is None:
            self.config = CircuitBreakerConfig()
        elif isinstance(config, dict):
            self.config = CircuitBreakerConfig.from_dict(config)
        else:
            self.config = config
        
        # Handle state path
        self.state_path = Path(state_path) if state_path else self.DEFAULT_STATE_PATH
        
        # Load state
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
                    if hours_since_trip >= self.config.cooldown_hours:
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
        """
        Update peak equity and drawdown tracking.
        
        Args:
            current_equity: Current portfolio value
            starting_capital: Initial capital for reference
        """
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (self.state.peak_equity - current_equity) / self.state.peak_equity
        
        # Check max drawdown
        if self.state.current_drawdown_pct >= self.config.max_drawdown_pct:
            self._trip(f"Max drawdown exceeded: {self.state.current_drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}")
        
        self._save_state()
    
    def record_trade_result(self, pnl: float, is_win: bool, portfolio_value: float):
        """
        Record a trade result and check limits.
        
        Args:
            pnl: Dollar P&L of the trade
            is_win: Whether the trade was profitable
            portfolio_value: Current portfolio value for percentage calculation
        """
        pnl_pct = pnl / portfolio_value if portfolio_value > 0 else 0
        
        self.state.daily_pnl += pnl_pct
        self.state.weekly_pnl += pnl_pct
        self.state.trades_today += 1
        
        if is_win:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
        
        # Check limits
        if self.state.daily_pnl <= -self.config.daily_loss_limit_pct:
            self._trip(f"Daily loss limit: {self.state.daily_pnl:.2%} <= -{self.config.daily_loss_limit_pct:.2%}")
        elif self.state.weekly_pnl <= -self.config.weekly_loss_limit_pct:
            self._trip(f"Weekly loss limit: {self.state.weekly_pnl:.2%} <= -{self.config.weekly_loss_limit_pct:.2%}")
        elif self.state.consecutive_losses >= self.config.consecutive_loss_limit:
            self._trip(f"Consecutive losses: {self.state.consecutive_losses} >= {self.config.consecutive_loss_limit}")
        
        self._save_state()
    
    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if self.state.is_tripped:
            return False, f"Circuit breaker tripped: {self.state.trip_reason}"
        
        if self.state.trades_today >= self.config.max_daily_trades:
            return False, f"Max daily trades reached: {self.state.trades_today}/{self.config.max_daily_trades}"
        
        return True, "OK"
    
    def get_status(self) -> dict:
        """Get current circuit breaker status as a dictionary."""
        return {
            "is_tripped": self.state.is_tripped,
            "trip_reason": self.state.trip_reason,
            "daily_pnl": f"{self.state.daily_pnl:.2%}",
            "weekly_pnl": f"{self.state.weekly_pnl:.2%}",
            "consecutive_losses": self.state.consecutive_losses,
            "drawdown": f"{self.state.current_drawdown_pct:.2%}",
            "trades_today": self.state.trades_today,
            "limits": {
                "daily_loss_limit": f"{self.config.daily_loss_limit_pct:.1%}",
                "weekly_loss_limit": f"{self.config.weekly_loss_limit_pct:.1%}",
                "max_drawdown": f"{self.config.max_drawdown_pct:.1%}",
                "consecutive_loss_limit": self.config.consecutive_loss_limit,
                "max_daily_trades": self.config.max_daily_trades,
            }
        }
    
    def force_reset(self):
        """Manually reset circuit breaker (use with caution)."""
        self._clear_trip("Manual force reset")
        self.state.daily_pnl = 0.0
        self.state.consecutive_losses = 0
        self._save_state()
        logger.warning("⚠️ Circuit breaker force reset!")
    
    def __repr__(self) -> str:
        status = "TRIPPED" if self.state.is_tripped else "OK"
        return f"CircuitBreaker(status={status}, daily_pnl={self.state.daily_pnl:.2%}, trades={self.state.trades_today})"
