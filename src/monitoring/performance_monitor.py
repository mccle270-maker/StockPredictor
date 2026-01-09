"""
Real-time Performance Monitoring System for Stock Predictor

Provides:
- Rolling Sharpe ratio calculations (21-day, 63-day)
- Drawdown tracking from peak equity
- Win rate tracking over last N trades
- Alert system with configurable thresholds
- Daily summary reports (JSON)
- Optional Slack webhook notifications

Usage:
    from src.monitoring import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Record a trade
    monitor.record_trade(
        symbol="AAPL",
        side="BUY",
        qty=10,
        entry_price=150.0,
        exit_price=155.0,
        pnl=50.0,
        strategy="LONG"
    )
    
    # Check for alerts
    alerts = monitor.check_alerts()
    for alert in alerts:
        print(f"[{alert.level}] {alert.message}")
    
    # Generate daily summary
    summary = monitor.generate_daily_summary()
    print(summary.to_json())
"""

import json
import logging
import os
import requests
from pathlib import Path
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(__file__).parent.parent.parent
PERFORMANCE_DATA_PATH = BASE_DIR / ".monitoring" / "performance_data.json"
DAILY_SUMMARIES_PATH = BASE_DIR / ".monitoring" / "daily_summaries"
ALERTS_LOG_PATH = BASE_DIR / ".monitoring" / "alerts.log"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Represents a performance alert."""
    level: AlertLevel
    category: str  # e.g., "sharpe", "drawdown", "win_rate"
    message: str
    value: float  # The value that triggered the alert
    threshold: float  # The threshold that was breached
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }
    
    def __str__(self) -> str:
        return f"[{self.level.value}] {self.category}: {self.message}"


@dataclass
class TradeRecord:
    """Record of a single trade for performance tracking."""
    trade_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    strategy: str = ""
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TradeRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class PerformanceMetrics:
    """Current performance metrics snapshot."""
    # Equity tracking
    current_equity: float = 0.0
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Rolling Sharpe ratios
    sharpe_21d: float = 0.0
    sharpe_63d: float = 0.0
    
    # Win rate
    win_rate_20: float = 0.0  # Last 20 trades
    win_rate_all: float = 0.0  # All time
    
    # P&L
    pnl_today: float = 0.0
    pnl_week: float = 0.0
    pnl_month: float = 0.0
    pnl_total: float = 0.0
    
    # Trade counts
    trades_today: int = 0
    trades_week: int = 0
    trades_total: int = 0
    
    # Timestamp
    last_updated: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TickerPerformance:
    """Performance breakdown for a single ticker."""
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    pnl_total: float = 0.0
    pnl_avg: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailySummary:
    """Daily performance summary report."""
    date: str
    metrics: PerformanceMetrics
    ticker_breakdown: Dict[str, TickerPerformance]
    active_alerts: List[Alert]
    trades_today: List[TradeRecord]
    
    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "metrics": self.metrics.to_dict(),
            "ticker_breakdown": {k: v.to_dict() for k, v in self.ticker_breakdown.items()},
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "trades_today": [t.to_dict() for t in self.trades_today],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Optional[Path] = None):
        """Save summary to JSON file."""
        if path is None:
            DAILY_SUMMARIES_PATH.mkdir(parents=True, exist_ok=True)
            path = DAILY_SUMMARIES_PATH / f"summary_{self.date}.json"
        
        path.write_text(self.to_json())
        logger.info(f"📊 Daily summary saved to {path}")


class AlertConfig:
    """Configuration for alert thresholds."""
    
    def __init__(
        self,
        sharpe_critical: float = -0.5,
        sharpe_warning: float = 0.0,
        drawdown_critical: float = 0.08,  # 8%
        drawdown_warning: float = 0.05,   # 5%
        win_rate_warning: float = 0.45,   # 45%
        win_rate_critical: float = 0.35,  # 35%
        consecutive_loss_warning: int = 3,
        consecutive_loss_critical: int = 5,
    ):
        self.sharpe_critical = sharpe_critical
        self.sharpe_warning = sharpe_warning
        self.drawdown_critical = drawdown_critical
        self.drawdown_warning = drawdown_warning
        self.win_rate_warning = win_rate_warning
        self.win_rate_critical = win_rate_critical
        self.consecutive_loss_warning = consecutive_loss_warning
        self.consecutive_loss_critical = consecutive_loss_critical


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Tracks trades, calculates rolling metrics, generates alerts,
    and produces daily summary reports.
    
    Args:
        starting_capital: Initial capital for return calculations
        data_path: Path to performance data JSON file
        alert_config: Alert threshold configuration
        slack_webhook_url: Optional Slack webhook for critical alerts
    
    Example:
        >>> monitor = PerformanceMonitor(starting_capital=50000.0)
        >>> monitor.record_trade("AAPL", "BUY", 10, 150.0, 155.0, 50.0)
        >>> alerts = monitor.check_alerts()
        >>> summary = monitor.generate_daily_summary()
    """
    
    def __init__(
        self,
        starting_capital: float = 50000.0,
        data_path: Optional[Path] = None,
        alert_config: Optional[AlertConfig] = None,
        slack_webhook_url: Optional[str] = None,
    ):
        self.starting_capital = starting_capital
        self.data_path = Path(data_path) if data_path else PERFORMANCE_DATA_PATH
        self.alert_config = alert_config or AlertConfig()
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        
        # Initialize data structures
        self.trades: List[TradeRecord] = []
        self.daily_returns: List[Dict[str, Any]] = []  # {date, return_pct, equity}
        self.metrics = PerformanceMetrics()
        self.active_alerts: List[Alert] = []
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load performance data from disk."""
        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            return
        
        try:
            data = json.loads(self.data_path.read_text())
            self.trades = [TradeRecord.from_dict(t) for t in data.get("trades", [])]
            self.daily_returns = data.get("daily_returns", [])
            
            # Reconstruct metrics
            if "metrics" in data:
                m = data["metrics"]
                self.metrics = PerformanceMetrics(**{
                    k: v for k, v in m.items() 
                    if k in PerformanceMetrics.__dataclass_fields__
                })
            
            logger.info(f"📂 Loaded {len(self.trades)} trades from {self.data_path}")
        except Exception as e:
            logger.warning(f"Failed to load performance data: {e}")
    
    def _save_data(self):
        """Save performance data to disk."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "trades": [t.to_dict() for t in self.trades],
                "daily_returns": self.daily_returns,
                "metrics": self.metrics.to_dict(),
                "last_saved": datetime.now(timezone.utc).isoformat(),
            }
            self.data_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save performance data: {e}")
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        trade_id: Optional[str] = None,
        entry_time: Optional[str] = None,
        exit_time: Optional[str] = None,
        strategy: str = "",
        notes: str = "",
    ) -> TradeRecord:
        """
        Record a completed trade for performance tracking.
        
        Args:
            symbol: Ticker symbol
            side: "BUY" or "SELL"
            qty: Number of shares/contracts
            entry_price: Entry price
            exit_price: Exit price
            pnl: Dollar P&L
            trade_id: Optional unique ID
            entry_time: ISO timestamp of entry
            exit_time: ISO timestamp of exit
            strategy: Strategy name (e.g., "LONG", "SHORT")
            notes: Additional notes
        
        Returns:
            TradeRecord object
        """
        now = datetime.now(timezone.utc)
        
        # Calculate P&L percentage
        position_value = qty * entry_price
        pnl_pct = (pnl / position_value * 100) if position_value > 0 else 0
        
        record = TradeRecord(
            trade_id=trade_id or f"{symbol}_{now.strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol.upper(),
            side=side.upper(),
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time or now.isoformat(),
            exit_time=exit_time or now.isoformat(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy=strategy,
            notes=notes,
        )
        
        self.trades.append(record)
        
        # Update metrics
        self._update_metrics_after_trade(record)
        
        # Save data
        self._save_data()
        
        logger.info(f"📝 Trade recorded: {symbol} {side} {qty}@{entry_price:.2f} → {exit_price:.2f} | P&L: ${pnl:+.2f}")
        
        return record
    
    def _update_metrics_after_trade(self, trade: TradeRecord):
        """Update metrics after a new trade."""
        today = date.today().isoformat()
        week_start = (date.today() - timedelta(days=date.today().weekday())).isoformat()
        month_start = date.today().replace(day=1).isoformat()
        
        # Update P&L totals
        self.metrics.pnl_total += trade.pnl
        
        # Count trades
        self.metrics.trades_total = len(self.trades)
        
        # Today's P&L and trades
        today_trades = [t for t in self.trades if t.exit_time.startswith(today)]
        self.metrics.pnl_today = sum(t.pnl for t in today_trades)
        self.metrics.trades_today = len(today_trades)
        
        # Week's P&L and trades
        week_trades = [t for t in self.trades if t.exit_time >= week_start]
        self.metrics.pnl_week = sum(t.pnl for t in week_trades)
        self.metrics.trades_week = len(week_trades)
        
        # Month's P&L
        month_trades = [t for t in self.trades if t.exit_time >= month_start]
        self.metrics.pnl_month = sum(t.pnl for t in month_trades)
        
        # Update equity
        self.metrics.current_equity = self.starting_capital + self.metrics.pnl_total
        if self.metrics.current_equity > self.metrics.peak_equity:
            self.metrics.peak_equity = self.metrics.current_equity
        
        # Calculate drawdown
        if self.metrics.peak_equity > 0:
            self.metrics.current_drawdown_pct = (
                (self.metrics.peak_equity - self.metrics.current_equity) 
                / self.metrics.peak_equity
            )
        
        # Calculate win rates
        self._calculate_win_rates()
        
        # Calculate rolling Sharpe ratios
        self._calculate_rolling_sharpe()
        
        # Update timestamp
        self.metrics.last_updated = datetime.now(timezone.utc).isoformat()
    
    def _calculate_win_rates(self):
        """Calculate win rates."""
        if not self.trades:
            return
        
        # All-time win rate
        wins = sum(1 for t in self.trades if t.is_win)
        self.metrics.win_rate_all = wins / len(self.trades)
        
        # Last 20 trades win rate
        recent_trades = self.trades[-20:]
        if recent_trades:
            recent_wins = sum(1 for t in recent_trades if t.is_win)
            self.metrics.win_rate_20 = recent_wins / len(recent_trades)
    
    def _calculate_rolling_sharpe(self):
        """Calculate rolling Sharpe ratios."""
        # Group trades by date and calculate daily returns
        daily_pnl: Dict[str, float] = {}
        for trade in self.trades:
            trade_date = trade.exit_time[:10]  # YYYY-MM-DD
            daily_pnl[trade_date] = daily_pnl.get(trade_date, 0) + trade.pnl
        
        if not daily_pnl:
            return
        
        # Convert to daily return percentages
        dates = sorted(daily_pnl.keys())
        equity = self.starting_capital
        daily_returns = []
        
        for d in dates:
            pnl = daily_pnl[d]
            ret_pct = pnl / equity if equity > 0 else 0
            daily_returns.append(ret_pct)
            equity += pnl
        
        # Store for persistence
        self.daily_returns = [
            {"date": d, "pnl": daily_pnl[d], "return_pct": r}
            for d, r in zip(dates, daily_returns)
        ]
        
        # Calculate 21-day Sharpe
        if len(daily_returns) >= 21:
            self.metrics.sharpe_21d = self._calculate_sharpe(daily_returns[-21:])
        elif len(daily_returns) >= 5:
            self.metrics.sharpe_21d = self._calculate_sharpe(daily_returns)
        
        # Calculate 63-day Sharpe
        if len(daily_returns) >= 63:
            self.metrics.sharpe_63d = self._calculate_sharpe(daily_returns[-63:])
        elif len(daily_returns) >= 21:
            self.metrics.sharpe_63d = self._calculate_sharpe(daily_returns)
    
    def _calculate_sharpe(self, returns: List[float], annualization_factor: float = 252) -> float:
        """
        Calculate Sharpe ratio from a list of returns.
        
        Args:
            returns: List of daily returns (as decimals, e.g., 0.01 = 1%)
            annualization_factor: Trading days per year (default 252)
        
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe (assuming 0 risk-free rate)
            sharpe = (mean_return / std_return) * math.sqrt(annualization_factor)
            return round(sharpe, 3)
        except Exception:
            return 0.0
    
    def update_equity(self, current_equity: float):
        """
        Update equity tracking (call when portfolio value changes).
        
        Args:
            current_equity: Current portfolio value
        """
        self.metrics.current_equity = current_equity
        
        if current_equity > self.metrics.peak_equity:
            self.metrics.peak_equity = current_equity
        
        if self.metrics.peak_equity > 0:
            self.metrics.current_drawdown_pct = (
                (self.metrics.peak_equity - current_equity) / self.metrics.peak_equity
            )
        
        self.metrics.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_data()
    
    def check_alerts(self) -> List[Alert]:
        """
        Check current metrics against alert thresholds.
        
        Returns:
            List of triggered alerts
        """
        alerts = []
        config = self.alert_config
        
        # Check rolling Sharpe (21-day)
        if self.metrics.sharpe_21d < config.sharpe_critical:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                category="sharpe",
                message=f"21-day Sharpe ratio critically low: {self.metrics.sharpe_21d:.2f}",
                value=self.metrics.sharpe_21d,
                threshold=config.sharpe_critical,
            ))
        elif self.metrics.sharpe_21d < config.sharpe_warning:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                category="sharpe",
                message=f"21-day Sharpe ratio below target: {self.metrics.sharpe_21d:.2f}",
                value=self.metrics.sharpe_21d,
                threshold=config.sharpe_warning,
            ))
        
        # Check drawdown
        if self.metrics.current_drawdown_pct > config.drawdown_critical:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                category="drawdown",
                message=f"Drawdown exceeds critical threshold: {self.metrics.current_drawdown_pct:.1%}",
                value=self.metrics.current_drawdown_pct,
                threshold=config.drawdown_critical,
            ))
        elif self.metrics.current_drawdown_pct > config.drawdown_warning:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                category="drawdown",
                message=f"Drawdown approaching critical: {self.metrics.current_drawdown_pct:.1%}",
                value=self.metrics.current_drawdown_pct,
                threshold=config.drawdown_warning,
            ))
        
        # Check win rate (only if we have enough trades)
        if len(self.trades) >= 10:
            if self.metrics.win_rate_20 < config.win_rate_critical:
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    category="win_rate",
                    message=f"Win rate critically low: {self.metrics.win_rate_20:.1%}",
                    value=self.metrics.win_rate_20,
                    threshold=config.win_rate_critical,
                ))
            elif self.metrics.win_rate_20 < config.win_rate_warning:
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="win_rate",
                    message=f"Win rate below target: {self.metrics.win_rate_20:.1%}",
                    value=self.metrics.win_rate_20,
                    threshold=config.win_rate_warning,
                ))
        
        # Check consecutive losses
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= config.consecutive_loss_critical:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                category="consecutive_losses",
                message=f"Consecutive losses: {consecutive_losses}",
                value=consecutive_losses,
                threshold=config.consecutive_loss_critical,
            ))
        elif consecutive_losses >= config.consecutive_loss_warning:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                category="consecutive_losses",
                message=f"Building consecutive losses: {consecutive_losses}",
                value=consecutive_losses,
                threshold=config.consecutive_loss_warning,
            ))
        
        # Store active alerts
        self.active_alerts = alerts
        
        # Log alerts
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                logger.critical(str(alert))
            elif alert.level == AlertLevel.WARNING:
                logger.warning(str(alert))
        
        # Send Slack notification for critical alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            self._send_slack_alert(critical_alerts)
        
        return alerts
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losses from most recent trades."""
        if not self.trades:
            return 0
        
        count = 0
        for trade in reversed(self.trades):
            if trade.is_win:
                break
            count += 1
        return count
    
    def _send_slack_alert(self, alerts: List[Alert]):
        """Send critical alerts to Slack webhook."""
        if not self.slack_webhook_url:
            return
        
        try:
            message_parts = ["🚨 *CRITICAL TRADING ALERTS*"]
            for alert in alerts:
                message_parts.append(f"• {alert.category.upper()}: {alert.message}")
            
            message_parts.append(f"\n📊 Current Metrics:")
            message_parts.append(f"  • Equity: ${self.metrics.current_equity:,.2f}")
            message_parts.append(f"  • Drawdown: {self.metrics.current_drawdown_pct:.1%}")
            message_parts.append(f"  • 21d Sharpe: {self.metrics.sharpe_21d:.2f}")
            message_parts.append(f"  • Win Rate: {self.metrics.win_rate_20:.1%}")
            
            payload = {
                "text": "\n".join(message_parts),
                "username": "StockPredictor Monitor",
                "icon_emoji": ":chart_with_downwards_trend:",
            }
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10,
            )
            
            if response.status_code == 200:
                logger.info("📱 Slack alert sent successfully")
            else:
                logger.warning(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Failed to send Slack alert: {e}")
    
    def get_ticker_breakdown(self) -> Dict[str, TickerPerformance]:
        """Get performance breakdown by ticker."""
        breakdown: Dict[str, TickerPerformance] = {}
        
        for trade in self.trades:
            symbol = trade.symbol
            if symbol not in breakdown:
                breakdown[symbol] = TickerPerformance(symbol=symbol)
            
            perf = breakdown[symbol]
            perf.trades += 1
            perf.pnl_total += trade.pnl
            
            if trade.is_win:
                perf.wins += 1
            else:
                perf.losses += 1
            
            if trade.pnl > perf.best_trade:
                perf.best_trade = trade.pnl
            if trade.pnl < perf.worst_trade:
                perf.worst_trade = trade.pnl
        
        # Calculate derived metrics
        for symbol, perf in breakdown.items():
            if perf.trades > 0:
                perf.win_rate = perf.wins / perf.trades
                perf.pnl_avg = perf.pnl_total / perf.trades
        
        return breakdown
    
    def generate_daily_summary(self, for_date: Optional[date] = None) -> DailySummary:
        """
        Generate a daily performance summary.
        
        Args:
            for_date: Date to generate summary for (default: today)
        
        Returns:
            DailySummary object
        """
        if for_date is None:
            for_date = date.today()
        
        date_str = for_date.isoformat()
        
        # Get trades for the day
        trades_today = [
            t for t in self.trades 
            if t.exit_time.startswith(date_str)
        ]
        
        # Refresh metrics
        self._calculate_win_rates()
        self._calculate_rolling_sharpe()
        
        # Check alerts
        alerts = self.check_alerts()
        
        summary = DailySummary(
            date=date_str,
            metrics=self.metrics,
            ticker_breakdown=self.get_ticker_breakdown(),
            active_alerts=alerts,
            trades_today=trades_today,
        )
        
        # Auto-save summary
        summary.save()
        
        return summary
    
    def get_status(self) -> dict:
        """Get current monitoring status as a dictionary."""
        return {
            "metrics": self.metrics.to_dict(),
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "trades_count": len(self.trades),
            "last_trade": self.trades[-1].to_dict() if self.trades else None,
            "ticker_breakdown": {k: v.to_dict() for k, v in self.get_ticker_breakdown().items()},
        }
    
    def reset(self, confirm: bool = False):
        """
        Reset all performance data (use with caution).
        
        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            logger.warning("Reset requires confirm=True")
            return
        
        self.trades = []
        self.daily_returns = []
        self.metrics = PerformanceMetrics()
        self.active_alerts = []
        self._save_data()
        logger.warning("⚠️ Performance data reset!")
    
    def __repr__(self) -> str:
        return (
            f"PerformanceMonitor(trades={len(self.trades)}, "
            f"equity=${self.metrics.current_equity:,.0f}, "
            f"sharpe_21d={self.metrics.sharpe_21d:.2f})"
        )


# ============================================================
# Convenience functions for integration with auto_paper_trade
# ============================================================

_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor(starting_capital: float = 50000.0) -> PerformanceMonitor:
    """
    Get or create the global performance monitor instance.
    
    Args:
        starting_capital: Initial capital (only used if creating new instance)
    
    Returns:
        PerformanceMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(starting_capital=starting_capital)
    return _global_monitor


def record_trade_to_monitor(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    pnl: float,
    **kwargs
) -> TradeRecord:
    """
    Convenience function to record a trade to the global monitor.
    
    Returns:
        TradeRecord object
    """
    monitor = get_monitor()
    return monitor.record_trade(
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        **kwargs
    )


def check_alerts_and_notify() -> List[Alert]:
    """
    Check alerts on the global monitor and return them.
    
    Returns:
        List of triggered alerts
    """
    monitor = get_monitor()
    return monitor.check_alerts()


def export_daily_summary() -> DailySummary:
    """
    Export today's daily summary from the global monitor.
    
    Returns:
        DailySummary object
    """
    monitor = get_monitor()
    return monitor.generate_daily_summary()
