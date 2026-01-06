"""
Trade Limiter Module
====================

Limits the number of trades per ticker per period to prevent overtrading.
Signals are ranked by |z-score|, confidence, or predicted return.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional, Literal

logger = logging.getLogger("trade_limiter")
logger.setLevel(logging.INFO)


@dataclass
class SkippedSignal:
    """Record of a signal that was skipped due to trade limits."""
    ticker: str
    reason: str
    rank: int
    ranking_value: float
    ranking_method: str
    action: str
    pred_next_ret: float
    z_score: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeCount:
    """Tracks trade counts per ticker."""
    ticker: str
    period: str
    count: int
    max_allowed: int
    last_trade_date: str = ""
    
    @property
    def can_trade(self) -> bool:
        return self.count < self.max_allowed
    
    @property
    def remaining(self) -> int:
        return max(0, self.max_allowed - self.count)


class TradeLimiter:
    """
    Limits trades per ticker per period.
    
    Features:
    - Configurable max trades per ticker
    - Ranking by z-score, confidence, or return
    - Per-ticker overrides
    - Skipped signal logging
    """
    
    def __init__(
        self,
        max_trades_per_ticker: int = 1,
        period: Literal["day", "week", "session"] = "day",
        ranking_method: Literal["zscore", "confidence", "return"] = "zscore",
        log_path: Optional[Path] = None,
        trade_history: Optional[dict[str, list[str]]] = None,
    ):
        """
        Initialize trade limiter.
        
        Args:
            max_trades_per_ticker: Maximum trades per ticker per period
            period: "day", "week", or "session"
            ranking_method: How to rank signals ("zscore", "confidence", "return")
            log_path: Path to log skipped signals
            trade_history: Dict of ticker -> list of trade dates (ISO format)
        """
        self.max_trades_per_ticker = max_trades_per_ticker
        self.period = period
        self.ranking_method = ranking_method
        self.log_path = Path(log_path) if log_path else None
        
        # Trade history: ticker -> list of trade dates
        self._trade_history: dict[str, list[str]] = trade_history or {}
        
        # Skipped signals for current session
        self._skipped_signals: list[SkippedSignal] = []
        
        # Per-ticker overrides
        self._ticker_limits: dict[str, int] = {}
    
    def set_ticker_limit(self, ticker: str, max_trades: int):
        """Set custom trade limit for a specific ticker."""
        self._ticker_limits[ticker.upper()] = max_trades
    
    def get_ticker_limit(self, ticker: str) -> int:
        """Get trade limit for a ticker (checks overrides first)."""
        return self._ticker_limits.get(ticker.upper(), self.max_trades_per_ticker)
    
    def get_trade_count(self, ticker: str) -> TradeCount:
        """Get current trade count for a ticker in the current period."""
        ticker = ticker.upper()
        today = date.today()
        
        # Get period start date
        if self.period == "day":
            period_start = today
        elif self.period == "week":
            period_start = today - timedelta(days=today.weekday())
        else:  # session
            period_start = today  # Session resets each run
        
        # Count trades in period
        trade_dates = self._trade_history.get(ticker, [])
        count = sum(1 for d in trade_dates if d >= period_start.isoformat())
        
        return TradeCount(
            ticker=ticker,
            period=self.period,
            count=count,
            max_allowed=self.get_ticker_limit(ticker),
            last_trade_date=trade_dates[-1] if trade_dates else "",
        )
    
    def record_trade(self, ticker: str, trade_date: Optional[str] = None):
        """Record a trade for a ticker."""
        ticker = ticker.upper()
        if trade_date is None:
            trade_date = date.today().isoformat()
        
        if ticker not in self._trade_history:
            self._trade_history[ticker] = []
        self._trade_history[ticker].append(trade_date)
    
    def _get_ranking_value(self, signal: dict) -> float:
        """Extract ranking value from signal based on ranking method."""
        if self.ranking_method == "zscore":
            return abs(float(signal.get("z_score", 0.0)))
        elif self.ranking_method == "confidence":
            # Use confidence_score or fall back to |pred_next_ret|
            conf = signal.get("confidence_score", 0.0)
            if conf == 0.0:
                conf = abs(float(signal.get("pred_next_ret", 0.0)))
            return float(conf)
        else:  # return
            return abs(float(signal.get("pred_next_ret", 0.0)))
    
    def rank_signals(
        self,
        signals: dict[str, dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any], float, int]]:
        """
        Rank signals by the configured ranking method.
        
        Returns:
            List of (ticker, signal_dict, ranking_value, rank) tuples
            sorted by ranking_value descending (best first)
        """
        ranked = []
        for ticker, signal in signals.items():
            value = self._get_ranking_value(signal)
            ranked.append((ticker, signal, value))
        
        # Sort by value descending
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        # Add rank numbers
        return [(tk, sig, val, i + 1) for i, (tk, sig, val) in enumerate(ranked)]
    
    def apply_limits(
        self,
        signals: dict[str, dict[str, Any]],
        include_skipped: bool = True,
    ) -> tuple[dict[str, dict[str, Any]], list[SkippedSignal]]:
        """
        Apply trade limits to signals.
        
        Args:
            signals: Dict of ticker -> signal data
            include_skipped: If True, include skipped signals with trade_allowed=False
            
        Returns:
            (filtered_signals, skipped_signals)
            
        Signals are:
        1. Ranked by the configured method
        2. For each ticker, only top N signals are allowed (based on limit)
        3. Skipped signals are logged and optionally included with flag
        """
        if not signals:
            return {}, []
        
        # Track signals per ticker
        ticker_signals: dict[str, list[tuple[dict, float, int]]] = defaultdict(list)
        
        # Rank all signals
        ranked = self.rank_signals(signals)
        
        for ticker, signal, ranking_value, rank in ranked:
            ticker_signals[ticker].append((signal, ranking_value, rank))
        
        # Apply limits
        allowed_signals: dict[str, dict[str, Any]] = {}
        skipped: list[SkippedSignal] = []
        
        for ticker, sig_list in ticker_signals.items():
            limit = self.get_ticker_limit(ticker)
            trade_count = self.get_trade_count(ticker)
            
            # Sort by ranking value (best first)
            sig_list.sort(key=lambda x: x[1], reverse=True)
            
            for i, (signal, ranking_value, rank) in enumerate(sig_list):
                # Check if we can take this trade
                current_session_count = i  # How many we've allowed this session
                total_period_count = trade_count.count + current_session_count
                
                can_trade = total_period_count < limit
                
                # Add limit metadata to signal
                signal["trade_allowed"] = can_trade
                signal["trade_rank"] = rank
                signal["trade_rank_value"] = ranking_value
                signal["trade_rank_method"] = self.ranking_method
                signal["ticker_trade_limit"] = limit
                signal["ticker_trade_count"] = total_period_count
                
                if can_trade:
                    allowed_signals[ticker] = signal
                else:
                    reason = f"limit_exceeded_{limit}_per_{self.period}"
                    skipped_record = SkippedSignal(
                        ticker=ticker,
                        reason=reason,
                        rank=rank,
                        ranking_value=ranking_value,
                        ranking_method=self.ranking_method,
                        action=signal.get("action", signal.get("strategy", "HOLD")),
                        pred_next_ret=float(signal.get("pred_next_ret", 0.0)),
                        z_score=float(signal.get("z_score", 0.0)),
                    )
                    skipped.append(skipped_record)
                    self._log_skipped_signal(skipped_record)
                    
                    if include_skipped:
                        signal["skip_reason"] = reason
                        allowed_signals[ticker] = signal
        
        self._skipped_signals.extend(skipped)
        return allowed_signals, skipped
    
    def _log_skipped_signal(self, record: SkippedSignal):
        """Log a skipped signal to file."""
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log skipped signal: {e}")
        
        logger.info(
            f"SKIPPED: {record.ticker} | rank={record.rank} ({record.ranking_method}={record.ranking_value:.4f}) | "
            f"reason={record.reason}"
        )
    
    def get_skipped_signals(self, last_n: int = 100) -> list[SkippedSignal]:
        """Get recent skipped signals."""
        return self._skipped_signals[-last_n:]
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary of trade limiting status."""
        return {
            "max_trades_per_ticker": self.max_trades_per_ticker,
            "period": self.period,
            "ranking_method": self.ranking_method,
            "skipped_count": len(self._skipped_signals),
            "trade_history_tickers": list(self._trade_history.keys()),
        }
    
    def reset_session(self):
        """Reset session-specific data (for new prediction run)."""
        self._skipped_signals = []
    
    def clear_history(self, ticker: Optional[str] = None):
        """Clear trade history for a ticker or all tickers."""
        if ticker:
            self._trade_history.pop(ticker.upper(), None)
        else:
            self._trade_history = {}


# Default global limiter instance
_default_limiter: Optional[TradeLimiter] = None


def get_trade_limiter() -> TradeLimiter:
    """Get or create the default trade limiter instance."""
    global _default_limiter
    
    if _default_limiter is None:
        try:
            from ..config import (
                TRADE_LIMIT_CONFIG,
                TRADE_LIMIT_OVERRIDES,
                get_skipped_log_path,
            )
            _default_limiter = TradeLimiter(
                max_trades_per_ticker=TRADE_LIMIT_CONFIG.get("max_trades_per_ticker", 1),
                period=TRADE_LIMIT_CONFIG.get("period", "day"),
                ranking_method=TRADE_LIMIT_CONFIG.get("ranking_method", "zscore"),
                log_path=get_skipped_log_path(),
            )
            # Apply per-ticker overrides
            for ticker, override in TRADE_LIMIT_OVERRIDES.items():
                if "max_trades_per_ticker" in override:
                    _default_limiter.set_ticker_limit(ticker, override["max_trades_per_ticker"])
        except ImportError:
            _default_limiter = TradeLimiter()
    
    return _default_limiter


def reset_trade_limiter():
    """Reset the default limiter (e.g., after config changes)."""
    global _default_limiter
    _default_limiter = None


def apply_trade_limits(
    signals: dict[str, dict[str, Any]],
    include_skipped: bool = True,
) -> tuple[dict[str, dict[str, Any]], list[SkippedSignal]]:
    """
    Convenience function to apply trade limits using default limiter.
    
    Args:
        signals: Dict of ticker -> signal data
        include_skipped: If True, include skipped signals with trade_allowed=False
        
    Returns:
        (limited_signals, skipped_signals)
    """
    limiter = get_trade_limiter()
    return limiter.apply_limits(signals, include_skipped=include_skipped)


def rank_signals_by_conviction(
    signals: dict[str, dict[str, Any]],
    method: Literal["zscore", "confidence", "return"] = "zscore",
) -> list[tuple[str, dict[str, Any], float, int]]:
    """
    Rank signals by conviction without applying limits.
    
    Returns list of (ticker, signal, value, rank) tuples.
    """
    limiter = TradeLimiter(ranking_method=method)
    return limiter.rank_signals(signals)
