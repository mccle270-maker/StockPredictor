"""
Market Regime Filter Module
===========================

Filters trades based on market regime indicators:
- SPY 200DMA (trend direction)
- VIX level (volatility regime)
- SPY RSI (overbought/oversold)
- Market breadth (advance/decline)

Long trades blocked in bearish regime, shorts blocked in bullish regime.
Tickers are NEVER disabled entirely - only specific trade directions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Literal, Optional
from enum import Enum

import pandas as pd
import numpy as np

logger = logging.getLogger("regime_filter")
logger.setLevel(logging.INFO)


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"      # SPY > 200DMA + RSI healthy + VIX low
    BULL = "bull"                     # SPY > 200DMA
    NEUTRAL = "neutral"               # Mixed signals
    BEAR = "bear"                     # SPY < 200DMA
    STRONG_BEAR = "strong_bear"       # SPY < 200DMA + RSI weak + VIX high
    CRASH = "crash"                   # VIX spike, extreme conditions


@dataclass
class RegimeState:
    """Current market regime state with all indicators."""
    regime: MarketRegime
    spy_price: float
    spy_200dma: float
    spy_vs_200dma_pct: float          # (price - 200dma) / 200dma * 100
    spy_rsi: float
    vix_level: float
    vix_percentile: float             # VIX percentile over lookback
    breadth_ratio: float              # Advance/decline ratio (if available)
    timestamp: str = ""
    
    # Trade direction allowances
    longs_allowed: bool = True
    shorts_allowed: bool = True
    long_reason: str = ""
    short_reason: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


@dataclass
class BlockedTrade:
    """Record of a trade blocked by regime filter."""
    ticker: str
    action: str                       # BUY, SELL, LONG, SHORT
    direction: Literal["long", "short"]
    reason: str
    regime: str
    spy_vs_200dma_pct: float
    vix_level: float
    pred_next_ret: float
    z_score: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)


class RegimeFilter:
    """
    Filters trades based on market regime.
    
    Rules:
    - STRONG_BEAR/CRASH: Block all longs, allow shorts
    - BEAR: Block longs on low-conviction signals, allow shorts
    - NEUTRAL: Allow both directions
    - BULL: Block shorts on low-conviction signals, allow longs
    - STRONG_BULL: Allow all longs, block most shorts
    
    Tickers are NEVER disabled entirely.
    """
    
    def __init__(
        self,
        spy_dma_period: int = 200,
        vix_high_threshold: float = 25.0,
        vix_extreme_threshold: float = 35.0,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        min_conviction_override: float = 2.0,  # |z-score| to override regime block
        log_path: Optional[Path] = None,
    ):
        """
        Initialize regime filter.
        
        Args:
            spy_dma_period: Period for SPY moving average (default 200)
            vix_high_threshold: VIX level considered "high" (default 25)
            vix_extreme_threshold: VIX level considered "extreme/crash" (default 35)
            rsi_oversold: RSI level considered oversold (default 30)
            rsi_overbought: RSI level considered overbought (default 70)
            min_conviction_override: Minimum |z-score| to override regime block
            log_path: Path to log blocked trades
        """
        self.spy_dma_period = spy_dma_period
        self.vix_high_threshold = vix_high_threshold
        self.vix_extreme_threshold = vix_extreme_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_conviction_override = min_conviction_override
        self.log_path = Path(log_path) if log_path else None
        
        # Cache for regime state
        self._current_regime: Optional[RegimeState] = None
        self._blocked_trades: list[BlockedTrade] = []
        
        # SPY data cache
        self._spy_data: Optional[pd.DataFrame] = None
        self._spy_data_date: Optional[date] = None
    
    def _fetch_spy_data(self, lookback_days: int = 252) -> pd.DataFrame:
        """Fetch SPY price data for regime calculation."""
        today = date.today()
        
        # Use cache if fresh
        if self._spy_data is not None and self._spy_data_date == today:
            return self._spy_data
        
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="2y")
            
            if hist.empty:
                logger.warning("Failed to fetch SPY data, using fallback")
                return pd.DataFrame()
            
            self._spy_data = hist
            self._spy_data_date = today
            return hist
            
        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return pd.DataFrame()
    
    def _fetch_vix_data(self) -> tuple[float, float]:
        """Fetch VIX level and percentile."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y")
            
            if hist.empty:
                return 20.0, 50.0  # Default neutral values
            
            current_vix = float(hist["Close"].iloc[-1])
            
            # Calculate percentile over 1 year
            vix_percentile = (hist["Close"] < current_vix).mean() * 100
            
            return current_vix, float(vix_percentile)
            
        except Exception as e:
            logger.warning(f"Error fetching VIX data: {e}")
            return 20.0, 50.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for a price series."""
        if len(prices) < period + 1:
            return 50.0  # Neutral default
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def get_regime_state(self, refresh: bool = False) -> RegimeState:
        """
        Calculate current market regime state.
        
        Args:
            refresh: Force refresh of cached data
            
        Returns:
            RegimeState with all indicators and trade allowances
        """
        if self._current_regime is not None and not refresh:
            # Check if cache is from today
            cache_date = self._current_regime.timestamp[:10]
            if cache_date == date.today().isoformat():
                return self._current_regime
        
        # Fetch data
        spy_data = self._fetch_spy_data()
        vix_level, vix_percentile = self._fetch_vix_data()
        
        # Calculate indicators
        if spy_data.empty:
            # Fallback neutral state
            state = RegimeState(
                regime=MarketRegime.NEUTRAL,
                spy_price=0.0,
                spy_200dma=0.0,
                spy_vs_200dma_pct=0.0,
                spy_rsi=50.0,
                vix_level=vix_level,
                vix_percentile=vix_percentile,
                breadth_ratio=1.0,
                longs_allowed=True,
                shorts_allowed=True,
                long_reason="Data unavailable - allowing all",
                short_reason="Data unavailable - allowing all",
            )
            self._current_regime = state
            return state
        
        spy_price = float(spy_data["Close"].iloc[-1])
        spy_200dma = float(spy_data["Close"].rolling(self.spy_dma_period).mean().iloc[-1])
        spy_vs_200dma_pct = ((spy_price - spy_200dma) / spy_200dma) * 100
        spy_rsi = self._calculate_rsi(spy_data["Close"])
        
        # Determine regime
        regime = self._classify_regime(
            spy_vs_200dma_pct, spy_rsi, vix_level, vix_percentile
        )
        
        # Determine trade allowances
        longs_allowed, long_reason = self._check_long_allowed(regime, spy_rsi, vix_level)
        shorts_allowed, short_reason = self._check_short_allowed(regime, spy_rsi, vix_level)
        
        state = RegimeState(
            regime=regime,
            spy_price=spy_price,
            spy_200dma=spy_200dma,
            spy_vs_200dma_pct=spy_vs_200dma_pct,
            spy_rsi=spy_rsi,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            breadth_ratio=1.0,  # TODO: Add breadth data
            longs_allowed=longs_allowed,
            shorts_allowed=shorts_allowed,
            long_reason=long_reason,
            short_reason=short_reason,
        )
        
        self._current_regime = state
        logger.info(
            f"Regime: {regime.value} | SPY: {spy_price:.2f} vs 200DMA: {spy_200dma:.2f} "
            f"({spy_vs_200dma_pct:+.1f}%) | VIX: {vix_level:.1f} | RSI: {spy_rsi:.1f}"
        )
        
        return state
    
    def _classify_regime(
        self,
        spy_vs_200dma_pct: float,
        spy_rsi: float,
        vix_level: float,
        vix_percentile: float,
    ) -> MarketRegime:
        """Classify market regime based on indicators."""
        
        # Extreme conditions
        if vix_level >= self.vix_extreme_threshold:
            return MarketRegime.CRASH
        
        above_200dma = spy_vs_200dma_pct > 0
        
        if above_200dma:
            # Bullish territory
            if (
                spy_vs_200dma_pct > 5.0  # Well above 200DMA
                and vix_level < self.vix_high_threshold
                and spy_rsi > 50
            ):
                return MarketRegime.STRONG_BULL
            return MarketRegime.BULL
        else:
            # Bearish territory
            if (
                spy_vs_200dma_pct < -5.0  # Well below 200DMA
                and (vix_level > self.vix_high_threshold or spy_rsi < 40)
            ):
                return MarketRegime.STRONG_BEAR
            if spy_vs_200dma_pct < -2.0:
                return MarketRegime.BEAR
            
            # Close to 200DMA either side
            return MarketRegime.NEUTRAL
    
    def _check_long_allowed(
        self,
        regime: MarketRegime,
        spy_rsi: float,
        vix_level: float,
    ) -> tuple[bool, str]:
        """Check if long trades are allowed in current regime."""
        
        if regime == MarketRegime.CRASH:
            return False, f"CRASH regime (VIX={vix_level:.1f}) - longs blocked"
        
        if regime == MarketRegime.STRONG_BEAR:
            return False, f"STRONG_BEAR regime - longs blocked (need |z|≥{self.min_conviction_override})"
        
        if regime == MarketRegime.BEAR:
            # Allow with warning
            return True, f"BEAR regime - longs allowed but risky"
        
        # NEUTRAL, BULL, STRONG_BULL - longs OK
        return True, f"{regime.value} - longs allowed"
    
    def _check_short_allowed(
        self,
        regime: MarketRegime,
        spy_rsi: float,
        vix_level: float,
    ) -> tuple[bool, str]:
        """Check if short trades are allowed in current regime."""
        
        if regime == MarketRegime.STRONG_BULL:
            return False, f"STRONG_BULL regime - shorts blocked (need |z|≥{self.min_conviction_override})"
        
        if regime == MarketRegime.BULL:
            # Allow with warning
            return True, f"BULL regime - shorts allowed but risky"
        
        # NEUTRAL, BEAR, STRONG_BEAR, CRASH - shorts OK
        return True, f"{regime.value} - shorts allowed"
    
    def filter_signal(
        self,
        ticker: str,
        signal: dict[str, Any],
        regime_state: Optional[RegimeState] = None,
    ) -> tuple[dict[str, Any], Optional[BlockedTrade]]:
        """
        Filter a single signal based on regime.
        
        Args:
            ticker: Stock ticker
            signal: Signal dictionary
            regime_state: Pre-computed regime state (optional)
            
        Returns:
            (modified_signal, blocked_trade_record or None)
            
        Signal is modified with regime metadata but NEVER removed.
        If blocked, signal gets regime_blocked=True flag.
        """
        if regime_state is None:
            regime_state = self.get_regime_state()
        
        # Determine trade direction from signal
        action = signal.get("action", signal.get("strategy", "HOLD")).upper()
        pred_ret = float(signal.get("pred_next_ret", 0.0))
        z_score = float(signal.get("z_score", 0.0))
        
        # Infer direction
        if action in ("BUY", "LONG", "CALL"):
            direction = "long"
        elif action in ("SELL", "SHORT", "PUT"):
            direction = "short"
        elif pred_ret > 0:
            direction = "long"
        elif pred_ret < 0:
            direction = "short"
        else:
            direction = "neutral"
        
        # Add regime metadata to signal
        signal["regime"] = regime_state.regime.value
        signal["spy_vs_200dma_pct"] = regime_state.spy_vs_200dma_pct
        signal["vix_level"] = regime_state.vix_level
        signal["regime_longs_allowed"] = regime_state.longs_allowed
        signal["regime_shorts_allowed"] = regime_state.shorts_allowed
        
        # Check if blocked
        blocked = False
        block_reason = ""
        
        if direction == "long" and not regime_state.longs_allowed:
            # Check for high-conviction override
            if abs(z_score) >= self.min_conviction_override:
                signal["regime_override"] = True
                signal["regime_note"] = f"High conviction override (|z|={abs(z_score):.2f})"
            else:
                blocked = True
                block_reason = regime_state.long_reason
        
        elif direction == "short" and not regime_state.shorts_allowed:
            # Check for high-conviction override
            if abs(z_score) >= self.min_conviction_override:
                signal["regime_override"] = True
                signal["regime_note"] = f"High conviction override (|z|={abs(z_score):.2f})"
            else:
                blocked = True
                block_reason = regime_state.short_reason
        
        if blocked:
            signal["regime_blocked"] = True
            signal["regime_block_reason"] = block_reason
            
            blocked_record = BlockedTrade(
                ticker=ticker,
                action=action,
                direction=direction,
                reason=block_reason,
                regime=regime_state.regime.value,
                spy_vs_200dma_pct=regime_state.spy_vs_200dma_pct,
                vix_level=regime_state.vix_level,
                pred_next_ret=pred_ret,
                z_score=z_score,
            )
            self._log_blocked_trade(blocked_record)
            return signal, blocked_record
        else:
            signal["regime_blocked"] = False
            return signal, None
    
    def filter_signals(
        self,
        signals: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], list[BlockedTrade]]:
        """
        Filter all signals based on regime.
        
        Args:
            signals: Dict of ticker -> signal data
            
        Returns:
            (filtered_signals, blocked_trades)
            
        All signals are returned (none removed), but blocked ones
        have regime_blocked=True flag.
        """
        if not signals:
            return {}, []
        
        # Get regime once for all signals
        regime_state = self.get_regime_state()
        
        filtered = {}
        blocked_list = []
        
        for ticker, signal in signals.items():
            filtered_signal, blocked = self.filter_signal(
                ticker, signal.copy(), regime_state
            )
            filtered[ticker] = filtered_signal
            if blocked:
                blocked_list.append(blocked)
        
        self._blocked_trades.extend(blocked_list)
        
        logger.info(
            f"Regime filter: {len(signals)} signals, "
            f"{len(blocked_list)} blocked, "
            f"{len(signals) - len(blocked_list)} allowed"
        )
        
        return filtered, blocked_list
    
    def _log_blocked_trade(self, record: BlockedTrade):
        """Log a blocked trade to file."""
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.warning(f"Failed to log blocked trade: {e}")
        
        logger.info(
            f"BLOCKED: {record.ticker} {record.direction.upper()} | "
            f"regime={record.regime} | reason={record.reason}"
        )
    
    def get_blocked_trades(self, last_n: int = 100) -> list[BlockedTrade]:
        """Get recent blocked trades."""
        return self._blocked_trades[-last_n:]
    
    def get_regime_summary(self) -> dict[str, Any]:
        """Get summary of current regime and filter status."""
        state = self.get_regime_state()
        return {
            "regime": state.regime.value,
            "spy_price": state.spy_price,
            "spy_200dma": state.spy_200dma,
            "spy_vs_200dma_pct": state.spy_vs_200dma_pct,
            "spy_rsi": state.spy_rsi,
            "vix_level": state.vix_level,
            "longs_allowed": state.longs_allowed,
            "shorts_allowed": state.shorts_allowed,
            "blocked_count": len(self._blocked_trades),
        }
    
    def reset(self):
        """Reset filter state (for new session)."""
        self._blocked_trades = []
        self._current_regime = None


# Default global filter instance
_default_filter: Optional[RegimeFilter] = None


def get_regime_filter() -> RegimeFilter:
    """Get or create the default regime filter instance."""
    global _default_filter
    
    if _default_filter is None:
        try:
            from ..config import REGIME_FILTER_CONFIG, get_blocked_trades_log_path
            _default_filter = RegimeFilter(
                spy_dma_period=REGIME_FILTER_CONFIG.get("spy_dma_period", 200),
                vix_high_threshold=REGIME_FILTER_CONFIG.get("vix_high_threshold", 25.0),
                vix_extreme_threshold=REGIME_FILTER_CONFIG.get("vix_extreme_threshold", 35.0),
                rsi_oversold=REGIME_FILTER_CONFIG.get("rsi_oversold", 30.0),
                rsi_overbought=REGIME_FILTER_CONFIG.get("rsi_overbought", 70.0),
                min_conviction_override=REGIME_FILTER_CONFIG.get("min_conviction_override", 2.0),
                log_path=get_blocked_trades_log_path(),
            )
        except ImportError:
            _default_filter = RegimeFilter()
    
    return _default_filter


def reset_regime_filter():
    """Reset the default filter."""
    global _default_filter
    _default_filter = None


def apply_regime_filter(
    signals: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[BlockedTrade]]:
    """
    Convenience function to apply regime filter using default instance.
    
    Returns:
        (filtered_signals, blocked_trades)
    """
    rf = get_regime_filter()
    return rf.filter_signals(signals)


def get_current_regime() -> RegimeState:
    """Get current market regime state."""
    rf = get_regime_filter()
    return rf.get_regime_state()
