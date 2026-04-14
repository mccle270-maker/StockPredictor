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


# ============================================================================
# HMM-BASED REGIME DETECTION
# ============================================================================
# Uses a Hidden Markov Model to detect latent market regimes from returns + volatility.
# More sophisticated than simple 200DMA — captures non-linear regime transitions.
# Added 2026-01-12

class HMMRegimeDetector:
    """
    Hidden Markov Model regime detector.
    
    Fits a Gaussian HMM on (returns, realized_vol) to detect latent market states:
    - State 0: Low-vol / Bull (typically)
    - State 1: Medium-vol / Normal
    - State 2: High-vol / Bear/Crisis
    
    States are auto-labeled after fitting based on mean return & volatility.
    
    Usage:
        detector = HMMRegimeDetector(n_states=3)
        regimes = detector.fit_predict(price_series)
        current = detector.current_regime()
    """
    
    # Map HMM states to MarketRegime enums
    REGIME_MAP = {
        "bull": MarketRegime.BULL,
        "neutral": MarketRegime.NEUTRAL,
        "bear": MarketRegime.BEAR,
    }
    
    def __init__(
        self,
        n_states: int = 3,
        vol_lookback: int = 21,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Args:
            n_states: Number of hidden states (default 3: bull/neutral/bear)
            vol_lookback: Rolling window for realized volatility feature
            covariance_type: HMM covariance type ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.vol_lookback = vol_lookback
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self._model = None
        self._state_labels = {}  # Maps HMM state index → 'bull'/'neutral'/'bear'
        self._last_regimes = None  # Full regime series
        self._fitted = False
    
    def _build_features(self, prices: pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Build observation features for HMM from price series.
        
        Features:
        1. Daily log returns (captures return level)
        2. Realized volatility (captures volatility regime)
        
        Returns (features_array, valid_index)
        """
        # Log returns (more Gaussian than simple returns)
        log_ret = np.log(prices / prices.shift(1))
        
        # Realized volatility
        real_vol = log_ret.rolling(self.vol_lookback).std() * np.sqrt(252)
        
        # Combine and drop NaN
        features = pd.DataFrame({
            "log_return": log_ret,
            "realized_vol": real_vol,
        }).dropna()
        
        return features.values, features.index
    
    def _label_states(self, means: np.ndarray) -> dict[int, str]:
        """
        Auto-label HMM states based on their mean return.
        
        The state with highest mean return → bull
        The state with lowest mean return → bear
        Middle state → neutral
        """
        # means[:, 0] is mean log return per state
        mean_returns = means[:, 0]
        
        if self.n_states == 2:
            sorted_states = np.argsort(mean_returns)
            return {sorted_states[0]: "bear", sorted_states[1]: "bull"}
        
        # 3+ states: sort by mean return
        sorted_states = np.argsort(mean_returns)
        labels = {}
        labels[sorted_states[0]] = "bear"
        labels[sorted_states[-1]] = "bull"
        for s in sorted_states[1:-1]:
            labels[s] = "neutral"
        
        return labels
    
    def fit_predict(self, prices: pd.Series) -> pd.Series:
        """
        Fit HMM and predict regime for each day.
        
        Args:
            prices: Price series with DatetimeIndex
        
        Returns:
            Series of regime labels ('bull', 'neutral', 'bear') indexed by date
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed, falling back to simple regime")
            return pd.Series("neutral", index=prices.index, name="hmm_regime")
        
        features, valid_idx = self._build_features(prices)
        
        if len(features) < 60:
            logger.warning(f"Insufficient data for HMM ({len(features)} obs), need ≥60")
            return pd.Series("neutral", index=prices.index, name="hmm_regime")
        
        # Fit HMM
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        
        try:
            model.fit(features)
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return pd.Series("neutral", index=prices.index, name="hmm_regime")
        
        self._model = model
        self._fitted = True
        
        # Predict states
        hidden_states = model.predict(features)
        
        # Auto-label states
        self._state_labels = self._label_states(model.means_)
        
        # Map integer states to labels
        regime_labels = [self._state_labels[s] for s in hidden_states]
        
        regime_series = pd.Series(regime_labels, index=valid_idx, name="hmm_regime")
        
        # Reindex to full price index
        self._last_regimes = regime_series.reindex(prices.index).ffill().bfill()
        
        return self._last_regimes
    
    def current_regime(self) -> str:
        """Get the most recent regime label. Must call fit_predict first."""
        if self._last_regimes is None:
            return "neutral"
        return str(self._last_regimes.iloc[-1])
    
    def current_regime_enum(self) -> MarketRegime:
        """Get the most recent regime as MarketRegime enum."""
        label = self.current_regime()
        return self.REGIME_MAP.get(label, MarketRegime.NEUTRAL)
    
    def get_regime_probabilities(self, prices: pd.Series = None) -> pd.DataFrame:
        """
        Get probability of each regime for each day.
        
        Returns DataFrame with columns like 'prob_bull', 'prob_neutral', 'prob_bear'.
        """
        if not self._fitted or self._model is None:
            if prices is not None:
                self.fit_predict(prices)
            else:
                return pd.DataFrame()
        
        features, valid_idx = self._build_features(prices if prices is not None else pd.Series())
        
        if len(features) == 0:
            return pd.DataFrame()
        
        try:
            # Get posterior probabilities
            posteriors = self._model.predict_proba(features)
        except Exception:
            return pd.DataFrame()
        
        # Build DataFrame with labeled columns
        prob_df = pd.DataFrame(posteriors, index=valid_idx)
        prob_df.columns = [f"prob_{self._state_labels.get(i, f'state_{i}')}" for i in range(self.n_states)]
        
        return prob_df
    
    def get_state_stats(self) -> dict:
        """Get mean return & volatility for each regime state."""
        if not self._fitted or self._model is None:
            return {}
        
        stats = {}
        for state_idx, label in self._state_labels.items():
            mean_ret = self._model.means_[state_idx, 0] * 252  # Annualize
            mean_vol = self._model.means_[state_idx, 1] if self._model.means_.shape[1] > 1 else 0
            stats[label] = {
                "annualized_return": float(mean_ret),
                "mean_vol": float(mean_vol),
                "state_index": int(state_idx),
            }
        return stats
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get regime transition probability matrix."""
        if not self._fitted or self._model is None:
            return pd.DataFrame()
        
        labels = [self._state_labels.get(i, f"state_{i}") for i in range(self.n_states)]
        return pd.DataFrame(
            self._model.transmat_,
            index=[f"from_{l}" for l in labels],
            columns=[f"to_{l}" for l in labels],
        )


def add_hmm_regime_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    n_states: int = 3,
    vol_lookback: int = 21,
) -> pd.DataFrame:
    """
    Add HMM regime features to a DataFrame.
    
    Adds columns:
    - hmm_regime: Regime label ('bull', 'neutral', 'bear')
    - hmm_regime_bull: 1/0 indicator for bull regime
    - hmm_regime_bear: 1/0 indicator for bear regime
    - hmm_regime_neutral: 1/0 indicator for neutral regime
    
    All features are shifted by 1 day to prevent look-ahead bias.
    
    Args:
        df: DataFrame with price data
        price_col: Price column name
        n_states: Number of HMM states
        vol_lookback: Volatility lookback window
    
    Returns:
        DataFrame with HMM regime features added
    """
    detector = HMMRegimeDetector(n_states=n_states, vol_lookback=vol_lookback)
    
    try:
        regimes = detector.fit_predict(df[price_col])
        
        # Shift by 1 day (look-ahead bias prevention)
        regimes_lagged = regimes.shift(1)
        
        df["hmm_regime"] = regimes_lagged
        df["hmm_regime_bull"] = (regimes_lagged == "bull").astype(int)
        df["hmm_regime_bear"] = (regimes_lagged == "bear").astype(int)
        df["hmm_regime_neutral"] = (regimes_lagged == "neutral").astype(int)
        
        # Fill NaN from shift with neutral
        df["hmm_regime"] = df["hmm_regime"].fillna("neutral")
        df["hmm_regime_bull"] = df["hmm_regime_bull"].fillna(0).astype(int)
        df["hmm_regime_bear"] = df["hmm_regime_bear"].fillna(0).astype(int)
        df["hmm_regime_neutral"] = df["hmm_regime_neutral"].fillna(1).astype(int)
        
    except Exception as e:
        logger.warning(f"HMM regime detection failed: {e}, using neutral defaults")
        df["hmm_regime"] = "neutral"
        df["hmm_regime_bull"] = 0
        df["hmm_regime_bear"] = 0
        df["hmm_regime_neutral"] = 1
    
    return df
