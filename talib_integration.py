"""
TA-Lib Integration Module
Validates and enhances technical indicators using TA-Lib (200+ validated indicators)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class TALibValidator:
    """Validate existing indicators against TA-Lib reference implementations"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Args:
            tolerance: Acceptable difference between our indicators and TA-Lib (in percent)
        """
        self.tolerance = tolerance
        self.validation_results = {}
    
    def validate_rsi(self, hist: pd.DataFrame, period: int = 14) -> Tuple[bool, pd.Series]:
        """
        Validate RSI against TA-Lib reference
        
        Args:
            hist: DataFrame with 'Close' column
            period: RSI period (default 14)
            
        Returns:
            (is_valid, talib_rsi_series)
        """
        close = hist["Close"].values
        
        # Get TA-Lib RSI
        talib_rsi = talib.RSI(close, timeperiod=period)
        talib_rsi = pd.Series(talib_rsi, index=hist.index)
        
        # Get our RSI if it exists
        if f"rsi{period}" in hist.columns:
            our_rsi = hist[f"rsi{period}"]
            
            # Compare (skip NaN values)
            valid_idx = ~our_rsi.isna() & ~talib_rsi.isna()
            if valid_idx.sum() > 0:
                diff = np.abs(our_rsi[valid_idx] - talib_rsi[valid_idx])
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                is_match = max_diff < self.tolerance
                
                self.validation_results[f"rsi{period}"] = {
                    "match": is_match,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "samples": valid_idx.sum()
                }
                
                if is_match:
                    print(f"✅ RSI{period} matches TA-Lib (max diff: {max_diff:.4f})")
                else:
                    print(f"⚠️  RSI{period} differs from TA-Lib (max diff: {max_diff:.4f})")
                
                return is_match, talib_rsi
        
        return True, talib_rsi
    
    def validate_macd(self, hist: pd.DataFrame, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[bool, Tuple]:
        """
        Validate MACD against TA-Lib reference
        """
        close = hist["Close"].values
        
        # Get TA-Lib MACD
        macd_talib, signal_talib, hist_talib = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        macd_talib = pd.Series(macd_talib, index=hist.index)
        signal_talib = pd.Series(signal_talib, index=hist.index)
        hist_talib = pd.Series(hist_talib, index=hist.index)
        
        # Compare with our MACD if exists
        if "macd" in hist.columns and "macdsignal" in hist.columns:
            our_macd = hist["macd"]
            our_signal = hist["macdsignal"]
            
            valid_idx = ~our_macd.isna() & ~macd_talib.isna()
            if valid_idx.sum() > 0:
                diff_macd = np.abs(our_macd[valid_idx] - macd_talib[valid_idx])
                diff_signal = np.abs(our_signal[valid_idx] - signal_talib[valid_idx])
                
                max_diff = max(diff_macd.max(), diff_signal.max())
                is_match = max_diff < self.tolerance
                
                self.validation_results["macd"] = {
                    "match": is_match,
                    "max_diff": max_diff,
                    "samples": valid_idx.sum()
                }
                
                if is_match:
                    print(f"✅ MACD matches TA-Lib (max diff: {max_diff:.6f})")
                else:
                    print(f"⚠️  MACD differs from TA-Lib (max diff: {max_diff:.6f})")
                
                return is_match, (macd_talib, signal_talib, hist_talib)
        
        return True, (macd_talib, signal_talib, hist_talib)
    
    def validate_bollinger_bands(self, hist: pd.DataFrame, period: int = 20) -> Tuple[bool, Tuple]:
        """
        Validate Bollinger Bands against TA-Lib reference
        """
        close = hist["Close"].values
        
        # Get TA-Lib Bollinger Bands
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=period)
        bb_upper = pd.Series(bb_upper, index=hist.index)
        bb_mid = pd.Series(bb_mid, index=hist.index)
        bb_lower = pd.Series(bb_lower, index=hist.index)
        
        # Compare with our BB if exists
        if "bb_upper" in hist.columns and "bb_lower" in hist.columns:
            our_upper = hist["bb_upper"]
            our_lower = hist["bb_lower"]
            
            valid_idx = ~our_upper.isna() & ~bb_upper.isna()
            if valid_idx.sum() > 0:
                diff_upper = np.abs(our_upper[valid_idx] - bb_upper[valid_idx])
                diff_lower = np.abs(our_lower[valid_idx] - bb_lower[valid_idx])
                
                max_diff = max(diff_upper.max(), diff_lower.max())
                is_match = max_diff < (self.tolerance * 10)  # More tolerant for BB
                
                self.validation_results["bollinger_bands"] = {
                    "match": is_match,
                    "max_diff": max_diff,
                    "samples": valid_idx.sum()
                }
                
                if is_match:
                    print(f"✅ Bollinger Bands match TA-Lib (max diff: ${max_diff:.4f})")
                else:
                    print(f"⚠️  Bollinger Bands differ from TA-Lib (max diff: ${max_diff:.4f})")
                
                return is_match, (bb_upper, bb_mid, bb_lower)
        
        return True, (bb_upper, bb_mid, bb_lower)
    
    def validate_atr(self, hist: pd.DataFrame, period: int = 14) -> Tuple[bool, pd.Series]:
        """Validate Average True Range against TA-Lib"""
        high = hist["High"].values
        low = hist["Low"].values
        close = hist["Close"].values
        
        # Get TA-Lib ATR
        atr_talib = talib.ATR(high, low, close, timeperiod=period)
        atr_talib = pd.Series(atr_talib, index=hist.index)
        
        # Compare with our ATR if exists
        if f"atr_{period}" in hist.columns:
            our_atr = hist[f"atr_{period}"]
            
            valid_idx = ~our_atr.isna() & ~atr_talib.isna()
            if valid_idx.sum() > 0:
                diff = np.abs(our_atr[valid_idx] - atr_talib[valid_idx])
                max_diff = diff.max()
                is_match = max_diff < (self.tolerance * 10)
                
                self.validation_results[f"atr_{period}"] = {
                    "match": is_match,
                    "max_diff": max_diff,
                    "samples": valid_idx.sum()
                }
                
                if is_match:
                    print(f"✅ ATR{period} matches TA-Lib (max diff: ${max_diff:.4f})")
                else:
                    print(f"⚠️  ATR{period} differs from TA-Lib (max diff: ${max_diff:.4f})")
                
                return is_match, atr_talib
        
        return True, atr_talib
    
    def get_summary(self) -> dict:
        """Get validation summary"""
        return self.validation_results
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("TA-Lib Validation Summary")
        print("="*60)
        
        if not self.validation_results:
            print("No validations performed yet")
            return
        
        matches = sum(1 for v in self.validation_results.values() if v.get("match", False))
        total = len(self.validation_results)
        
        print(f"\nMatches: {matches}/{total} indicators")
        
        for indicator, results in self.validation_results.items():
            status = "✅" if results.get("match") else "⚠️"
            print(f"{status} {indicator}: max_diff={results.get('max_diff', 'N/A')}")
        
        print("="*60)


def add_talib_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Add TA-Lib indicators to existing DataFrame
    
    Args:
        hist: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional TA-Lib indicators (or original if TA-Lib unavailable)
    """
    if not TALIB_AVAILABLE:
        return hist
    
    hist = hist.copy()
    
    close = hist["Close"].values
    high = hist["High"].values if "High" in hist.columns else None
    low = hist["Low"].values if "Low" in hist.columns else None
    
    # Technical Indicators
    try:
        # Momentum
        hist["talib_rsi14"] = talib.RSI(close, timeperiod=14)
        hist["talib_rsi21"] = talib.RSI(close, timeperiod=21)
        
        # Trend
        macd, signal, hist_vals = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        hist["talib_macd"] = macd
        hist["talib_macd_signal"] = signal
        hist["talib_macd_hist"] = hist_vals
        
        # Volatility
        if high is not None and low is not None:
            hist["talib_atr14"] = talib.ATR(high, low, close, timeperiod=14)
            bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            hist["talib_bb_upper"] = bb_upper
            hist["talib_bb_mid"] = bb_mid
            hist["talib_bb_lower"] = bb_lower
        
        # Overlap (Moving Averages)
        hist["talib_sma20"] = talib.SMA(close, timeperiod=20)
        hist["talib_sma50"] = talib.SMA(close, timeperiod=50)
        hist["talib_ema12"] = talib.EMA(close, timeperiod=12)
        hist["talib_ema26"] = talib.EMA(close, timeperiod=26)
        
        # Volume-based
        if "Volume" in hist.columns:
            volume = hist["Volume"].values
            hist["talib_obv"] = talib.OBV(close, volume)
            hist["talib_ad"] = talib.AD(high, low, close, volume) if high is not None and low is not None else np.nan
        
    except Exception as e:
        print(f"Warning: Could not add TA-Lib indicators: {e}")
    
    return hist
