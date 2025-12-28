"""
Pandas-TA Integration Module
Easy 150+ indicator API for technical analysis
"""

import pandas as pd
from typing import Dict, List, Optional

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

class PandasTAWrapper:
    """Convenient wrapper for Pandas-TA indicators"""
    
    def __init__(self, hist: pd.DataFrame):
        """
        Args:
            hist: DataFrame with OHLCV data (requires Close, High, Low, Volume)
        """
        self.hist = hist.copy()
        self.added_indicators = []
    
    def add_momentum_indicators(self) -> pd.DataFrame:
        """Add momentum indicators (RSI, STOCH, CMO, etc.)"""
        try:
            # RSI
            self.hist.ta.rsi(length=14, append=True)
            self.hist.ta.rsi(length=21, append=True)
            
            # Stochastic
            self.hist.ta.stoch(length=14, append=True)
            
            # MACD variations
            self.hist.ta.macd(append=True)
            
            # CMO (Chande Momentum)
            self.hist.ta.cmo(length=14, append=True)
            
            # Rate of Change
            self.hist.ta.roc(length=12, append=True)
            
            self.added_indicators.extend(["RSI", "Stochastic", "MACD", "CMO", "ROC"])
            print("✅ Added momentum indicators")
            
        except Exception as e:
            print(f"⚠️  Could not add momentum indicators: {e}")
        
        return self.hist
    
    def add_trend_indicators(self) -> pd.DataFrame:
        """Add trend indicators (ADX, Aroon, KAMA, etc.)"""
        try:
            # ADX
            self.hist.ta.adx(length=14, append=True)
            
            # Aroon
            self.hist.ta.aroon(append=True)
            
            # Kaufman Adaptive MA
            self.hist.ta.kama(length=10, append=True)
            
            # TEMA (Triple EMA)
            self.hist.ta.tema(length=10, append=True)
            
            self.added_indicators.extend(["ADX", "Aroon", "KAMA", "TEMA"])
            print("✅ Added trend indicators")
            
        except Exception as e:
            print(f"⚠️  Could not add trend indicators: {e}")
        
        return self.hist
    
    def add_volatility_indicators(self) -> pd.DataFrame:
        """Add volatility indicators (ATR, Keltner, NATR, etc.)"""
        try:
            # ATR
            self.hist.ta.atr(length=14, append=True)
            
            # Keltner Channel
            self.hist.ta.kc(length=20, append=True)
            
            # Normalized ATR
            self.hist.ta.natr(length=14, append=True)
            
            # Historical Volatility
            self.hist.ta.bbands(length=20, append=True)
            
            self.added_indicators.extend(["ATR", "KeltnerChannel", "NATR", "BBands"])
            print("✅ Added volatility indicators")
            
        except Exception as e:
            print(f"⚠️  Could not add volatility indicators: {e}")
        
        return self.hist
    
    def add_volume_indicators(self) -> pd.DataFrame:
        """Add volume indicators (OBV, MFI, AD, VMAP, etc.)"""
        try:
            # On Balance Volume
            self.hist.ta.obv(append=True)
            
            # Money Flow Index
            self.hist.ta.mfi(length=14, append=True)
            
            # Accumulation/Distribution
            self.hist.ta.ad(append=True)
            
            # Volume Weighted Moving Average
            self.hist.ta.vwma(length=20, append=True)
            
            # Volume Rate of Change
            self.hist.ta.pvol(append=True)
            
            self.added_indicators.extend(["OBV", "MFI", "AD", "VWMA", "PVOL"])
            print("✅ Added volume indicators")
            
        except Exception as e:
            print(f"⚠️  Could not add volume indicators: {e}")
        
        return self.hist
    
    def add_cycle_indicators(self) -> pd.DataFrame:
        """Add cycle/oscillator indicators (HMA, HULL MA, etc.)"""
        try:
            # Hull Moving Average
            self.hist.ta.hma(length=20, append=True)
            
            # Linear Regression
            self.hist.ta.linreg(length=20, append=True)
            
            self.added_indicators.extend(["HMA", "LinReg"])
            print("✅ Added cycle indicators")
            
        except Exception as e:
            print(f"⚠️  Could not add cycle indicators: {e}")
        
        return self.hist
    
    def add_all_indicators(self) -> pd.DataFrame:
        """Add all available indicator categories"""
        self.add_momentum_indicators()
        self.add_trend_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_cycle_indicators()
        
        return self.hist
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return enhanced DataFrame"""
        return self.hist
    
    def get_summary(self) -> Dict:
        """Get summary of added indicators"""
        new_cols = [col for col in self.hist.columns if col not in pd.DataFrame()]
        return {
            "original_columns": len(self.hist.columns) - len(new_cols),
            "new_columns": len(new_cols),
            "added_indicators": self.added_indicators
        }


def add_pandas_ta_indicators(hist: pd.DataFrame, categories: List[str] = None) -> pd.DataFrame:
    """
    Convenient function to add Pandas-TA indicators
    
    Args:
        hist: DataFrame with OHLCV data
        categories: List of categories to add
                   Options: ["momentum", "trend", "volatility", "volume", "cycle", "all"]
    
    Returns:
        DataFrame with added indicators (or original if pandas-ta unavailable)
    """
    if not PANDAS_TA_AVAILABLE:
        return hist
    
    if categories is None:
        categories = ["momentum", "trend", "volatility", "volume"]
    
    wrapper = PandasTAWrapper(hist)
    
    if "all" in categories:
        return wrapper.add_all_indicators()
    
    if "momentum" in categories:
        wrapper.add_momentum_indicators()
    if "trend" in categories:
        wrapper.add_trend_indicators()
    if "volatility" in categories:
        wrapper.add_volatility_indicators()
    if "volume" in categories:
        wrapper.add_volume_indicators()
    if "cycle" in categories:
        wrapper.add_cycle_indicators()
    
    return wrapper.get_dataframe()
