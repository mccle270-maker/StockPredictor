"""
Aggressive file-based cache manager for data providers.
"""
from __future__ import annotations

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import threading


@dataclass
class CacheConfig:
    """Configuration for cache TTL by data type."""
    price_ttl_hours: float = 1.0        # Intraday price updates
    fundamentals_ttl_hours: float = 24.0  # Fundamentals change rarely
    macro_ttl_hours: float = 6.0         # Macro updates several times a day
    sentiment_ttl_hours: float = 2.0     # News sentiment more volatile
    filings_ttl_hours: float = 24.0      # SEC filings - daily check
    default_ttl_hours: float = 4.0       # Default for unknown types


class CacheManager:
    """
    Thread-safe file-based cache manager.
    
    Features:
    - Configurable TTL per data type
    - Automatic cache directory creation
    - Pickle serialization for DataFrames
    - JSON serialization for dicts
    - Thread-safe operations
    - Automatic cleanup of expired entries
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        config: Optional[CacheConfig] = None,
    ):
        self.cache_dir = Path(cache_dir or ".cache/data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or CacheConfig()
        self._lock = threading.Lock()
        self._memory_cache: Dict[str, tuple[Any, datetime]] = {}
    
    def _get_ttl(self, data_type: str) -> timedelta:
        """Get TTL for a data type."""
        ttl_map = {
            "price": self.config.price_ttl_hours,
            "fundamentals": self.config.fundamentals_ttl_hours,
            "macro": self.config.macro_ttl_hours,
            "sentiment": self.config.sentiment_ttl_hours,
            "filings": self.config.filings_ttl_hours,
        }
        hours = ttl_map.get(data_type, self.config.default_ttl_hours)
        return timedelta(hours=hours)
    
    def _make_key(self, data_type: str, identifier: str, **kwargs) -> str:
        """Create a unique cache key."""
        # Include kwargs in key (e.g., period, interval)
        params = sorted(kwargs.items())
        key_string = f"{data_type}:{identifier}:{params}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str, data_type: str) -> Path:
        """Get the file path for a cache entry."""
        subdir = self.cache_dir / data_type
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.cache"
    
    def get(
        self,
        data_type: str,
        identifier: str,
        **kwargs,
    ) -> Optional[Any]:
        """
        Get cached data if available and not expired.
        
        Args:
            data_type: Type of data (price, fundamentals, macro, etc.)
            identifier: Unique identifier (ticker, FRED symbol, etc.)
            **kwargs: Additional parameters that affect the cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        key = self._make_key(data_type, identifier, **kwargs)
        ttl = self._get_ttl(data_type)
        
        # Check memory cache first
        with self._lock:
            if key in self._memory_cache:
                data, timestamp = self._memory_cache[key]
                if datetime.now() - timestamp < ttl:
                    return data
                else:
                    del self._memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key, data_type)
        if not cache_path.exists():
            return None
        
        try:
            # Read cache entry
            with open(cache_path, "rb") as f:
                entry = pickle.load(f)
            
            timestamp = entry.get("timestamp")
            if timestamp and datetime.now() - timestamp < ttl:
                data = entry.get("data")
                # Populate memory cache
                with self._lock:
                    self._memory_cache[key] = (data, timestamp)
                return data
            else:
                # Expired - delete file
                cache_path.unlink(missing_ok=True)
                return None
                
        except Exception:
            # Corrupted cache - delete
            cache_path.unlink(missing_ok=True)
            return None
    
    def set(
        self,
        data_type: str,
        identifier: str,
        data: Any,
        **kwargs,
    ) -> bool:
        """
        Store data in cache.
        
        Args:
            data_type: Type of data (price, fundamentals, macro, etc.)
            identifier: Unique identifier (ticker, FRED symbol, etc.)
            data: Data to cache (DataFrame, dict, etc.)
            **kwargs: Additional parameters that affect the cache key
            
        Returns:
            True if cached successfully
        """
        key = self._make_key(data_type, identifier, **kwargs)
        timestamp = datetime.now()
        
        # Store in memory cache
        with self._lock:
            self._memory_cache[key] = (data, timestamp)
        
        # Store in file cache
        cache_path = self._get_cache_path(key, data_type)
        
        try:
            entry = {
                "timestamp": timestamp,
                "data": data,
                "data_type": data_type,
                "identifier": identifier,
                "params": kwargs,
            }
            
            with open(cache_path, "wb") as f:
                pickle.dump(entry, f)
            
            return True
            
        except Exception as e:
            # Silently fail - caching is best-effort
            return False
    
    def invalidate(
        self,
        data_type: str,
        identifier: str,
        **kwargs,
    ) -> bool:
        """Invalidate a specific cache entry."""
        key = self._make_key(data_type, identifier, **kwargs)
        
        # Remove from memory cache
        with self._lock:
            self._memory_cache.pop(key, None)
        
        # Remove from file cache
        cache_path = self._get_cache_path(key, data_type)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
    
    def invalidate_type(self, data_type: str) -> int:
        """Invalidate all entries of a specific type."""
        subdir = self.cache_dir / data_type
        if not subdir.exists():
            return 0
        
        count = 0
        for cache_file in subdir.glob("*.cache"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        
        # Clear memory cache for this type
        with self._lock:
            keys_to_remove = [k for k in self._memory_cache if k.startswith(data_type)]
            for k in keys_to_remove:
                del self._memory_cache[k]
        
        return count
    
    def clear_all(self) -> int:
        """Clear entire cache."""
        count = 0
        
        with self._lock:
            self._memory_cache.clear()
        
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                for cache_file in subdir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                        count += 1
                    except Exception:
                        pass
        
        return count
    
    def cleanup_expired(self) -> int:
        """Remove all expired cache entries."""
        count = 0
        
        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            data_type = subdir.name
            ttl = self._get_ttl(data_type)
            
            for cache_file in subdir.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)
                    
                    timestamp = entry.get("timestamp")
                    if timestamp and datetime.now() - timestamp >= ttl:
                        cache_file.unlink()
                        count += 1
                        
                except Exception:
                    # Corrupted - remove
                    cache_file.unlink()
                    count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_entries": len(self._memory_cache),
            "file_entries": 0,
            "total_size_bytes": 0,
            "by_type": {},
        }
        
        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            data_type = subdir.name
            type_count = 0
            type_size = 0
            
            for cache_file in subdir.glob("*.cache"):
                type_count += 1
                type_size += cache_file.stat().st_size
            
            stats["by_type"][data_type] = {
                "count": type_count,
                "size_bytes": type_size,
            }
            stats["file_entries"] += type_count
            stats["total_size_bytes"] += type_size
        
        return stats


# Global cache instance
_global_cache: Optional[CacheManager] = None


def get_cache() -> CacheManager:
    """Get or create the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cache_get(data_type: str, identifier: str, **kwargs) -> Optional[Any]:
    """Convenience function for global cache get."""
    return get_cache().get(data_type, identifier, **kwargs)


def cache_set(data_type: str, identifier: str, data: Any, **kwargs) -> bool:
    """Convenience function for global cache set."""
    return get_cache().set(data_type, identifier, data, **kwargs)
