"""
Caching utilities for expensive operations.

This module provides caching mechanisms for:
- RAG query results
- LLM responses
- Embedding computations

Uses TTL (time-to-live) based caching to balance freshness with performance.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with TTL."""

    value: T
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 minutes default
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


class TTLCache(Generic[T]):
    """
    Thread-safe TTL-based LRU cache.

    Features:
    - Time-based expiration (TTL)
    - Size-based eviction (LRU)
    - Thread-safe operations
    - Hit/miss statistics

    Examples:
        >>> cache = TTLCache[str](max_size=100, default_ttl=300)
        >>> cache.set("key1", "value1")
        >>> cache.get("key1")
        'value1'
        >>> cache.get("missing")  # Returns None
    """

    def __init__(
        self,
        max_size: int = 100,
        default_ttl: float = 300.0,
        name: str = "cache",
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
            name: Name for logging
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.name = name
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                logger.debug("cache_expired", cache=self.name, key=key[:50])
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug("cache_evicted", cache=self.name, key=oldest_key[:50])

            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
            )

    def delete(self, key: str) -> bool:
        """
        Delete an entry from the cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """
        Clear all entries from the cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("cache_cleared", cache=self.name, entries=count)
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(
                    "cache_cleanup",
                    cache=self.name,
                    removed=len(expired_keys),
                )
            return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "default_ttl": self.default_ttl,
            }

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Get current cache size (including expired entries)."""
        return len(self._cache)


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Create a cache key from arguments.

    Uses MD5 hash for consistent, fixed-length keys.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        MD5 hash string

    Examples:
        >>> make_cache_key("query", top_k=10)
        '5f4dcc3b5aa765d61d8327deb882cf99'
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


# Pre-configured caches for different use cases

# RAG results cache (5 minute TTL, 200 entries max)
_rag_cache: TTLCache[dict[str, Any]] | None = None


def get_rag_cache() -> TTLCache[dict[str, Any]]:
    """Get or create the RAG results cache."""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = TTLCache(
            max_size=200,
            default_ttl=300.0,  # 5 minutes
            name="rag_cache",
        )
    return _rag_cache


# Embedding cache (longer TTL since embeddings don't change)
_embedding_cache: TTLCache[list[float]] | None = None


def get_embedding_cache() -> TTLCache[list[float]]:
    """Get or create the embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = TTLCache(
            max_size=500,
            default_ttl=3600.0,  # 1 hour
            name="embedding_cache",
        )
    return _embedding_cache


# LLM response cache (shorter TTL for freshness)
_llm_cache: TTLCache[str] | None = None


def get_llm_cache() -> TTLCache[str]:
    """Get or create the LLM response cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = TTLCache(
            max_size=100,
            default_ttl=180.0,  # 3 minutes
            name="llm_cache",
        )
    return _llm_cache


def cached_rag_query(query: str, top_k: int = 300) -> dict[str, Any] | None:
    """
    Check RAG cache for a query.

    Args:
        query: Search query
        top_k: Number of results requested

    Returns:
        Cached result or None
    """
    cache = get_rag_cache()
    key = make_cache_key(query.lower().strip(), top_k=top_k)
    return cache.get(key)


def cache_rag_result(query: str, result: dict[str, Any], top_k: int = 300) -> None:
    """
    Cache a RAG query result.

    Args:
        query: Search query
        result: RAG result to cache
        top_k: Number of results
    """
    cache = get_rag_cache()
    key = make_cache_key(query.lower().strip(), top_k=top_k)
    cache.set(key, result)
    logger.debug("rag_result_cached", query=query[:50], top_k=top_k)


def get_all_cache_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all caches."""
    stats = {}
    if _rag_cache:
        stats["rag"] = _rag_cache.stats()
    if _embedding_cache:
        stats["embedding"] = _embedding_cache.stats()
    if _llm_cache:
        stats["llm"] = _llm_cache.stats()
    return stats


__all__ = [
    "TTLCache",
    "CacheEntry",
    "make_cache_key",
    "get_rag_cache",
    "get_embedding_cache",
    "get_llm_cache",
    "cached_rag_query",
    "cache_rag_result",
    "get_all_cache_stats",
]
