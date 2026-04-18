"""
cache.py
Redis caching wrapper with JSON serialization.
Falls back gracefully if Redis is unavailable (dev mode).
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"


class CacheService:
    """
    Redis-backed cache for prediction results.
    Gracefully degrades to no-op if Redis is unavailable.
    """

    def __init__(self):
        self._client = None
        if CACHE_ENABLED:
            self._connect()

    def _connect(self):
        try:
            import redis
            self._client = redis.from_url(REDIS_URL, decode_responses=True)
            self._client.ping()
            logger.info(f"Redis connected: {REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}). Cache disabled — running without cache.")
            self._client = None

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None on miss or error."""
        if not self._client:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Cache GET error for '{key}': {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Cache a value with a TTL (in seconds).
        Returns True on success.
        """
        if not self._client:
            return False
        try:
            self._client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache SET '{key}' (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Cache SET error for '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """Invalidate a cache entry."""
        if not self._client:
            return False
        try:
            self._client.delete(key)
            logger.debug(f"Cache DELETE '{key}'")
            return True
        except Exception as e:
            logger.warning(f"Cache DELETE error for '{key}': {e}")
            return False

    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern. Returns count deleted."""
        if not self._client:
            return 0
        try:
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
            logger.info(f"Flushed {len(keys)} cache keys matching '{pattern}'")
            return len(keys)
        except Exception as e:
            logger.warning(f"Cache flush error: {e}")
            return 0
