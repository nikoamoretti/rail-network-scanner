"""
File-based JSON caching for API responses.

Wraps any API call with disk caching keyed by request parameters.
Cache files are stored as JSON in the .cache/ directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from config import CACHE_DIR

logger = logging.getLogger(__name__)


class FileCache:
    """JSON file-based cache for API responses."""

    def __init__(self, namespace: str, cache_dir: Path = CACHE_DIR, ttl_hours: float = 0):
        """
        Args:
            namespace: Cache subdirectory name (e.g., 'overpass', 'places').
            cache_dir: Root cache directory.
            ttl_hours: Time-to-live in hours. 0 means no expiration.
        """
        self.cache_dir = cache_dir / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else 0

    def _make_key(self, params: Dict[str, Any]) -> str:
        """Generate a deterministic cache key from parameters."""
        serialized = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, params: Dict[str, Any]) -> Optional[Any]:
        """
        Look up cached response.

        Args:
            params: The request parameters used as cache key.

        Returns:
            Cached data or None if not found / expired.
        """
        key = self._make_key(params)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                entry = json.load(f)

            # Check TTL
            if self.ttl_seconds > 0:
                cached_at = entry.get("_cached_at", 0)
                if time.time() - cached_at > self.ttl_seconds:
                    logger.debug(f"Cache expired for key {key}")
                    path.unlink()
                    return None

            logger.debug(f"Cache hit for key {key}")
            return entry.get("data")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt cache entry {key}: {e}")
            path.unlink(missing_ok=True)
            return None

    def set(self, params: Dict[str, Any], data: Any) -> None:
        """
        Store a response in cache.

        Args:
            params: The request parameters used as cache key.
            data: The response data to cache.
        """
        key = self._make_key(params)
        path = self._cache_path(key)

        entry = {
            "_cached_at": time.time(),
            "_params": params,
            "data": data,
        }

        with open(path, "w") as f:
            json.dump(entry, f)

        logger.debug(f"Cached response for key {key}")

    def has(self, params: Dict[str, Any]) -> bool:
        """Check if a valid cache entry exists."""
        return self.get(params) is not None

    def clear(self) -> int:
        """Remove all cache entries. Returns count of removed files."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        logger.info(f"Cleared {count} cache entries from {self.cache_dir.name}")
        return count

    @property
    def size(self) -> int:
        """Number of cache entries."""
        return len(list(self.cache_dir.glob("*.json")))


def make_location_key(lat: float, lon: float, precision: int = 4) -> Dict[str, float]:
    """
    Create a cache key from coordinates, rounded to given precision.

    Rounding to 4 decimal places (~11m precision) prevents redundant
    API calls for nearby coordinates that would return the same results.
    """
    return {
        "lat": round(lat, precision),
        "lon": round(lon, precision),
    }
