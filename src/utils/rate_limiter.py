"""
Generic rate limiter for API calls.

Configurable per-API with different requests-per-second limits.
Thread-safe using a simple lock.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, requests_per_second: float, name: str = ""):
        """
        Args:
            requests_per_second: Maximum sustained request rate.
            name: Human-readable name for logging.
        """
        self.min_interval = 1.0 / requests_per_second
        self.name = name or f"limiter({requests_per_second}/s)"
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """
        Block until the next request is allowed.

        Returns:
            Seconds waited.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_time = max(0.0, self.min_interval - elapsed)

            if wait_time > 0:
                logger.debug(f"[{self.name}] Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)

            self._last_request_time = time.monotonic()
            return wait_time

    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *args):
        pass
