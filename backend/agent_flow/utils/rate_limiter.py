"""
Rate limiting implementation using token bucket algorithm.

Prevents abuse by limiting requests per user per time window.
Thread-safe implementation suitable for production use.
"""

import time
from collections import defaultdict
from threading import Lock

from backend.agent_flow.config import config
from backend.agent_flow.utils.errors import RateLimitError
from backend.agent_flow.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.

    Limits requests per minute and per hour per user.
    Thread-safe implementation.

    Example:
        limiter = RateLimiter(requests_per_minute=60, requests_per_hour=500)
        limiter.check_rate_limit(user_id="user123")  # Raises RateLimitError if exceeded
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        requests_per_hour: int | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute per user (uses config default if None)
            requests_per_hour: Max requests per hour per user (uses config default if None)
        """
        self.rpm = requests_per_minute or config.safety.MAX_REQUESTS_PER_MINUTE
        self.rph = requests_per_hour or config.safety.MAX_REQUESTS_PER_HOUR

        # Storage: user_id -> [timestamp1, timestamp2, ...]
        self.minute_buckets: dict[str, list[float]] = defaultdict(list)
        self.hour_buckets: dict[str, list[float]] = defaultdict(list)

        self.lock = Lock()

        logger.info("rate_limiter_initialized", rpm=self.rpm, rph=self.rph)

    def check_rate_limit(self, user_id: str) -> tuple[bool, int]:
        """
        Check if user has exceeded rate limit.

        Args:
            user_id: User identifier

        Returns:
            (allowed, retry_after_seconds)

        Raises:
            RateLimitError: If rate limit exceeded
        """
        with self.lock:
            now = time.time()

            # Clean old entries
            self._clean_old_entries(user_id, now)

            # Check per-minute limit
            minute_count = len(self.minute_buckets[user_id])
            if minute_count >= self.rpm:
                retry_after = 60
                logger.warning(
                    "rate_limit_exceeded_minute",
                    user_id=user_id,
                    count=minute_count,
                    limit=self.rpm,
                )
                raise RateLimitError(retry_after=retry_after)

            # Check per-hour limit
            hour_count = len(self.hour_buckets[user_id])
            if hour_count >= self.rph:
                retry_after = 3600
                logger.warning(
                    "rate_limit_exceeded_hour",
                    user_id=user_id,
                    count=hour_count,
                    limit=self.rph,
                )
                raise RateLimitError(retry_after=retry_after)

            # Record this request
            self.minute_buckets[user_id].append(now)
            self.hour_buckets[user_id].append(now)

            logger.debug(
                "rate_limit_check_passed",
                user_id=user_id,
                minute_count=minute_count + 1,
                hour_count=hour_count + 1,
            )

            return True, 0

    def _clean_old_entries(self, user_id: str, now: float):
        """Remove entries older than time window."""
        # Clean minute buckets (older than 60s)
        minute_cutoff = now - 60
        self.minute_buckets[user_id] = [
            ts for ts in self.minute_buckets[user_id] if ts > minute_cutoff
        ]

        # Clean hour buckets (older than 3600s)
        hour_cutoff = now - 3600
        self.hour_buckets[user_id] = [
            ts for ts in self.hour_buckets[user_id] if ts > hour_cutoff
        ]

    def get_usage(self, user_id: str) -> dict:
        """
        Get current usage for user.

        Args:
            user_id: User identifier

        Returns:
            dict with usage statistics
        """
        with self.lock:
            now = time.time()
            self._clean_old_entries(user_id, now)

            return {
                "requests_last_minute": len(self.minute_buckets[user_id]),
                "requests_last_hour": len(self.hour_buckets[user_id]),
                "limit_per_minute": self.rpm,
                "limit_per_hour": self.rph,
                "remaining_minute": max(
                    0, self.rpm - len(self.minute_buckets[user_id])
                ),
                "remaining_hour": max(0, self.rph - len(self.hour_buckets[user_id])),
            }

    def reset_user(self, user_id: str):
        """
        Reset rate limit for a specific user.

        Args:
            user_id: User identifier
        """
        with self.lock:
            self.minute_buckets.pop(user_id, None)
            self.hour_buckets.pop(user_id, None)
            logger.info("rate_limit_reset", user_id=user_id)

    def reset_all(self):
        """Reset rate limits for all users."""
        with self.lock:
            self.minute_buckets.clear()
            self.hour_buckets.clear()
            logger.info("rate_limit_reset_all")


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """
    Get global rate limiter instance (singleton).

    Returns:
        RateLimiter: Global rate limiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
