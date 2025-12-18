"""
Unit tests for rate limiter.

Tests the token bucket rate limiting implementation.
"""

import time
from threading import Thread

import pytest

from backend.agent_flow.utils.errors import RateLimitError
from backend.agent_flow.utils.rate_limiter import RateLimiter


class TestRateLimiterBasics:
    """Test basic rate limiter functionality."""

    def test_creates_with_defaults(self):
        """Should create with default config values."""
        limiter = RateLimiter()
        assert limiter.rpm > 0
        assert limiter.rph > 0

    def test_creates_with_custom_limits(self):
        """Should accept custom rate limits."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        assert limiter.rpm == 10
        assert limiter.rph == 100

    def test_allows_first_request(self):
        """First request should always be allowed."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=50)
        allowed, retry_after = limiter.check_rate_limit("user1")
        assert allowed is True
        assert retry_after == 0

    def test_allows_multiple_requests_under_limit(self):
        """Should allow requests under the limit."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        for i in range(5):
            allowed, retry_after = limiter.check_rate_limit("user1")
            assert allowed is True
            assert retry_after == 0


class TestRateLimitEnforcement:
    """Test rate limit enforcement."""

    def test_blocks_when_minute_limit_exceeded(self):
        """Should block when per-minute limit exceeded."""
        limiter = RateLimiter(requests_per_minute=3, requests_per_hour=100)

        # Use up the limit
        for i in range(3):
            limiter.check_rate_limit("user1")

        # Next request should be blocked
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_rate_limit("user1")

        assert exc_info.value.retry_after == 60

    def test_blocks_when_hour_limit_exceeded(self):
        """Should block when per-hour limit exceeded."""
        limiter = RateLimiter(requests_per_minute=100, requests_per_hour=5)

        # Use up the limit
        for i in range(5):
            limiter.check_rate_limit("user1")

        # Next request should be blocked
        with pytest.raises(RateLimitError) as exc_info:
            limiter.check_rate_limit("user1")

        assert exc_info.value.retry_after == 3600

    def test_different_users_independent(self):
        """Rate limits should be independent per user."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=10)

        # User1 uses up their limit
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user1")

        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("user1")

        # User2 should still be allowed
        allowed, _ = limiter.check_rate_limit("user2")
        assert allowed is True


class TestRateLimitRecovery:
    """Test rate limit recovery over time."""

    def test_allows_request_after_window_expires(self):
        """Should allow requests after time window expires."""
        limiter = RateLimiter(requests_per_minute=1, requests_per_hour=10)

        # First request
        limiter.check_rate_limit("user1")

        # Second request should fail
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("user1")

        # Wait for window to expire (simulate with manual cleanup)
        # In real scenario would wait 60+ seconds
        limiter.minute_buckets["user1"].clear()

        # Should work now
        allowed, _ = limiter.check_rate_limit("user1")
        assert allowed is True


class TestUsageTracking:
    """Test usage statistics."""

    def test_get_usage_empty(self):
        """Should return zero usage for new user."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
        usage = limiter.get_usage("user1")

        assert usage["requests_last_minute"] == 0
        assert usage["requests_last_hour"] == 0
        assert usage["limit_per_minute"] == 10
        assert usage["limit_per_hour"] == 100
        assert usage["remaining_minute"] == 10
        assert usage["remaining_hour"] == 100

    def test_get_usage_after_requests(self):
        """Should track usage correctly."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        # Make 3 requests
        for i in range(3):
            limiter.check_rate_limit("user1")

        usage = limiter.get_usage("user1")

        assert usage["requests_last_minute"] == 3
        assert usage["requests_last_hour"] == 3
        assert usage["remaining_minute"] == 7
        assert usage["remaining_hour"] == 97

    def test_get_usage_calculates_remaining(self):
        """Should calculate remaining quota correctly."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_hour=50)

        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user1")

        usage = limiter.get_usage("user1")

        assert usage["remaining_minute"] == 3
        assert usage["remaining_hour"] == 48


class TestReset:
    """Test reset functionality."""

    def test_reset_user(self):
        """Should reset limits for specific user."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=10)

        # Use up limit
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user1")

        # Should be blocked
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("user1")

        # Reset
        limiter.reset_user("user1")

        # Should work now
        allowed, _ = limiter.check_rate_limit("user1")
        assert allowed is True

    def test_reset_user_doesnt_affect_others(self):
        """Resetting one user shouldn't affect others."""
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=10)

        # Both users make requests
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user2")

        # Reset user1
        limiter.reset_user("user1")

        # User1 usage should be reset
        usage1 = limiter.get_usage("user1")
        assert usage1["requests_last_minute"] == 0

        # User2 usage should remain
        usage2 = limiter.get_usage("user2")
        assert usage2["requests_last_minute"] == 1

    def test_reset_all(self):
        """Should reset all users."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        # Multiple users make requests
        limiter.check_rate_limit("user1")
        limiter.check_rate_limit("user2")
        limiter.check_rate_limit("user3")

        # Reset all
        limiter.reset_all()

        # All should have zero usage
        for user_id in ["user1", "user2", "user3"]:
            usage = limiter.get_usage(user_id)
            assert usage["requests_last_minute"] == 0
            assert usage["requests_last_hour"] == 0


class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_requests_same_user(self):
        """Should handle concurrent requests safely."""
        limiter = RateLimiter(requests_per_minute=100, requests_per_hour=1000)
        errors = []

        def make_requests():
            try:
                for _ in range(10):
                    limiter.check_rate_limit("user1")
            except Exception as e:
                errors.append(e)

        # Spawn multiple threads
        threads = [Thread(target=make_requests) for _ in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should not have threading errors
        assert len(errors) == 0

        # Total requests should be 50
        usage = limiter.get_usage("user1")
        assert usage["requests_last_minute"] == 50

    def test_concurrent_requests_different_users(self):
        """Should handle multiple users concurrently."""
        limiter = RateLimiter(requests_per_minute=50, requests_per_hour=500)
        errors = []

        def make_requests(user_id):
            try:
                for _ in range(5):
                    limiter.check_rate_limit(user_id)
            except Exception as e:
                errors.append(e)

        # Spawn threads for different users
        threads = [Thread(target=make_requests, args=(f"user{i}",)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should not have threading errors
        assert len(errors) == 0

        # Each user should have 5 requests
        for i in range(10):
            usage = limiter.get_usage(f"user{i}")
            assert usage["requests_last_minute"] == 5


class TestCleanup:
    """Test automatic cleanup of old entries."""

    def test_cleanup_removes_old_entries(self):
        """Should clean up entries outside time window."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        # Make request
        limiter.check_rate_limit("user1")

        # Manually add old timestamp (more than 60 seconds ago)
        old_time = time.time() - 61
        limiter.minute_buckets["user1"].append(old_time)

        # Get usage (triggers cleanup)
        usage = limiter.get_usage("user1")

        # Old entry should be cleaned up
        # Only the recent request should remain
        assert usage["requests_last_minute"] == 1

    def test_cleanup_preserves_recent_entries(self):
        """Should preserve entries within time window."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)

        # Make 3 requests
        for _ in range(3):
            limiter.check_rate_limit("user1")

        # Get usage
        usage = limiter.get_usage("user1")

        # All recent requests should be preserved
        assert usage["requests_last_minute"] == 3
        assert usage["requests_last_hour"] == 3
