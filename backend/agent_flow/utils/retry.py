"""
Retry utilities with exponential backoff.

This module provides decorators and utilities for retrying operations
that may fail due to transient errors (network issues, rate limits, etc.).
"""

import asyncio
import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from .logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RetryError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self, message: str, last_exception: Exception | None = None, attempts: int = 0
    ):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        backoff_factor: Multiplier for delay between attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between attempts
        jitter: Add random jitter to prevent thundering herd
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback called on each retry (exception, attempt_number)

    Returns:
        Decorated function that retries on failure

    Examples:
        >>> @retry(max_attempts=3, backoff_factor=2)
        ... def fetch_data():
        ...     return requests.get(url)

        >>> @retry(retryable_exceptions=(ConnectionError, TimeoutError))
        ... def connect_to_database():
        ...     return db.connect()
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e),
                        )
                        raise RetryError(
                            f"All {max_attempts} attempts failed for {func.__name__}",
                            last_exception=e,
                            attempts=attempt,
                        ) from e

                    # Calculate delay with optional jitter
                    actual_delay = min(delay, max_delay)
                    if jitter:
                        actual_delay = actual_delay * (0.5 + random.random())

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=actual_delay,
                        error=str(e),
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(actual_delay)
                    delay *= backoff_factor

            # Should not reach here, but just in case
            raise RetryError(
                f"Unexpected state in retry for {func.__name__}",
                last_exception=last_exception,
                attempts=max_attempts,
            )

        return wrapper  # type: ignore

    return decorator


def async_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    """
    Async version of retry decorator.

    Same parameters as retry(), but for async functions.

    Examples:
        >>> @async_retry(max_attempts=3)
        ... async def fetch_data():
        ...     return await client.get(url)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            "async_retry_exhausted",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e),
                        )
                        raise RetryError(
                            f"All {max_attempts} attempts failed for {func.__name__}",
                            last_exception=e,
                            attempts=attempt,
                        ) from e

                    actual_delay = min(delay, max_delay)
                    if jitter:
                        actual_delay = actual_delay * (0.5 + random.random())

                    logger.warning(
                        "async_retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=actual_delay,
                        error=str(e),
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    await asyncio.sleep(actual_delay)
                    delay *= backoff_factor

            raise RetryError(
                f"Unexpected state in async retry for {func.__name__}",
                last_exception=last_exception,
                attempts=max_attempts,
            )

        return wrapper  # type: ignore

    return decorator


def retry_on_exception(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> Any:
    """
    Functional style retry without decorator.

    Useful when you can't or don't want to use a decorator.

    Args:
        func: Function to call
        *args: Positional arguments for func
        max_attempts: Maximum attempts
        backoff_factor: Delay multiplier
        **kwargs: Keyword arguments for func

    Returns:
        Result of func(*args, **kwargs)

    Examples:
        >>> result = retry_on_exception(requests.get, url, max_attempts=3)
    """

    @retry(max_attempts=max_attempts, backoff_factor=backoff_factor)
    def wrapped() -> Any:
        return func(*args, **kwargs)

    return wrapped()


# Pre-configured retry decorators for common use cases

# For API calls (network errors, timeouts)
retry_api = retry(
    max_attempts=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

# For database operations
retry_db = retry(
    max_attempts=3,
    backoff_factor=1.5,
    initial_delay=0.5,
    max_delay=10.0,
)

# For LLM calls (may have rate limits)
retry_llm = retry(
    max_attempts=3,
    backoff_factor=2.0,
    initial_delay=2.0,
    max_delay=30.0,
    jitter=True,
)


__all__ = [
    "retry",
    "async_retry",
    "retry_on_exception",
    "retry_api",
    "retry_db",
    "retry_llm",
    "RetryError",
]
