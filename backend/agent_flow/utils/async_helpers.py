"""
Async helper functions for parallel operations.

Provides utilities for:
- Running tasks in parallel
- Timeouts
- Sync-to-async wrapping
"""

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from backend.agent_flow.utils.constants import AGENT_TIMEOUT_SECONDS
from backend.agent_flow.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


async def run_parallel(*tasks: Coroutine) -> list[Any]:
    """
    Run multiple async tasks in parallel.

    Args:
        *tasks: Variable number of coroutines to run

    Returns:
        List of results in same order as tasks

    Example:
        results = await run_parallel(
            fetch_context(query),
            fetch_knowledge(query),
            fetch_sentiment(query)
        )
    """
    logger.debug("running_parallel_tasks", count=len(tasks))

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions in results
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.warning("parallel_tasks_had_errors", error_count=len(errors))

        return results
    except Exception as e:
        logger.error("parallel_execution_failed", error=str(e))
        raise


async def run_with_timeout(
    coro: Coroutine[Any, Any, T], timeout: float = AGENT_TIMEOUT_SECONDS
) -> T:
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds

    Returns:
        Result of coroutine

    Raises:
        asyncio.TimeoutError: If operation times out

    Example:
        result = await run_with_timeout(
            slow_operation(),
            timeout=10.0
        )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("operation_timed_out", timeout=timeout)
        raise


def async_wrap(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Wrap synchronous function to run in executor (async context).

    Useful for running sync functions in async code without blocking.

    Args:
        func: Synchronous function to wrap

    Returns:
        Async function that runs in executor

    Example:
        @async_wrap
        def sync_operation(x):
            return x * 2

        result = await sync_operation(5)  # Runs in executor
    """

    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        bound_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


async def run_in_executor(
    func: Callable[..., T], *args, executor: ThreadPoolExecutor | None = None, **kwargs
) -> T:
    """
    Run synchronous function in executor.

    Args:
        func: Synchronous function to run
        *args: Positional arguments for func
        executor: Optional executor (uses default if None)
        **kwargs: Keyword arguments for func

    Returns:
        Result of function

    Example:
        result = await run_in_executor(
            sync_function,
            arg1, arg2,
            key=value
        )
    """
    loop = asyncio.get_event_loop()
    bound_func = partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, bound_func)


async def run_with_retries(
    coro_func: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> T:
    """
    Run coroutine with exponential backoff retries.

    Args:
        coro_func: Function that returns a coroutine
        max_retries: Maximum number of retry attempts
        backoff_seconds: Initial backoff time (doubles each retry)

    Returns:
        Result of successful execution

    Raises:
        Exception: Last exception if all retries fail

    Example:
        result = await run_with_retries(
            lambda: unstable_api_call(params),
            max_retries=3,
            backoff_seconds=1.0
        )
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = backoff_seconds * (2**attempt)
                logger.warning(
                    "retry_attempt",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error("max_retries_exceeded", attempts=max_retries, error=str(e))

    raise last_exception


def is_async_context() -> bool:
    """
    Check if currently in async context.

    Returns:
        True if in async context, False otherwise

    Example:
        if is_async_context():
            result = await async_operation()
        else:
            result = sync_operation()
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run coroutine in sync context (creates event loop if needed).

    Args:
        coro: Coroutine to run

    Returns:
        Result of coroutine

    Example:
        # From sync code
        result = run_async(async_function())
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)
    else:
        # Already in async context, just await
        logger.warning("run_async_called_in_async_context")
        raise RuntimeError(
            "Cannot use run_async() from async context. Use await instead."
        )
