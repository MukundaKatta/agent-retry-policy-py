"""agent-retry-policy-py — composable retry policies with backoff for LLM/agent calls."""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Any, Callable, Type


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Failed after {attempts} attempt(s): {last_error!r}")


@dataclass
class RetryAttempt:
    attempt: int
    error: Exception
    wait_seconds: float


class RetryPolicy:
    """
    Composable retry policy with configurable backoff and jitter.

    Example::

        policy = RetryPolicy(max_attempts=3, base_delay=1.0, backoff=2.0)

        @policy.wrap
        def call_llm(prompt):
            return client.complete(prompt)

        result = call_llm("Hello")
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable: tuple[Type[Exception], ...] = (Exception,),
        on_retry: Callable[[RetryAttempt], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {max_attempts}"
            )
        if base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {base_delay}")
        if max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff = backoff
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable = retryable
        self.on_retry = on_retry

    def delay_for(self, attempt: int) -> float:
        """Return the backoff delay (seconds) for a 0-indexed attempt number.

        The base delay is ``base_delay * backoff ** attempt`` capped at
        ``max_delay``. When ``jitter`` is enabled the result is a uniformly
        random value in ``[0, delay]`` ("full jitter"), which spreads
        retries out and avoids thundering-herd behaviour.
        """
        d = min(self.base_delay * (self.backoff**attempt), self.max_delay)
        if self.jitter:
            d = random.uniform(0, d)
        return d

    def execute(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute fn with retries. Raises RetryExhausted on final failure."""
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return fn(*args, **kwargs)
            except self.retryable as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                wait = self.delay_for(attempt)
                if self.on_retry:
                    self.on_retry(
                        RetryAttempt(attempt=attempt + 1, error=exc, wait_seconds=wait)
                    )
                if wait > 0:
                    time.sleep(wait)
        assert last_error is not None  # guaranteed by max_attempts >= 1
        raise RetryExhausted(self.max_attempts, last_error) from last_error

    def wrap(self, fn: Callable) -> Callable:
        """Decorator that applies this policy to a function."""
        import functools

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.execute(fn, *args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Async support
    # ------------------------------------------------------------------

    async def execute_async(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute an async ``fn`` with retries.

        Mirrors :meth:`execute` but awaits ``fn`` and sleeps with
        ``asyncio.sleep`` between attempts. Raises :class:`RetryExhausted`
        when all attempts fail.
        """
        import asyncio

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                return await fn(*args, **kwargs)
            except self.retryable as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                wait = self.delay_for(attempt)
                if self.on_retry:
                    self.on_retry(
                        RetryAttempt(attempt=attempt + 1, error=exc, wait_seconds=wait)
                    )
                if wait > 0:
                    await asyncio.sleep(wait)
        assert last_error is not None  # guaranteed by max_attempts >= 1
        raise RetryExhausted(self.max_attempts, last_error) from last_error

    def wrap_async(self, fn: Callable) -> Callable:
        """Decorator that applies this policy to an async function."""
        import functools

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.execute_async(fn, *args, **kwargs)

        return wrapper


# ------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------


def exponential(max_attempts: int = 3, base_delay: float = 1.0) -> RetryPolicy:
    """Exponential backoff with jitter."""
    return RetryPolicy(
        max_attempts=max_attempts, base_delay=base_delay, backoff=2.0, jitter=True
    )


def fixed(max_attempts: int = 3, delay: float = 1.0) -> RetryPolicy:
    """Fixed delay between retries."""
    return RetryPolicy(
        max_attempts=max_attempts, base_delay=delay, backoff=1.0, jitter=False
    )


def no_retry() -> RetryPolicy:
    """Single attempt, no retries."""
    return RetryPolicy(max_attempts=1, base_delay=0)


__all__ = [
    "RetryPolicy",
    "RetryAttempt",
    "RetryExhausted",
    "exponential",
    "fixed",
    "no_retry",
]
