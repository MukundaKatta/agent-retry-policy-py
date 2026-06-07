"""Tests for agent-retry-policy-py."""

import pytest
from agent_retry_policy import (
    RetryPolicy,
    RetryExhausted,
    RetryAttempt,
    exponential,
    fixed,
    no_retry,
)


def test_success_first_attempt():
    policy = RetryPolicy(max_attempts=3)
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert policy.execute(fn) == "ok"
    assert len(calls) == 1


def test_retries_on_failure_then_succeeds():
    policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "done"

    assert policy.execute(fn) == "done"
    assert len(calls) == 3


def test_raises_retry_exhausted():
    policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)

    def fn():
        raise RuntimeError("always fails")

    with pytest.raises(RetryExhausted) as exc_info:
        policy.execute(fn)
    assert exc_info.value.attempts == 2
    assert isinstance(exc_info.value.last_error, RuntimeError)


def test_on_retry_callback():
    attempts_seen = []

    def on_retry(attempt: RetryAttempt):
        attempts_seen.append(attempt.attempt)

    policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False, on_retry=on_retry)

    def fn():
        raise IOError("fail")

    with pytest.raises(RetryExhausted):
        policy.execute(fn)
    assert attempts_seen == [1, 2]


def test_retryable_filter():
    policy = RetryPolicy(
        max_attempts=3, base_delay=0, jitter=False, retryable=(ValueError,)
    )
    calls = []

    def fn():
        calls.append(1)
        raise TypeError("not retryable")

    with pytest.raises(TypeError):
        policy.execute(fn)
    assert len(calls) == 1  # no retries


def test_delay_for_exponential():
    policy = RetryPolicy(base_delay=1.0, backoff=2.0, max_delay=60.0, jitter=False)
    assert policy.delay_for(0) == 1.0
    assert policy.delay_for(1) == 2.0
    assert policy.delay_for(2) == 4.0


def test_delay_for_capped():
    policy = RetryPolicy(base_delay=1.0, backoff=2.0, max_delay=3.0, jitter=False)
    assert policy.delay_for(10) == 3.0


def test_wrap_decorator():
    policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)
    calls = []

    @policy.wrap
    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("retry")
        return "wrapped"

    assert fn() == "wrapped"
    assert len(calls) == 2


def test_presets_exponential():
    p = exponential(max_attempts=4, base_delay=0.5)
    assert p.max_attempts == 4
    assert p.base_delay == 0.5
    assert p.backoff == 2.0
    assert p.jitter is True


def test_presets_fixed():
    p = fixed(max_attempts=5, delay=2.0)
    assert p.max_attempts == 5
    assert p.base_delay == 2.0
    assert p.backoff == 1.0
    assert p.jitter is False


def test_presets_no_retry():
    p = no_retry()
    assert p.max_attempts == 1
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError("fail")

    with pytest.raises(RetryExhausted) as exc_info:
        p.execute(fn)
    assert exc_info.value.attempts == 1
    assert len(calls) == 1


def test_execute_with_args():
    policy = RetryPolicy(max_attempts=1)

    def add(a, b):
        return a + b

    assert policy.execute(add, 3, b=4) == 7


@pytest.mark.asyncio
async def test_execute_async_success():
    policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)

    async def fn():
        return "async_ok"

    result = await policy.execute_async(fn)
    assert result == "async_ok"


@pytest.mark.asyncio
async def test_execute_async_retries():
    policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)
    calls = []

    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("try again")
        return "done"

    result = await policy.execute_async(fn)
    assert result == "done"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_wrap_async():
    policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)
    calls = []

    @policy.wrap_async
    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("retry async")
        return "wrapped_async"

    assert await fn() == "wrapped_async"
    assert len(calls) == 2
