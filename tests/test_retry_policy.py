"""Tests for agent-retry-policy-py.

These tests use only the Python standard library (``unittest`` and
``asyncio``) so they run with::

    python3 -m unittest discover -s tests

No third-party packages (pytest, pytest-asyncio, ...) are required.
"""

import asyncio
import os
import sys
import unittest

# Make the ``src`` layout importable when running the tests directly
# (e.g. ``python3 -m unittest discover -s tests``) without an editable install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_retry_policy import (  # noqa: E402
    RetryAttempt,
    RetryExhausted,
    RetryPolicy,
    exponential,
    fixed,
    no_retry,
)


class TestExecute(unittest.TestCase):
    def test_success_first_attempt(self):
        policy = RetryPolicy(max_attempts=3)
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(policy.execute(fn), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_on_failure_then_succeeds(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not yet")
            return "done"

        self.assertEqual(policy.execute(fn), "done")
        self.assertEqual(len(calls), 3)

    def test_raises_retry_exhausted(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)

        def fn():
            raise RuntimeError("always fails")

        with self.assertRaises(RetryExhausted) as ctx:
            policy.execute(fn)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertIsInstance(ctx.exception.last_error, RuntimeError)

    def test_retry_exhausted_chains_cause(self):
        """RetryExhausted should chain the underlying error as its cause."""
        policy = RetryPolicy(max_attempts=1, base_delay=0, jitter=False)
        boom = KeyError("boom")

        def fn():
            raise boom

        with self.assertRaises(RetryExhausted) as ctx:
            policy.execute(fn)
        self.assertIs(ctx.exception.__cause__, boom)

    def test_execute_with_args(self):
        policy = RetryPolicy(max_attempts=1)

        def add(a, b):
            return a + b

        self.assertEqual(policy.execute(add, 3, b=4), 7)


class TestCallbacksAndFiltering(unittest.TestCase):
    def test_on_retry_callback(self):
        attempts_seen = []

        def on_retry(attempt: RetryAttempt):
            attempts_seen.append(attempt.attempt)

        policy = RetryPolicy(
            max_attempts=3, base_delay=0, jitter=False, on_retry=on_retry
        )

        def fn():
            raise OSError("fail")

        with self.assertRaises(RetryExhausted):
            policy.execute(fn)
        # Callback fires before each retry, not after the final failure.
        self.assertEqual(attempts_seen, [1, 2])

    def test_on_retry_receives_attempt_details(self):
        seen = []

        def on_retry(attempt: RetryAttempt):
            seen.append(attempt)

        policy = RetryPolicy(
            max_attempts=2, base_delay=0, jitter=False, on_retry=on_retry
        )

        def fn():
            raise ValueError("nope")

        with self.assertRaises(RetryExhausted):
            policy.execute(fn)
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], RetryAttempt)
        self.assertIsInstance(seen[0].error, ValueError)
        self.assertEqual(seen[0].wait_seconds, 0.0)

    def test_retryable_filter(self):
        policy = RetryPolicy(
            max_attempts=3, base_delay=0, jitter=False, retryable=(ValueError,)
        )
        calls = []

        def fn():
            calls.append(1)
            raise TypeError("not retryable")

        with self.assertRaises(TypeError):
            policy.execute(fn)
        self.assertEqual(len(calls), 1)  # no retries for non-retryable errors


class TestDelay(unittest.TestCase):
    def test_delay_for_exponential(self):
        policy = RetryPolicy(
            base_delay=1.0, backoff=2.0, max_delay=60.0, jitter=False
        )
        self.assertEqual(policy.delay_for(0), 1.0)
        self.assertEqual(policy.delay_for(1), 2.0)
        self.assertEqual(policy.delay_for(2), 4.0)

    def test_delay_for_capped(self):
        policy = RetryPolicy(
            base_delay=1.0, backoff=2.0, max_delay=3.0, jitter=False
        )
        self.assertEqual(policy.delay_for(10), 3.0)

    def test_delay_for_fixed(self):
        policy = RetryPolicy(base_delay=2.0, backoff=1.0, jitter=False)
        self.assertEqual(policy.delay_for(0), 2.0)
        self.assertEqual(policy.delay_for(5), 2.0)

    def test_delay_for_jitter_within_bounds(self):
        policy = RetryPolicy(
            base_delay=1.0, backoff=2.0, max_delay=60.0, jitter=True
        )
        for attempt in range(5):
            upper = min(1.0 * (2.0**attempt), 60.0)
            for _ in range(20):
                d = policy.delay_for(attempt)
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, upper)


class TestValidation(unittest.TestCase):
    def test_max_attempts_must_be_positive(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=0)
        with self.assertRaises(ValueError):
            RetryPolicy(max_attempts=-1)

    def test_negative_base_delay_rejected(self):
        with self.assertRaises(ValueError):
            RetryPolicy(base_delay=-1.0)

    def test_negative_max_delay_rejected(self):
        with self.assertRaises(ValueError):
            RetryPolicy(max_delay=-5.0)


class TestDecorators(unittest.TestCase):
    def test_wrap_decorator(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)
        calls = []

        @policy.wrap
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("retry")
            return "wrapped"

        self.assertEqual(fn(), "wrapped")
        self.assertEqual(len(calls), 2)

    def test_wrap_preserves_metadata(self):
        policy = RetryPolicy(max_attempts=1)

        @policy.wrap
        def documented():
            """A documented function."""
            return 1

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "A documented function.")


class TestPresets(unittest.TestCase):
    def test_presets_exponential(self):
        p = exponential(max_attempts=4, base_delay=0.5)
        self.assertEqual(p.max_attempts, 4)
        self.assertEqual(p.base_delay, 0.5)
        self.assertEqual(p.backoff, 2.0)
        self.assertTrue(p.jitter)

    def test_presets_fixed(self):
        p = fixed(max_attempts=5, delay=2.0)
        self.assertEqual(p.max_attempts, 5)
        self.assertEqual(p.base_delay, 2.0)
        self.assertEqual(p.backoff, 1.0)
        self.assertFalse(p.jitter)

    def test_presets_no_retry_success(self):
        p = no_retry()
        self.assertEqual(p.max_attempts, 1)
        self.assertEqual(p.execute(lambda: "ok"), "ok")

    def test_presets_no_retry_failure(self):
        p = no_retry()
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("fail")

        with self.assertRaises(RetryExhausted) as ctx:
            p.execute(fn)
        self.assertEqual(ctx.exception.attempts, 1)
        self.assertEqual(len(calls), 1)


class TestAsync(unittest.TestCase):
    def test_execute_async_success(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)

        async def fn():
            return "async_ok"

        self.assertEqual(asyncio.run(policy.execute_async(fn)), "async_ok")

    def test_execute_async_retries(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0, jitter=False)
        calls = []

        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("try again")
            return "done"

        self.assertEqual(asyncio.run(policy.execute_async(fn)), "done")
        self.assertEqual(len(calls), 2)

    def test_execute_async_exhausted(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)

        async def fn():
            raise RuntimeError("always")

        with self.assertRaises(RetryExhausted) as ctx:
            asyncio.run(policy.execute_async(fn))
        self.assertEqual(ctx.exception.attempts, 2)

    def test_wrap_async(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0, jitter=False)
        calls = []

        @policy.wrap_async
        async def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("retry async")
            return "wrapped_async"

        self.assertEqual(asyncio.run(fn()), "wrapped_async")
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
