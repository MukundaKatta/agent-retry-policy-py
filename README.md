# agent-retry-policy-py

Composable retry policies with configurable backoff and jitter for LLM/agent calls.

Network-bound LLM and tool calls fail transiently — timeouts, rate limits,
flaky connections. This library wraps any sync or async callable with a
configurable retry policy: exponential/fixed backoff, full jitter, an upper
delay cap, a filter for which exceptions are retryable, and an `on_retry`
callback for logging. It has **zero runtime dependencies** (standard library
only).

## Install

The package targets Python 3.9+. Install from a local checkout:

```bash
pip install .
```

or directly from GitHub:

```bash
pip install git+https://github.com/MukundaKatta/agent-retry-policy-py.git
```

## Usage

```python
from agent_retry_policy import RetryPolicy, exponential, fixed, no_retry

# Exponential backoff with jitter
policy = RetryPolicy(max_attempts=3, base_delay=1.0, backoff=2.0)

@policy.wrap
def call_llm(prompt):
    return client.complete(prompt)

result = call_llm("Hello")

# Preset shortcuts
policy = exponential(max_attempts=4, base_delay=0.5)
policy = fixed(max_attempts=3, delay=2.0)
policy = no_retry()

# Async support
@policy.wrap_async
async def async_call(prompt):
    return await client.acomplete(prompt)

# On-retry callback
def log_retry(attempt):
    print(f"Retry {attempt.attempt} after {attempt.wait_seconds:.1f}s: {attempt.error}")

policy = RetryPolicy(max_attempts=3, base_delay=1.0, on_retry=log_retry)

# Filter which exceptions trigger retries
policy = RetryPolicy(max_attempts=3, retryable=(TimeoutError, ConnectionError))
```

## Runnable example

This snippet runs as-is (no network or API key required):

```python
from agent_retry_policy import RetryPolicy, RetryExhausted

# Fail twice, then succeed — base_delay=0 keeps the example fast.
attempts = {"n": 0}

def flaky_call():
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError(f"transient failure #{attempts['n']}")
    return "success"

policy = RetryPolicy(max_attempts=5, base_delay=0, jitter=False)
print(policy.execute(flaky_call))  # -> "success" after 3 tries

# When every attempt fails, RetryExhausted is raised and the original
# error is preserved both as `.last_error` and as the chained `__cause__`.
def always_fails():
    raise TimeoutError("upstream down")

try:
    RetryPolicy(max_attempts=2, base_delay=0).execute(always_fails)
except RetryExhausted as exc:
    print(exc.attempts)        # -> 2
    print(repr(exc.last_error))  # -> TimeoutError('upstream down')
```

## Behavior notes

- `delay_for(attempt)` returns `base_delay * backoff ** attempt`, capped at
  `max_delay`. With `jitter=True` (the default) the actual sleep is a uniform
  random value in `[0, delay]` ("full jitter"), which avoids thundering-herd
  retries.
- The `on_retry` callback fires *before* each retry sleep, once per retry — so
  with `max_attempts=3` it fires at most twice.
- `retryable` filters which exception types trigger a retry; anything not
  matching propagates immediately without retrying.
- Constructor arguments are validated: `max_attempts` must be `>= 1` and
  `base_delay`/`max_delay` must be non-negative, otherwise `ValueError` is
  raised.

## API

- `RetryPolicy(max_attempts=3, base_delay=1.0, backoff=2.0, max_delay=60.0, jitter=True, retryable=(Exception,), on_retry=None)` — full config
- `.execute(fn, *args, **kwargs)` — run a sync callable with retries
- `.execute_async(fn, *args, **kwargs)` — run an async callable with retries
- `.wrap(fn)` / `.wrap_async(fn)` — decorators that apply the policy
- `.delay_for(attempt)` — compute the backoff delay for a 0-indexed attempt
- `exponential(max_attempts=3, base_delay=1.0)` / `fixed(max_attempts=3, delay=1.0)` / `no_retry()` — preset factories
- `RetryAttempt(attempt, error, wait_seconds)` — passed to `on_retry`
- `RetryExhausted(attempts, last_error)` — raised when all attempts fail

## Development

The test suite uses only the Python standard library (`unittest`), so no
third-party packages are needed:

```bash
python3 -m unittest discover -s tests
```

## License

MIT
