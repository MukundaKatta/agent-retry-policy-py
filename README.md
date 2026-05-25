# agent-retry-policy-py

Composable retry policies with configurable backoff and jitter for LLM/agent calls.

## Install

```bash
pip install agent-retry-policy-py
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

## API

- `RetryPolicy(max_attempts, base_delay, backoff, max_delay, jitter, retryable, on_retry)` — full config
- `.execute(fn, *args, **kwargs)` — run with retries
- `.wrap(fn)` / `.wrap_async(fn)` — decorators
- `.execute_async(fn, ...)` — async execute
- `exponential()` / `fixed()` / `no_retry()` — preset factories
- `RetryExhausted` — raised when all attempts fail

## License

MIT
