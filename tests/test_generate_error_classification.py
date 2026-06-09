"""Regression tests for generate-error classification in robust_generate.

generate() runs the model call AND tool execution, so exceptions from tool/
harness code surface here too. Two failures motivated this:
  - Deterministic tool bugs (a MuJoCo TypeError, a missing motion.pt ValueError)
    were logged as "API ERROR" and retried with backoff, thrashing the circuit
    breaker instead of surfacing the bug.
  - The rate-limit heuristic matched the bare substring "limit", so tool errors
    like "...tokens allowed 32768" / "token limit exceeded" risked being treated
    as provider rate limits.
"""

import pytest

from inspect_eval.tasks import classify_generate_error


class _RateLimitError(Exception):
    """Stand-in for a provider RateLimitError (classified by its type name)."""


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("body(): incompatible function arguments"),
        ValueError("The provided filename .../motion.pt does not exist"),
        KeyError("missing_key"),
        AttributeError("'NoneType' object has no attribute 'x'"),
        FileNotFoundError("no such file"),
        ModuleNotFoundError("No module named 'anthropic'"),
    ],
)
def test_deterministic_python_errors_are_harness_bugs(exc):
    """Tool/harness bugs must be flagged so they fail fast, not retried."""
    assert classify_generate_error(exc) == "harness_bug"


@pytest.mark.parametrize(
    "exc",
    [
        _RateLimitError("slow down"),
        RuntimeError("Error 429: too many requests"),
        RuntimeError("quota exceeded for project"),
        RuntimeError("RESOURCE_EXHAUSTED"),
        RuntimeError("rate limit reached"),
    ],
)
def test_genuine_rate_limits_are_detected(exc):
    assert classify_generate_error(exc) == "rate_limit"


@pytest.mark.parametrize(
    "exc",
    [
        # Must NOT be misread as rate limits despite containing "limit".
        RuntimeError("input token count exceeds the maximum number of tokens allowed 32768"),
        RuntimeError("token limit exceeded"),
        # Genuinely transient API/network errors worth a bounded retry.
        ConnectionError("connection reset by peer"),
        RuntimeError("503 service unavailable"),
    ],
)
def test_other_errors_are_transient(exc):
    assert classify_generate_error(exc) == "transient"
