"""Retry policy helper for OpenRouter calls.

Spec Section 4.1: each exception class routes to a declared retry behaviour.
Auth and InsufficientCredits fail fast (silent retry would mask the very
problem the user needs to know about).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Literal

import httpx


class OpenRouterError(Exception):
    """Base class for OpenRouter errors caught by the retry helper."""


class AuthError(OpenRouterError):
    """HTTP 401/403 — invalid or revoked API key. Never retry."""


class InsufficientCreditsError(OpenRouterError):
    """HTTP 402 — out of credits. Never retry."""


class RateLimitError(OpenRouterError):
    """HTTP 429 — back off and retry. retry_after_seconds from Retry-After header."""

    def __init__(self, message: str, retry_after_seconds: float = 1.0):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ServerError(OpenRouterError):
    """HTTP 5xx — transient server issue. Exponential backoff retry."""


class ValidationRetryError(OpenRouterError):
    """Raised by a role module when the model returned malformed output.

    The role's prompt should be re-issued with a format-correction nudge.
    Allowed exactly one retry; persistent failure propagates.
    """


Policy = Literal["default", "validation"]


async def with_retries(
    func: Callable[[], Awaitable[Any]],
    policy: Policy = "default",
    *,
    backoff_base: float = 1.0,
) -> Any:
    """Call `func` with the retry policy.

    Args:
        func: Async callable taking no arguments.
        policy: "default" for OpenRouter calls (3 attempts with exponential
            backoff on ServerError/RateLimit/Timeout; immediate fail on
            Auth/Credits). "validation" for model-output reparses (1 retry).
        backoff_base: Seconds for exponential backoff base. Pass 0.0 in
            tests to skip sleeps.

    Returns:
        Whatever `func` returns on success.

    Raises:
        The last exception encountered when retries are exhausted.
    """
    if policy == "validation":
        attempts = 2  # 1 try + 1 retry
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return await func()
            except ValidationRetryError as exc:
                last_exc = exc
                if i + 1 < attempts:
                    continue
                raise
        assert last_exc is not None  # unreachable
        raise last_exc

    # default policy
    max_server_attempts = 3
    max_timeout_attempts = 2
    server_count = 0
    timeout_count = 0
    last_exc: Exception | None = None

    while True:
        try:
            return await func()
        except (AuthError, InsufficientCreditsError):
            raise  # no retry
        except RateLimitError as exc:
            last_exc = exc
            await asyncio.sleep(exc.retry_after_seconds)
            continue  # retry indefinitely on rate limit, capped at server_count
        except ServerError as exc:
            last_exc = exc
            server_count += 1
            if server_count >= max_server_attempts:
                raise
            await asyncio.sleep(backoff_base * (3 ** (server_count - 1)))
        except httpx.TimeoutException as exc:
            last_exc = exc
            timeout_count += 1
            if timeout_count >= max_timeout_attempts:
                raise
            await asyncio.sleep(backoff_base * 5 * timeout_count)
