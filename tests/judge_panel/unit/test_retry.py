"""Unit tests for the retry policy helper.

Spec: Section 4.1. Each exception class routes to its declared retry behaviour;
auth/credits errors fail immediately; rate limits respect Retry-After.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from judge_panel.retry import (
    AuthError,
    InsufficientCreditsError,
    RateLimitError,
    ServerError,
    ValidationRetryError,
    with_retries,
)


@pytest.mark.asyncio
async def test_succeeds_on_first_try():
    func = AsyncMock(return_value="ok")
    result = await with_retries(func, policy="default")
    assert result == "ok"
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_retries_server_error_then_succeeds():
    func = AsyncMock(side_effect=[ServerError("500"), ServerError("500"), "ok"])
    result = await with_retries(func, policy="default", backoff_base=0.0)
    assert result == "ok"
    assert func.call_count == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts():
    func = AsyncMock(side_effect=ServerError("500"))
    with pytest.raises(ServerError):
        await with_retries(func, policy="default", backoff_base=0.0)
    assert func.call_count == 3  # max attempts


@pytest.mark.asyncio
async def test_auth_error_does_not_retry():
    func = AsyncMock(side_effect=AuthError("401"))
    with pytest.raises(AuthError):
        await with_retries(func, policy="default")
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_credits_error_does_not_retry():
    func = AsyncMock(side_effect=InsufficientCreditsError("402"))
    with pytest.raises(InsufficientCreditsError):
        await with_retries(func, policy="default")
    assert func.call_count == 1


@pytest.mark.asyncio
async def test_rate_limit_respects_retry_after():
    # First call raises RateLimitError with retry_after_seconds; second succeeds.
    func = AsyncMock(side_effect=[RateLimitError("429", retry_after_seconds=0.01), "ok"])
    result = await with_retries(func, policy="default", backoff_base=0.0)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_timeout_retries_twice_then_gives_up():
    func = AsyncMock(side_effect=httpx.TimeoutException("read"))
    with pytest.raises(httpx.TimeoutException):
        await with_retries(func, policy="default", backoff_base=0.0)
    assert func.call_count == 2  # 2 attempts for timeouts per spec


@pytest.mark.asyncio
async def test_validation_retry_one_attempt():
    # ValidationRetryError indicates malformed model output; retry once.
    func = AsyncMock(side_effect=[ValidationRetryError("bad json"), "ok"])
    result = await with_retries(func, policy="validation", backoff_base=0.0)
    assert result == "ok"
    assert func.call_count == 2


@pytest.mark.asyncio
async def test_validation_retry_max_one_retry():
    func = AsyncMock(side_effect=ValidationRetryError("bad json"))
    with pytest.raises(ValidationRetryError):
        await with_retries(func, policy="validation", backoff_base=0.0)
    assert func.call_count == 2  # 1 try + 1 retry
