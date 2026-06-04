"""Unit tests for the OpenRouter client wrapper.

Uses httpx.MockTransport for stubbed responses; no live API calls.
"""

from __future__ import annotations

import json

import httpx
import pytest

from judge_panel.models import (
    OpenRouterClient,
    OpenRouterCallResult,
)
from judge_panel.retry import (
    AuthError,
    InsufficientCreditsError,
    RateLimitError,
    ServerError,
)


def _mock_transport(responses: list[httpx.Response]) -> httpx.AsyncBaseTransport:
    iterator = iter(responses)
    def handler(request: httpx.Request) -> httpx.Response:
        return next(iterator)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_successful_call_returns_parsed_result():
    body = {
        "id": "test-1",
        "model": "xiaomi/mimo-v2.5-pro",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }
    transport = _mock_transport([httpx.Response(200, json=body)])
    client = OpenRouterClient(api_key="test", transport=transport)
    result = await client.call(model="xiaomi/mimo-v2.5-pro", cached_prefix="A", fresh_suffix="B")
    assert isinstance(result, OpenRouterCallResult)
    assert result.content == "hello"
    assert result.input_tokens == 100
    assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_401_raises_auth_error():
    transport = _mock_transport([httpx.Response(401, json={"error": "bad key"})])
    client = OpenRouterClient(api_key="test", transport=transport)
    with pytest.raises(AuthError):
        await client.call(model="x", cached_prefix="A", fresh_suffix="B")


@pytest.mark.asyncio
async def test_402_raises_credits_error():
    transport = _mock_transport([httpx.Response(402, json={"error": "no credits"})])
    client = OpenRouterClient(api_key="test", transport=transport)
    with pytest.raises(InsufficientCreditsError):
        await client.call(model="x", cached_prefix="A", fresh_suffix="B")


@pytest.mark.asyncio
async def test_429_raises_rate_limit_with_retry_after():
    transport = _mock_transport([httpx.Response(429, headers={"Retry-After": "5"}, json={"error": "slow"})])
    client = OpenRouterClient(api_key="test", transport=transport)
    with pytest.raises(RateLimitError) as exc:
        await client.call(model="x", cached_prefix="A", fresh_suffix="B")
    assert exc.value.retry_after_seconds == 5.0


@pytest.mark.asyncio
async def test_500_raises_server_error():
    transport = _mock_transport([httpx.Response(500, json={"error": "oops"})])
    client = OpenRouterClient(api_key="test", transport=transport)
    with pytest.raises(ServerError):
        await client.call(model="x", cached_prefix="A", fresh_suffix="B")


@pytest.mark.asyncio
async def test_records_timing_and_cost():
    body = {
        "id": "x",
        "model": "moonshotai/kimi-k2.6",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7000, "completion_tokens": 3000, "total_tokens": 10000},
    }
    transport = _mock_transport([httpx.Response(200, json=body)])
    client = OpenRouterClient(api_key="test", transport=transport)
    result = await client.call(model="moonshotai/kimi-k2.6", cached_prefix="A", fresh_suffix="B")
    assert result.duration_ms >= 0
    # Cost: 7000 input × $0.684/M + 3000 output × $3.42/M = $0.0048 + $0.01026 ≈ $0.015
    assert 0.010 < result.cost_usd < 0.020
