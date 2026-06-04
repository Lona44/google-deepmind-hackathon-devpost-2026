"""OpenRouter client wrapper.

Single-purpose: take (model, cached_prefix, fresh_suffix), call OpenRouter,
return a structured result with timing + token usage + cost estimate. Maps
HTTP errors onto the typed exception hierarchy in judge_panel.retry.

Cost rates are hard-coded from the screenshots the user provided
(2026-06-04 OpenRouter prices). If providers/prices change, update the
PRICING table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from judge_panel.retry import (
    AuthError,
    InsufficientCreditsError,
    OpenRouterError,
    RateLimitError,
    ServerError,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-million-token rates verified from the user's OpenRouter screenshots on
# 2026-06-04. Cache rate applied when the response usage indicates cached
# input tokens; otherwise full input rate.
PRICING = {
    "xiaomi/mimo-v2.5-pro": {"input": 0.435, "cached": 0.0036, "output": 0.87},
    "moonshotai/kimi-k2.6": {"input": 0.684, "cached": 0.144, "output": 3.42},
}


@dataclass(frozen=True)
class OpenRouterCallResult:
    content: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    duration_ms: int
    cost_usd: float
    raw_response: dict


def _compute_cost(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    rates = PRICING.get(model)
    if rates is None:
        # Unknown model — return 0 rather than error so cost tracking degrades gracefully.
        return 0.0
    fresh_input = max(0, prompt_tokens - cached_tokens)
    cost = (
        fresh_input * rates["input"] / 1_000_000
        + cached_tokens * rates["cached"] / 1_000_000
        + completion_tokens * rates["output"] / 1_000_000
    )
    return round(cost, 6)


def _raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    body_preview = response.text[:200]
    if response.status_code == 401 or response.status_code == 403:
        raise AuthError(f"HTTP {response.status_code}: {body_preview}")
    if response.status_code == 402:
        raise InsufficientCreditsError(f"HTTP {response.status_code}: {body_preview}")
    if response.status_code == 429:
        retry_after = float(response.headers.get("Retry-After", "1"))
        raise RateLimitError(f"HTTP 429: {body_preview}", retry_after_seconds=retry_after)
    if 500 <= response.status_code < 600:
        raise ServerError(f"HTTP {response.status_code}: {body_preview}")
    raise OpenRouterError(f"HTTP {response.status_code}: {body_preview}")


class OpenRouterClient:
    """Thin async wrapper around OpenRouter's chat-completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def call(
        self,
        *,
        model: str,
        cached_prefix: str,
        fresh_suffix: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
    ) -> OpenRouterCallResult:
        """One chat completion call. The prompt is the concatenation of
        cached_prefix + fresh_suffix; we keep them split in the API so
        future versions can pass OpenRouter cache-control hints."""
        body = {
            "model": model,
            "messages": [{"role": "user", "content": cached_prefix + fresh_suffix}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        t0 = time.time()
        response = await self._client.post(OPENROUTER_URL, json=body)
        elapsed_ms = int((time.time() - t0) * 1000)
        _raise_for_status(response)
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cached_tokens = int(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
        if cached_tokens == 0:
            # Some providers report under a different key. Section 0 exploration
            # finalises the exact path; document.
            cached_tokens = int(usage.get("cache_read_input_tokens", 0))

        # Some reasoning models (e.g. Kimi K2.6) return message.content=null with
        # the actual response text in message.reasoning. The verbatim string
        # `or ""` guards against both missing key and explicit JSON null.
        message = choices[0].get("message", {}) if (choices := data.get("choices", [])) else {}
        content = message.get("content") or message.get("reasoning") or ""

        return OpenRouterCallResult(
            content=content,
            input_tokens=prompt_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=completion_tokens,
            duration_ms=elapsed_ms,
            cost_usd=_compute_cost(model, prompt_tokens, cached_tokens, completion_tokens),
            raw_response=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
