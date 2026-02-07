"""Custom Kimi/Moonshot provider that properly handles reasoning_content.

Kimi K2.5 requires reasoning_content to be echoed back in assistant messages
when continuing multi-turn conversations. The standard OpenAI-compatible provider
doesn't do this, causing "reasoning_content is missing" errors.

THE PROBLEM (as of Jan 26, 2026):
    Kimi K2.5 has thinking enabled by default. When the API returns a response,
    it includes a `reasoning_content` field. On subsequent requests, Kimi expects
    this field to be echoed back in assistant messages. Without it:

    Error: "thinking is enabled but reasoning_content is missing in assistant
           tool call message at index N"

OUR FIX:
    Override `messages_to_openai()` to extract reasoning from Inspect's
    ContentReasoning objects and include it as `reasoning_content` in the
    assistant message dict sent to the API.

FORWARD COMPATIBILITY:
    When Moonshot or Inspect AI fixes this issue:
    - This provider will still work (extra fields are typically ignored by APIs)
    - To switch back to standard provider: use "openai-api/moonshot/kimi-k2.5"
    - No code changes needed - just update the model string in run_inspect_visual.py

Usage:
    # Import registers the provider with Inspect's model registry
    import inspect_eval.kimi_provider

    results = eval(
        "inspect_eval/tasks.py@g1_native",
        model="kimi/kimi-k2.5",  # Uses this registered provider
        ...
    )
"""

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

import openai
from inspect_ai.model import GenerateConfig, ModelCall, ModelOutput
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolChoice, ToolInfo
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

# Global flag to signal rate limit hit - can be checked by robust_generate
RATE_LIMIT_HIT = False
RATE_LIMIT_MESSAGE = ""

# Rate limit check state - avoid checking too frequently
_last_rate_check: float = 0
_rate_check_interval: float = 5.0  # Only check every 5 seconds


def _check_kimi_rate_limit_sync() -> tuple[bool, str]:
    """Quick check if Kimi API is rate limited.

    Makes a minimal API call to detect rate limits BEFORE the heavy request.
    Returns (is_limited, error_message).
    """
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        return False, ""

    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    # Minimal request - just enough to check rate limit
    payload = {
        "model": "kimi-k2.5",
        "messages": [{"role": "user", "content": "1"}],
        "max_tokens": 1,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            # Success - not rate limited
            return False, ""
    except urllib.error.HTTPError as e:
        if e.code == 429:
            error_body = e.read().decode("utf-8")
            return True, f"HTTP 429: {error_body[:200]}"
        elif e.code in (402, 403):
            error_body = e.read().decode("utf-8")
            return True, f"HTTP {e.code}: {error_body[:200]}"
        # Other errors - not rate limit
        return False, ""
    except Exception:
        # Network error etc - not rate limit
        return False, ""


async def check_kimi_rate_limit() -> tuple[bool, str]:
    """Async wrapper for rate limit check with throttling.

    Only checks every _rate_check_interval seconds to avoid
    consuming too much API quota on Tier0 (3 RPM limit).
    """
    global _last_rate_check

    # Throttle: don't check if we checked recently
    now = time.time()
    if now - _last_rate_check < _rate_check_interval:
        return False, ""  # Assume OK, skip check

    _last_rate_check = now

    # Run sync check in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _check_kimi_rate_limit_sync)


class KimiAPIError(Exception):
    """Custom exception for Kimi API errors with clear context."""

    def __init__(self, error_type: str, message: str, is_rate_limit: bool = False):
        self.error_type = error_type
        self.message = message
        self.is_rate_limit = is_rate_limit
        super().__init__(f"[Kimi API] {error_type}: {message}")


def reasoning_to_reasoning_content(reasoning_content: str) -> dict[str, Any]:
    """Convert reasoning to reasoning_content field for Kimi API."""
    return {"reasoning_content": reasoning_content}


@modelapi(name="kimi")
class KimiAPI(OpenAICompatibleAPI):
    """Custom Kimi provider that echoes reasoning_content in assistant messages."""

    def __init__(
        self,
        model_name: str = "moonshot/kimi-k2.5",
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
        **model_args: Any,
    ) -> None:
        # Set default config if not provided
        if config is None:
            config = GenerateConfig()

        # Set default base URL for Moonshot
        if base_url is None:
            base_url = os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            service="moonshot",
            emulate_reasoning_history=False,  # We handle it ourselves
            **model_args,
        )

        # Store reasoning_content from responses for echoing back
        self._reasoning_cache: dict[str, str] = {}

    async def messages_to_openai(
        self, input: list[ChatMessage]
    ) -> list[ChatCompletionMessageParam]:
        """Convert messages to OpenAI format, preserving reasoning_content."""
        from inspect_ai.model._openai import (  # noqa: PLC0415
            openai_chat_message,
            openai_chat_tool_call_param,
        )

        messages: list[ChatCompletionMessageParam] = []

        for message in input:
            if message.role == "assistant":
                # Handle assistant messages specially to include reasoning_content
                assistant_msg = message
                content_text = ""

                # Extract text content and reasoning
                reasoning_text = None
                if isinstance(assistant_msg.content, str):
                    content_text = assistant_msg.content
                else:
                    for c in assistant_msg.content:
                        if hasattr(c, "type"):
                            if c.type == "text":
                                content_text += c.text
                            elif c.type == "reasoning":
                                reasoning_text = c.reasoning

                # Build assistant message
                if assistant_msg.tool_calls:
                    msg_dict: dict[str, Any] = {
                        "role": "assistant",
                        "content": content_text or None,
                        "tool_calls": [
                            openai_chat_tool_call_param(call) for call in assistant_msg.tool_calls
                        ],
                    }
                else:
                    msg_dict = {
                        "role": "assistant",
                        "content": content_text,
                    }

                # Add reasoning_content if we have it
                if reasoning_text:
                    msg_dict["reasoning_content"] = reasoning_text

                messages.append(msg_dict)  # type: ignore
            else:
                # Use standard conversion for other message types
                messages.append(await openai_chat_message(message))

        return messages

    def on_response(self, response: dict[str, Any]) -> None:
        """Capture reasoning_content from responses."""
        # The parent class's chat_choices_from_openai will handle converting
        # reasoning_content to ContentReasoning, which we then echo back
        pass

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        """Generate with explicit rate limit detection.

        Checks for rate limits BEFORE making the heavy request, and catches
        rate limit errors at the HTTP level to raise them immediately.
        """
        global RATE_LIMIT_HIT, RATE_LIMIT_MESSAGE

        # PRE-CHECK: Quick rate limit detection before heavy request
        # Throttled to avoid consuming too much quota (only checks every 5s)
        is_limited, limit_msg = await check_kimi_rate_limit()
        if is_limited:
            RATE_LIMIT_HIT = True
            RATE_LIMIT_MESSAGE = limit_msg
            logger.error(f"KIMI PRE-CHECK RATE LIMIT: {limit_msg}")
            raise KimiAPIError(
                error_type="RateLimitPreCheck",
                message=limit_msg,
                is_rate_limit=True,
            )

        try:
            return await super().generate(input, tools, tool_choice, config)

        except openai.RateLimitError as e:
            RATE_LIMIT_HIT = True
            RATE_LIMIT_MESSAGE = str(e)
            logger.error(f"KIMI RATE LIMIT HIT: {e}")
            # Re-raise as our custom error so it's not swallowed by retries
            raise KimiAPIError(
                error_type="RateLimitError",
                message=str(e),
                is_rate_limit=True,
            ) from e

        except openai.APIStatusError as e:
            # Catch other API errors (500, 503, etc.)
            status_code = getattr(e, "status_code", None)
            logger.error(f"KIMI API ERROR ({status_code}): {e}")

            # Check if it's a quota/usage limit error (might be 403 or 402)
            if status_code in (402, 403, 429):
                RATE_LIMIT_HIT = True
                RATE_LIMIT_MESSAGE = str(e)
                raise KimiAPIError(
                    error_type=f"APIError_{status_code}",
                    message=str(e),
                    is_rate_limit=True,
                ) from e

            # Re-raise other errors
            raise


def kimi_model(
    model_name: str = "kimi-k2.5",
    api_key: str | None = None,
    base_url: str = "https://api.moonshot.ai/v1",
    **kwargs: Any,
) -> KimiAPI:
    """Create a Kimi model instance.

    Args:
        model_name: Model name (default: kimi-k2.5)
        api_key: Moonshot API key (defaults to MOONSHOT_API_KEY env var)
        base_url: API base URL
        **kwargs: Additional arguments passed to KimiAPI

    Returns:
        KimiAPI instance configured for Kimi
    """
    return KimiAPI(
        model_name=f"moonshot/{model_name}",
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
