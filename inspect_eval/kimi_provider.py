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

import os
from typing import Any

from inspect_ai.model import GenerateConfig
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.model._registry import modelapi
from openai.types.chat import ChatCompletionMessageParam


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
