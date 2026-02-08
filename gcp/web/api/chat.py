"""
Chat API - Secure proxy for Gemini API calls.

Supports two modes:
1. Vertex AI mode (when GOOGLE_CLOUD_PROJECT is set) - uses ADC credentials
2. API Key mode (when GEMINI_API_KEY is set) - uses direct API
"""

import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Gemini model configuration - Gemini 3 for the hackathon
GEMINI_MODEL = "gemini-3-pro-preview"


def is_vertex_mode() -> bool:
    """Check if Vertex AI mode is enabled."""
    return bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: str  # 'user' or 'model'
    parts: list[dict[str, Any]]


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    success: bool
    content: dict[str, Any] | None = None
    error: str | None = None


async def _call_vertex_ai(contents: list, tools: list | None) -> dict[str, Any]:
    """Call Gemini via Vertex AI using the google-genai SDK."""
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = genai.Client(vertexai=True, project=project, location=location)

    # Build config
    config: dict[str, Any] = {
        "temperature": 0.7,
        "max_output_tokens": 1024,
    }

    if tools:
        config["tools"] = [{"function_declarations": tools}]

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

    # Convert response to dict format matching the REST API
    result: dict[str, Any] = {"candidates": []}

    if response.candidates:
        candidate = response.candidates[0]
        parts = []

        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append({"text": part.text})
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    parts.append({
                        "functionCall": {
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {}
                        }
                    })

        result["candidates"].append({
            "content": {
                "role": "model",
                "parts": parts
            },
            "finishReason": candidate.finish_reason.name if candidate.finish_reason else "STOP"
        })

    return result


async def _call_direct_api(contents: list, tools: list | None) -> dict[str, Any]:
    """Call Gemini via direct REST API with API key."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not configured")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    request_body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    }

    if tools:
        request_body["tools"] = [{"functionDeclarations": tools}]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{api_url}?key={api_key}",
            json=request_body,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get(
                "message", f"API error: {response.status_code}"
            )
            raise ValueError(error_msg)

        return response.json()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Proxy chat requests to Gemini API.

    Supports two modes:
    - Vertex AI: Uses ADC credentials (GOOGLE_CLOUD_PROJECT env var)
    - Direct API: Uses GEMINI_API_KEY env var
    """
    # Check credentials
    has_vertex = is_vertex_mode()
    has_api_key = bool(os.environ.get("GEMINI_API_KEY"))

    if not has_vertex and not has_api_key:
        raise HTTPException(
            status_code=500,
            detail="No credentials configured. Set GOOGLE_CLOUD_PROJECT (Vertex AI) or GEMINI_API_KEY.",
        )

    # Build contents
    contents = []

    # Add system prompt as first user message (Gemini pattern)
    if request.system_prompt:
        contents.append({"role": "user", "parts": [{"text": request.system_prompt}]})
        contents.append(
            {
                "role": "model",
                "parts": [
                    {
                        "text": "I understand. I'm ready to help you explore the G1 alignment experiments."
                    }
                ],
            }
        )

    # Add conversation messages
    for msg in request.messages:
        contents.append({"role": msg.role, "parts": msg.parts})

    try:
        if has_vertex:
            data = await _call_vertex_ai(contents, request.tools)
        else:
            data = await _call_direct_api(contents, request.tools)

        return ChatResponse(success=True, content=data)

    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.get("/status")
async def chat_status() -> dict[str, Any]:
    """Check if chat API is properly configured."""
    has_vertex = is_vertex_mode()
    has_api_key = bool(os.environ.get("GEMINI_API_KEY"))

    return {
        "configured": has_vertex or has_api_key,
        "mode": "vertex" if has_vertex else ("api_key" if has_api_key else "none"),
        "model": GEMINI_MODEL,
    }
