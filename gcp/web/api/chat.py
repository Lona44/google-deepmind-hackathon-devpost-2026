"""
Chat API - Secure proxy for Gemini API calls.

Keeps the API key server-side, never exposed to frontend.
"""

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Gemini API configuration - Gemini 3 for the hackathon
GEMINI_MODEL = "gemini-3-pro-preview"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


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


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Proxy chat requests to Gemini API.

    The API key is read from environment variables, never exposed to frontend.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not configured on server",
        )

    # Build the request body
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

    request_body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
    }

    # Add tools if provided
    if request.tools:
        request_body["tools"] = [{"functionDeclarations": request.tools}]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get(
                    "message", f"API error: {response.status_code}"
                )
                return ChatResponse(success=False, error=error_msg)

            data = response.json()
            return ChatResponse(success=True, content=data)

    except httpx.TimeoutException:
        return ChatResponse(success=False, error="Request timed out")
    except Exception as e:
        return ChatResponse(success=False, error=str(e))


@router.get("/status")
async def chat_status() -> dict[str, Any]:
    """Check if chat API is properly configured."""
    api_key = os.environ.get("GEMINI_API_KEY")
    return {
        "configured": bool(api_key),
        "model": GEMINI_MODEL,
    }
