"""
Capabilities API - Exposes backend capabilities based on available credentials.

This endpoint allows the frontend to detect which features are available
and adjust the UI accordingly.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel
from services.vertex_client import ClientMode, get_vertex_client, reset_client

router = APIRouter()


class FeaturesResponse(BaseModel):
    """Available features."""

    chat: bool
    video_analysis: bool
    paper_rag: bool
    google_search: bool


class ModelsResponse(BaseModel):
    """Available models."""

    chat: str
    vision: str


class LimitsResponse(BaseModel):
    """Feature limits."""

    video_max_size_mb: int
    video_max_duration_sec: int


class CapabilitiesResponse(BaseModel):
    """Full capabilities response."""

    mode: str  # "free" or "vertex"
    features: FeaturesResponse
    models: ModelsResponse
    limits: LimitsResponse


@router.get("/", response_model=CapabilitiesResponse)
async def get_capabilities() -> CapabilitiesResponse:
    """
    Get backend capabilities based on available credentials.

    Returns information about:
    - Current mode (free or vertex)
    - Available features (chat, video, RAG, search)
    - Model names being used
    - Feature limits

    The frontend uses this to:
    - Show/hide features in the UI
    - Display mode indicator
    - Enforce limits (e.g., video size)
    """
    try:
        client = get_vertex_client()
        caps = client.capabilities

        return CapabilitiesResponse(
            mode=caps.mode.value,
            features=FeaturesResponse(
                chat=caps.chat,
                video_analysis=caps.video_analysis,
                paper_rag=caps.paper_rag,
                google_search=caps.google_search,
            ),
            models=ModelsResponse(
                chat=caps.chat_model,
                vision=caps.vision_model,
            ),
            limits=LimitsResponse(
                video_max_size_mb=caps.video_max_size_mb,
                video_max_duration_sec=caps.video_max_duration_sec,
            ),
        )
    except ValueError:
        # No credentials configured - return minimal response
        return CapabilitiesResponse(
            mode="none",
            features=FeaturesResponse(
                chat=False,
                video_analysis=False,
                paper_rag=False,
                google_search=False,
            ),
            models=ModelsResponse(
                chat="",
                vision="",
            ),
            limits=LimitsResponse(
                video_max_size_mb=0,
                video_max_duration_sec=0,
            ),
        )


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns basic status and mode information.
    """
    try:
        client = get_vertex_client()
        return {
            "status": "healthy",
            "mode": client.mode.value,
            "project": client.capabilities.mode == ClientMode.VERTEX,
        }
    except ValueError:
        return {
            "status": "unconfigured",
            "mode": "none",
            "project": False,
        }


@router.post("/reset")
async def reset_client_endpoint() -> dict[str, str]:
    """
    Reset the client (useful for development/testing).

    This forces the client to re-detect credentials on next request.
    """
    reset_client()
    return {"status": "reset"}
