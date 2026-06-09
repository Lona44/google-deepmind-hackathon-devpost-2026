"""
Vertex Client - Unified client supporting both direct Gemini API and Vertex AI modes.

Automatically detects available credentials and provides appropriate client.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig


class ClientMode(str, Enum):
    """Available client modes."""

    FREE = "free"  # Direct Gemini API with API key
    VERTEX = "vertex"  # Vertex AI with GCP project


@dataclass
class ClientCapabilities:
    """Capabilities available in current mode."""

    mode: ClientMode
    chat: bool = True
    video_analysis: bool = False
    paper_rag: bool = False
    google_search: bool = False

    # Model names
    chat_model: str = "gemini-2.0-flash"
    vision_model: str = "gemini-2.0-flash"

    # Limits
    video_max_size_mb: int = 100
    video_max_duration_sec: int = 300


class VertexClient:
    """
    Unified Gemini client supporting both free (API key) and Vertex AI modes.

    Usage:
        client = VertexClient()
        response = client.generate_content("Hello!")
        print(client.capabilities)
    """

    def __init__(self) -> None:
        self._mode = self._detect_mode()
        self._client = self._init_client()
        self._capabilities = self._init_capabilities()

    @property
    def mode(self) -> ClientMode:
        """Current client mode."""
        return self._mode

    @property
    def client(self) -> genai.Client:
        """Underlying genai client."""
        return self._client

    @property
    def capabilities(self) -> ClientCapabilities:
        """Available capabilities in current mode."""
        return self._capabilities

    def _detect_mode(self) -> ClientMode:
        """Detect which mode to use based on available credentials."""
        # Check for Vertex AI credentials first (preferred if available)
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")

        if gcp_project:
            # Verify we can actually use Vertex AI
            # (credentials should be available via ADC or service account)
            return ClientMode.VERTEX

        # Fall back to direct API key mode
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            return ClientMode.FREE

        raise ValueError(
            "No credentials found. Set either GOOGLE_CLOUD_PROJECT (for Vertex AI) "
            "or GEMINI_API_KEY (for direct API access)."
        )

    def _init_client(self) -> genai.Client:
        """Initialize the appropriate genai client."""
        if self._mode == ClientMode.VERTEX:
            return genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            )
        else:
            return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _init_capabilities(self) -> ClientCapabilities:
        """Determine capabilities based on mode and available resources."""
        if self._mode == ClientMode.VERTEX:
            # Check for optional Vertex AI Search datastore
            has_datastore = bool(os.getenv("VERTEX_SEARCH_DATASTORE_ID"))
            # Check for GCS bucket (needed for video uploads)
            has_gcs = bool(os.getenv("GCS_BUCKET_NAME"))

            return ClientCapabilities(
                mode=ClientMode.VERTEX,
                chat=True,
                video_analysis=has_gcs,  # Need GCS to upload videos
                paper_rag=has_datastore,  # Need datastore for RAG
                google_search=True,  # Available in Vertex AI
                chat_model="gemini-3.1-pro-preview",
                vision_model="gemini-3.1-pro-preview",
            )
        else:
            # Free mode - basic capabilities only
            return ClientCapabilities(
                mode=ClientMode.FREE,
                chat=True,
                video_analysis=False,
                paper_rag=False,
                google_search=False,
                chat_model="gemini-2.0-flash",
                vision_model="gemini-2.0-flash",
            )

    def generate_content(
        self,
        prompt: str,
        *,
        model: str | None = None,
        config: GenerateContentConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate content using the appropriate model.

        Args:
            prompt: The prompt to send
            model: Model name (defaults to chat_model from capabilities)
            config: Generation config
            **kwargs: Additional arguments passed to generate_content

        Returns:
            Response from the model
        """
        model_name = model or self._capabilities.chat_model

        return self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
            **kwargs,
        )

    def generate_content_with_video(
        self,
        prompt: str,
        video_uri: str,
        *,
        model: str | None = None,
        config: GenerateContentConfig | None = None,
    ) -> Any:
        """
        Generate content with video input (Vertex AI mode only).

        Args:
            prompt: The question about the video
            video_uri: GCS URI of the video (gs://bucket/path/video.mp4)
            model: Model name (defaults to vision_model)
            config: Generation config

        Returns:
            Response from the model

        Raises:
            ValueError: If video analysis not available in current mode
        """
        if not self._capabilities.video_analysis:
            raise ValueError(
                "Video analysis not available. Requires Vertex AI mode with GCS bucket configured."
            )

        model_name = model or self._capabilities.vision_model

        # Build multimodal content using Part.from_uri for GCS video
        video_part = types.Part.from_uri(
            file_uri=video_uri,
            mime_type="video/mp4",
        )

        contents = [video_part, prompt]

        return self._client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )


# Singleton instance (lazy initialization)
_client_instance: VertexClient | None = None


def get_vertex_client() -> VertexClient:
    """Get the singleton VertexClient instance."""
    global _client_instance  # noqa: PLW0603 - Singleton pattern
    if _client_instance is None:
        _client_instance = VertexClient()
    return _client_instance


def reset_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _client_instance  # noqa: PLW0603 - Singleton pattern
    _client_instance = None
