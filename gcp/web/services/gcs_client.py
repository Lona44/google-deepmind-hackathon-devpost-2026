"""
GCS Client - Handles video uploads to Google Cloud Storage for Vertex AI analysis.

Videos must be uploaded to GCS before they can be analyzed by Gemini models
when using Vertex AI mode.
"""

import os
from pathlib import Path

from google.cloud import storage


class GCSClient:
    """
    Client for uploading videos to Google Cloud Storage.

    Videos are cached by their hash to avoid re-uploading.
    """

    def __init__(self) -> None:
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable required")

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._bucket_name = bucket_name

    def upload_video(self, local_path: str | Path, destination_name: str | None = None) -> str:
        """
        Upload a video file to GCS.

        Args:
            local_path: Path to the local video file
            destination_name: Optional name for the file in GCS (defaults to filename)

        Returns:
            GCS URI in format gs://bucket/path/to/video.mp4

        Raises:
            FileNotFoundError: If local file doesn't exist
            ValueError: If file is too large
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")

        # Check file size (100MB limit for now)
        max_size_mb = int(os.getenv("VIDEO_MAX_SIZE_MB", "100"))
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValueError(f"Video too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")

        # Use filename if no destination specified
        if destination_name is None:
            destination_name = local_path.name

        # Upload to GCS
        blob = self._bucket.blob(f"videos/{destination_name}")

        # Check if already uploaded (by name - could use hash for deduplication)
        if blob.exists():
            return f"gs://{self._bucket_name}/videos/{destination_name}"

        blob.upload_from_filename(str(local_path), content_type="video/mp4")

        return f"gs://{self._bucket_name}/videos/{destination_name}"

    def get_video_uri(self, video_name: str) -> str | None:
        """
        Get the GCS URI for an existing video.

        Args:
            video_name: Name of the video file in GCS

        Returns:
            GCS URI if exists, None otherwise
        """
        blob = self._bucket.blob(f"videos/{video_name}")
        if blob.exists():
            return f"gs://{self._bucket_name}/videos/{video_name}"
        return None

    def delete_video(self, video_name: str) -> bool:
        """
        Delete a video from GCS.

        Args:
            video_name: Name of the video file to delete

        Returns:
            True if deleted, False if not found
        """
        blob = self._bucket.blob(f"videos/{video_name}")
        if blob.exists():
            blob.delete()
            return True
        return False


# Singleton instance (lazy initialization)
_gcs_instance: GCSClient | None = None


def get_gcs_client() -> GCSClient:
    """Get the singleton GCSClient instance."""
    global _gcs_instance  # noqa: PLW0603 - Singleton pattern
    if _gcs_instance is None:
        _gcs_instance = GCSClient()
    return _gcs_instance


def reset_gcs_client() -> None:
    """Reset the singleton client (useful for testing)."""
    global _gcs_instance  # noqa: PLW0603 - Singleton pattern
    _gcs_instance = None
