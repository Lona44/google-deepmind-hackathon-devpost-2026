"""
GCS Client - Handles video/extraction serving from Google Cloud Storage.

Supports:
- Video uploads for Gemini vision analysis
- Extraction data serving (when running on Cloud Run without local files)
- Signed URL generation for video streaming
"""

import datetime
import json
import os
from pathlib import Path

from google.cloud import storage


# Prefix in GCS where extractions are stored
EXTRACTIONS_PREFIX = "extractions"

# Known scenarios
SCENARIOS = ["barrels_corrupt", "barrels_lo", "barrels_hi"]


class GCSClient:
    """
    Client for Google Cloud Storage operations.

    Handles video uploads, extraction data access, and signed URL generation.
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

    # --- Extraction data access ---

    def find_extraction_video(self, run_id: str) -> str | None:
        """
        Find a video for a run ID in GCS extractions.

        Searches: extractions/{scenario}/{run_id}/media/full_run.mp4

        Returns:
            GCS blob path if found, None otherwise
        """
        for scenario in SCENARIOS:
            blob_path = f"{EXTRACTIONS_PREFIX}/{scenario}/{run_id}/media/full_run.mp4"
            blob = self._bucket.blob(blob_path)
            if blob.exists():
                return blob_path
        return None

    def get_video_signed_url(self, blob_path: str, expiration_minutes: int = 60) -> str:
        """
        Generate a signed URL for streaming a video from GCS.

        On Cloud Run (compute engine credentials), uses the IAM signBlob API
        by passing service_account_email and access_token.

        Args:
            blob_path: Path to the blob in the bucket
            expiration_minutes: URL validity in minutes

        Returns:
            Signed URL string
        """
        import google.auth
        from google.auth.transport import requests as auth_requests

        blob = self._bucket.blob(blob_path)

        credentials, _ = google.auth.default()
        if hasattr(credentials, "service_account_email"):
            # Refresh to ensure we have a valid token
            credentials.refresh(auth_requests.Request())
            return blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=expiration_minutes),
                method="GET",
                service_account_email=credentials.service_account_email,
                access_token=credentials.token,
            )

        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
        )

    def get_extraction_video_uri(self, run_id: str) -> str | None:
        """
        Get the gs:// URI for a run's video (for Gemini vision).

        Returns:
            GCS URI if found, None otherwise
        """
        blob_path = self.find_extraction_video(run_id)
        if blob_path:
            return f"gs://{self._bucket_name}/{blob_path}"
        return None

    def read_extraction_json(self, run_id: str, filename: str) -> dict | None:
        """
        Read a JSON file from a run's extraction directory.

        Args:
            run_id: The experiment run ID
            filename: JSON filename (e.g., 'extraction.json', 'judge_analysis.json')

        Returns:
            Parsed JSON dict if found, None otherwise
        """
        for scenario in SCENARIOS:
            blob_path = f"{EXTRACTIONS_PREFIX}/{scenario}/{run_id}/{filename}"
            blob = self._bucket.blob(blob_path)
            if blob.exists():
                content = blob.download_as_text()
                return json.loads(content)
        return None

    def list_extraction_runs(self) -> list[dict]:
        """
        List all extraction runs in GCS.

        Returns:
            List of {run_id, scenario} dicts
        """
        runs = []
        seen = set()

        for scenario in SCENARIOS:
            prefix = f"{EXTRACTIONS_PREFIX}/{scenario}/"
            blobs = self._client.list_blobs(
                self._bucket_name, prefix=prefix, delimiter="/"
            )
            # Iterate through the prefixes (subdirectories)
            # list_blobs with delimiter returns "directory" prefixes
            for page in blobs.pages:
                for blob_prefix in page.prefixes:
                    # prefix looks like: extractions/barrels_corrupt/2026-02-06T04-28_kimi-k2.5/
                    run_id = blob_prefix.rstrip("/").split("/")[-1]
                    if run_id not in seen:
                        seen.add(run_id)
                        runs.append({"run_id": run_id, "scenario": scenario})

        return runs


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
