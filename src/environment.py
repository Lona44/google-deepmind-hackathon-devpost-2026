"""
Centralized environment configuration for G1 Alignment experiments.

This module provides a single source of truth for all environment variables
used throughout the project. Configuration is loaded once and cached.

Environment Variables:
    G1_VERBOSE: Enable progress logging (default: false)
    G1_HEADLESS: Run without viewer window (default: false)
    G1_RECORD_VIDEO: Record experiment videos (default: false)
    G1_VIDEO_WIDTH: Video width in pixels (default: 1280)
    G1_VIDEO_HEIGHT: Video height in pixels (default: 720)
    G1_SCENARIO: Scenario to run (default: barrels_corrupt)
    G1_DEBUG_API: Enable API debug logging (default: false)
    G1_MODEL_NAME: Model name for video overlay (default: empty)
    G1_JUDGE_MODEL: Model for LLM judge scoring (default: google/gemini-3-pro-preview)
"""

import os
from dataclasses import dataclass
from functools import lru_cache


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse boolean from environment variable string."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _parse_int(value: str | None, default: int) -> int:
    """Parse integer from environment variable string."""
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class G1Config:
    """Centralized environment configuration.

    This is a frozen dataclass - values are set at load time and cannot
    be modified. This ensures consistency throughout a run.
    """

    # Logging
    verbose: bool
    debug_api: bool

    # Simulation
    headless: bool
    record_video: bool
    video_width: int
    video_height: int
    scenario: str

    # Model identification
    model_name: str
    judge_model: str

    @classmethod
    def load(cls) -> "G1Config":
        """Load configuration from environment variables.

        Note: Does NOT call load_dotenv() here to avoid import-time side effects.
        The caller should load .env if needed before calling this.
        """
        return cls(
            verbose=_parse_bool(os.environ.get("G1_VERBOSE")),
            debug_api=_parse_bool(os.environ.get("G1_DEBUG_API")),
            headless=_parse_bool(os.environ.get("G1_HEADLESS")),
            record_video=_parse_bool(os.environ.get("G1_RECORD_VIDEO")),
            video_width=_parse_int(os.environ.get("G1_VIDEO_WIDTH"), 1280),
            video_height=_parse_int(os.environ.get("G1_VIDEO_HEIGHT"), 720),
            scenario=os.environ.get("G1_SCENARIO", "barrels_corrupt"),
            model_name=os.environ.get("G1_MODEL_NAME", ""),
            judge_model=os.environ.get("G1_JUDGE_MODEL", "google/gemini-3-pro-preview"),
        )

    @classmethod
    def reload(cls) -> "G1Config":
        """Force reload configuration (clears cache).

        Use this after modifying environment variables programmatically.
        """
        get_config.cache_clear()
        return get_config()


@lru_cache(maxsize=1)
def get_config() -> G1Config:
    """Get the singleton configuration instance.

    Configuration is loaded lazily on first access and cached.
    Use G1Config.reload() to force a refresh after env changes.
    """
    return G1Config.load()


# Convenience functions for common checks
def is_verbose() -> bool:
    """Check if verbose logging is enabled."""
    return get_config().verbose


def is_headless() -> bool:
    """Check if headless mode is enabled."""
    return get_config().headless


def is_recording() -> bool:
    """Check if video recording is enabled."""
    return get_config().record_video
