"""
Centralized path handling for G1 Alignment experiments.

This module defines all paths used throughout the project, with support
for environment variable overrides to enable cloud deployment (Modal).

Environment Variables:
    G1_PROJECT_ROOT: Override project root (default: auto-detected)
    G1_LOGS_DIR: Override logs directory (default: {project_root}/logs)
    G1_DATA_DIR: Override data/extractions directory (default: {project_root}/extractions)
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get project root, supports local and Modal /app structure.

    Returns the project root directory, which can be overridden via
    the G1_PROJECT_ROOT environment variable for cloud deployment.
    """
    if custom := os.environ.get("G1_PROJECT_ROOT"):
        return Path(custom)
    # Default: parent of parent of this file (src/paths.py -> project root)
    return Path(__file__).resolve().parent.parent


def get_logs_dir() -> Path:
    """Get logs directory (respects G1_LOGS_DIR env var).

    Creates the directory if it doesn't exist.
    """
    if custom := os.environ.get("G1_LOGS_DIR"):  # noqa: SIM108
        d = Path(custom)
    else:
        d = get_project_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    """Get data/extractions directory (respects G1_DATA_DIR env var).

    Creates the directory if it doesn't exist.
    Used for storing experiment extractions and results.
    """
    if custom := os.environ.get("G1_DATA_DIR"):
        d = Path(custom)
    else:
        d = get_project_root() / "extractions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_experiments_dir() -> Path:
    """Get experiments directory for legacy experiment data.

    Creates the directory if it doesn't exist.
    """
    d = get_project_root() / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Computed Path Constants (use functions above for dynamic resolution)
# =============================================================================

PROJECT_ROOT = get_project_root()
LEGGED_GYM_ROOT = PROJECT_ROOT / "unitree_rl_gym"
SCENARIOS_DIR = PROJECT_ROOT / "src" / "scenarios"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Robot config paths
ROBOT_CONFIG_PATH = LEGGED_GYM_ROOT / "deploy/deploy_mujoco/configs/g1.yaml"
SCENE_XML_PATH = LEGGED_GYM_ROOT / "resources/robots/g1_description/scene_alignment.xml"
SCENE_DIR = LEGGED_GYM_ROOT / "resources/robots/g1_description"

# API environment file
ENV_PATH = PROJECT_ROOT / ".env"

# Available scenes (use --scene flag to switch)
SCENES = {
    "barrels": "scene_barrels.xml",  # Full facility with walls, signs, background
    "minimal": "scene_barrels_minimal.xml",  # Just robot, barrels, and charger
    "alignment": "scene_alignment.xml",  # Original alignment scene
}


def get_scene_path(scene_name: str | None = None) -> Path:
    """Get path to scene XML file.

    Args:
        scene_name: Scene name (e.g., 'minimal', 'barrels') or filename (e.g., 'scene_barrels.xml').
                   Uses default if None.

    Returns:
        Full path to scene XML file.
    """
    if scene_name is None:
        return SCENE_XML_PATH
    # Check if it's a short name
    if scene_name in SCENES:
        return SCENE_DIR / SCENES[scene_name]
    # Otherwise treat as filename
    return SCENE_DIR / scene_name
