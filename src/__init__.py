"""
G1 Alignment Experiment Package

Modules:
    paths - Centralized path handling (no external dependencies)
    environment - Centralized environment configuration
    config - Configuration and constants
    robot - Robot controller (requires mujoco)
    simulation_state - Stateful simulation interface for Inspect AI

Usage:
    # Path utilities (no external dependencies)
    from src.paths import PROJECT_ROOT, get_logs_dir, get_data_dir

    # Config utilities (no external dependencies)
    from src.config import ScenarioConfig, load_scenario

    # Inspect AI integration
    # See inspect_eval/tasks.py for task definitions
"""

# Export paths - no external dependencies
# Export config - no external dependencies
from .config import (
    ForbiddenZone,
    ScenarioConfig,
    load_scenario,
)
from .paths import (
    LEGGED_GYM_ROOT,
    PROJECT_ROOT,
    SCENARIOS_DIR,
    get_data_dir,
    get_logs_dir,
    get_project_root,
)

__all__ = [
    "LEGGED_GYM_ROOT",
    "PROJECT_ROOT",
    "SCENARIOS_DIR",
    "ForbiddenZone",
    "ScenarioConfig",
    "get_data_dir",
    "get_logs_dir",
    "get_project_root",
    "load_scenario",
]
