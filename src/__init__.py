"""
G1 Alignment Experiment Package

Modules:
    config - Configuration and constants
    robot - Robot controller (requires mujoco)
    simulation_state - Stateful simulation interface for Inspect AI

Usage:
    # Config utilities (no external dependencies)
    from src.config import ScenarioConfig, load_scenario

    # Inspect AI integration
    # See inspect_eval/tasks.py for task definitions
"""

# Only export config - no external dependencies
from .config import (
    LEGGED_GYM_ROOT,
    PROJECT_ROOT,
    ForbiddenZone,
    ScenarioConfig,
    load_scenario,
)

__all__ = [
    "LEGGED_GYM_ROOT",
    "PROJECT_ROOT",
    "ForbiddenZone",
    "ScenarioConfig",
    "load_scenario",
]
