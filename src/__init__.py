"""
G1 Alignment Experiment Package

Modules:
    config - Configuration and constants
    logger - Experiment logging
    robot - Robot controller (requires mujoco)
    gemini_client - Gemini API client (requires GEMINI_API_KEY)
    simulation - Simulation runner (requires mujoco + API key)
    main - Entry point

Usage:
    # Config utilities (no external dependencies)
    from src.config import ScenarioConfig, load_scenario

    # Full stack (requires mujoco + API key)
    from src.gemini_client import GeminiNavigator
    from src.simulation import SimulationRunner
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
