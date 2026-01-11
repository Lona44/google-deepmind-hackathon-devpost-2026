"""
G1 Alignment Experiment Package

Modules:
    config - Configuration and constants
    logger - Experiment logging
    robot - Robot controller
    gemini_client - Gemini API client
    simulation - Simulation runner
    main - Entry point
"""

from .config import (
    PROJECT_ROOT,
    LEGGED_GYM_ROOT,
    ScenarioConfig,
    load_scenario,
)
from .logger import ExperimentLogger
from .robot import RobotController
from .gemini_client import GeminiNavigator
from .simulation import SimulationRunner, SimulationResult
from .main import run_experiment

__all__ = [
    "PROJECT_ROOT",
    "LEGGED_GYM_ROOT",
    "ScenarioConfig",
    "load_scenario",
    "ExperimentLogger",
    "RobotController",
    "GeminiNavigator",
    "SimulationRunner",
    "SimulationResult",
    "run_experiment",
]
