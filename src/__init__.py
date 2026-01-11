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
    LEGGED_GYM_ROOT,
    PROJECT_ROOT,
    ScenarioConfig,
    load_scenario,
)
from .gemini_client import GeminiNavigator
from .logger import ExperimentLogger
from .main import run_experiment
from .robot import RobotController
from .simulation import SimulationResult, SimulationRunner

__all__ = [
    "LEGGED_GYM_ROOT",
    "PROJECT_ROOT",
    "ExperimentLogger",
    "GeminiNavigator",
    "RobotController",
    "ScenarioConfig",
    "SimulationResult",
    "SimulationRunner",
    "load_scenario",
    "run_experiment",
]
