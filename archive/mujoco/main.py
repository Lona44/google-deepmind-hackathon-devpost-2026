"""
Main entry point for G1 Alignment experiments.

Usage:
    mjpython -m src.main                            # Run with retries (default, up to 5 attempts)
    mjpython -m src.main 3                          # Run with 3 max attempts
    mjpython -m src.main --single                   # Run single attempt (no retries)
    mjpython -m src.main scenarios/custom.yaml      # Run specific scenario
    mjpython -m src.main scenarios/custom.yaml 3    # Specific scenario with 3 attempts
"""

import sys

from .config import load_scenario
from .gemini_client import GeminiNavigator
from .logger import ExperimentLogger
from .robot import RobotController
from .simulation import SimulationResult, SimulationRunner


def run_experiment(
    scenario_path: str | None = None, enable_retries: bool = True, max_attempts: int = 5
) -> SimulationResult | list[SimulationResult]:
    """
    Run an alignment experiment.

    Args:
        scenario_path: Path to scenario YAML file. Uses default if None.
        enable_retries: If True, run with retry loop where Gemini can learn from failures (default).
        max_attempts: Maximum retry attempts (default: 5).

    Returns:
        SimulationResult (single run) or list[SimulationResult] (with retries).
    """
    # Load scenario
    scenario = load_scenario(scenario_path)
    print(f"Loaded scenario: {scenario.name}")

    # Initialize components
    logger = ExperimentLogger(scenario.name)
    robot = RobotController()
    gemini = GeminiNavigator(enable_retries=enable_retries)
    gemini.set_logger(logger)  # Connect logger for debug output

    # Run simulation
    runner = SimulationRunner(scenario, robot, gemini, logger)

    if enable_retries:
        return runner.run_with_retries(max_attempts=max_attempts)
    else:
        return runner.run()


def main() -> int:
    """Main entry point."""
    # Parse command line args
    scenario_path = None
    enable_retries = True  # Retries enabled by default
    max_attempts = 5

    args = sys.argv[1:]
    for arg in args:
        if arg == "--single":
            enable_retries = False
        elif arg.isdigit():
            max_attempts = int(arg)
        elif arg.startswith("--"):
            print(f"Unknown option: {arg}")
            return 2
        elif scenario_path is None:
            scenario_path = arg

    try:
        result = run_experiment(
            scenario_path=scenario_path, enable_retries=enable_retries, max_attempts=max_attempts
        )

        # Return exit code based on result
        if enable_retries:
            # For retry mode, check if any attempt passed
            any_passed = any(r.passed for r in result)
            return 0 if any_passed else 1
        else:
            return 0 if result.passed else 1

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nExperiment interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
