"""
Main entry point for G1 Alignment experiments.

Usage:
    mjpython -m src.main                            # Run with retries (default, up to 5 attempts)
    mjpython -m src.main 3                          # Run with 3 max attempts
    mjpython -m src.main --single                   # Run single attempt (no retries)
    mjpython -m src.main --scene scene_barrels.xml  # Use alternate scene
    mjpython -m src.main --model gemini2.5          # Use gemini-2.5-pro instead of gemini-3
    mjpython -m src.main --model robotics           # Use gemini-robotics-er-1.5 (spatial reasoning)
    mjpython -m src.main --model flash              # Use gemini-2.0-flash (fast, no thinking)
    mjpython -m src.main scenarios/custom.yaml      # Run specific scenario
    mjpython -m src.main scenarios/custom.yaml 3    # Specific scenario with 3 attempts

Available models (via --model):
    gemini3   - gemini-3-pro-preview (default, best thinking, daily quota)
    gemini2.5 - gemini-2.5-pro (good thinking, separate quota)
    robotics  - gemini-robotics-er-1.5-preview (spatial reasoning, trajectory planning)
    flash     - gemini-2.0-flash (fast, no thinking mode)
"""

import sys

from .config import GEMINI_MODEL, GEMINI_MODELS, get_scene_path, load_scenario
from .gemini_client import GeminiNavigator
from .logger import ExperimentLogger
from .robot import RobotController
from .simulation import SimulationResult, SimulationRunner


def run_experiment(
    scenario_path: str | None = None,
    enable_retries: bool = True,
    max_attempts: int = 5,
    scene_name: str | None = None,
    model: str = GEMINI_MODEL,
) -> SimulationResult | list[SimulationResult]:
    """
    Run an alignment experiment.

    Args:
        scenario_path: Path to scenario YAML file. Uses default if None.
        enable_retries: If True, run with retry loop where Gemini can learn from failures (default).
        max_attempts: Maximum retry attempts (default: 5).
        scene_name: Scene XML filename (e.g., 'scene_barrels.xml'). Uses default if None.
        model: Gemini model to use (default: GEMINI_MODEL from config).

    Returns:
        SimulationResult (single run) or list[SimulationResult] (with retries).
    """
    # Load scenario
    scenario = load_scenario(scenario_path)
    print(f"Loaded scenario: {scenario.name}")

    # Get scene path
    scene_path = get_scene_path(scene_name)
    print(f"Using scene: {scene_path.name}")
    print(f"Using model: {model}")

    # Initialize components
    logger = ExperimentLogger(scenario.name, model=model)
    robot = RobotController()
    gemini = GeminiNavigator(model=model, enable_retries=enable_retries)
    gemini.set_logger(logger)  # Connect logger for debug output

    # Run simulation
    runner = SimulationRunner(scenario, robot, gemini, logger, scene_path=scene_path)

    if enable_retries:
        return runner.run_with_retries(max_attempts=max_attempts)
    else:
        return runner.run()


def _parse_args(args: list[str]) -> dict | int:
    """Parse command line arguments. Returns dict of options or error code."""
    opts = {
        "scenario_path": None,
        "enable_retries": True,
        "max_attempts": 5,
        "scene_name": None,
        "model": GEMINI_MODEL,
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--single":
            opts["enable_retries"] = False
        elif arg == "--scene":
            if i + 1 < len(args):
                opts["scene_name"] = args[i + 1]
                i += 1
            else:
                print("Error: --scene requires a filename")
                return 2
        elif arg == "--model":
            if i + 1 < len(args):
                model_key = args[i + 1]
                if model_key in GEMINI_MODELS:
                    opts["model"] = GEMINI_MODELS[model_key]
                else:
                    print(f"Error: Unknown model '{model_key}'")
                    print(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
                    return 2
                i += 1
            else:
                print("Error: --model requires a model name")
                print(f"Available models: {', '.join(GEMINI_MODELS.keys())}")
                return 2
        elif arg.isdigit():
            opts["max_attempts"] = int(arg)
        elif arg.startswith("--"):
            print(f"Unknown option: {arg}")
            return 2
        elif opts["scenario_path"] is None:
            opts["scenario_path"] = arg
        i += 1

    return opts


def main() -> int:
    """Main entry point."""
    opts = _parse_args(sys.argv[1:])
    if isinstance(opts, int):
        return opts

    try:
        result = run_experiment(
            scenario_path=opts["scenario_path"],
            enable_retries=opts["enable_retries"],
            max_attempts=opts["max_attempts"],
            scene_name=opts["scene_name"],
            model=opts["model"],
        )

        # Return exit code based on result
        if isinstance(result, list):
            # Retry mode returns list - check if any attempt passed
            any_passed = any(r.passed for r in result)
            return 0 if any_passed else 1
        else:
            # Single run mode returns single result
            return 0 if result.passed else 1

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nExperiment interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
