"""
Interactive G1 Alignment Demo with Vision & Sensors

This is a thin wrapper for backwards compatibility.
The actual implementation is in the modular components:
    - config.py: Configuration and scenarios
    - logger.py: Experiment logging
    - robot.py: Robot controller
    - gemini_client.py: Gemini API client
    - simulation.py: Simulation runner
    - main.py: Entry point

Usage:
    python src/interactive_alignment_demo.py
    # or
    python -m src.main
"""

from .main import run_experiment


def run_interactive_demo():
    """Run the interactive alignment demo with vision and sensors."""
    return run_experiment()


if __name__ == "__main__":
    run_interactive_demo()
