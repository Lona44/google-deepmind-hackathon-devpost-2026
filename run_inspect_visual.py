"""Run Inspect eval with MuJoCo viewer via mjpython.

Usage:
    ./venv/bin/mjpython run_inspect_visual.py                    # Default: robotics model
    ./venv/bin/mjpython run_inspect_visual.py --model gemini3    # Gemini 3 Pro
    ./venv/bin/mjpython run_inspect_visual.py --model gemini2.5  # Gemini 2.5 Pro
    ./venv/bin/mjpython run_inspect_visual.py --model robotics   # Robotics ER (default)
    ./venv/bin/mjpython run_inspect_visual.py --video            # Enable video recording
    ./venv/bin/mjpython run_inspect_visual.py --headless         # No MuJoCo viewer
    ./venv/bin/mjpython run_inspect_visual.py --verbose          # Terminal logging
"""

import argparse
import os

from inspect_ai import eval
from inspect_ai.model import GenerateConfig

# Model shortcuts -> full Inspect model names
MODELS = {
    "gemini3": "google/gemini-3-pro-preview",
    "gemini2.5": "google/gemini-2.5-pro",
    "robotics": "google/gemini-robotics-er-1.5-preview",
    "claude": "anthropic/claude-3-5-sonnet-latest",
    "gpt4": "openai/gpt-4o",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run G1 alignment eval with Inspect AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mjpython run_inspect_visual.py                     # Default settings
  mjpython run_inspect_visual.py --model gemini3     # Use Gemini 3 Pro
  mjpython run_inspect_visual.py --video --headless  # Record video, no viewer
  mjpython run_inspect_visual.py --model claude      # Test Claude 3.5 Sonnet
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        default="robotics",
        help=f"Model to use. Shortcuts: {', '.join(MODELS.keys())}. "
        "Or full name like 'google/gemini-2.5-pro' (default: robotics)",
    )
    parser.add_argument(
        "--reasoning",
        "-r",
        default="high",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (default: high)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video recording",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without MuJoCo viewer (faster)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose terminal logging",
    )

    args = parser.parse_args()

    # Resolve model shortcut to full name
    model = MODELS.get(args.model, args.model)

    # Set environment variables based on flags
    if args.video:
        os.environ["G1_RECORD_VIDEO"] = "true"
    if args.headless:
        os.environ["G1_HEADLESS"] = "true"
    if args.verbose:
        os.environ["G1_VERBOSE"] = "true"

    print(f"Running G1 alignment eval")
    print(f"  Model: {model}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Video: {args.video}")
    print(f"  Headless: {args.headless}")
    print(f"  Verbose: {args.verbose}")
    print()

    # Run the native task
    results = eval(
        "inspect_eval/tasks.py@g1_native",
        model=model,
        config=GenerateConfig(reasoning_effort=args.reasoning),
        metadata={
            "reasoning_effort": args.reasoning,
            "model": model,
        },
        limit=1,
    )

    print("\n=== Eval Complete ===")
    for result in results:
        print(f"Status: {result.status}")
        for sample in result.samples:
            print(f"Score: {sample.scores}")


if __name__ == "__main__":
    main()
