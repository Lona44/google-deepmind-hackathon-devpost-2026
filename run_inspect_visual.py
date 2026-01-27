"""Run Inspect eval with MuJoCo viewer via mjpython.

Usage:
    ./venv/bin/mjpython run_inspect_visual.py                    # Default: robotics model
    ./venv/bin/mjpython run_inspect_visual.py --model gemini2.5  # Gemini 2.5 Pro
    ./venv/bin/mjpython run_inspect_visual.py --model robotics   # Robotics ER (default)
    ./venv/bin/mjpython run_inspect_visual.py --temperature 0.5  # Set temperature
    ./venv/bin/mjpython run_inspect_visual.py --video            # Enable video recording
    ./venv/bin/mjpython run_inspect_visual.py --headless         # No MuJoCo viewer
    ./venv/bin/mjpython run_inspect_visual.py --verbose          # Terminal logging
    ./venv/bin/mjpython run_inspect_visual.py --no-extract       # Skip auto-extraction

Note: Gemini 3 Pro is reserved for evaluation/orchestration and not available as a test model.
"""

import argparse
import os
from pathlib import Path

from inspect_ai import eval
from inspect_ai.model import GenerateConfig

# Model shortcuts -> full Inspect model names
# Note: Gemini 3 Pro is reserved for evaluation/orchestration
MODELS = {
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
  mjpython run_inspect_visual.py                     # Default settings (robotics)
  mjpython run_inspect_visual.py --model gemini2.5   # Use Gemini 2.5 Pro
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
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help="Model temperature (0.0=deterministic, 1.0+=random). Default: model's default",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable API debug logging to logs/api_debug.log",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip automatic extraction after eval",
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
    if args.debug:
        os.environ["G1_DEBUG_API"] = "true"
        # Clear previous debug log
        debug_log = Path("logs/api_debug.log")
        if debug_log.exists():
            debug_log.unlink()
        print(f"Debug logging enabled: {debug_log}")

    print("Running G1 alignment eval")
    print(f"  Model: {model}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Temperature: {args.temperature or 'default'}")
    print(f"  Video: {args.video}")
    print(f"  Headless: {args.headless}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Debug: {args.debug}")
    print()

    # Build generation config
    gen_config = GenerateConfig(
        reasoning_effort=args.reasoning,
        reasoning_summary="detailed",  # Full reasoning traces
    )
    if args.temperature is not None:
        gen_config = GenerateConfig(
            reasoning_effort=args.reasoning,
            reasoning_summary="detailed",
            temperature=args.temperature,
        )

    # Run the native task
    results = eval(
        "inspect_eval/tasks.py@g1_native",
        model=model,
        config=gen_config,
        metadata={
            "reasoning_effort": args.reasoning,
            "temperature": args.temperature,
            "model": model,
        },
        limit=1,
    )

    print("\n=== Eval Complete ===")
    for result in results:
        print(f"Status: {result.status}")
        for sample in result.samples:
            print(f"Score: {sample.scores}")

    # Auto-extract results
    if not args.no_extract and results and results[0].status == "success":
        print("\n=== Extracting Results ===")
        try:
            # Lazy import to avoid slow startup (extraction is optional)
            from scripts.extract_eval_data import (  # noqa: PLC0415
                _generate_folder_name,
                extract_eval,
                extraction_to_dict,
                save_extraction,
            )

            # Get the log file path from the result
            log_path = results[0].location
            if log_path:
                extraction = extract_eval(log_path, full_content=True)
                data = extraction_to_dict(extraction, include_full_prompts=True)

                folder_name = _generate_folder_name(
                    data["metadata"]["created"], data["metadata"]["model"]
                )
                output_dir = Path("extractions") / folder_name
                save_extraction(data, output_dir / "extraction.json", extraction)
        except Exception as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
