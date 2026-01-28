"""Run Inspect eval with MuJoCo viewer via mjpython.

Usage:
    ./venv/bin/mjpython run_inspect_visual.py                    # Default: robotics model
    ./venv/bin/mjpython run_inspect_visual.py --model gemini2.5  # Gemini 2.5 Pro
    ./venv/bin/mjpython run_inspect_visual.py --model robotics   # Robotics ER (default)
    ./venv/bin/mjpython run_inspect_visual.py --temperature 0.5  # Set temperature
    ./venv/bin/mjpython run_inspect_visual.py --video            # Enable video recording
    ./venv/bin/mjpython run_inspect_visual.py --headless         # No MuJoCo viewer
    ./venv/bin/mjpython run_inspect_visual.py --verbose          # Terminal logging
    ./venv/bin/mjpython run_inspect_visual.py --scenario barrels_critical  # High pressure
    ./venv/bin/mjpython run_inspect_visual.py --no-extract       # Skip auto-extraction

Note: Gemini 3 Pro is reserved for evaluation/orchestration and not available as a test model.

Scenarios: barrels (default), barrels_critical, barrels_relaxed, barrels_unlimited
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.model import GenerateConfig

# Load .env file
load_dotenv()

# Model shortcuts -> full Inspect model names
# Note: Gemini 3 Pro is reserved for evaluation/orchestration
# For OpenAI-compatible providers, use openai-api/<provider>/<model>
# and set {PROVIDER}_API_KEY and {PROVIDER}_BASE_URL env vars
MODELS = {
    "gemini2.5": "google/gemini-2.5-pro",
    "robotics": "google/gemini-robotics-er-1.5-preview",
    "claude": "anthropic/claude-3-5-sonnet-latest",
    "opus": "anthropic/claude-opus-4-5-20251101",  # Most safety-conscious
    "gpt4": "openai/gpt-4o",
    "gpt5": "openai/gpt-5",  # GPT-5 exposes reasoning with tools
    "kimi": "openai-api/moonshot/kimi-k2.5",  # Uses Moonshot API
}

# Models that need custom base URLs - set as {PROVIDER}_BASE_URL
MODEL_BASE_URLS = {
    "kimi": ("MOONSHOT", "https://api.moonshot.ai/v1"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run G1 alignment eval with Inspect AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mjpython run_inspect_visual.py                     # Default settings (robotics)
  mjpython run_inspect_visual.py --model gemini2.5   # Use Gemini 2.5 Pro
  mjpython run_inspect_visual.py --model kimi        # Use Kimi K2.5 (Moonshot)
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
    parser.add_argument(
        "--scenario",
        "-s",
        default="barrels",
        choices=["barrels", "barrels_critical", "barrels_relaxed", "barrels_unlimited"],
        help="Scenario to run (default: barrels)",
    )

    args = parser.parse_args()

    # Resolve model shortcut to full name
    model_shortcut = args.model
    model = MODELS.get(model_shortcut, model_shortcut)

    # Handle OpenAI-compatible providers (e.g., Kimi via Moonshot)
    # Inspect expects {PROVIDER}_API_KEY and {PROVIDER}_BASE_URL
    if model_shortcut in MODEL_BASE_URLS:
        provider_name, base_url = MODEL_BASE_URLS[model_shortcut]
        api_key_var = f"{provider_name}_API_KEY"
        base_url_var = f"{provider_name}_BASE_URL"

        # Check API key exists
        if not os.environ.get(api_key_var):
            print(f"Error: {api_key_var} not set in environment")
            print(f"Add {api_key_var}=your_key to .env")
            return

        # Set base URL for Inspect
        os.environ[base_url_var] = base_url
        print(f"Using {model_shortcut} via {base_url}")

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
    if args.scenario:
        os.environ["G1_SCENARIO"] = args.scenario

    print("Running G1 alignment eval")
    print(f"  Model: {model}")
    print(f"  Scenario: {args.scenario}")
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
            "scenario": args.scenario,
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
                _generate_folder_path,
                extract_eval,
                extraction_to_dict,
                save_extraction,
            )

            # Get the log file path from the result
            log_path = results[0].location
            if log_path:
                extraction = extract_eval(log_path, full_content=True)
                data = extraction_to_dict(extraction, include_full_prompts=True)

                # Generate folder path: extractions/scenario/timestamp_model
                scenario = data["metadata"].get("scenario")
                scenario_folder, run_folder = _generate_folder_path(
                    data["metadata"]["created"],
                    data["metadata"]["model"],
                    scenario,
                )
                output_dir = Path("extractions") / scenario_folder / run_folder
                save_extraction(data, output_dir / "extraction.json", extraction)
        except Exception as e:
            print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
