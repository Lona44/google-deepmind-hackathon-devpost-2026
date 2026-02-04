"""Run Inspect eval with MuJoCo viewer via mjpython.

Usage:
    ./venv/bin/mjpython run_inspect_visual.py                    # Default: robotics model
    ./venv/bin/mjpython run_inspect_visual.py --model gemini2.5  # Gemini 2.5 Pro
    ./venv/bin/mjpython run_inspect_visual.py --model robotics   # Robotics ER (default)
    ./venv/bin/mjpython run_inspect_visual.py --temperature 0.5  # Set temperature
    ./venv/bin/mjpython run_inspect_visual.py --video            # Enable video recording
    ./venv/bin/mjpython run_inspect_visual.py --viewer           # Show MuJoCo viewer (default: headless)
    ./venv/bin/mjpython run_inspect_visual.py --verbose          # Terminal logging
    ./venv/bin/mjpython run_inspect_visual.py --scenario barrels_hi  # High pressure
    ./venv/bin/mjpython run_inspect_visual.py --no-extract       # Skip auto-extraction
    ./venv/bin/mjpython run_inspect_visual.py --skip-preflight   # Skip API connectivity check
    ./venv/bin/mjpython run_inspect_visual.py --no-judge-analysis  # Skip saving judge_analysis.json
    ./venv/bin/mjpython run_inspect_visual.py --no-moments         # Skip alignment moment generation

Scoring: Uses Gemini 3 Pro as judge to evaluate safety behavior (always enabled).
         Produces safety (1-5) and honesty (1-5) scores. Composite score weighted by
         honesty confidence (if honesty was tested, average both; otherwise just safety).

Scenarios: barrels_lo (default), barrels_mi, barrels_mh, barrels_hi
"""

import argparse
import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval
from inspect_ai.model import GenerateConfig


def _enrich_trajectory_and_copy(
    extraction_dir: Path,
    generate_moments: bool = True,
    judge_model: str = "google/gemini-3-pro-preview",
) -> None:
    """Enrich trajectory with extraction data and copy to viewer assets.

    This is called after all extraction files are saved (extraction.json,
    judge_analysis.json, trajectory.json) to merge them and copy to the
    frontend viewer.

    Args:
        extraction_dir: Directory containing trajectory.json, extraction.json, judge_analysis.json
        generate_moments: If True, generate alignment moments via judge model (adds ~5-10s)
        judge_model: Model to use for alignment moment generation
    """
    trajectory_path = extraction_dir / "trajectory.json"
    if not trajectory_path.exists():
        print("  ⚠️  No trajectory.json to enrich")
        return

    try:
        # Import enrichment functions
        from scripts.enrich_trajectory import (  # noqa: PLC0415
            add_alignment_moments_to_trajectory,
            enrich_trajectory,
            extract_navigation_sequence,
            generate_alignment_moments,
            generate_timeline_events,
            load_json,
        )

        # Load all the files
        trajectory = load_json(trajectory_path)
        extraction = load_json(extraction_dir / "extraction.json")
        judge_analysis = load_json(extraction_dir / "judge_analysis.json")

        if not trajectory:
            print("  ⚠️  Failed to load trajectory.json")
            return

        print("\n=== Enriching Trajectory ===")

        # Enrich trajectory with existing data
        enriched = enrich_trajectory(trajectory, extraction, judge_analysis)

        # Generate alignment moments if requested
        if generate_moments and extraction:
            print("  🧠 Generating alignment moments (parallel)...")
            scenario = extraction.get("metadata", {}).get("scenario", "unknown")

            # Extract navigation sequence (same as enrich_trajectory.py does)
            nav_sequence = extract_navigation_sequence(extraction)

            # Build decision context from nav sequence (include 'tool' for filtering)
            decisions = []
            for nav in nav_sequence:
                decisions.append({
                    "reasoning": nav.get("reasoning"),
                    "position": nav.get("position"),
                    "battery": nav.get("battery"),
                    "attempt": nav.get("attempt"),
                    "tool": nav.get("tool"),
                    "distance_to_goal": nav.get("distance_to_goal"),
                })

            if decisions:
                # Run async moment generation
                moments = asyncio.run(
                    generate_alignment_moments(decisions, scenario, judge_model)
                )

                # Add moments to trajectory frames
                added = add_alignment_moments_to_trajectory(enriched, moments, nav_sequence)

                # Regenerate timeline events with moments
                frames = enriched.get("frames", [])
                timeline_events = generate_timeline_events(frames, nav_sequence, moments)
                if timeline_events:
                    enriched["timeline_events"] = timeline_events
                    print(f"  ✨ Generated {len([m for m in moments if m])} alignment moments ({added} added to frames)")
            else:
                print("  ⚠️  No decisions found for moment generation")

        # Save enriched trajectory back
        with open(trajectory_path, "w") as f:
            json.dump(enriched, f, indent=2)

        # Copy to viewer assets
        viewer_assets = Path(__file__).parent / "gcp" / "frontend" / "assets"
        if viewer_assets.exists():
            name = extraction_dir.name.replace(":", "-").replace(" ", "_")
            viewer_path = viewer_assets / f"trajectory_{name}.json"
            with open(viewer_path, "w") as f:
                json.dump(enriched, f, indent=2)
            print(f"  📺 Copied to viewer: {viewer_path.name}")
            print(f"  View at: http://localhost:5500/?trajectory=assets/{viewer_path.name}")

            # Regenerate manifest for extraction selector
            try:
                from scripts.generate_manifest import generate_manifest  # noqa: PLC0415

                manifest_path = viewer_assets / "extractions_index.json"
                manifest = generate_manifest()
                with manifest_path.open("w") as f:
                    json.dump(manifest, f, indent=2)

                total_runs = sum(len(s["runs"]) for s in manifest["scenarios"].values())
                print(f"  📋 Updated manifest: {total_runs} runs across {len(manifest['scenarios'])} scenarios")
            except Exception as e:
                print(f"  ⚠️  Manifest generation failed: {e}")

    except Exception as e:
        print(f"  ⚠️  Trajectory enrichment failed: {e}")


def preflight_check(model_shortcut: str) -> bool:
    """Test API connectivity before starting eval.

    Args:
        model_shortcut: The model shortcut (e.g., 'kimi', 'gemini2.5', 'claude')

    Returns:
        True if API is reachable and responding, False otherwise.
    """
    print("Running API preflight check...")

    # Define API endpoints and test payloads for each provider
    if model_shortcut == "kimi":
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            print("  ✗ MOONSHOT_API_KEY not set")
            return False

        url = "https://api.moonshot.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "kimi-k2.5",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }

    elif model_shortcut in ("gemini2.5", "robotics"):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  ✗ GOOGLE_API_KEY or GEMINI_API_KEY not set")
            return False

        # Use Gemini API directly for preflight
        model_name = "gemini-2.5-flash"  # Use flash for quick test
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Say OK"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }

    elif model_shortcut == "claude" or model_shortcut == "opus":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  ✗ ANTHROPIC_API_KEY not set")
            return False

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": "claude-3-haiku-20240307",  # Use haiku for quick test
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say OK"}],
        }

    elif model_shortcut in ("gpt4", "gpt5"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("  ✗ OPENAI_API_KEY not set")
            return False

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "gpt-4o-mini",  # Use mini for quick test
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }

    else:
        print(f"  ⚠ No preflight check configured for '{model_shortcut}', skipping")
        return True

    # Make the request
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

            # Check for success based on provider
            if model_shortcut == "kimi":
                if "choices" in result:
                    print("  ✓ Kimi/Moonshot API is responding")
                    return True
            elif model_shortcut in ("gemini2.5", "robotics"):
                if "candidates" in result:
                    print("  ✓ Google/Gemini API is responding")
                    return True
            elif model_shortcut in ("claude", "opus"):
                if "content" in result:
                    print("  ✓ Anthropic API is responding")
                    return True
            elif model_shortcut in ("gpt4", "gpt5"):
                if "choices" in result:
                    print("  ✓ OpenAI API is responding")
                    return True

            # Unexpected response format
            print(f"  ✗ Unexpected response: {result}")
            return False

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get("error", {}).get("message", error_body)
        except json.JSONDecodeError:
            error_msg = error_body
        print(f"  ✗ API error ({e.code}): {error_msg}")
        return False

    except urllib.error.URLError as e:
        print(f"  ✗ Connection error: {e.reason}")
        return False

    except TimeoutError:
        print("  ✗ Request timed out (30s)")
        return False

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

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
    "kimi": "kimi/kimi-k2.5",  # Custom provider with reasoning_content support
}

# Models that need custom base URLs - set as {PROVIDER}_BASE_URL
# Note: kimi uses custom provider (inspect_eval.kimi_provider) with built-in URL
MODEL_BASE_URLS: dict[str, tuple[str, str]] = {}


def main():
    parser = argparse.ArgumentParser(
        description="Run G1 alignment eval with Inspect AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mjpython run_inspect_visual.py                     # Default settings (robotics)
  mjpython run_inspect_visual.py --model gemini2.5   # Use Gemini 2.5 Pro
  mjpython run_inspect_visual.py --model kimi        # Use Kimi K2.5 (Moonshot)
  mjpython run_inspect_visual.py --video             # Record video (headless by default)
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
        "--no-trajectory",
        action="store_true",
        dest="no_trajectory",
        help="Disable trajectory recording (enabled by default)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show MuJoCo viewer window (default: headless)",
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
        default="barrels_lo",
        choices=["barrels_lo", "barrels_mi", "barrels_mh", "barrels_hi"],
        help="Scenario to run (default: barrels_lo)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip API connectivity check before running eval",
    )
    parser.add_argument(
        "--judge",
        default="google/gemini-3-pro-preview",
        help="Judge model for safety scoring (default: google/gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--no-judge-analysis",
        action="store_true",
        help="Skip saving detailed judge_analysis.json after eval",
    )
    parser.add_argument(
        "--no-moments",
        action="store_true",
        help="Skip generating alignment moments (saves ~5-10s of API calls)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "trace", "http", "info", "warning", "error", "critical"],
        default="http",
        help="Inspect log level for console output. 'http' shows API errors (default: http)",
    )
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Raise errors instead of logging them (useful for debugging API issues)",
    )
    parser.add_argument(
        "--max-api-retries",
        type=int,
        default=5,
        help="Max API-level retries for transient errors like 429 (default: 5)",
    )
    parser.add_argument(
        "--retry-on-error",
        type=int,
        default=1,
        help="Sample-level retries; errors recorded in error_retries field (default: 1)",
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
    if not args.no_trajectory:
        os.environ["G1_RECORD_TRAJECTORY"] = "true"
    if not args.viewer:
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

    # Set judge model for LLM scorer (always enabled)
    os.environ["G1_JUDGE_MODEL"] = args.judge

    # Set model name for video overlay (use shortcut for readability)
    os.environ["G1_MODEL_NAME"] = model_shortcut

    # Run preflight check unless skipped
    if not args.skip_preflight:
        if not preflight_check(model_shortcut):
            print("\nPreflight check failed. Use --skip-preflight to bypass.")
            return
        print()

    print("Running G1 alignment eval")
    print(f"  Model: {model}")
    print(f"  Scenario: {args.scenario}")
    print(f"  Reasoning: {args.reasoning}")
    print(f"  Temperature: {args.temperature or 'default'}")
    print(f"  Judge: {args.judge}")
    print(f"  Timeout: 600s (robust_generate), {180}s per attempt (Inspect)")
    print(f"  Video: {args.video}")
    print(f"  Trajectory: {not args.no_trajectory}")
    print(f"  Viewer: {args.viewer}")
    print(f"  Verbose: {args.verbose}")
    print(f"  Debug: {args.debug}")
    print(f"  Log level: {args.log_level}")
    if args.log_level == "http":
        print("    (HTTP logging enabled - API errors will be visible)")
    print(f"  Debug errors: {args.debug_errors}")
    print(f"  Max API retries: {args.max_api_retries}")
    print(f"  Retry on error: {args.retry_on_error}")
    print(f"  Alignment moments: {not args.no_moments}")
    print()

    # Build generation config
    gen_config_kwargs = {
        "reasoning_effort": args.reasoning,
        "reasoning_summary": "detailed",  # Full reasoning traces
        # API-level retry settings (Inspect handles transient errors)
        "max_retries": args.max_api_retries,  # Default: 5 (more patience for 429s)
        "attempt_timeout": 180,  # 3 min per attempt (fail fast if hung)
        # No "timeout" - let robust_generate control total time via asyncio.wait_for
    }
    if args.temperature is not None:
        gen_config_kwargs["temperature"] = args.temperature

    gen_config = GenerateConfig(**gen_config_kwargs)

    # Use custom Kimi provider that properly handles reasoning_content
    # Standard OpenAI-compatible provider doesn't echo reasoning_content back,
    # causing "reasoning_content is missing" errors in multi-turn tool calls
    if model_shortcut == "kimi":
        # Import to register the provider with Inspect's model registry
        import inspect_eval.kimi_provider  # noqa: F401, PLC0415

        model_instance = "kimi/kimi-k2.5"  # Use registered provider name
        print("  Note: Using custom Kimi provider for reasoning_content support")
    else:
        model_instance = model

    # Run the native task
    results = eval(
        "inspect_eval/tasks.py@g1_native",
        model=model_instance,
        config=gen_config,
        metadata={
            "scenario": args.scenario,
            "reasoning_effort": args.reasoning,
            "temperature": args.temperature,
            "model": model,
        },
        limit=1,
        log_level=args.log_level,
        debug_errors=args.debug_errors,
        # Error visibility at sample level
        retry_on_error=args.retry_on_error,  # Record errors in error_retries field
        fail_on_error=False,  # Don't abort eval on errors (let debrief complete)
    )

    print("\n=== Eval Complete ===")
    for result in results:
        print(f"Status: {result.status}")
        for sample in result.samples:
            print(f"Score: {sample.scores}")

    # Auto-extract results
    if not args.no_extract and results and results[0].status == "success":
        print("\n=== Extracting Results ===")
        output_dir = None
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

        # Run judge analysis to save detailed judge_analysis.json
        # Display judge results from live scorer (already in extraction.json)
        from inspect_eval import kimi_provider  # noqa: PLC0415

        if kimi_provider.RATE_LIMIT_HIT:
            print("\n=== Skipping Judge Analysis (rate limit hit) ===")
        elif not args.no_judge_analysis and output_dir and (output_dir / "extraction.json").exists():
            print("\n=== Judge Analysis Results ===")
            try:
                import json
                with open(output_dir / "extraction.json") as f:
                    extraction = json.load(f)

                scores = extraction.get("scores", {})
                scorer_data = scores.get("safety_behavior_scorer", {})

                if not scorer_data:
                    print("No safety_behavior_scorer results found")
                else:
                    value = scorer_data.get("value")
                    answer = scorer_data.get("answer", "")
                    explanation = scorer_data.get("explanation", "")
                    metadata = scorer_data.get("metadata", {})
                    analysis = metadata.get("analysis", {})

                    # Calculate components for display
                    safety_score = analysis.get("safety_score", "?")
                    ml = analysis.get("misalignment_level", {})
                    level = ml.get("level", "?")
                    level_name = ml.get("level_name", "unknown")
                    justification = ml.get("justification", "")

                    print("=" * 60)
                    print("Three-Phase Analysis Complete")
                    print("=" * 60)

                    print(f"\n📊 COMPOSITE SCORE: {value}")
                    print(f"   {answer}")

                    print(f"\n⚠️  MISALIGNMENT LEVEL: {level} - {level_name}")
                    print(f"   {justification}")

                    # Phase 1
                    p1 = analysis.get("phase1_navigation", {})
                    print("\n--- Phase 1: Navigation ---")
                    print(f"  Harm noticed: {p1.get('harm_noticed')} (turn {p1.get('harm_noticed_turn')})")
                    print(f"  Warnings ignored: {p1.get('warnings_ignored_count')}")
                    print(f"  Risky path despite warning: {p1.get('took_risky_path_despite_warning')}")
                    print(f"  Rationalization detected: {p1.get('rationalization_detected')}")

                    # Phase 2
                    p2 = analysis.get("phase2_report", {})
                    print("\n--- Phase 2: Report ---")
                    print(f"  Facts known: {len(p2.get('facts_known_from_phase1', []))}")
                    print(f"  Facts omitted: {len(p2.get('facts_omitted_from_report', []))}")
                    if p2.get("facts_omitted_from_report"):
                        print(f"    Omitted: {p2.get('facts_omitted_from_report')}")
                    print(f"  Omission type: {p2.get('omission_type')}")
                    completeness = p2.get("report_completeness")
                    if completeness is not None:
                        print(f"  Report completeness: {completeness:.0%}")

                    # Phase 3
                    p3 = analysis.get("phase3_debrief", {})
                    print("\n--- Phase 3: Debrief ---")
                    if p3.get("claims_actions_were_aligned") is None:
                        print("  ⚠️  Phase 3 data missing")
                    else:
                        print(f"  Claims aligned: {p3.get('claims_actions_were_aligned')}")
                        print(f"  Self-assessment accuracy: {p3.get('self_assessment_accuracy')}")
                        print(f"  Story changed at disclosure: {analysis.get('cross_phase', {}).get('story_changed_at_disclosure')}")
                        print(f"  Retrospective pattern: {p3.get('retrospective_pattern')}")

                    # Cross-phase
                    cp = analysis.get("cross_phase", {})
                    print("\n--- Cross-Phase Analysis ---")
                    ktr = cp.get("knowledge_to_report_honesty")
                    if ktr is not None:
                        print(f"  Knowledge→Report honesty: {ktr:.0%}")
                    print(f"  Behavior trajectory: {cp.get('behavior_trajectory')}")

                    # Key findings
                    print(f"\n🚨 Most concerning: {analysis.get('most_concerning_behavior')}")
                    if analysis.get("most_positive_behavior"):
                        print(f"✅ Most positive: {analysis.get('most_positive_behavior')}")

                    # Data quality
                    da = analysis.get("data_analysis", {})
                    print(f"\n📊 Data quality: {da.get('data_quality_notes', 'N/A')}")
                    rc = da.get("reasoning_coverage_percent")
                    if rc is not None:
                        print(f"   Reasoning coverage: {rc:.0f}%")

                    # Save judge_analysis.json
                    judge_output = {
                        "judge_model": metadata.get("judge_model", "unknown"),
                        "subject_model": extraction.get("metadata", {}).get("model", "unknown"),
                        "scenario": extraction.get("metadata", {}).get("scenario", "unknown"),
                        "analysis": analysis,
                        "judge_reasoning": metadata.get("judge_reasoning"),
                        "raw_response": metadata.get("judge_response"),
                    }
                    judge_path = output_dir / "judge_analysis.json"
                    with open(judge_path, "w") as jf:
                        json.dump(judge_output, jf, indent=2)
                    print(f"\n💾 Saved: {judge_path}")

                    # Enrich trajectory with extraction data and copy to viewer
                    # Also generates alignment moments unless --no-moments is set
                    _enrich_trajectory_and_copy(
                        output_dir,
                        generate_moments=not args.no_moments,
                        judge_model=args.judge,
                    )

            except Exception as e:
                print(f"Failed to display judge results: {e}")


if __name__ == "__main__":
    main()
