#!/usr/bin/env python3
"""
Local runner for G1 Alignment Experiments.

Mirrors the Modal deployment interface for local testing.
Use this to verify experiments work before deploying to Modal.

Usage:
    # Run single experiment (same as Modal)
    ./venv/bin/mjpython local_runner.py --scenario barrels_corrupt --model robotics

    # Run batch of experiments
    ./venv/bin/mjpython local_runner.py --scenario barrels_hi --model gemini2.5 --num-runs 3

    # Compare with Modal (after deploying)
    modal run modal/app.py::run_experiment --scenario barrels_lo --model robotics
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


def _enrich_trajectory_and_copy(extraction_dir: Path) -> None:
    """Enrich trajectory with extraction data and copy to viewer assets."""
    trajectory_path = extraction_dir / "trajectory.json"
    if not trajectory_path.exists():
        print("  ⚠️  No trajectory.json to enrich")
        return

    try:
        from scripts.enrich_trajectory import (
            enrich_trajectory,
            load_json,
        )

        trajectory = load_json(trajectory_path)
        extraction = load_json(extraction_dir / "extraction.json")
        judge_analysis = load_json(extraction_dir / "judge_analysis.json")

        if not trajectory:
            return

        print("\n=== Enriching Trajectory ===")
        enriched = enrich_trajectory(trajectory, extraction, judge_analysis)

        with open(trajectory_path, "w") as f:
            json.dump(enriched, f, indent=2)

        viewer_assets = Path(__file__).parent / "gcp" / "frontend" / "assets"
        if viewer_assets.exists():
            name = extraction_dir.name.replace(":", "-").replace(" ", "_")
            viewer_path = viewer_assets / f"trajectory_{name}.json"
            with open(viewer_path, "w") as f:
                json.dump(enriched, f, indent=2)
            print(f"  📺 Copied to viewer: {viewer_path.name}")

    except Exception as e:
        print(f"  ⚠️  Trajectory enrichment failed: {e}")


def run_experiment(
    run_id: str,
    scenario: str = "barrels_lo",
    model: str = "robotics",
    reasoning: str = "high",
    judge_model: str = "google/gemini-3.1-pro-preview",
) -> dict:
    """Run a single alignment experiment locally.

    This function mirrors modal/app.py::run_experiment exactly.

    Args:
        run_id: Unique identifier for this run
        scenario: Scenario name (barrels_lo, barrels_mi, barrels_mh, barrels_hi)
        model: Model shortcut (robotics, gemini2.5, claude, etc.)
        reasoning: Reasoning effort level
        judge_model: Model to use for judging safety behavior

    Returns:
        Dict with experiment results including scores and extraction path
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import GenerateConfig

    # Set up environment (same as Modal)
    os.environ["G1_SCENARIO"] = scenario
    os.environ["G1_HEADLESS"] = "true"
    os.environ["G1_RECORD_VIDEO"] = "true"
    os.environ["G1_MODEL_NAME"] = model
    os.environ["G1_JUDGE_MODEL"] = judge_model

    # Model shortcuts (same as Modal)
    MODELS = {
        "gemini2.5": "google/gemini-2.5-pro",
        "robotics": "google/gemini-robotics-er-1.6-preview",
        "claude": "anthropic/claude-3-5-sonnet-latest",
        "opus": "anthropic/claude-opus-4-5-20251101",
        "gpt4": "openai/gpt-4o",
        "gpt5": "openai/gpt-5",
        "kimi": "kimi/kimi-k2.5",
    }
    model_full = MODELS.get(model, model)

    # Handle Kimi provider
    if model == "kimi":
        import inspect_eval.kimi_provider  # Registers provider with Inspect

        _ = inspect_eval.kimi_provider  # Ensure import is used
        model_instance = "kimi/kimi-k2.5"
    else:
        model_instance = model_full

    # Build generation config (same as Modal)
    gen_config = GenerateConfig(
        reasoning_effort=reasoning,
        reasoning_summary="detailed",
        max_retries=5,
        attempt_timeout=180,
    )

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {run_id}")
    print(f"  Scenario: {scenario}")
    print(f"  Model: {model_full}")
    print(f"  Reasoning: {reasoning}")
    print(f"  Judge: {judge_model}")
    print(f"{'=' * 60}\n")

    # Run the experiment
    results = inspect_eval(
        "inspect_eval/tasks.py@g1_native",
        model=model_instance,
        config=gen_config,
        metadata={
            "scenario": scenario,
            "reasoning_effort": reasoning,
            "model": model_full,
            "run_id": run_id,
            "local": True,
        },
        limit=1,
        log_level="info",
    )

    # Extract results (same structure as Modal)
    output = {
        "run_id": run_id,
        "status": "error",
        "scenario": scenario,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if results and results[0].status == "success":
        output["status"] = "success"

        # Get scores from samples
        for sample in results[0].samples:
            if sample.scores:
                output["scores"] = {}
                for scorer_name, score in sample.scores.items():
                    output["scores"][scorer_name] = {
                        "value": score.value,
                        "answer": score.answer,
                        "explanation": score.explanation,
                    }

        # Run extraction
        try:
            from scripts.extract_eval_data import (
                _generate_folder_path,
                extract_eval,
                extraction_to_dict,
                save_extraction,
            )

            log_path = results[0].location
            if log_path:
                extraction = extract_eval(log_path, full_content=True)
                data = extraction_to_dict(extraction, include_full_prompts=True)

                # Generate folder path
                scenario_folder, run_folder = _generate_folder_path(
                    data["metadata"]["created"],
                    data["metadata"]["model"],
                    scenario,
                )
                output_dir = Path("extractions") / scenario_folder / run_folder
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save extraction
                save_extraction(data, output_dir / "extraction.json", extraction)
                output["extraction_path"] = str(output_dir / "extraction.json")

                # Save judge analysis
                scores = data.get("scores", {})
                scorer_data = scores.get("safety_behavior_scorer", {})
                if scorer_data:
                    judge_output = {
                        "judge_model": scorer_data.get("metadata", {}).get(
                            "judge_model", "unknown"
                        ),
                        "subject_model": model_full,
                        "scenario": scenario,
                        "analysis": scorer_data.get("metadata", {}).get("analysis", {}),
                    }
                    judge_path = output_dir / "judge_analysis.json"
                    with judge_path.open("w") as f:
                        json.dump(judge_output, f, indent=2)
                    output["judge_analysis_path"] = str(judge_path)

                # Enrich trajectory and copy to viewer
                _enrich_trajectory_and_copy(output_dir)

                print(f"\nResults saved to: {output_dir}")

        except Exception as e:
            output["extraction_error"] = str(e)
            print(f"\nExtraction error: {e}")

    else:
        output["error"] = "Experiment failed"
        if results:
            output["eval_status"] = results[0].status

    return output


def run_batch(
    batch_id: str,
    num_runs: int,
    scenario: str = "barrels_lo",
    model: str = "robotics",
    reasoning: str = "high",
) -> dict:
    """Run multiple experiments sequentially.

    Note: Unlike Modal which runs in parallel, local runs are sequential.

    Args:
        batch_id: Unique identifier for this batch
        num_runs: Number of experiments to run
        scenario: Scenario name
        model: Model shortcut
        reasoning: Reasoning effort level

    Returns:
        Dict with batch results summary
    """
    print(f"\n{'=' * 60}")
    print(f"Running batch: {batch_id}")
    print(f"  Runs: {num_runs}")
    print(f"  Scenario: {scenario}")
    print(f"  Model: {model}")
    print(f"{'=' * 60}\n")

    results = []
    for i in range(num_runs):
        run_id = f"{batch_id}_{i:03d}"
        print(f"\n--- Run {i + 1}/{num_runs} ---")
        result = run_experiment(
            run_id=run_id,
            scenario=scenario,
            model=model,
            reasoning=reasoning,
        )
        results.append(result)

    # Summarize results (same as Modal)
    success_count = sum(1 for r in results if r.get("status") == "success")
    scores = []
    for r in results:
        if r.get("scores", {}).get("safety_behavior_scorer"):
            score_val = r["scores"]["safety_behavior_scorer"].get("value")
            if isinstance(score_val, (int, float)):
                scores.append(score_val)

    summary = {
        "batch_id": batch_id,
        "num_runs": num_runs,
        "success_count": success_count,
        "failure_count": num_runs - success_count,
        "average_score": sum(scores) / len(scores) if scores else None,
        "scores": scores,
        "results": results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"\n{'=' * 60}")
    print("Batch Complete")
    print(f"  Success: {success_count}/{num_runs}")
    if scores:
        print(f"  Average Score: {summary['average_score']:.2%}")
    print(f"{'=' * 60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run G1 alignment experiments locally (mirrors Modal interface)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  mjpython local_runner.py --scenario barrels_lo --model robotics

  # Batch of 3 experiments
  mjpython local_runner.py --scenario barrels_hi --model gemini2.5 --num-runs 3

  # Compare with Modal
  modal run modal/app.py::run_experiment --scenario barrels_lo --model robotics
        """,
    )

    parser.add_argument(
        "--scenario",
        "-s",
        default="barrels_corrupt",
        choices=["barrels_lo", "barrels_mi", "barrels_mh", "barrels_hi", "barrels_corrupt"],
        help="Scenario to run (default: barrels_corrupt)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="robotics",
        choices=["robotics", "gemini2.5", "claude", "opus", "gpt4", "gpt5", "kimi"],
        help="Model to use (default: robotics)",
    )
    parser.add_argument(
        "--reasoning",
        "-r",
        default="high",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (default: high)",
    )
    parser.add_argument(
        "--num-runs",
        "-n",
        type=int,
        default=1,
        help="Number of runs (default: 1, use >1 for batch)",
    )
    parser.add_argument(
        "--judge",
        default="google/gemini-3.1-pro-preview",
        help="Judge model for safety scoring (default: google/gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Generate IDs
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.num_runs == 1:
        # Single experiment
        run_id = f"local_{timestamp}"
        result = run_experiment(
            run_id=run_id,
            scenario=args.scenario,
            model=args.model,
            reasoning=args.reasoning,
            judge_model=args.judge,
        )
    else:
        # Batch
        batch_id = f"local_batch_{timestamp}"
        result = run_batch(
            batch_id=batch_id,
            num_runs=args.num_runs,
            scenario=args.scenario,
            model=args.model,
            reasoning=args.reasoning,
        )

    # Output
    print("\n=== Result ===")
    print(json.dumps(result, indent=2, default=str))

    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()
