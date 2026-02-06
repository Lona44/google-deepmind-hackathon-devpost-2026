#!/usr/bin/env python3
"""
Generate manifest of all extractions for the browser viewer.

Scans extractions/ folder and creates gcp/frontend/assets/extractions_index.json
with metadata from extraction.json, judge_analysis.json, and trajectory.json.

Usage:
    python scripts/generate_manifest.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Scenario display names
SCENARIO_NAMES = {
    "barrels_lo": "Barrels (Low Pressure)",
    "barrels_mi": "Barrels (Medium-Low Pressure)",
    "barrels_mh": "Barrels (Medium-High Pressure)",
    "barrels_hi": "Barrels (High Pressure)",
    "barrels_corrupt": "Barrels (Telemetry Corruption)",
}

PROJECT_ROOT = Path(__file__).parent.parent
EXTRACTIONS_DIR = PROJECT_ROOT / "extractions"
OUTPUT_PATH = PROJECT_ROOT / "gcp" / "frontend" / "assets" / "extractions_index.json"


def load_json(path: Path) -> dict | None:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def extract_run_metadata(run_dir: Path) -> dict | None:
    """Extract metadata from a single run directory."""
    extraction = load_json(run_dir / "extraction.json")
    judge = load_json(run_dir / "judge_analysis.json")
    trajectory = load_json(run_dir / "trajectory.json")

    if not trajectory:
        # No trajectory = can't view in browser
        return None

    # Build run metadata
    run_id = run_dir.name

    # Parse model and timestamp from run_id (format: 2026-02-01T20-36_gpt-5)
    parts = run_id.rsplit("_", 1)
    timestamp = parts[0] if parts else run_id
    model = parts[1] if len(parts) > 1 else "unknown"

    # Get scores from judge analysis or trajectory
    analysis = judge.get("analysis", {}) if judge else {}
    misalignment = analysis.get("misalignment_level", {})

    # Try trajectory.judge first (enriched), then fall back to judge_analysis
    traj_judge = trajectory.get("judge", {})

    composite_score = traj_judge.get("composite_score")
    if composite_score is None and extraction:
        scores = extraction.get("scores", {})
        scorer_data = scores.get("safety_behavior_scorer", {})
        composite_score = scorer_data.get("value")

    # Use explicit None checks since 0 is a valid score
    safety_score = traj_judge.get("safety_score")
    if safety_score is None:
        safety_score = analysis.get("safety_score")
    honesty_score = traj_judge.get("honesty_score")
    if honesty_score is None:
        honesty_score = analysis.get("honesty_score")
    alignment_level = traj_judge.get("alignment_level")
    if alignment_level is None:
        alignment_level = misalignment.get("level")
    alignment_name = traj_judge.get("alignment_name")
    if alignment_name is None:
        alignment_name = misalignment.get("level_name")

    # Count attempts from trajectory
    attempts = set()
    for frame in trajectory.get("frames", []):
        attempts.add(frame.get("attempt", 1))
    num_attempts = len(attempts)

    # Frame count and duration
    frames = trajectory.get("frames", [])
    num_frames = len(frames)
    duration = frames[-1].get("time", 0) if frames else 0

    # Trajectory filename (as copied to assets)
    trajectory_file = f"trajectory_{run_id}.json"

    return {
        "id": run_id,
        "model": model,
        "timestamp": timestamp.replace("T", " ").replace("-", ":"),
        "composite_score": composite_score,
        "safety_score": safety_score,
        "honesty_score": honesty_score,
        "alignment_level": alignment_level,
        "alignment_name": alignment_name,
        "attempts": num_attempts,
        "frames": num_frames,
        "duration": round(duration, 1),
        "trajectory_file": trajectory_file,
    }


def generate_manifest() -> dict:
    """Generate manifest from all extractions."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "scenarios": {},
    }

    if not EXTRACTIONS_DIR.exists():
        print(f"Warning: {EXTRACTIONS_DIR} does not exist")
        return manifest

    # Scan scenario directories
    for scenario_dir in sorted(EXTRACTIONS_DIR.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
            continue

        scenario_id = scenario_dir.name
        scenario_name = SCENARIO_NAMES.get(scenario_id, scenario_id)

        runs = []
        for run_dir in sorted(scenario_dir.iterdir(), reverse=True):  # Most recent first
            if not run_dir.is_dir():
                continue

            run_meta = extract_run_metadata(run_dir)
            if run_meta:
                runs.append(run_meta)

        if runs:
            manifest["scenarios"][scenario_id] = {
                "name": scenario_name,
                "runs": runs,
            }

    return manifest


def main():
    print("Generating extractions manifest...")

    manifest = generate_manifest()

    # Count totals
    total_runs = sum(len(s["runs"]) for s in manifest["scenarios"].values())
    total_scenarios = len(manifest["scenarios"])

    print(f"  Found {total_runs} runs across {total_scenarios} scenarios")

    # Save manifest
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved to: {OUTPUT_PATH}")

    # Print summary
    for _scenario_id, scenario in manifest["scenarios"].items():
        print(f"\n  {scenario['name']}:")
        for run in scenario["runs"]:
            score_str = f"{run['composite_score']:.2f}" if run["composite_score"] else "?"
            print(f"    {run['model']} • {score_str} • {run['timestamp']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
