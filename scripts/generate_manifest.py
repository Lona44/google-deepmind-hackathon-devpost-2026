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

    # Need at least judge analysis to show anything meaningful
    if not judge and not extraction:
        return None

    has_trajectory = trajectory is not None

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
    traj_judge = trajectory.get("judge", {}) if trajectory else {}

    composite_score = traj_judge.get("composite_score")
    if composite_score is None and extraction:
        scores = extraction.get("scores", {})
        scorer_data = scores.get("safety_behavior_scorer", {})
        raw_score = scorer_data.get("value")
        # Only use numeric scores (judge errors return strings like "E")
        if isinstance(raw_score, (int, float)):
            composite_score = raw_score
        else:
            try:
                composite_score = float(raw_score) if raw_score is not None else None
            except (ValueError, TypeError):
                composite_score = None

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

    # Get multi-dimensional scoring from scorer metadata or trajectory
    risk_class = traj_judge.get("risk_class")
    deployment_status = traj_judge.get("deployment_status")
    if (risk_class is None or deployment_status is None) and extraction:
        scores = extraction.get("scores", {})
        scorer_data = scores.get("safety_behavior_scorer", {})
        scorer_meta = scorer_data.get("metadata", {})
        if risk_class is None:
            risk_class = scorer_meta.get("risk_class")
        if deployment_status is None:
            deployment_status = scorer_meta.get("deployment_status")

    # Trajectory-dependent fields (null for aborted runs)
    if has_trajectory:
        # Count attempts from trajectory
        attempts = set()
        for frame in trajectory.get("frames", []):
            attempts.add(frame.get("attempt", 1))
        num_attempts = len(attempts)

        # Frame count and duration
        frames = trajectory.get("frames", [])
        num_frames = len(frames)
        duration = frames[-1].get("time", 0) if frames else 0
        trajectory_file = f"trajectory_{run_id}.json"
    else:
        # Aborted run - no trajectory data
        num_attempts = None
        num_frames = None
        duration = None
        trajectory_file = None

    # For aborted runs, include full judge data in manifest (can't fetch separately)
    judge_data = None
    if not has_trajectory and judge:
        scorer_meta = {}
        if extraction:
            scores = extraction.get("scores", {})
            scorer_data = scores.get("safety_behavior_scorer", {})
            scorer_meta = scorer_data.get("metadata", {})

        judge_data = {
            "most_positive": analysis.get("most_positive_behavior"),
            "most_concerning": analysis.get("most_concerning_behavior"),
            "key_quotes": analysis.get("key_quotes", [])[:5],
            "alignment_flags": scorer_meta.get("alignment_flags"),
            "judge_model": judge.get("judge_model"),
        }

    return {
        "id": run_id,
        "model": model,
        "timestamp": timestamp.replace("T", " ").replace("-", ":"),
        "composite_score": composite_score,
        "safety_score": safety_score,
        "honesty_score": honesty_score,
        "alignment_level": alignment_level,
        "alignment_name": alignment_name,
        "risk_class": risk_class,
        "deployment_status": deployment_status,
        "judge_data": judge_data,
        "attempts": num_attempts,
        "frames": num_frames,
        "duration": round(duration, 1) if duration is not None else None,
        "trajectory_file": trajectory_file,
        "has_trajectory": has_trajectory,
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
    all_runs = [r for s in manifest["scenarios"].values() for r in s["runs"]]
    total_runs = len(all_runs)
    aborted_runs = sum(1 for r in all_runs if not r.get("has_trajectory"))
    total_scenarios = len(manifest["scenarios"])

    print(f"  Found {total_runs} runs across {total_scenarios} scenarios")
    if aborted_runs:
        print(f"  ({aborted_runs} aborted runs without trajectory)")

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
            aborted_marker = " [ABORTED]" if not run.get("has_trajectory") else ""
            print(f"    {run['model']} • {score_str} • {run['timestamp']}{aborted_marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
