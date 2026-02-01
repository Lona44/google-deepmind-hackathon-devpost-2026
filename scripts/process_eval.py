#!/usr/bin/env python3
"""
Complete eval processing pipeline.

Extracts data from .eval file, enriches trajectory, and copies to viewer.

Usage:
    # Process latest eval
    python scripts/process_eval.py

    # Process specific eval
    python scripts/process_eval.py logs/2026-02-01T20-36_gpt-5.eval

    # Process without opening viewer
    python scripts/process_eval.py --no-view
"""

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inspect_ai.log import list_eval_logs


def find_latest_eval(log_dir: Path) -> Path | None:
    """Find the most recently created eval file."""
    logs = list_eval_logs(str(log_dir))
    if not logs:
        return None
    # logs are sorted by creation time, most recent last
    return Path(logs[-1].name)


def find_extraction_dir(extractions_base: Path, eval_path: Path) -> Path | None:
    """Find the extraction directory for a given eval file.

    Matches by timestamp in the eval filename to the extraction folder name.
    """
    # Try to parse timestamp from eval filename
    # e.g., "2026-02-01T20-36-53+00-00_g1_native_gpt-5.eval"
    eval_name = eval_path.stem

    # Look for matching extraction directories across all scenarios
    for scenario_dir in extractions_base.iterdir():
        if not scenario_dir.is_dir():
            continue
        for run_dir in scenario_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # Check if this run directory matches
            # run_dir.name is like "2026-02-01T20-36_gpt-5"
            if run_dir.name.replace("-", "").replace("_", "") in eval_name.replace("-", "").replace("_", ""):
                return run_dir

    # If no match, return the most recently modified directory
    all_dirs = []
    for scenario_dir in extractions_base.iterdir():
        if not scenario_dir.is_dir():
            continue
        for run_dir in scenario_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "extraction.json").exists():
                all_dirs.append((run_dir.stat().st_mtime, run_dir))

    if all_dirs:
        all_dirs.sort(reverse=True)
        return all_dirs[0][1]

    return None


def run_extraction(eval_path: Path, output_base: Path) -> Path | None:
    """Run extract_eval_data.py and return the extraction directory."""
    script = PROJECT_ROOT / "scripts" / "extract_eval_data.py"

    cmd = [
        sys.executable, str(script),
        str(eval_path),
        "-o", str(output_base)
    ]

    print(f"\n{'='*60}")
    print("Step 1: Extracting eval data")
    print(f"{'='*60}")
    print(f"  Eval: {eval_path.name}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return None

    print(result.stdout)

    # Find the created extraction directory
    return find_extraction_dir(output_base, eval_path)


def run_enrichment(extraction_dir: Path) -> bool:
    """Run enrich_trajectory.py with --copy-to-viewer."""
    script = PROJECT_ROOT / "scripts" / "enrich_trajectory.py"

    cmd = [
        sys.executable, str(script),
        str(extraction_dir),
        "--copy-to-viewer"
    ]

    print(f"\n{'='*60}")
    print("Step 2: Enriching trajectory")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False

    return True


def get_viewer_url(extraction_dir: Path) -> str:
    """Get the URL to view the trajectory in the browser."""
    name = extraction_dir.name.replace(":", "-").replace(" ", "_")
    return f"http://localhost:5500/?trajectory=assets/trajectory_{name}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Complete eval processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "eval_path",
        nargs="?",
        type=Path,
        help="Path to .eval file (default: latest eval)"
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Don't open the viewer in browser"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Directory containing eval logs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "extractions",
        help="Base directory for extractions"
    )

    args = parser.parse_args()

    # Find eval file
    if args.eval_path:
        eval_path = args.eval_path
        if not eval_path.exists():
            print(f"Error: {eval_path} not found")
            return 1
    else:
        eval_path = find_latest_eval(args.log_dir)
        if not eval_path:
            print(f"Error: No eval files found in {args.log_dir}")
            return 1
        print(f"Using latest eval: {eval_path.name}")

    # Step 1: Extract eval data
    extraction_dir = run_extraction(eval_path, args.output_dir)
    if not extraction_dir:
        print("\nExtraction failed!")
        return 1

    print(f"\n  Extraction dir: {extraction_dir}")

    # Check if trajectory exists
    trajectory_path = extraction_dir / "trajectory.json"
    if not trajectory_path.exists():
        print(f"\n  Warning: No trajectory.json found in {extraction_dir}")
        print("  The trajectory may not have been recorded for this run.")
        return 1

    # Step 2: Enrich trajectory and copy to viewer
    if not run_enrichment(extraction_dir):
        print("\nEnrichment failed!")
        return 1

    # Step 3: Open viewer
    viewer_url = get_viewer_url(extraction_dir)
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"\nView at: {viewer_url}")

    if not args.no_view:
        print("Opening in browser...")
        webbrowser.open(viewer_url)

    return 0


if __name__ == "__main__":
    sys.exit(main())
