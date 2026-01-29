#!/usr/bin/env python3
"""
Tally experiment runs by scenario and model.

Usage:
    python3 scripts/tally_runs.py [extractions_dir]
"""

import sys
from collections import defaultdict
from pathlib import Path


def tally_runs(extractions_dir: Path) -> dict:
    """Tally runs by scenario and model."""
    # Structure: {scenario: {model: count}}
    tally = defaultdict(lambda: defaultdict(int))

    # Also track individual runs for details
    runs = []

    for scenario_dir in extractions_dir.iterdir():
        if not scenario_dir.is_dir() or scenario_dir.name.startswith('.'):
            continue

        scenario = scenario_dir.name

        for run_dir in scenario_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.'):
                continue

            # Parse folder name: YYYY-MM-DDTHH-MM_model-name
            parts = run_dir.name.split('_', 1)
            if len(parts) == 2:
                timestamp, model = parts
                tally[scenario][model] += 1
                runs.append({
                    'scenario': scenario,
                    'model': model,
                    'timestamp': timestamp,
                    'path': run_dir
                })

    return dict(tally), runs


def print_tally(tally: dict, runs: list) -> None:
    """Print tally as a table."""
    if not tally:
        print("No runs found.")
        return

    # Get all unique models
    all_models = set()
    for scenario_models in tally.values():
        all_models.update(scenario_models.keys())
    all_models = sorted(all_models)

    # Get all scenarios
    scenarios = sorted(tally.keys())

    # Calculate column widths
    model_width = max(len(m) for m in all_models) if all_models else 10
    scenario_width = max(len(s) for s in scenarios) if scenarios else 10

    # Print header
    print("=" * 70)
    print("EXPERIMENT RUN TALLY")
    print("=" * 70)
    print()

    # Print table header
    header = f"{'Scenario':<{scenario_width}} | " + " | ".join(f"{m:<{max(len(m), 3)}}" for m in all_models) + " | Total"
    print(header)
    print("-" * len(header))

    # Print rows
    model_totals = defaultdict(int)
    for scenario in scenarios:
        row = f"{scenario:<{scenario_width}} | "
        scenario_total = 0
        for model in all_models:
            count = tally[scenario].get(model, 0)
            row += f"{count:<{max(len(model), 3)}} | "
            scenario_total += count
            model_totals[model] += count
        row += str(scenario_total)
        print(row)

    # Print totals row
    print("-" * len(header))
    totals_row = f"{'TOTAL':<{scenario_width}} | "
    grand_total = 0
    for model in all_models:
        totals_row += f"{model_totals[model]:<{max(len(model), 3)}} | "
        grand_total += model_totals[model]
    totals_row += str(grand_total)
    print(totals_row)

    # Print balance analysis
    print()
    print("=" * 70)
    print("BALANCE ANALYSIS")
    print("=" * 70)

    # Find max count per scenario to identify gaps
    for scenario in scenarios:
        counts = tally[scenario]
        if not counts:
            continue
        max_count = max(counts.values())
        missing = []
        for model in all_models:
            count = counts.get(model, 0)
            if count < max_count:
                missing.append(f"{model} (+{max_count - count})")

        if missing:
            print(f"{scenario}: needs {', '.join(missing)}")
        else:
            print(f"{scenario}: balanced at {max_count} runs each")


def main():
    if len(sys.argv) > 1:
        extractions_dir = Path(sys.argv[1])
    else:
        extractions_dir = Path("extractions")

    if not extractions_dir.exists():
        print(f"Error: Directory not found: {extractions_dir}")
        sys.exit(1)

    tally, runs = tally_runs(extractions_dir)
    print_tally(tally, runs)


if __name__ == "__main__":
    main()
