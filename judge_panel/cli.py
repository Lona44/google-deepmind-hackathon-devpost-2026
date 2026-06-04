"""judge_panel CLI — process a directory of behavioral_data.json files.

Usage:
    python -m judge_panel.cli \\
        --input-dir <dir of behavioral_data.json files> \\
        --verdicts-dir <output dir> \\
        [--force-regrade] \\
        [--max-session-cost-usd N]

Honors env vars OPENROUTER_API_KEY (required), JUDGE_PANEL_MAX_COST_PER_RUN,
JUDGE_PANEL_MAX_SESSION_COST.

Exit codes:
    0 — success or all-skipped or partial_failure
    1 — error (auth, cost cap, no input files, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from judge_panel.cost_tracker import PerSessionTracker
from judge_panel.models import OpenRouterClient
from judge_panel.orchestrator import run_panel, run_panel_idempotent
from judge_panel.verdicts import write_verdict


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-judge panel on a batch of experiments.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing behavioral_data.json files")
    parser.add_argument("--verdicts-dir", type=Path, required=True,
                        help="Where to write verdicts/<run_id>/panel_verdict.{json,md}")
    parser.add_argument("--force-regrade", action="store_true",
                        help="Re-grade even if a verdict already exists. New verdicts get timestamped filenames.")
    parser.add_argument("--max-session-cost-usd", type=float, default=None,
                        help="Hard stop CLI when cumulative cost crosses this. Default from JUDGE_PANEL_MAX_SESSION_COST or $5.")
    return parser.parse_args(argv)


def _discover_inputs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.glob("*.json") if p.is_file())


async def _process_one(
    path: Path, verdicts_dir: Path, client: OpenRouterClient,
    session: PerSessionTracker, force_regrade: bool,
) -> str:
    data = json.loads(path.read_text())
    run_id = data.get("run_id") or path.stem

    async def panel_call(**kwargs):
        return await run_panel(behavioral_data=data, client=client, run_id=run_id)

    result = await run_panel_idempotent(
        run_id=run_id, behavioral_data=data, verdicts_dir=verdicts_dir,
        panel_func=panel_call, force_regrade=force_regrade,
    )
    if result == "skipped":
        return "skipped"
    verdict = result
    write_verdict(verdict, base_dir=verdicts_dir, force_regrade=force_regrade)
    session.record_experiment(run_id, verdict.metadata.total_cost_usd)
    return verdict.status


async def _main_async(args: argparse.Namespace) -> int:
    # Search for .env from the user's CWD upward, not from this source file's
    # directory. This makes the CLI behave the way users expect when invoked
    # from arbitrary working directories and lets tests run in isolated dirs.
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    inputs = _discover_inputs(args.input_dir)
    if not inputs:
        print(f"ERROR: no *.json files in {args.input_dir}", file=sys.stderr)
        return 1

    session_cap = args.max_session_cost_usd
    if session_cap is None:
        session_cap = float(os.environ.get("JUDGE_PANEL_MAX_SESSION_COST", "5.0"))
    session = PerSessionTracker(max_cost_usd=session_cap)

    print(f"judge_panel: {len(inputs)} inputs, session cap ${session_cap:.2f}, force_regrade={args.force_regrade}")

    client = OpenRouterClient(api_key=api_key)
    try:
        for path in inputs:
            if session.would_exceed(0.10):  # 10× expected per-run cost as headroom check
                print(f"session cap close to limit (used ${session.total_cost_usd:.4f} / ${session_cap:.2f}), stopping")
                break
            status = await _process_one(path, args.verdicts_dir, client, session, args.force_regrade)
            print(f"  {path.name}  →  {status}")
    finally:
        await client.aclose()

    print(f"done. processed {session.experiment_count} experiments, ${session.total_cost_usd:.4f} total")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
