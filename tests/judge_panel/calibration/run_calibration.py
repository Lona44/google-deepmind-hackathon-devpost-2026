#!/usr/bin/env python3
"""Live-API calibration runner.

Runs the multi-judge panel against the recorded G1 experiments, compares
each verdict to the existing Gemini 3 Pro baseline, computes inter-rater
agreement, emits a REPORT.md citable in the AISI research note.

Expected cost: ~$0.15 for the default 3 inputs. Hard-stops at the per-
session cost cap.

Usage:
    python tests/judge_panel/calibration/run_calibration.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "judge_panel" / "fixtures"
RESULTS_ROOT = REPO_ROOT / "tests" / "judge_panel" / "calibration" / "results"

sys.path.insert(0, str(REPO_ROOT))

from judge_panel.calibration import (  # noqa: E402
    aggregate_within_panel_honesty,
    aggregate_within_panel_safety,
    compute_calibration_result,
    load_behavioral_data,
    load_gemini_baseline,
    render_report_markdown,
)
from judge_panel.metrics import krippendorffs_alpha  # introduced in Task 26
from judge_panel.models import OpenRouterClient  # noqa: E402
from judge_panel.orchestrator import run_panel  # noqa: E402


def _default_inputs() -> list[tuple[str, Path, Path]]:
    """(run_id, behavioral_data_path, gemini_baseline_path) for each input.

    Mapping note: Task 4's behavioral_data fixtures are `canonical.json`
    (a slice of the gpt-5 trajectory) and `synthetic_edge_case.json`.
    The canonical slice is paired with the gpt-5 Gemini baseline; the
    kimi-k2.5 baseline has no matching behavioral_data fixture yet, so
    its entry is skipped via the `bd_path.exists()` guard. The synthetic
    edge case has no Gemini baseline by design.
    """
    bd = FIXTURES / "behavioral_data"
    gb = FIXTURES / "gemini_baselines"
    pairs: list[tuple[str, Path, Path]] = []

    # Map canonical fixture to the gpt-5 Gemini baseline (Task 4 sliced
    # canonical.json from trajectory_2026-02-06T05-01_gpt-5.json).
    canonical_bd = bd / "canonical.json"
    canonical_gb = gb / "2026-02-06T05-01_gpt-5.json"
    if canonical_bd.exists():
        pairs.append(("2026-02-06T05-01_gpt-5", canonical_bd, canonical_gb))

    # The second recorded G1 run (kimi-k2.5) — included for forward
    # compatibility once a matching behavioral_data fixture lands.
    kimi_bd = bd / "2026-02-06T04-28_kimi-k2.5.json"
    kimi_gb = gb / "2026-02-06T04-28_kimi-k2.5.json"
    if kimi_bd.exists():
        pairs.append(("2026-02-06T04-28_kimi-k2.5", kimi_bd, kimi_gb))

    # The synthetic edge case (no Gemini baseline)
    pairs.append(
        (
            "synthetic-edge-case",
            bd / "synthetic_edge_case.json",
            gb / "synthetic-edge-case.json",
        )
    )
    return pairs


async def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = RESULTS_ROOT / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenRouterClient(api_key=api_key)
    cal_results = []
    verdicts = []
    try:
        for run_id, bd_path, gb_path in _default_inputs():
            print(f"\n=== {run_id} ===")
            bd = load_behavioral_data(bd_path)
            verdict = await run_panel(behavioral_data=bd, client=client, run_id=run_id)
            verdicts.append(verdict)

            verdict_out = out_dir / f"verdict-{run_id}.json"
            verdict_out.write_text(verdict.model_dump_json(indent=2))
            print(f"  verdict: safety={verdict.final_safety_score} honesty={verdict.final_honesty_score} level={verdict.final_misalignment_level} cost=${verdict.metadata.total_cost_usd:.4f}")

            gemini = load_gemini_baseline(gb_path)
            if gemini is not None:
                cr = compute_calibration_result(verdict, gemini)
                cal_results.append(cr)
                cmp_out = out_dir / f"comparison-{run_id}.json"
                cmp_out.write_text(cr.model_dump_json(indent=2))
                print(f"  comparison vs Gemini: Δsafety={cr.safety_score_delta:+d} Δhonesty={cr.honesty_score_delta:+d} level_match={cr.misalignment_level_match}")
            else:
                print("  no Gemini baseline available for this run")
    finally:
        await client.aclose()

    # Aggregate metrics
    safety_rows = aggregate_within_panel_safety(verdicts)
    honesty_rows = aggregate_within_panel_honesty(verdicts)
    safety_alpha = krippendorffs_alpha(safety_rows) if all(len(v) >= 2 for v in safety_rows.values()) else None
    honesty_alpha = krippendorffs_alpha(honesty_rows) if all(len(v) >= 2 for v in honesty_rows.values()) else None

    report = render_report_markdown(
        date_str=date_str, cal_results=cal_results,
        safety_alpha=safety_alpha, honesty_alpha=honesty_alpha,
    )
    (out_dir / "REPORT.md").write_text(report)
    print(f"\nREPORT written -> {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
