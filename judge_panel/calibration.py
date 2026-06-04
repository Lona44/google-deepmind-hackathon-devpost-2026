"""Calibration utilities: load behavioral_data, load Gemini baselines,
compute agreement metrics, emit reports.

Tier C of the test strategy. Lives in judge_panel/ rather than
tests/judge_panel/calibration/ so it's importable from anywhere; the
actual CLI runner is in tests/judge_panel/calibration/run_calibration.py.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from judge_panel.types import CalibrationResult, Verdict


def load_behavioral_data(path: Path) -> dict:
    return json.loads(path.read_text())


def load_gemini_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compute_calibration_result(verdict: Verdict, gemini: dict, notes: str = "") -> CalibrationResult:
    """Build a CalibrationResult comparing one Verdict to one Gemini baseline."""
    return CalibrationResult(
        verdict=verdict,
        gemini_safety_score=gemini["safety_score"],
        gemini_honesty_score=gemini["honesty_score"],
        gemini_misalignment_level=gemini["misalignment_level"],
        safety_score_delta=verdict.final_safety_score - gemini["safety_score"],
        honesty_score_delta=verdict.final_honesty_score - gemini["honesty_score"],
        misalignment_level_match=verdict.final_misalignment_level == gemini["misalignment_level"],
        notes=notes,
    )


def aggregate_within_panel_safety(verdicts: list[Verdict]) -> dict:
    """Per-judge safety scores aggregated across multiple verdicts.

    Returns a dict {judge_name: [scores...]} suitable for feeding into
    a Krippendorff's alpha calculator.
    """
    rows = {"auditor": [], "critic": [], "arbiter": []}
    for v in verdicts:
        if v.auditor.safety_score is not None:
            rows["auditor"].append(v.auditor.safety_score)
        if v.critic.safety_score is not None:
            rows["critic"].append(v.critic.safety_score)
        if v.arbiter.safety_score is not None:
            rows["arbiter"].append(v.arbiter.safety_score)
    return rows


def aggregate_within_panel_honesty(verdicts: list[Verdict]) -> dict:
    rows = {"auditor": [], "detector": [], "critic": [], "arbiter": []}
    for v in verdicts:
        if v.auditor.honesty_score is not None:
            rows["auditor"].append(v.auditor.honesty_score)
        if v.detector.honesty_score is not None:
            rows["detector"].append(v.detector.honesty_score)
        if v.critic.honesty_score is not None:
            rows["critic"].append(v.critic.honesty_score)
        if v.arbiter.honesty_score is not None:
            rows["arbiter"].append(v.arbiter.honesty_score)
    return rows


def render_report_markdown(
    *,
    date_str: str,
    cal_results: list[CalibrationResult],
    safety_alpha: float | None,
    honesty_alpha: float | None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Calibration report — {date_str}")
    lines.append("")
    lines.append(f"Inputs: {len(cal_results)} runs. Panel: 5-role RADAR cascade (MiMo + Kimi K2.6).")
    lines.append("")
    lines.append("## Per-run comparison vs Gemini 3 Pro baseline")
    lines.append("")
    lines.append("| run_id | panel safety | gemini safety | Δ | panel honesty | gemini honesty | Δ | panel level | gemini level | match |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cr in cal_results:
        v = cr.verdict
        lines.append(
            f"| {v.metadata.run_id} | {v.final_safety_score} | {cr.gemini_safety_score} | {cr.safety_score_delta:+d} | "
            f"{v.final_honesty_score} | {cr.gemini_honesty_score} | {cr.honesty_score_delta:+d} | "
            f"{v.final_misalignment_level} | {cr.gemini_misalignment_level} | {'✓' if cr.misalignment_level_match else '✗'} |"
        )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    if safety_alpha is not None:
        lines.append(f"- Within-panel safety Krippendorff's α (Auditor/Critic/Arbiter): **{safety_alpha:.3f}**")
    else:
        lines.append("- Within-panel safety Krippendorff's α: not computable (need ≥2 runs)")
    if honesty_alpha is not None:
        lines.append(f"- Within-panel honesty Krippendorff's α (Auditor/Detector/Critic/Arbiter): **{honesty_alpha:.3f}**")
    else:
        lines.append("- Within-panel honesty Krippendorff's α: not computable")
    lines.append("")
    lines.append("## Acknowledged limits")
    lines.append("")
    lines.append(
        "Agreement with the Gemini baseline is a CONCORDANCE metric, not a TRUTH metric. "
        "If Gemini's verdict was wrong on a run, the panel agreeing with it is not evidence "
        "of panel correctness. See judge_panel design spec Section 5.7."
    )
    lines.append("")
    return "\n".join(lines)
