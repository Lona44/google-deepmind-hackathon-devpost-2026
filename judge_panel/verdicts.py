"""Verdict serialisation: JSON for forensic completeness + markdown for humans.

Output layout under base_dir:
    verdicts/<run_id>/panel_verdict.json      <- canonical
    verdicts/<run_id>/panel_verdict.md        <- human summary

When force_regrade=True, new writes go to timestamped filenames so prior
verdicts are never overwritten (spec Section 3.5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from judge_panel.types import Verdict


def verdict_paths(base_dir: Path, run_id: str) -> tuple[Path, Path]:
    """Return (json_path, md_path) for a run_id."""
    run_dir = base_dir / run_id
    return run_dir / "panel_verdict.json", run_dir / "panel_verdict.md"


def render_verdict_markdown(verdict: Verdict) -> str:
    """Human-readable summary of a verdict.

    Includes final scores, status, per-role one-line summary, within-panel
    variance, and a Dissenting Reviewer note if flagged.
    """
    lines: list[str] = []
    m = verdict.metadata
    lines.append(f"# Panel verdict — {m.run_id}")
    lines.append("")
    lines.append(f"- Status: **{verdict.status}**")
    lines.append(f"- Safety: {verdict.final_safety_score}/5")
    lines.append(f"- Honesty: {verdict.final_honesty_score}/5")
    lines.append(f"- Misalignment level: {verdict.final_misalignment_level}")
    lines.append(f"- Confidence: {verdict.final_confidence:.2f}")
    lines.append(f"- Cost: ${m.total_cost_usd:.4f}")
    lines.append(f"- Wall-clock: {m.total_duration_ms / 1000:.1f}s")
    lines.append("")
    lines.append("## Per-role summary")
    lines.append("")
    for role_out in (verdict.auditor, verdict.detector, verdict.critic, verdict.arbiter, verdict.dissenter):
        status = "FAILED" if role_out.error else "ok"
        score = ""
        if role_out.safety_score is not None or role_out.honesty_score is not None:
            score = f" — safety={role_out.safety_score} honesty={role_out.honesty_score}"
        lines.append(f"- **{role_out.role}** ({role_out.model}, {status}){score}")
    lines.append("")
    lines.append("## Within-panel variance")
    lines.append("")
    lines.append(f"- Safety variance (Auditor / Critic / Arbiter): {verdict.safety_score_variance:.2f}")
    lines.append(f"- Honesty variance (Auditor / Detector / Critic / Arbiter): {verdict.honesty_score_variance:.2f}")
    if verdict.dissenter.dissent_flag:
        lines.append("")
        lines.append("## Dissenting Reviewer flag raised")
        lines.append("")
        lines.append(verdict.dissenter.dissent_reason or "(no reason given)")
    return "\n".join(lines) + "\n"


def write_verdict(
    verdict: Verdict,
    *,
    base_dir: Path,
    force_regrade: bool = False,
) -> tuple[Path, Path]:
    """Write verdict JSON + markdown. Returns (json_path, md_path)."""
    run_id = verdict.metadata.run_id
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if force_regrade:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        json_path = run_dir / f"panel_verdict-{ts}.json"
        md_path = run_dir / f"panel_verdict-{ts}.md"
    else:
        json_path = run_dir / "panel_verdict.json"
        md_path = run_dir / "panel_verdict.md"

    json_path.write_text(verdict.model_dump_json(indent=2))
    md_path.write_text(render_verdict_markdown(verdict))
    return json_path, md_path


def load_verdict(json_path: Path) -> Verdict:
    return Verdict.model_validate_json(json_path.read_text())
