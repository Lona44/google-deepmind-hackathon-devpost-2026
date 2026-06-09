"""Unit tests for calibration.render_report_markdown.

Critical contract: when a run's verdict.status == "error", the row in the
per-run comparison table must NOT present the orchestrator's placeholder
scores (typically safety=3, honesty=3, level=L0) as if they were real
measurements. Doing so would let downstream readers mistake a failed run
for a clean panel verdict — a research-integrity bug.
"""

from __future__ import annotations

from datetime import datetime, timezone

from judge_panel.calibration import render_report_markdown
from judge_panel.types import (
    CalibrationResult,
    Evidence,
    PanelMetadata,
    RoleOutput,
    Verdict,
)


def _role(role: str, **overrides) -> RoleOutput:
    defaults = {
        "role": role,
        "model": "m",
        "prompt_sha": "s",
        "safety_score": 3,
        "honesty_score": 3,
        "reasoning": "ok",
        "evidence": [Evidence(step_id=0, quote="x", interpretation="y")],
        "confidence": 0.7,
        "duration_ms": 1000,
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.001,
        "raw_response": {},
    }
    defaults.update(overrides)
    return RoleOutput(**defaults)


def _metadata(run_id: str = "run-1") -> PanelMetadata:
    return PanelMetadata(
        run_id=run_id,
        panel_version="0.1.0",
        panel_commit_sha="deadbeef",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        total_cost_usd=0.05,
        total_duration_ms=30000,
    )


def _verdict(status: str, run_id: str = "run-1") -> Verdict:
    return Verdict(
        metadata=_metadata(run_id),
        final_safety_score=3,
        final_honesty_score=3,
        final_misalignment_level="L0",
        final_confidence=0.5,
        status=status,  # type: ignore[arg-type]
        auditor=_role("auditor"),
        detector=_role(
            "detector",
            safety_score=None,
            honesty_score=3,
            evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
        ),
        critic=_role("critic"),
        arbiter=_role("arbiter"),
        dissenter=_role("dissenter", safety_score=None, honesty_score=None, evidence=[]),
        safety_score_variance=0,
        honesty_score_variance=0,
    )


def _cal(verdict: Verdict) -> CalibrationResult:
    return CalibrationResult(
        verdict=verdict,
        gemini_safety_score=2,
        gemini_honesty_score=5,
        gemini_misalignment_level="L2",
        safety_score_delta=verdict.final_safety_score - 2,
        honesty_score_delta=verdict.final_honesty_score - 5,
        misalignment_level_match=verdict.final_misalignment_level == "L2",
    )


class TestReportRenderingForErrorRuns:
    def test_header_includes_status_column(self):
        md = render_report_markdown(
            date_str="2026-06-04",
            cal_results=[_cal(_verdict("success"))],
            safety_alpha=None,
            honesty_alpha=None,
        )
        assert "| status |" in md or "status |" in md.split("\n")[5]

    def test_error_run_row_marks_status_and_suppresses_scores(self):
        md = render_report_markdown(
            date_str="2026-06-04",
            cal_results=[_cal(_verdict("error", run_id="failed-run"))],
            safety_alpha=None,
            honesty_alpha=None,
        )
        row = next(line for line in md.split("\n") if "failed-run" in line)

        assert "error" in row, "row must label the run as errored"
        cells = [c.strip() for c in row.split("|")[1:-1]]
        # cells: [run_id, status, panel_s, gem_s, Δs, panel_h, gem_h, Δh, panel_l, gem_l, match]
        panel_safety_cell = cells[2]
        panel_honesty_cell = cells[5]
        panel_level_cell = cells[8]
        delta_safety_cell = cells[4]
        delta_honesty_cell = cells[7]
        match_cell = cells[10]

        assert panel_safety_cell in {"—", "-", "n/a"}, (
            f"panel safety must NOT be a number for errored runs (got {panel_safety_cell!r})"
        )
        assert panel_honesty_cell in {"—", "-", "n/a"}
        assert panel_level_cell in {"—", "-", "n/a"}
        assert delta_safety_cell in {"—", "-", "n/a"}
        assert delta_honesty_cell in {"—", "-", "n/a"}
        assert match_cell in {"—", "-", "n/a"}

    def test_success_run_still_shows_scores(self):
        md = render_report_markdown(
            date_str="2026-06-04",
            cal_results=[_cal(_verdict("success", run_id="ok-run"))],
            safety_alpha=None,
            honesty_alpha=None,
        )
        row = next(line for line in md.split("\n") if "ok-run" in line)
        assert "success" in row
        cells = [c.strip() for c in row.split("|")[1:-1]]
        # status column shouldn't displace numeric scores
        assert cells[2] == "3", "panel safety should still render for success rows"
        assert cells[5] == "3", "panel honesty should still render for success rows"

    def test_partial_failure_renders_scores(self):
        """partial_failure runs DO produce real scores (Arbiter succeeded) — keep them."""
        md = render_report_markdown(
            date_str="2026-06-04",
            cal_results=[_cal(_verdict("partial_failure", run_id="partial-run"))],
            safety_alpha=None,
            honesty_alpha=None,
        )
        row = next(line for line in md.split("\n") if "partial-run" in line)
        cells = [c.strip() for c in row.split("|")[1:-1]]
        assert cells[2] == "3"
        assert cells[5] == "3"
