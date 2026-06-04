"""Unit tests for verdict serialisation to disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judge_panel.types import Evidence, PanelMetadata, RoleOutput, Verdict
from judge_panel.verdicts import (
    load_verdict,
    render_verdict_markdown,
    verdict_paths,
    write_verdict,
)


def _make_verdict(run_id: str = "test") -> Verdict:
    from datetime import datetime, timezone
    def role(role_name, **f):
        defaults = dict(
            role=role_name, model="m", prompt_sha="s",
            reasoning="ok",
            evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
            confidence=0.8,
            duration_ms=100, input_tokens=100, output_tokens=20, cost_usd=0.001,
            raw_response={},
        )
        defaults.update(f)
        return RoleOutput(**defaults)
    return Verdict(
        metadata=PanelMetadata(
            run_id=run_id, panel_version="0.1.0", panel_commit_sha="abc",
            started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
            total_cost_usd=0.05, total_duration_ms=30000,
        ),
        final_safety_score=4, final_honesty_score=5,
        final_misalignment_level="L0", final_confidence=0.85,
        status="success",
        auditor=role("auditor", safety_score=4, honesty_score=5),
        detector=role("detector", safety_score=None, honesty_score=4, detector_risk_flag="low"),
        critic=role("critic", safety_score=3, honesty_score=4, critic_argument="x"),
        arbiter=role("arbiter", safety_score=4, honesty_score=5, misalignment_level="L0"),
        dissenter=role("dissenter", safety_score=None, honesty_score=None, evidence=[], dissent_flag=False),
        safety_score_variance=0.5, honesty_score_variance=0.3,
    )


def test_verdict_paths(tmp_path: Path):
    json_path, md_path = verdict_paths(tmp_path, "run-abc")
    assert json_path == tmp_path / "run-abc" / "panel_verdict.json"
    assert md_path == tmp_path / "run-abc" / "panel_verdict.md"


def test_write_and_load_verdict_roundtrip(tmp_path: Path):
    v = _make_verdict("rt")
    json_path, md_path = write_verdict(v, base_dir=tmp_path)
    assert json_path.exists()
    assert md_path.exists()
    loaded = load_verdict(tmp_path / "rt" / "panel_verdict.json")
    assert loaded.final_safety_score == 4
    assert loaded.metadata.run_id == "rt"


def test_markdown_contains_key_fields():
    v = _make_verdict()
    md = render_verdict_markdown(v)
    assert "Safety: 4/5" in md or "safety_score" in md.lower()
    assert "L0" in md
    assert "test" in md  # run_id


def test_force_regrade_writes_timestamped(tmp_path: Path):
    v = _make_verdict("rt")
    write_verdict(v, base_dir=tmp_path)
    # Second write with force_regrade=True creates a timestamped file
    json2, md2 = write_verdict(v, base_dir=tmp_path, force_regrade=True)
    assert json2.name != "panel_verdict.json"  # different filename
    assert json2.name.startswith("panel_verdict-")
    assert json2.suffix == ".json"
