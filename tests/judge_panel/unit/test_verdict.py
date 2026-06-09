"""Unit tests for PanelMetadata, Verdict, and CalibrationResult."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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


def _metadata() -> PanelMetadata:
    return PanelMetadata(
        run_id="test-run",
        panel_version="0.1.0",
        panel_commit_sha="deadbeef",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        total_cost_usd=0.05,
        total_duration_ms=30000,
    )


class TestPanelMetadata:
    def test_constructs(self):
        m = _metadata()
        assert m.run_id == "test-run"
        assert m.total_cost_usd == 0.05

    def test_total_cost_non_negative(self):
        with pytest.raises(ValidationError):
            PanelMetadata(
                run_id="x",
                panel_version="0.1.0",
                panel_commit_sha="s",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                total_cost_usd=-0.01,
                total_duration_ms=1,
            )


class TestVerdict:
    def test_constructs_success_verdict(self):
        v = Verdict(
            metadata=_metadata(),
            final_safety_score=4,
            final_honesty_score=5,
            final_misalignment_level="L0",
            final_confidence=0.85,
            status="success",
            auditor=_role("auditor"),
            detector=_role(
                "detector",
                safety_score=None,
                honesty_score=4,
                evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
            ),
            critic=_role("critic", safety_score=2, honesty_score=3),
            arbiter=_role("arbiter", misalignment_level="L1"),
            dissenter=_role(
                "dissenter", safety_score=None, honesty_score=None, evidence=[], dissent_flag=False
            ),
            safety_score_variance=0.5,
            honesty_score_variance=0.3,
        )
        assert v.status == "success"
        assert v.final_safety_score == 4

    def test_status_must_be_known_value(self):
        with pytest.raises(ValidationError):
            Verdict(
                metadata=_metadata(),
                final_safety_score=3,
                final_honesty_score=3,
                final_misalignment_level="L0",
                final_confidence=0.5,
                status="banana",
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


class TestCalibrationResult:
    def test_constructs(self):
        v = Verdict(
            metadata=_metadata(),
            final_safety_score=3,
            final_honesty_score=4,
            final_misalignment_level="L1",
            final_confidence=0.7,
            status="success",
            auditor=_role("auditor"),
            detector=_role(
                "detector",
                safety_score=None,
                honesty_score=4,
                evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
            ),
            critic=_role("critic"),
            arbiter=_role("arbiter", misalignment_level="L1"),
            dissenter=_role("dissenter", safety_score=None, honesty_score=None, evidence=[]),
            safety_score_variance=0,
            honesty_score_variance=0,
        )
        cal = CalibrationResult(
            verdict=v,
            gemini_safety_score=2,
            gemini_honesty_score=4,
            gemini_misalignment_level="L2",
            safety_score_delta=1,
            honesty_score_delta=0,
            misalignment_level_match=False,
            notes="panel disagreed on misalignment level",
        )
        assert cal.safety_score_delta == 1
        assert cal.misalignment_level_match is False
