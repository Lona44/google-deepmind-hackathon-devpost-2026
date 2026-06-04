"""Unit tests for the RoleOutput pydantic type.

The verbatim-quote check (semantic validation against behavioral_data) is
in test_validation.py. The mandatory-evidence-when-score-given rule IS
enforced at the schema level here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from judge_panel.types import ErrorDetail, Evidence, RoleOutput


def _ev(n: int = 1) -> list[Evidence]:
    return [
        Evidence(step_id=i, quote=f"quote {i}", interpretation=f"interp {i}")
        for i in range(n)
    ]


class TestRoleOutputSchema:
    def test_minimal_auditor_output(self):
        out = RoleOutput(
            role="auditor",
            model="xiaomi/mimo-v2.5-pro",
            prompt_sha="abc123",
            safety_score=4,
            honesty_score=5,
            reasoning="The agent took the safe detour.",
            evidence=_ev(2),
            confidence=0.85,
            duration_ms=6800,
            input_tokens=12000,
            output_tokens=2500,
            cached_input_tokens=7000,
            cost_usd=0.0058,
            raw_response={"id": "test-123"},
        )
        assert out.role == "auditor"
        assert out.safety_score == 4
        assert out.misalignment_level is None  # not set by Auditor

    def test_safety_score_range_0_to_5(self):
        for bad in [-1, 6, 10]:
            with pytest.raises(ValidationError):
                RoleOutput(
                    role="auditor", model="m", prompt_sha="s", safety_score=bad,
                    honesty_score=3, reasoning="x", evidence=_ev(1), confidence=0.5,
                    duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                    raw_response={},
                )

    def test_honesty_score_range_1_to_5(self):
        for bad in [0, 6]:
            with pytest.raises(ValidationError):
                RoleOutput(
                    role="auditor", model="m", prompt_sha="s", safety_score=3,
                    honesty_score=bad, reasoning="x", evidence=_ev(1), confidence=0.5,
                    duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                    raw_response={},
                )

    def test_confidence_must_be_in_0_to_1(self):
        for bad in [-0.1, 1.1, 2.0]:
            with pytest.raises(ValidationError):
                RoleOutput(
                    role="auditor", model="m", prompt_sha="s", reasoning="x",
                    evidence=[], confidence=bad,
                    duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                    raw_response={},
                )

    def test_misalignment_level_only_allows_L0_through_L4(self):
        with pytest.raises(ValidationError):
            RoleOutput(
                role="arbiter", model="m", prompt_sha="s", safety_score=3,
                honesty_score=3, misalignment_level="L5",
                reasoning="x", evidence=_ev(1), confidence=0.5,
                duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                raw_response={},
            )

    def test_role_must_be_one_of_five(self):
        with pytest.raises(ValidationError):
            RoleOutput(
                role="referee", model="m", prompt_sha="s", reasoning="x",
                evidence=[], confidence=0.5,
                duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                raw_response={},
            )

    def test_evidence_required_when_safety_score_set(self):
        with pytest.raises(ValidationError, match="evidence"):
            RoleOutput(
                role="auditor", model="m", prompt_sha="s", safety_score=3,
                honesty_score=3, reasoning="x", evidence=[], confidence=0.5,
                duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
                raw_response={},
            )

    def test_evidence_not_required_for_dissenter(self):
        """The Dissenter does not produce numerical scores so it never needs evidence."""
        out = RoleOutput(
            role="dissenter", model="m", prompt_sha="s",
            reasoning="The arbiter's verdict looks sound.",
            evidence=[], confidence=0.9,
            dissent_flag=False,
            duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
            raw_response={},
        )
        assert out.dissent_flag is False

    def test_error_detail_optional(self):
        out = RoleOutput(
            role="auditor", model="m", prompt_sha="s",
            reasoning="failed", evidence=[], confidence=0.0,
            duration_ms=1, input_tokens=1, output_tokens=1, cost_usd=0.0,
            raw_response={},
            error=ErrorDetail(kind="api_timeout", message="role timed out after 120s"),
        )
        assert out.error.kind == "api_timeout"
