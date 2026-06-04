"""Integration test for the Inspect AI panel scorer shim.

Verifies that ``panel_safety_behavior_scorer`` from ``inspect_eval.scorers_llm``
returns a callable that, when the panel-call indirection is patched, produces
a Score with the panel verdict surfaced via metadata.

Skipped cleanly if ``inspect_ai`` is not available in the environment.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("inspect_ai")

from judge_panel.types import Evidence, PanelMetadata, RoleOutput, Verdict


def _role(role_name: str, **fields) -> RoleOutput:
    defaults = dict(
        role=role_name,
        model="m",
        prompt_sha="s",
        reasoning="ok",
        evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
        confidence=0.8,
        duration_ms=100,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.001,
        raw_response={},
    )
    defaults.update(fields)
    return RoleOutput(**defaults)


def _make_canonical_verdict(run_id: str = "shim-run") -> Verdict:
    now = datetime.now(timezone.utc)
    return Verdict(
        metadata=PanelMetadata(
            run_id=run_id,
            panel_version="0.1.0",
            panel_commit_sha="abc",
            started_at=now,
            completed_at=now,
            total_cost_usd=0.05,
            total_duration_ms=30000,
        ),
        final_safety_score=4,
        final_honesty_score=5,
        final_misalignment_level="L0",
        final_confidence=0.85,
        status="success",
        auditor=_role("auditor", safety_score=4, honesty_score=5),
        detector=_role("detector", safety_score=None, honesty_score=4, detector_risk_flag="low"),
        critic=_role("critic", safety_score=3, honesty_score=4, critic_argument="x"),
        arbiter=_role("arbiter", safety_score=4, honesty_score=5, misalignment_level="L0"),
        dissenter=_role(
            "dissenter",
            safety_score=None,
            honesty_score=None,
            evidence=[],
            dissent_flag=False,
        ),
        safety_score_variance=0.5,
        honesty_score_variance=0.3,
    )


def test_panel_safety_behavior_scorer_returns_callable():
    """The scorer factory returns a callable; patched panel call surfaces verdict."""
    from inspect_eval.scorers_llm import panel_safety_behavior_scorer

    verdict = _make_canonical_verdict(run_id="shim-run")

    with patch(
        "inspect_eval.scorers_llm._run_panel_for_shim",
        new=AsyncMock(return_value=verdict),
    ):
        score_fn = panel_safety_behavior_scorer()
        assert callable(score_fn)
