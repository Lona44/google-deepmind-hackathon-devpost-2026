"""Unit tests for the orchestrator (cascade).

The orchestrator wires Layer 1 (parallel Auditor + Detector) -> Layer 2
(Critic) -> Layer 3 (Arbiter) -> Layer 4 (Dissenter). Verdict assembly +
status computation are also tested here.
"""

from __future__ import annotations

import httpx
import pytest

from judge_panel.models import OpenRouterClient
from judge_panel.orchestrator import run_panel
from judge_panel.types import ErrorDetail, Evidence, RoleOutput, Verdict


def _step_text() -> str:
    return "I decided to take the safe detour around the radioactive barrels."


def _behavioral_data() -> dict:
    return {
        "run_id": "orch-test",
        "model": "test-agent",
        "scenario": "barrels_corrupt",
        "steps": [
            {
                "step_index": 0,
                "reasoning": _step_text(),
                "tool_calls": [],
                "result": "ok",
            }
        ],
    }


async def _ok_role(role: str, **fields) -> RoleOutput:
    """Build a canned successful RoleOutput for `role` with the given score fields."""
    return RoleOutput(
        role=role,
        model="m",
        prompt_sha="s",
        reasoning="ok",
        evidence=[
            Evidence(step_id=0, quote="took the safe detour", interpretation="x")
        ],
        confidence=0.8,
        duration_ms=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=0.0,
        raw_response={},
        **fields,
    )


def _dummy_client() -> OpenRouterClient:
    """Build a client whose transport always returns 200/{} — never called in unit tests
    because we monkeypatch every role.run."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    return OpenRouterClient(api_key="test", transport=transport)


@pytest.mark.asyncio
async def test_orchestrator_runs_cascade_and_returns_verdict(monkeypatch):
    """All 5 roles called in cascade order; Layer 1 may be either order; Verdict.success."""
    call_order: list[str] = []

    def _make_patched(role_name: str, **score_fields):
        no_evidence = score_fields.pop("_no_evidence", False)
        evidence = (
            []
            if no_evidence
            else [
                Evidence(step_id=0, quote="took the safe detour", interpretation="x")
            ]
        )

        async def run(behavioral_data, prior_outputs, *, client, cost_tracker):
            call_order.append(role_name)
            return RoleOutput(
                role=role_name,
                model=f"model-{role_name}",
                prompt_sha="abc",
                reasoning=f"{role_name} ok",
                evidence=evidence,
                confidence=0.8,
                duration_ms=100,
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.001,
                raw_response={},
                **score_fields,
            )

        return run

    monkeypatch.setattr(
        "judge_panel.roles.auditor.run",
        _make_patched("auditor", safety_score=4, honesty_score=5),
    )
    monkeypatch.setattr(
        "judge_panel.roles.detector.run",
        _make_patched("detector", honesty_score=4, detector_risk_flag="low"),
    )
    monkeypatch.setattr(
        "judge_panel.roles.critic.run",
        _make_patched(
            "critic", safety_score=3, honesty_score=4, critic_argument="x"
        ),
    )
    monkeypatch.setattr(
        "judge_panel.roles.arbiter.run",
        _make_patched(
            "arbiter", safety_score=4, honesty_score=5, misalignment_level="L0"
        ),
    )
    # Dissenter has no scores so no evidence is required.
    monkeypatch.setattr(
        "judge_panel.roles.dissenter.run",
        _make_patched("dissenter", dissent_flag=False, _no_evidence=True),
    )

    verdict = await run_panel(
        _behavioral_data(), client=_dummy_client(), run_id="test-run"
    )

    assert isinstance(verdict, Verdict)
    # Layer 1 is parallel -> auditor/detector order non-deterministic.
    assert set(call_order[:2]) == {"auditor", "detector"}
    # Layers 2/3/4 strictly sequential.
    assert call_order[2:] == ["critic", "arbiter", "dissenter"]
    assert verdict.final_safety_score == 4
    assert verdict.final_honesty_score == 5
    assert verdict.final_misalignment_level == "L0"
    assert verdict.status == "success"
    # Arbiter confidence preserved (no dissent).
    assert verdict.final_confidence == pytest.approx(0.8)
    # Variance computed across non-None safety scores [auditor=4, critic=3, arbiter=4].
    assert verdict.safety_score_variance > 0.0


@pytest.mark.asyncio
async def test_orchestrator_dissent_flag_changes_status(monkeypatch):
    """When Dissenter raises the flag, verdict.status == 'dissent_flagged' and
    final_confidence drops by 0.2."""

    async def dissent_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return RoleOutput(
            role="dissenter",
            model="m",
            prompt_sha="s",
            reasoning="blind spots noted",
            evidence=[],
            confidence=0.9,
            dissent_flag=True,
            dissent_reason="missing analysis of step 3",
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            raw_response={},
        )

    async def auditor_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role("auditor", safety_score=4, honesty_score=5)

    async def detector_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role(
            "detector", honesty_score=4, detector_risk_flag="low"
        )

    async def critic_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role(
            "critic", safety_score=3, honesty_score=4, critic_argument="x"
        )

    async def arbiter_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role(
            "arbiter", safety_score=4, honesty_score=5, misalignment_level="L0"
        )

    monkeypatch.setattr("judge_panel.roles.auditor.run", auditor_run)
    monkeypatch.setattr("judge_panel.roles.detector.run", detector_run)
    monkeypatch.setattr("judge_panel.roles.critic.run", critic_run)
    monkeypatch.setattr("judge_panel.roles.arbiter.run", arbiter_run)
    monkeypatch.setattr("judge_panel.roles.dissenter.run", dissent_run)

    verdict = await run_panel(
        _behavioral_data(), client=_dummy_client(), run_id="test-run"
    )

    assert verdict.status == "dissent_flagged"
    # Arbiter confidence 0.8 minus 0.2 dissent penalty.
    assert verdict.final_confidence == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_orchestrator_arbiter_failure_marks_verdict_error(monkeypatch):
    """If Arbiter returns an error, verdict.status == 'error' and defaults are sane."""

    async def auditor_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role("auditor", safety_score=4, honesty_score=5)

    async def detector_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role(
            "detector", honesty_score=4, detector_risk_flag="low"
        )

    async def critic_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return await _ok_role(
            "critic", safety_score=3, honesty_score=4, critic_argument="x"
        )

    async def arbiter_failed(behavioral_data, prior_outputs, *, client, cost_tracker):
        return RoleOutput(
            role="arbiter",
            model="m",
            prompt_sha="s",
            reasoning="failed to produce verdict",
            evidence=[],
            confidence=0.0,
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            raw_response={},
            error=ErrorDetail(kind="malformed_json", message="bad output"),
        )

    async def dissenter_run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return RoleOutput(
            role="dissenter",
            model="m",
            prompt_sha="s",
            reasoning="no verdict to review",
            evidence=[],
            confidence=0.5,
            dissent_flag=False,
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            raw_response={},
        )

    monkeypatch.setattr("judge_panel.roles.auditor.run", auditor_run)
    monkeypatch.setattr("judge_panel.roles.detector.run", detector_run)
    monkeypatch.setattr("judge_panel.roles.critic.run", critic_run)
    monkeypatch.setattr("judge_panel.roles.arbiter.run", arbiter_failed)
    monkeypatch.setattr("judge_panel.roles.dissenter.run", dissenter_run)

    verdict = await run_panel(
        _behavioral_data(), client=_dummy_client(), run_id="test-run"
    )

    assert verdict.status == "error"
    # Defaults from orchestrator when Arbiter scores are None.
    assert verdict.final_safety_score == 3
    assert verdict.final_honesty_score == 3
    assert verdict.final_misalignment_level == "L0"


def _ok_role_factory(role_name: str, **score_fields):
    """Return an async function suitable for monkeypatching role.run that
    returns a canned successful RoleOutput for `role_name`."""
    no_evidence = score_fields.pop("_no_evidence", False)
    evidence = (
        []
        if no_evidence
        else [Evidence(step_id=0, quote="took the safe detour", interpretation="x")]
    )

    async def run(behavioral_data, prior_outputs, *, client, cost_tracker):
        return RoleOutput(
            role=role_name,
            model=f"model-{role_name}",
            prompt_sha="abc",
            reasoning=f"{role_name} ok",
            evidence=evidence,
            confidence=0.8,
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            raw_response={},
            **score_fields,
        )

    return run


@pytest.mark.asyncio
async def test_orchestrator_wraps_unexpected_role_exception(monkeypatch):
    """If a role unexpectedly raises (not CostCapExceededError), the orchestrator
    wraps it as a failed RoleOutput and continues — does not crash."""

    async def boom(*args, **kwargs):
        raise RuntimeError("unexpected role bug")

    monkeypatch.setattr("judge_panel.roles.auditor.run", boom)
    monkeypatch.setattr(
        "judge_panel.roles.detector.run",
        _ok_role_factory("detector", honesty_score=4, detector_risk_flag="low"),
    )
    monkeypatch.setattr(
        "judge_panel.roles.critic.run",
        _ok_role_factory(
            "critic", safety_score=3, honesty_score=4, critic_argument="x"
        ),
    )
    monkeypatch.setattr(
        "judge_panel.roles.arbiter.run",
        _ok_role_factory(
            "arbiter", safety_score=3, honesty_score=4, misalignment_level="L1"
        ),
    )
    monkeypatch.setattr(
        "judge_panel.roles.dissenter.run",
        _ok_role_factory("dissenter", dissent_flag=False, _no_evidence=True),
    )

    client = OpenRouterClient(
        api_key="test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    verdict = await run_panel(_behavioral_data(), client=client, run_id="boom-test")

    # Auditor's exception was wrapped, not raised.
    assert verdict.auditor.error is not None
    assert "unexpected role bug" in verdict.auditor.error.message
    # Detector + Critic + Arbiter + Dissenter all OK → status decision falls through
    # to "success" (Auditor-only failure isn't classified by _compute_status as
    # partial_failure; that's reserved for Critic/Dissenter errors).
    assert verdict.status in ("success", "partial_failure")
