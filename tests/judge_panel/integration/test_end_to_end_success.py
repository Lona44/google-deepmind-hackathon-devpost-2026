"""End-to-end integration test of the full cascade with recorded responses.

NO live API calls. Uses httpx.MockTransport with a per-role response router
that inspects the prompt content for role-identifying markdown headers
(multiple roles share a model, so per-model routing is insufficient).

The verbatim quote used in every canned response is a real substring of
step 0's `result` text in the canonical fixture, so it survives the
verbatim-quote semantic check in judge_panel.validation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from judge_panel.models import OpenRouterClient
from judge_panel.orchestrator import run_panel

# Verbatim substring of canonical fixture step 0's `result`:
#   "Selected safe detour path at y=-1.6 to avoid barrel contact despite battery risk."
_QUOTE = "safe detour path"


def _payload_for(role: str) -> str:
    """Canonical valid payload per role with a verbatim quote from step 0."""
    common_evidence = [{"step_id": 0, "quote": _QUOTE, "interpretation": role}]
    if role == "auditor":
        return json.dumps(
            {
                "safety_score": 4,
                "honesty_score": 5,
                "reasoning": f"{role} verdict",
                "evidence": common_evidence,
                "confidence": 0.85,
            }
        )
    if role == "detector":
        return json.dumps(
            {
                "honesty_score": 4,
                "detector_risk_flag": "low",
                "reasoning": f"{role} verdict",
                "evidence": common_evidence,
                "confidence": 0.8,
            }
        )
    if role == "critic":
        return json.dumps(
            {
                "safety_score": 3,
                "honesty_score": 4,
                "critic_argument": "could be tighter",
                "reasoning": f"{role} verdict",
                "evidence": common_evidence,
                "confidence": 0.7,
            }
        )
    if role == "arbiter":
        return json.dumps(
            {
                "safety_score": 4,
                "honesty_score": 5,
                "misalignment_level": "L0",
                "reasoning": f"{role} verdict",
                "evidence": common_evidence,
                "confidence": 0.85,
            }
        )
    if role == "dissenter":
        return json.dumps(
            {
                "dissent_flag": False,
                "dissent_reason": "",
                "reasoning": f"{role} verdict",
                "confidence": 0.9,
            }
        )
    raise ValueError(role)


# Role -> identifying header in the role's prompt markdown.
_ROLE_HEADERS: tuple[tuple[str, str], ...] = (
    ("auditor", "# Security Criterion Auditor"),
    ("detector", "# Vulnerability Detector"),
    ("critic", "# Counterargument Critic"),
    ("arbiter", "# Holistic Arbiter"),
    ("dissenter", "# Dissenting Reviewer"),
)


def _routing_transport() -> httpx.AsyncBaseTransport:
    """Route requests by inspecting the prompt content — the cached prefix
    contains the role's prompt markdown which mentions the role name."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        for role, header in _ROLE_HEADERS:
            if header in content:
                payload = _payload_for(role)
                model = body["model"]
                return httpx.Response(
                    200,
                    json={
                        "id": "x",
                        "model": model,
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": payload},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 6000,
                            "completion_tokens": 800,
                            "total_tokens": 6800,
                            "prompt_tokens_details": {"cached_tokens": 5000},
                        },
                    },
                )
        raise ValueError("could not route request: prompt did not match any role")

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_full_cascade_success(canonical_behavioral_data):
    """Run the full panel on the canonical fixture and verify Verdict shape."""
    client = OpenRouterClient(api_key="test", transport=_routing_transport())
    try:
        verdict = await run_panel(canonical_behavioral_data, client=client, run_id="e2e-test")
        assert verdict.status == "success"
        assert verdict.final_safety_score == 4
        assert verdict.final_honesty_score == 5
        assert verdict.final_misalignment_level == "L0"
        assert all(
            role.error is None
            for role in (
                verdict.auditor,
                verdict.detector,
                verdict.critic,
                verdict.arbiter,
                verdict.dissenter,
            )
        )
        # Cost should be roughly 5 x per-role cost (~$0.05 total per spec)
        assert 0.01 < verdict.metadata.total_cost_usd < 0.20
    finally:
        await client.aclose()
