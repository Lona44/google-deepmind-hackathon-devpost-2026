"""End-to-end integration tests for failure paths.

NO live API calls. Uses httpx.MockTransport with a per-role response router
that inspects the prompt content for role-identifying markdown headers.

Covers:
- Hallucinated-quote semantic failure in the Auditor: parse-time validation
  rejects the response; after retry the role is marked failed with an error,
  but the cascade continues past Layer 1 (Detector still succeeds).
- Cost cap exceeded mid-cascade: tiny per-experiment budget causes the very
  first role's cost charge to blow the cap, raising CostCapExceededError.
"""

from __future__ import annotations

import json

import httpx
import pytest

from judge_panel.cost_tracker import CostCapExceededError, PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.orchestrator import run_panel

# Verbatim substring of canonical fixture step 0's `result` text:
#   "Selected safe detour path at y=-1.6 to avoid barrel contact despite battery risk."
_QUOTE = "safe detour path"


# Role -> identifying header in the role's prompt markdown (deterministic).
_ROLE_HEADERS: tuple[tuple[str, str], ...] = (
    ("auditor", "# Security Criterion Auditor"),
    ("detector", "# Vulnerability Detector"),
    ("critic", "# Counterargument Critic"),
    ("arbiter", "# Holistic Arbiter"),
    ("dissenter", "# Dissenting Reviewer"),
)


def _detect_role(content: str) -> str:
    for role, header in _ROLE_HEADERS:
        if header in content:
            return role
    raise ValueError("could not route request: prompt did not match any role")


def _hallucinated_auditor_payload() -> str:
    """Auditor response with a quote that does NOT appear in the fixture —
    semantic validation must reject this at parse time."""
    return json.dumps(
        {
            "safety_score": 4,
            "honesty_score": 5,
            "reasoning": "auditor verdict",
            "evidence": [
                {"step_id": 0, "quote": "completely fabricated quote", "interpretation": "x"}
            ],
            "confidence": 0.8,
        }
    )


def _clean_payload(role: str) -> str:
    """Valid per-role payload citing a verbatim substring of fixture step 0."""
    common_evidence = [{"step_id": 0, "quote": _QUOTE, "interpretation": role}]
    if role == "auditor":
        return json.dumps(
            {
                "safety_score": 4,
                "honesty_score": 5,
                "reasoning": "auditor verdict",
                "evidence": common_evidence,
                "confidence": 0.85,
            }
        )
    if role == "detector":
        return json.dumps(
            {
                "honesty_score": 4,
                "detector_risk_flag": "low",
                "reasoning": "detector verdict",
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
                "reasoning": "critic verdict",
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
                "reasoning": "arbiter verdict",
                "evidence": common_evidence,
                "confidence": 0.85,
            }
        )
    if role == "dissenter":
        return json.dumps(
            {
                "dissent_flag": False,
                "dissent_reason": "",
                "reasoning": "dissenter verdict",
                "confidence": 0.9,
            }
        )
    raise ValueError(role)


@pytest.mark.asyncio
async def test_hallucinated_quote_failure_in_auditor_marks_role_failed(
    canonical_behavioral_data,
):
    """Auditor returns a hallucinated quote.

    Expected behaviour:
    - Semantic verbatim-quote check rejects the response at parse time.
    - Retry yields the same hallucinated response (deterministic transport).
    - Auditor RoleOutput is finalised with error set.
    - Cascade continues: Detector (a Layer 1 peer of Auditor) still succeeds.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        role = _detect_role(content)
        payload = _hallucinated_auditor_payload() if role == "auditor" else _clean_payload(role)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": body["model"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": payload},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5000,
                    "completion_tokens": 500,
                    "total_tokens": 5500,
                    "prompt_tokens_details": {"cached_tokens": 4000},
                },
            },
        )

    client = OpenRouterClient(api_key="test", transport=httpx.MockTransport(handler))
    try:
        verdict = await run_panel(canonical_behavioral_data, client=client, run_id="bad-quote")
        # Auditor was marked failed because the quote could not be verified
        # against the canonical fixture text.
        assert verdict.auditor.error is not None
        assert "quote" in verdict.auditor.error.message.lower()
        # Cascade continued past Layer 1: Detector ran cleanly.
        assert verdict.detector.error is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cost_cap_aborts_cascade(canonical_behavioral_data):
    """A tiny per-experiment cost cap means the very first role's charge
    blows the budget and the orchestrator raises CostCapExceededError."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        role = _detect_role(content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": body["model"],
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": _clean_payload(role),
                        },
                        "finish_reason": "stop",
                    }
                ],
                # Huge tokens with NO caching -> expensive single call.
                "usage": {
                    "prompt_tokens": 100000,
                    "completion_tokens": 50000,
                    "total_tokens": 150000,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )

    client = OpenRouterClient(api_key="test", transport=httpx.MockTransport(handler))
    tiny_tracker = PerExperimentTracker(max_cost_usd=0.01)
    try:
        with pytest.raises(CostCapExceededError):
            await run_panel(
                canonical_behavioral_data,
                client=client,
                run_id="capped",
                cost_tracker=tiny_tracker,
            )
    finally:
        await client.aclose()
