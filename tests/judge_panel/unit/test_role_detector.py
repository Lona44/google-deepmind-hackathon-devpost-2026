"""Unit tests for the Detector role.

Uses MockTransport to inject canned model responses. The Detector's job:
find implicit failure modes the literal Auditor would miss; produces
honesty_score (not safety) plus detector_risk_flag.
"""

from __future__ import annotations

import json

import httpx
import pytest

from judge_panel.cost_tracker import PerExperimentTracker
from judge_panel.models import OpenRouterClient
from judge_panel.roles.detector import run as detector_run


def _step_text() -> str:
    return "I decided to take the safe detour around the radioactive barrels despite low battery."


def _behavioral_data() -> dict:
    return {
        "run_id": "test-run",
        "model": "test-agent",
        "scenario": "barrels_corrupt",
        "steps": [
            {"step_index": 0, "reasoning": _step_text(), "tool_calls": [], "result": "ok"}
        ],
    }


def _mock_response(json_body: str, model: str = "moonshotai/kimi-k2.6") -> dict:
    return {
        "id": "test-1",
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": json_body}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5000, "completion_tokens": 800, "total_tokens": 5800,
                  "prompt_tokens_details": {"cached_tokens": 4000}},
    }


def _mock_transport(content: str, model: str = "moonshotai/kimi-k2.6") -> httpx.MockTransport:
    body = _mock_response(content, model)
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


@pytest.fixture
def cost_tracker() -> PerExperimentTracker:
    return PerExperimentTracker(max_cost_usd=0.50)


@pytest.mark.asyncio
async def test_detector_returns_parsed_role_output(cost_tracker):
    content = json.dumps({
        "honesty_score": 4,
        "detector_risk_flag": "low",
        "reasoning": "Agent's report tracks its actions; no implicit failure mode surfaced.",
        "evidence": [
            {"step_id": 0, "quote": "I decided to take the safe detour", "interpretation": "matches stated reasoning"}
        ],
        "confidence": 0.8,
    })
    client = OpenRouterClient(api_key="test", transport=_mock_transport(content))
    out = await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert out.role == "detector"
    assert out.honesty_score == 4
    # Detector does NOT produce safety_score or misalignment_level.
    assert out.safety_score is None
    assert out.misalignment_level is None
    assert len(out.evidence) == 1
    assert out.evidence[0].quote == "I decided to take the safe detour"


@pytest.mark.asyncio
async def test_detector_returns_risk_flag(cost_tracker):
    content = json.dumps({
        "honesty_score": 3,
        "detector_risk_flag": "medium",
        "reasoning": "Subtle hedging in step 0 could mask a deferred risky action.",
        "evidence": [
            {"step_id": 0, "quote": "take the safe detour", "interpretation": "framing is suspiciously confident"}
        ],
        "confidence": 0.6,
    })
    client = OpenRouterClient(api_key="test", transport=_mock_transport(content))
    out = await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert out.detector_risk_flag == "medium"


@pytest.mark.asyncio
async def test_detector_strips_markdown_fences(cost_tracker):
    fenced = "```json\n" + json.dumps({
        "honesty_score": 4,
        "detector_risk_flag": "low",
        "reasoning": "ok",
        "evidence": [{"step_id": 0, "quote": "take the safe detour", "interpretation": "x"}],
        "confidence": 0.7,
    }) + "\n```"
    client = OpenRouterClient(api_key="test", transport=_mock_transport(fenced))
    out = await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert out.honesty_score == 4
    assert out.detector_risk_flag == "low"


@pytest.mark.asyncio
async def test_detector_records_cost_in_tracker(cost_tracker):
    content = json.dumps({
        "honesty_score": 4,
        "detector_risk_flag": "low",
        "reasoning": "ok",
        "evidence": [{"step_id": 0, "quote": "take the safe detour", "interpretation": "x"}],
        "confidence": 0.8,
    })
    client = OpenRouterClient(api_key="test", transport=_mock_transport(content))
    await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert "detector" in cost_tracker.per_role_costs
    assert cost_tracker.per_role_costs["detector"] > 0


@pytest.mark.asyncio
async def test_detector_fails_semantically_on_hallucinated_quote(cost_tracker):
    """When the model cites a quote not present in behavioral_data, the role
    returns a RoleOutput with error set rather than raising."""
    content = json.dumps({
        "honesty_score": 4,
        "detector_risk_flag": "low",
        "reasoning": "ok",
        "evidence": [{"step_id": 0, "quote": "completely fabricated text", "interpretation": "x"}],
        "confidence": 0.8,
    })
    client = OpenRouterClient(api_key="test", transport=_mock_transport(content))
    out = await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert out.error is not None
    assert "quote" in out.error.message.lower()


@pytest.mark.asyncio
async def test_detector_uses_kimi_model(cost_tracker):
    content = json.dumps({
        "honesty_score": 3,
        "detector_risk_flag": "low",
        "reasoning": "ok",
        "evidence": [{"step_id": 0, "quote": "take the safe detour", "interpretation": "x"}],
        "confidence": 0.7,
    })
    captured_request = {}
    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(200, json=_mock_response(content))
    client = OpenRouterClient(api_key="test", transport=httpx.MockTransport(handler))
    await detector_run(_behavioral_data(), prior_outputs=[], client=client, cost_tracker=cost_tracker)
    assert captured_request["body"]["model"] == "moonshotai/kimi-k2.6"
