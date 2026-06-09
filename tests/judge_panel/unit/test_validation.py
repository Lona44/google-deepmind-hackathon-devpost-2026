"""Unit tests for the validation pipeline.

Stage 1 (schema): pydantic ValidationError detection -> retry with format
correction prompt. Tested in test_role_output.py too.

Stage 2 (semantic): verbatim quote check against behavioral_data, step_id
validity, score ranges sanity. The verbatim check is THE load-bearing rule
that kills hallucinated evidence.
"""

from __future__ import annotations

import pytest

from judge_panel.types import Evidence, RoleOutput
from judge_panel.validation import SemanticValidationError, validate_role_output


def _step(idx: int, text: str) -> dict:
    return {"step_index": idx, "reasoning": text, "tool_calls": [], "result": "ok"}


def _behavioral_data(*step_texts: str) -> dict:
    return {
        "run_id": "test",
        "model": "test",
        "scenario": "test",
        "steps": [_step(i, t) for i, t in enumerate(step_texts)],
    }


def _role(**overrides) -> RoleOutput:
    defaults = {
        "role": "auditor",
        "model": "m",
        "prompt_sha": "s",
        "safety_score": 4,
        "honesty_score": 5,
        "reasoning": "ok",
        "evidence": [],
        "confidence": 0.8,
        "duration_ms": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "cost_usd": 0.0,
        "raw_response": {},
    }
    defaults.update(overrides)
    return RoleOutput(**defaults)


class TestVerbatimQuoteCheck:
    def test_passes_when_quote_appears_in_step_text(self):
        data = _behavioral_data("took the safe detour around the barrels")
        out = _role(
            evidence=[Evidence(step_id=0, quote="took the safe detour", interpretation="aligned")]
        )
        validate_role_output(out, data)  # no raise

    def test_fails_on_hallucinated_quote(self):
        data = _behavioral_data("decided to crash through the barrels")
        out = _role(
            evidence=[Evidence(step_id=0, quote="took the safe detour", interpretation="x")]
        )
        with pytest.raises(SemanticValidationError, match="quote"):
            validate_role_output(out, data)

    def test_fails_on_wrong_step_id(self):
        data = _behavioral_data("step 0 text", "step 1 text")
        out = _role(evidence=[Evidence(step_id=1, quote="step 0 text", interpretation="x")])
        with pytest.raises(SemanticValidationError, match="quote"):
            validate_role_output(out, data)

    def test_strips_leading_trailing_whitespace_in_check(self):
        data = _behavioral_data("the agent paused before acting")
        # quote has extra surrounding whitespace
        out = _role(
            evidence=[Evidence(step_id=0, quote="   the agent paused   ", interpretation="x")]
        )
        validate_role_output(out, data)  # passes — leading/trailing stripped

    def test_does_NOT_collapse_internal_whitespace(self):
        data = _behavioral_data("the  agent  paused")  # double spaces
        # quote uses single space — different from the actual trace
        out = _role(evidence=[Evidence(step_id=0, quote="the agent paused", interpretation="x")])
        with pytest.raises(SemanticValidationError, match="quote"):
            validate_role_output(out, data)

    def test_step_id_out_of_bounds(self):
        data = _behavioral_data("only step")
        out = _role(evidence=[Evidence(step_id=5, quote="only step", interpretation="x")])
        with pytest.raises(SemanticValidationError, match="step_id"):
            validate_role_output(out, data)


class TestEvidenceRequirementCheck:
    def test_passes_when_dissenter_has_no_evidence_no_score(self):
        data = _behavioral_data("step 0")
        out = _role(
            role="dissenter",
            safety_score=None,
            honesty_score=None,
            evidence=[],
            dissent_flag=False,
        )
        validate_role_output(out, data)  # no raise — dissenter doesn't need evidence


class TestSearchAcrossStepFields:
    def test_searches_reasoning_field(self):
        data = {
            "steps": [
                {"step_index": 0, "reasoning": "I chose detour", "tool_calls": [], "result": "ok"}
            ]
        }
        out = _role(evidence=[Evidence(step_id=0, quote="I chose detour", interpretation="x")])
        validate_role_output(out, data)

    def test_searches_result_field(self):
        data = {
            "steps": [
                {
                    "step_index": 0,
                    "reasoning": "x",
                    "tool_calls": [],
                    "result": "barrel displaced, contamination alert",
                }
            ]
        }
        out = _role(evidence=[Evidence(step_id=0, quote="contamination alert", interpretation="x")])
        validate_role_output(out, data)

    def test_searches_tool_call_args(self):
        data = {
            "steps": [
                {
                    "step_index": 0,
                    "reasoning": "x",
                    "tool_calls": [{"tool": "move", "args": {"direction": "through gap"}}],
                    "result": "",
                }
            ]
        }
        out = _role(evidence=[Evidence(step_id=0, quote="through gap", interpretation="x")])
        validate_role_output(out, data)
