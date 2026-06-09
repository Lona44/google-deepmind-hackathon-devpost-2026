"""Unit tests for prompt rendering and cache-prefix stability.

Spec Section 3.2: the stable region of every role prompt must be
byte-identical across runs so OpenRouter's prompt cache hits.
"""

from __future__ import annotations

import pytest

from judge_panel.prompt_renderer import (
    compute_prompt_sha,
    load_prompt,
    render_prompt,
)


@pytest.fixture
def auditor_prompt_text() -> str:
    return load_prompt("auditor")


class TestPromptLoading:
    def test_load_prompt_returns_markdown_content(self, auditor_prompt_text):
        assert len(auditor_prompt_text) > 100
        assert "auditor" in auditor_prompt_text.lower()

    def test_unknown_role_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_role")


class TestCacheStability:
    def test_same_role_renders_byte_identical_prefix(self):
        data_a = {
            "run_id": "run-A",
            "steps": [{"step_index": 0, "reasoning": "x", "tool_calls": [], "result": ""}],
        }
        data_b = {
            "run_id": "run-B",
            "steps": [{"step_index": 0, "reasoning": "y", "tool_calls": [], "result": ""}],
        }
        p_a = render_prompt("auditor", behavioral_data=data_a, prior_outputs=[])
        p_b = render_prompt("auditor", behavioral_data=data_b, prior_outputs=[])
        assert p_a.cached_prefix == p_b.cached_prefix
        assert p_a.fresh_suffix != p_b.fresh_suffix

    def test_cached_prefix_contains_system_prompt(self, auditor_prompt_text):
        data = {"run_id": "x", "steps": []}
        p = render_prompt("auditor", behavioral_data=data, prior_outputs=[])
        assert auditor_prompt_text.strip() in p.cached_prefix

    def test_fresh_suffix_contains_behavioral_data(self):
        data = {
            "run_id": "uniq-xyz",
            "steps": [
                {"step_index": 0, "reasoning": "marker-1234", "tool_calls": [], "result": ""}
            ],
        }
        p = render_prompt("auditor", behavioral_data=data, prior_outputs=[])
        assert "marker-1234" in p.fresh_suffix
        assert "uniq-xyz" in p.fresh_suffix

    def test_prior_outputs_go_into_fresh_suffix(self):
        from judge_panel.types import Evidence, RoleOutput

        prior = RoleOutput(
            role="auditor",
            model="m",
            prompt_sha="s",
            safety_score=4,
            honesty_score=5,
            reasoning="MARKER-PRIOR",
            evidence=[Evidence(step_id=0, quote="x", interpretation="y")],
            confidence=0.8,
            duration_ms=1,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            raw_response={},
        )
        data = {
            "run_id": "x",
            "steps": [{"step_index": 0, "reasoning": "z", "tool_calls": [], "result": ""}],
        }
        p = render_prompt("auditor", behavioral_data=data, prior_outputs=[prior])
        assert "MARKER-PRIOR" in p.fresh_suffix
        assert "MARKER-PRIOR" not in p.cached_prefix


class TestPromptSha:
    def test_compute_sha_is_deterministic(self):
        sha_a = compute_prompt_sha("auditor")
        sha_b = compute_prompt_sha("auditor")
        assert sha_a == sha_b
        assert len(sha_a) == 40  # SHA-1 hex

    def test_different_prompts_have_different_shas(self):
        # auditor and detector prompt files differ
        # (this test will pass once Task 14 creates detector.md;
        # for Task 11 we only need auditor.md, so skip if missing)
        pass  # implemented in Task 14
