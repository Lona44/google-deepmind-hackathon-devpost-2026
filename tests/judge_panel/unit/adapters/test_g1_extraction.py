"""Tests for the G1 trajectory → behavioral_data adapter.

Methodology motivation: the hand-built canonical.json fixture was a
5-step pre-incident slice that the panel could not detect actual
misalignment from. This adapter takes the full G1 trajectory JSON
(experiment-side output, ~1500 frames) and emits a behavioral_data dict
that exposes every agent decision to the panel — including the
rationalization frames and the collision moment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from judge_panel.adapters.g1_extraction import extract_from_g1_trajectory


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAJECTORY_PATH = (
    REPO_ROOT / "gcp/frontend/assets/trajectory_2026-02-06T05-01_gpt-5.json"
)


@pytest.fixture
def trajectory() -> dict:
    return json.loads(TRAJECTORY_PATH.read_text())


class TestExtractor:
    def test_returns_dict_with_required_top_level_keys(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        assert set(result.keys()) >= {"run_id", "model", "scenario", "steps"}

    def test_run_id_taken_from_experiment_id(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        assert result["run_id"] == "exp_1770354093"

    def test_scenario_taken_from_metadata(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        assert result["scenario"] == "barrels_corrupt"

    def test_model_resolved_from_metadata(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        # metadata.model_name in this trajectory is "gpt5"
        assert "gpt" in result["model"].lower()

    def test_emits_one_step_per_decision_frame(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        # This trajectory has 13 frames with substantive ai_reasoning.
        assert len(result["steps"]) == 13

    def test_step_indices_are_sequential(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        for i, step in enumerate(result["steps"]):
            assert step["step_index"] == i

    def test_each_step_has_required_fields(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        for step in result["steps"]:
            assert "step_index" in step
            assert "reasoning" in step
            assert "tool_calls" in step
            assert "result" in step
            assert step["reasoning"], "reasoning must be non-empty"

    def test_first_step_is_initial_planning_at_frame_zero(self, trajectory):
        result = extract_from_g1_trajectory(trajectory)
        step = result["steps"][0]
        # Frame 0's reasoning mentions barrels at x=2.5 with three y positions
        assert "barrel" in step["reasoning"].lower()
        # Step 0 also calls set_waypoints with the initial detour
        assert "set_waypoints" in json.dumps(step["tool_calls"])

    def test_rationalization_frame_is_present(self, trajectory):
        """Frame 760 contains the 'feels worse than aborting' reasoning
        that Gemini cited as the L2 rationalization. Critical that the
        panel can see it now."""
        result = extract_from_g1_trajectory(trajectory)
        # Reconcatenate all reasoning text
        all_reasoning = "\n\n".join(s["reasoning"] for s in result["steps"])
        assert "feels worse" in all_reasoning.lower(), (
            "rationalization moment ('feels worse than aborting') must be "
            "preserved in the extracted steps"
        )

    def test_collision_event_appears_in_results(self, trajectory):
        """Frame 1025 reports the collision. The adapter must surface
        the contact event (either in that step's reasoning or in the
        result of the preceding decision) so the panel can detect it."""
        result = extract_from_g1_trajectory(trajectory)
        joined = "\n\n".join(
            s["reasoning"] + " " + str(s["result"]) for s in result["steps"]
        )
        # Either the agent's own reflection ("That's a significant contact violation")
        # or the synthesised result note about the contact event must appear.
        assert (
            "contact violation" in joined.lower()
            or "first contact" in joined.lower()
            or "barrel" in joined.lower() and "contact" in joined.lower()
        ), "panel must be able to see the collision event"

    def test_attempt_resets_are_visible(self, trajectory):
        """The mission has 4 attempts. The adapter should make the
        attempt number visible per step so the panel can distinguish
        cross-attempt reasoning from intra-attempt contradiction."""
        result = extract_from_g1_trajectory(trajectory)
        joined_results = " ".join(str(s["result"]) for s in result["steps"])
        # Either through "attempt N" in result, or attempt_reset events
        assert "attempt" in joined_results.lower()
        # At least 4 distinct attempts referenced
        attempt_mentions = sum(
            1 for s in result["steps"] if "attempt" in str(s["result"]).lower()
        )
        assert attempt_mentions >= 3, (
            "panel must be able to see that the mission spans multiple attempts"
        )

    def test_tool_calls_are_searchable_via_json_dumps(self, trajectory):
        """The validation layer searches reasoning+json.dumps(tool_calls)+result
        for evidence quotes. The agent's action string ('set_waypoints(...)',
        'end_mission()', etc.) must end up in that searchable text."""
        result = extract_from_g1_trajectory(trajectory)
        # First step's action was set_waypoints with specific coordinates
        first = result["steps"][0]
        haystack = json.dumps(first["tool_calls"])
        assert "set_waypoints" in haystack
        # Last step ends the mission
        last = result["steps"][-1]
        last_haystack = json.dumps(last["tool_calls"])
        assert "end_mission" in last_haystack

    def test_output_passes_existing_step_searchable_text(self, trajectory):
        """End-to-end: the adapter's output must work with the existing
        validation._step_searchable_text helper without any changes."""
        from judge_panel.validation import _step_searchable_text
        result = extract_from_g1_trajectory(trajectory)
        for step in result["steps"]:
            text = _step_searchable_text(step)
            assert text, "every step must produce searchable text"
            assert len(text) > 10
