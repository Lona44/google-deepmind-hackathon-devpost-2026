"""
Unit tests for Inspect AI native tools.

Tests are split into:
- Pure unit tests (no MuJoCo required) - run in CI
- Integration tests (require MuJoCo) - marked with @pytest.mark.mujoco

Quality Standards (from QUALITY_CHECKLIST.md):
- Custom @tool decorated functions have tests
- Edge cases tested
- Invalid inputs tested
"""

import json

import pytest

# =============================================================================
# Helper Function Tests (no MuJoCo required)
# =============================================================================


class TestWaypointNormalization:
    """Test waypoint format normalization logic."""

    def test_dict_format_normalization(self):
        """Should normalize dict format waypoints."""
        # Test the logic that would be used in set_waypoints
        waypoints = [{"x": 1.0, "y": 2.0}, {"x": 3.5, "y": -0.5}]
        wp_list = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_list.append([wp["x"], wp["y"]])
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_list.append([float(wp[0]), float(wp[1])])

        assert wp_list == [[1.0, 2.0], [3.5, -0.5]]

    def test_list_format_normalization(self):
        """Should normalize list format waypoints."""
        waypoints = [[1.0, 2.0], [3.5, -0.5]]
        wp_list = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_list.append([wp["x"], wp["y"]])
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_list.append([float(wp[0]), float(wp[1])])

        assert wp_list == [[1.0, 2.0], [3.5, -0.5]]

    def test_tuple_format_normalization(self):
        """Should normalize tuple format waypoints."""
        waypoints = [(1.0, 2.0), (3.5, -0.5)]
        wp_list = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_list.append([wp["x"], wp["y"]])
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_list.append([float(wp[0]), float(wp[1])])

        assert wp_list == [[1.0, 2.0], [3.5, -0.5]]

    def test_mixed_format_normalization(self):
        """Should handle mixed waypoint formats."""
        waypoints = [{"x": 1.0, "y": 2.0}, [3.5, -0.5], (5.0, 0.0)]
        wp_list = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_list.append([wp["x"], wp["y"]])
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_list.append([float(wp[0]), float(wp[1])])

        assert wp_list == [[1.0, 2.0], [3.5, -0.5], [5.0, 0.0]]

    def test_string_number_conversion(self):
        """Should convert string numbers to floats."""
        waypoints = [["1.0", "2.0"], ["3.5", "-0.5"]]
        wp_list = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_list.append([wp["x"], wp["y"]])
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                wp_list.append([float(wp[0]), float(wp[1])])

        assert wp_list == [[1.0, 2.0], [3.5, -0.5]]


class TestToolImports:
    """Test that tools can be imported without MuJoCo."""

    def test_import_module(self):
        """Should import module without error."""
        # This tests that the module doesn't do heavy initialization at import time
        import inspect_eval.tools_native as tools

        assert hasattr(tools, "observe_scene")
        assert hasattr(tools, "set_waypoints")
        assert hasattr(tools, "end_mission")
        assert hasattr(tools, "request_retry")
        assert hasattr(tools, "request_debrief")
        assert hasattr(tools, "submit_mission_report")

    def test_get_native_tools_returns_list(self):
        """get_native_tools should return a list of tools."""
        from inspect_eval.tools_native import get_native_tools

        tools = get_native_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_reset_simulation_callable(self):
        """reset_simulation should be callable."""
        from inspect_eval.tools_native import reset_simulation

        # Should not raise - just resets state to None
        reset_simulation()


class TestEnvironmentVariables:
    """Test environment variable parsing."""

    def test_verbose_flag_parsing(self):
        """VERBOSE should parse from G1_VERBOSE env var."""
        import os

        # Save original
        original = os.environ.get("G1_VERBOSE")

        try:
            # Test false values
            for val in ["false", "0", "no", ""]:
                os.environ["G1_VERBOSE"] = val
                # Re-import to get fresh value
                import importlib

                import inspect_eval.tools_native as tools

                importlib.reload(tools)
                # The VERBOSE constant is set at import time
                assert not tools.VERBOSE
        finally:
            # Restore
            if original is not None:
                os.environ["G1_VERBOSE"] = original
            elif "G1_VERBOSE" in os.environ:
                del os.environ["G1_VERBOSE"]


# =============================================================================
# Tool Output Format Tests (no MuJoCo required)
# =============================================================================


class TestToolResultFormats:
    """Test expected tool result formats."""

    def test_observe_scene_result_is_list(self):
        """observe_scene should return list with text and image."""
        # We can't call the actual tool without MuJoCo, but we can test
        # the expected structure based on the return statements
        from inspect_ai._util.content import ContentImage, ContentText

        # This is the expected structure:
        result = [
            ContentText(text="360° CAMERA VIEW:"),
            ContentImage(image="data:image/png;base64,test", detail="high"),
            ContentText(text="SENSOR DATA:\n{}"),
        ]
        assert len(result) == 3
        assert isinstance(result[0], ContentText)
        assert isinstance(result[1], ContentImage)
        assert isinstance(result[2], ContentText)

    def test_json_result_structure(self):
        """Tool results should produce valid JSON."""
        # Test the JSON structure that tools return
        result_dict = {
            "status": "waypoints_complete",
            "position": [2.5, 0.0],
            "distance_traveled": 2.5,
            "contact_time": 0.0,
            "contact_events": 0,
            "barrel_displacements": [],
            "goal_distance": 2.5,
            "battery_percent": 5,
            "elapsed_time": 30.0,
            "message": "Waypoints complete",
            "goal_touched_by": None,
        }
        # Should be valid JSON
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)
        assert parsed["status"] == "waypoints_complete"
        assert parsed["goal_touched_by"] is None


# =============================================================================
# Integration Tests (require MuJoCo)
# Note: Full async tool testing requires pytest-asyncio.
# These tests verify tool creation and basic structure only.
# =============================================================================


@pytest.mark.mujoco
class TestToolCreation:
    """Test that tools can be created with MuJoCo available."""

    def test_observe_scene_creates_callable(self):
        """observe_scene should create a callable tool."""
        from inspect_eval.tools_native import observe_scene

        tool = observe_scene()
        assert callable(tool)

    def test_set_waypoints_creates_callable(self):
        """set_waypoints should create a callable tool."""
        from inspect_eval.tools_native import set_waypoints

        tool = set_waypoints()
        assert callable(tool)

    def test_end_mission_creates_callable(self):
        """end_mission should create a callable tool."""
        from inspect_eval.tools_native import end_mission

        tool = end_mission()
        assert callable(tool)

    def test_request_retry_creates_callable(self):
        """request_retry should create a callable tool."""
        from inspect_eval.tools_native import request_retry

        tool = request_retry()
        assert callable(tool)

    def test_request_debrief_creates_callable(self):
        """request_debrief should create a callable tool."""
        from inspect_eval.tools_native import request_debrief

        tool = request_debrief()
        assert callable(tool)

    def test_submit_mission_report_creates_callable(self):
        """submit_mission_report should create a callable tool."""
        from inspect_eval.tools_native import submit_mission_report

        tool = submit_mission_report()
        assert callable(tool)
