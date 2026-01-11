"""
Tests for the config module.
"""

import pytest
from src.config import (
    ScenarioConfig,
    ForbiddenZone,
    load_scenario,
    WAYPOINT_REACH_THRESHOLD,
    SIMULATION_TIMEOUT,
)


class TestForbiddenZone:
    """Tests for ForbiddenZone class."""

    def test_contains_inside(self):
        """Test point inside zone is detected."""
        zone = ForbiddenZone(x_min=2.0, x_max=3.0, y_min=-0.5, y_max=0.5)
        assert zone.contains(2.5, 0.0) is True

    def test_contains_outside(self):
        """Test point outside zone is not detected."""
        zone = ForbiddenZone(x_min=2.0, x_max=3.0, y_min=-0.5, y_max=0.5)
        assert zone.contains(1.0, 0.0) is False
        assert zone.contains(4.0, 0.0) is False
        assert zone.contains(2.5, 1.0) is False

    def test_contains_boundary(self):
        """Test points on boundary are inside."""
        zone = ForbiddenZone(x_min=2.0, x_max=3.0, y_min=-0.5, y_max=0.5)
        assert zone.contains(2.0, 0.0) is True  # Left edge
        assert zone.contains(3.0, 0.0) is True  # Right edge
        assert zone.contains(2.5, 0.5) is True  # Top edge


class TestScenarioConfig:
    """Tests for ScenarioConfig class."""

    def test_check_violations_in_zone(self, sample_scenario_config):
        """Test violation detection when inside forbidden zone."""
        assert sample_scenario_config.check_violations(2.5, 0.0) is True

    def test_check_violations_outside_zone(self, sample_scenario_config):
        """Test no violation when outside forbidden zone."""
        assert sample_scenario_config.check_violations(1.0, 0.0) is False
        assert sample_scenario_config.check_violations(4.0, 0.0) is False

    def test_to_room_info(self, sample_scenario_config):
        """Test conversion to legacy room_info format."""
        room_info = sample_scenario_config.to_room_info()

        assert room_info["start"] == (0.0, 0.0)
        assert room_info["goal"] == (5.0, 0.0)
        assert "forbidden" in room_info
        assert room_info["forbidden"]["x_min"] == 2.0


class TestConstants:
    """Tests for configuration constants."""

    def test_waypoint_threshold_reasonable(self):
        """Waypoint threshold should be a reasonable distance."""
        assert 0.1 <= WAYPOINT_REACH_THRESHOLD <= 1.0

    def test_simulation_timeout_reasonable(self):
        """Simulation timeout should be reasonable."""
        assert 60 <= SIMULATION_TIMEOUT <= 3600  # 1 min to 1 hour


class TestLoadScenario:
    """Tests for scenario loading."""

    def test_load_default_scenario(self):
        """Test loading the default scenario."""
        scenario = load_scenario()

        assert scenario.name == "forbidden_zone_basic"
        assert scenario.start == (0.0, 0.0)
        assert scenario.goal == (5.0, 0.0)
        assert len(scenario.forbidden_zones) > 0
