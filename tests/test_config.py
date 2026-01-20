"""
Tests for the config module.
"""

from src.config import (
    GOAL_REACH_THRESHOLD,
    SIMULATION_TIMEOUT,
    ForbiddenZone,
    load_scenario,
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

    def test_contains_with_margin(self):
        """Test margin expands the effective zone."""
        zone = ForbiddenZone(x_min=2.0, x_max=3.0, y_min=-0.5, y_max=0.5)
        # Point just outside zone, but within margin
        assert zone.contains(1.9, 0.0, margin=0.0) is False  # No margin: outside
        assert zone.contains(1.9, 0.0, margin=0.2) is True  # With margin: inside
        # Point on y edge with margin
        assert zone.contains(2.5, 0.6, margin=0.0) is False  # No margin: outside
        assert zone.contains(2.5, 0.6, margin=0.2) is True  # With margin: inside


class TestScenarioConfig:
    """Tests for ScenarioConfig class."""

    def test_check_violations_in_zone(self, sample_scenario_config):
        """Test violation detection when inside forbidden zone."""
        assert sample_scenario_config.check_violations(2.5, 0.0) is True

    def test_check_violations_outside_zone(self, sample_scenario_config):
        """Test no violation when outside forbidden zone."""
        assert sample_scenario_config.check_violations(1.0, 0.0) is False
        assert sample_scenario_config.check_violations(4.0, 0.0) is False

    def test_check_violations_accounts_for_robot_width(self, sample_scenario_config):
        """Test that violation check accounts for robot body, not just center.

        Zone is x=[2.0, 3.0], y=[-0.5, 0.5].
        Robot width is 0.45m, so half-width is 0.225m.
        A center at y=0.6 would have robot edge at y=0.375, inside the zone.
        """
        # Center at y=0.6, edge at y=0.375 (inside zone y<0.5)
        assert sample_scenario_config.check_violations(2.5, 0.6) is True
        # Center at y=0.8, edge at y=0.575 (outside zone y>0.5)
        assert sample_scenario_config.check_violations(2.5, 0.8) is False
        # Center at x=1.8, edge at x=2.025 (inside zone x>2.0)
        assert sample_scenario_config.check_violations(1.8, 0.0) is True
        # Center at x=1.7, edge at x=1.925 (outside zone x<2.0)
        assert sample_scenario_config.check_violations(1.7, 0.0) is False

    def test_to_room_info(self, sample_scenario_config):
        """Test conversion to legacy room_info format."""
        room_info = sample_scenario_config.to_room_info()

        assert room_info["start"] == (0.0, 0.0)
        assert room_info["goal"] == (5.0, 0.0)
        assert "forbidden" in room_info
        assert room_info["forbidden"]["x_min"] == 2.0


class TestConstants:
    """Tests for configuration constants."""

    def test_goal_threshold_reasonable(self):
        """Goal reach threshold should be a reasonable distance."""
        assert 0.1 <= GOAL_REACH_THRESHOLD <= 1.0

    def test_simulation_timeout_reasonable(self):
        """Simulation timeout should be reasonable."""
        assert 60 <= SIMULATION_TIMEOUT <= 3600  # 1 min to 1 hour


class TestLoadScenario:
    """Tests for scenario loading."""

    def test_load_default_scenario(self):
        """Test loading the default scenario."""
        scenario = load_scenario()

        assert scenario.name == "navigation_barrels"
        assert scenario.start == (0.0, 0.0)
        assert scenario.goal == (5.0, 0.0)
        # Barrel scenario uses physics-based collision, no forbidden zones
        assert len(scenario.forbidden_zones) == 0
