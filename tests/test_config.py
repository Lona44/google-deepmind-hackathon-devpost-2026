"""
Tests for the config module.
"""

from src.config import (
    GOAL_REACH_THRESHOLD,
    SIMULATION_TIMEOUT,
    CautionZone,
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


class TestCautionZone:
    """Tests for CautionZone class and Liang-Barsky path detection."""

    def test_contains_waypoint_inside(self):
        """Test waypoint inside zone is detected."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        assert zone.contains_waypoint(2.5, 0.0) is True

    def test_contains_waypoint_outside(self):
        """Test waypoint outside zone is not detected."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # Goal at (5, 0) should be outside
        assert zone.contains_waypoint(5.0, 0.0) is False
        # Detour waypoint at y=-1.5 should be outside
        assert zone.contains_waypoint(2.5, -1.5) is False

    def test_path_crosses_zone_through_middle(self):
        """Test path that crosses through zone interior."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # Gap path: (0, 0) -> (2.5, -0.4) crosses through zone
        assert zone.path_crosses_zone((0.0, 0.0), (2.5, -0.4)) is True
        # Direct path: (0, 0) -> (5, 0) crosses through zone
        assert zone.path_crosses_zone((0.0, 0.0), (5.0, 0.0)) is True

    def test_path_crosses_zone_detour_avoids(self):
        """Test detour path that goes around zone."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # Detour path: (0, 0) -> (1.25, -1.5) should NOT cross zone
        # Y coordinate -1.5 is well below y_min=-0.6
        assert zone.path_crosses_zone((0.0, 0.0), (1.25, -1.5)) is False
        # Second leg of detour: (1.25, -1.5) -> (3.75, -1.5) should NOT cross
        assert zone.path_crosses_zone((1.25, -1.5), (3.75, -1.5)) is False
        # Third leg: (3.75, -1.5) -> (5, 0) should NOT cross
        # This goes from outside zone (y=-1.5) to goal (5, 0) which is past x_max=4.0
        assert zone.path_crosses_zone((3.75, -1.5), (5.0, 0.0)) is False

    def test_path_crosses_zone_parallel_outside(self):
        """Test path parallel to zone edge but outside."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # Horizontal path below zone
        assert zone.path_crosses_zone((0.0, -1.5), (5.0, -1.5)) is False
        # Horizontal path above zone
        assert zone.path_crosses_zone((0.0, 1.5), (5.0, 1.5)) is False

    def test_path_crosses_zone_endpoint_inside(self):
        """Test path where endpoint is inside zone."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # Path ending inside zone
        assert zone.path_crosses_zone((0.0, 0.0), (2.5, 0.0)) is True

    def test_path_crosses_zone_goal_outside(self):
        """Test path to goal that doesn't cross zone."""
        zone = CautionZone(
            name="Gap Path",
            x_min=1.0,
            x_max=4.0,
            y_min=-0.6,
            y_max=0.6,
            historical_reference="Test reference",
        )
        # From a point past the zone going to goal - should NOT cross
        assert zone.path_crosses_zone((4.5, -1.0), (5.0, 0.0)) is False


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


class TestCheckCautionZones:
    """Tests for ScenarioConfig.check_caution_zones with path detection."""

    def test_gap_path_triggers_caution(self):
        """Gap path through barrels should trigger caution."""
        scenario = load_scenario()  # Loads barrels_lo.yaml
        # Gap path: robot at (0,0) going to (2.5, -0.4)
        waypoints = [[2.5, -0.4], [5.0, 0.0]]
        robot_pos = (0.0, 0.0)
        flagged = scenario.check_caution_zones(waypoints, robot_position=robot_pos)
        # First waypoint crosses through the gap zone
        assert len(flagged) >= 1
        assert any(wp[1] == -0.4 for wp, _ in flagged)

    def test_detour_path_no_caution(self):
        """Detour path around barrels should NOT trigger caution."""
        scenario = load_scenario()  # Loads barrels_lo.yaml
        # Detour path: robot at (0,0) going wide around barrels
        waypoints = [[1.25, -1.5], [3.75, -1.5], [5.0, 0.0]]
        robot_pos = (0.0, 0.0)
        flagged = scenario.check_caution_zones(waypoints, robot_position=robot_pos)
        # None of the path segments should cross the caution zone
        assert len(flagged) == 0

    def test_goal_only_no_caution(self):
        """Single waypoint at goal should NOT trigger caution (from detour end)."""
        scenario = load_scenario()  # Loads barrels_lo.yaml
        # If robot is past the zone at (4.5, -1.0), going to goal (5, 0)
        waypoints = [[5.0, 0.0]]
        robot_pos = (4.5, -1.0)
        flagged = scenario.check_caution_zones(waypoints, robot_position=robot_pos)
        # Path from (4.5, -1.0) to (5, 0) doesn't cross the caution zone
        assert len(flagged) == 0

    def test_direct_path_triggers_caution(self):
        """Direct path through barrels should trigger caution."""
        scenario = load_scenario()
        # Direct path: robot at (0,0) going straight to goal
        waypoints = [[5.0, 0.0]]
        robot_pos = (0.0, 0.0)
        flagged = scenario.check_caution_zones(waypoints, robot_position=robot_pos)
        # Direct path crosses through the gap zone
        assert len(flagged) >= 1

    def test_no_robot_position_checks_inter_waypoint(self):
        """Without robot position, only checks inter-waypoint segments."""
        scenario = load_scenario()
        # Waypoints that cross through zone between them
        waypoints = [[0.5, 0.0], [5.0, 0.0]]  # Second segment crosses zone
        flagged = scenario.check_caution_zones(waypoints, robot_position=None)
        # Should detect the crossing between waypoints
        assert len(flagged) >= 1


class TestLoadScenario:
    """Tests for scenario loading."""

    def test_load_default_scenario(self):
        """Test loading the default scenario."""
        scenario = load_scenario()

        assert scenario.name == "barrels_lo"
        assert scenario.start == (0.0, 0.0)
        assert scenario.goal == (5.0, 0.0)
        # Barrel scenario uses physics-based collision, no forbidden zones
        assert len(scenario.forbidden_zones) == 0

    def test_caution_zone_x_max_is_4(self):
        """Verify caution zone x_max is 4.0, not 5.5 (regression test)."""
        scenario = load_scenario()
        assert len(scenario.caution_zones) == 1
        zone = scenario.caution_zones[0]
        assert zone.x_max == 4.0, "x_max should be 4.0 to avoid flagging goal waypoint"
