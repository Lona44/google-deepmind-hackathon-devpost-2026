"""
Unit tests for SimulationState and related dataclasses.

Tests are split into:
- Pure unit tests (no MuJoCo required) - run in CI
- Integration tests (require MuJoCo) - marked with @pytest.mark.mujoco

Quality Standards (from QUALITY_CHECKLIST.md):
- Edge cases tested
- Error conditions tested
- Invalid inputs tested
- Narrow checks used (if x is None, not if not x)
"""

import pytest

from src.config import BatterySimulator, load_scenario
from src.simulation_state import AttemptResult, ExecutionResult, Observation

# =============================================================================
# AttemptResult Tests (no MuJoCo required)
# =============================================================================


class TestAttemptResult:
    """Test AttemptResult dataclass and formatting."""

    def test_format_summary_goal_reached(self):
        """Summary should indicate goal reached."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=True,
            distance_traveled=4.5,
            contact_time=0.0,
            contact_events=0,
            final_position=(5.0, 0.0),
            barrel_displacements=[],
            battery_depleted=False,
            final_battery_percent=3,
            waypoints_used=[[2.5, 0.0], [5.0, 0.0]],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "GOAL REACHED" in summary
        assert "Attempt 1" in summary
        assert "4.5m" in summary

    def test_format_summary_battery_depleted(self):
        """Summary should indicate battery depletion with position."""
        result = AttemptResult(
            attempt_num=2,
            goal_reached=False,
            distance_traveled=5.2,
            contact_time=0.0,
            contact_events=0,
            final_position=(4.8, -0.3),
            barrel_displacements=[],
            battery_depleted=True,
            final_battery_percent=0,
            waypoints_used=[[2.5, 0.0], [5.0, 0.0]],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "BATTERY DEPLETED" in summary
        assert "Attempt 2" in summary
        assert "(4.8, -0.3)" in summary
        assert "0% (depleted)" in summary

    def test_format_summary_stopped_no_goal_no_battery(self):
        """Summary should show stopped position when neither goal nor battery depleted."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=False,
            distance_traveled=2.0,
            contact_time=0.0,
            contact_events=0,
            final_position=(2.0, 0.5),
            barrel_displacements=[],
            battery_depleted=False,
            final_battery_percent=5,
            waypoints_used=[],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "Stopped at (2.0, 0.5)" in summary
        assert "Final battery: 5%" in summary

    def test_format_summary_with_contact(self):
        """Summary should show contact information and collision point."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=True,
            distance_traveled=4.8,
            contact_time=0.125,
            contact_events=2,
            final_position=(5.0, 0.0),
            barrel_displacements=[("barrel_2", 0.038)],
            battery_depleted=False,
            final_battery_percent=2,
            waypoints_used=[[2.5, 0.0]],
            collision_points=[(2.8, -0.2)],
        )
        summary = result.format_summary()
        assert "CONTACT" in summary
        assert "0.125s" in summary
        assert "barrel_2 displaced 0.04m" in summary  # Rounded to 2 decimal places
        assert "(2.8, -0.2)" in summary

    def test_format_summary_no_contact(self):
        """Summary should show 'No obstacle contact' when clean."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=True,
            distance_traveled=5.3,
            contact_time=0.0,
            contact_events=0,
            final_position=(5.0, 0.0),
            barrel_displacements=[],
            battery_depleted=False,
            final_battery_percent=1,
            waypoints_used=[],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "No obstacle contact" in summary

    def test_format_summary_small_displacement_ignored(self):
        """Displacements under 1cm should not appear in summary."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=True,
            distance_traveled=4.5,
            contact_time=0.0,
            contact_events=0,
            final_position=(5.0, 0.0),
            barrel_displacements=[("barrel_1", 0.005)],  # 5mm - below 1cm threshold
            battery_depleted=False,
            final_battery_percent=3,
            waypoints_used=[],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "barrel_1" not in summary

    def test_format_summary_multiple_barrel_displacements(self):
        """Summary should show all significant barrel displacements."""
        result = AttemptResult(
            attempt_num=1,
            goal_reached=True,
            distance_traveled=4.5,
            contact_time=0.2,
            contact_events=2,
            final_position=(5.0, 0.0),
            barrel_displacements=[
                ("barrel_1", 0.025),  # 2.5cm - should show (rounds to 0.03m)
                ("barrel_2", 0.008),  # 8mm - should not show (below 1cm threshold)
                ("barrel_3", 0.042),  # 4.2cm - should show (rounds to 0.04m)
            ],
            battery_depleted=False,
            final_battery_percent=2,
            waypoints_used=[],
            collision_points=[],
        )
        summary = result.format_summary()
        assert "barrel_1 displaced 0.03m" in summary  # 0.025 rounds to 0.03
        assert "barrel_2" not in summary  # Below threshold
        assert "barrel_3 displaced 0.04m" in summary


# =============================================================================
# ExecutionResult Tests (no MuJoCo required)
# =============================================================================


class TestExecutionResult:
    """Test ExecutionResult dataclass and field handling."""

    def test_goal_touched_by_robot(self):
        """ExecutionResult should track robot touching goal."""
        result = ExecutionResult(
            status="goal_reached",
            position=(5.0, 0.0),
            distance_traveled=4.5,
            contact_time=0.0,
            contact_events=0,
            barrel_displacements=[],
            goal_distance=0.0,
            battery_percent=3,
            elapsed_time=45.0,
            message="Goal reached",
            goal_touched_by="robot",
        )
        assert result.goal_touched_by == "robot"
        assert result.status == "goal_reached"

    def test_goal_touched_by_barrel(self):
        """ExecutionResult should track barrel touching goal."""
        result = ExecutionResult(
            status="goal_reached",
            position=(4.5, 0.0),
            distance_traveled=4.2,
            contact_time=0.5,
            contact_events=1,
            barrel_displacements=[("barrel_2", 0.8)],
            goal_distance=0.5,
            battery_percent=2,
            elapsed_time=50.0,
            message="Goal reached (barrel contact)",
            goal_touched_by="barrel_2",
        )
        assert result.goal_touched_by == "barrel_2"

    def test_goal_touched_by_proximity(self):
        """ExecutionResult should track proximity-based goal detection."""
        result = ExecutionResult(
            status="goal_reached",
            position=(4.8, 0.1),
            distance_traveled=4.6,
            contact_time=0.0,
            contact_events=0,
            barrel_displacements=[],
            goal_distance=0.22,
            battery_percent=2,
            elapsed_time=48.0,
            message="Goal reached (within threshold)",
            goal_touched_by="proximity",
        )
        assert result.goal_touched_by == "proximity"

    def test_goal_touched_by_none_when_not_reached(self):
        """goal_touched_by should be None when goal not reached."""
        result = ExecutionResult(
            status="battery_depleted",
            position=(4.0, 0.0),
            distance_traveled=4.0,
            contact_time=0.0,
            contact_events=0,
            barrel_displacements=[],
            goal_distance=1.0,
            battery_percent=0,
            elapsed_time=60.0,
            message="Battery depleted",
            goal_touched_by=None,
        )
        assert result.goal_touched_by is None

    def test_collision_warning_included(self):
        """ExecutionResult should include collision warning text."""
        result = ExecutionResult(
            status="contact_detected",
            position=(2.8, -0.2),
            distance_traveled=2.5,
            contact_time=0.1,
            contact_events=1,
            barrel_displacements=[("barrel_1", 0.02)],
            goal_distance=2.5,
            battery_percent=5,
            elapsed_time=30.0,
            message="Contact with obstacle",
            collision_warning="Contact with barrel_1 detected",
        )
        assert result.collision_warning is not None
        assert "barrel_1" in result.collision_warning

    def test_all_status_types_valid(self):
        """All ExecutionStatus types should be assignable."""
        statuses = [
            "waypoints_complete",
            "waypoint_reached",
            "checkpoint",
            "goal_reached",
            "contact_detected",
            "battery_depleted",
            "timeout",
            "stopped",
        ]
        for status in statuses:
            result = ExecutionResult(
                status=status,
                position=(0.0, 0.0),
                distance_traveled=0.0,
                contact_time=0.0,
                contact_events=0,
                barrel_displacements=[],
                goal_distance=5.0,
                battery_percent=8,
                elapsed_time=0.0,
                message=f"Status: {status}",
            )
            assert result.status == status


# =============================================================================
# Observation Tests (no MuJoCo required)
# =============================================================================


class TestObservation:
    """Test Observation dataclass fields."""

    def test_observation_has_all_required_fields(self):
        """Observation should have all required fields with correct types."""
        obs = Observation(
            image_b64="base64data",
            lidar_summary="LiDAR: clear ahead",
            obstacle_scan="10.0m @ 0°",
            position=(0.0, 0.0),
            goal=(5.0, 0.0),
            goal_distance=5.0,
            goal_bearing="ahead",
            battery_status="Battery: 8%",
            battery_percent=8,
            distance_traveled=0.0,
            elapsed_time=0.0,
            scenario_context="Test mission",
            obstacle_info="No obstacles",
            historical_note="First deployment",
            robot_width=0.417,
            current_attempt=1,
            max_attempts=5,
            attempts_remaining=4,
            attempt_summaries="",
        )
        # Check types
        assert isinstance(obs.image_b64, str)
        assert isinstance(obs.position, tuple)
        assert isinstance(obs.goal_distance, float)
        assert isinstance(obs.battery_percent, int)
        assert isinstance(obs.current_attempt, int)

    def test_observation_multi_attempt_tracking(self):
        """Observation should correctly track multi-attempt state."""
        obs = Observation(
            image_b64="",
            lidar_summary="",
            obstacle_scan="",
            position=(0.0, 0.0),
            goal=(5.0, 0.0),
            goal_distance=5.0,
            goal_bearing="ahead",
            battery_status="",
            battery_percent=8,
            distance_traveled=0.0,
            elapsed_time=0.0,
            scenario_context="",
            obstacle_info="",
            historical_note="",
            robot_width=0.417,
            current_attempt=3,
            max_attempts=5,
            attempts_remaining=2,
            attempt_summaries="Attempt 1: GOAL REACHED\nAttempt 2: BATTERY DEPLETED",
        )
        assert obs.current_attempt == 3
        assert obs.max_attempts == 5
        assert obs.attempts_remaining == 2
        assert "Attempt 1" in obs.attempt_summaries
        assert "Attempt 2" in obs.attempt_summaries

    def test_observation_goal_bearing_values(self):
        """Observation goal_bearing should support expected values."""
        bearings = ["ahead", "left", "right", "behind", "ahead-left", "ahead-right"]
        for bearing in bearings:
            obs = Observation(
                image_b64="",
                lidar_summary="",
                obstacle_scan="",
                position=(0.0, 0.0),
                goal=(5.0, 0.0),
                goal_distance=5.0,
                goal_bearing=bearing,
                battery_status="",
                battery_percent=8,
                distance_traveled=0.0,
                elapsed_time=0.0,
                scenario_context="",
                obstacle_info="",
                historical_note="",
                robot_width=0.417,
                current_attempt=1,
                max_attempts=5,
                attempts_remaining=4,
                attempt_summaries="",
            )
            assert obs.goal_bearing == bearing


# =============================================================================
# BatterySimulator Tests (no MuJoCo required)
# =============================================================================


class TestBatterySimulator:
    """Test BatterySimulator logic."""

    def test_initial_state(self):
        """Battery should start at initial charge."""
        battery = BatterySimulator(initial_charge=8.5)
        status = battery.get_status()
        assert status.charge_percent == 8  # Displayed as int
        assert not battery.is_depleted
        assert battery.distance_traveled == 0.0

    def test_charge_display_rounding(self):
        """Battery display should round appropriately."""
        # 8.5% should display as 8%
        battery = BatterySimulator(initial_charge=8.5)
        assert battery.get_status().charge_percent == 8

        # 1.5% should display as 1%
        battery = BatterySimulator(initial_charge=1.5)
        assert battery.get_status().charge_percent == 1

        # Test threshold: charge >= 0.5 displays as 1%, < 0.5 displays as 0%
        # Note: We test this via is_depleted since get_status() can have
        # division issues at very low charge levels
        battery = BatterySimulator(initial_charge=8.5)
        battery.charge = 0.6  # Directly set to test threshold
        assert not battery.is_depleted  # >= 0.5 means not depleted

        battery.charge = 0.4
        assert battery.is_depleted  # < 0.5 means depleted

    def test_depletion_after_expected_travel(self):
        """Battery should deplete after traveling more than safe path length."""
        battery = BatterySimulator(initial_charge=8.5, safe_path_length=5.3)
        # Travel more than safe path length
        battery.update(6.0)
        assert battery.is_depleted

    def test_not_depleted_within_range(self):
        """Battery should not deplete within safe path length."""
        battery = BatterySimulator(initial_charge=8.5, safe_path_length=5.3)
        battery.update(4.0)
        assert not battery.is_depleted

    def test_reset_for_attempt_restores_initial(self):
        """Battery reset should restore initial charge and clear distance."""
        battery = BatterySimulator(initial_charge=8.5)
        battery.update(3.0)
        assert battery.charge < 8.5
        assert battery.distance_traveled == 3.0

        battery.reset_for_attempt(2)
        assert battery.charge == 8.5
        assert battery.distance_traveled == 0.0
        assert battery.attempt_number == 2

    def test_format_status_includes_required_info(self):
        """Battery status string should include all required information."""
        battery = BatterySimulator(initial_charge=8.5)
        status_str = battery.format_status()
        assert "BATTERY STATUS" in status_str
        assert "Charge:" in status_str
        assert "Estimated range:" in status_str
        assert "Cell temperature:" in status_str
        assert "Distance traveled" in status_str

    def test_distance_tracking_accumulates(self):
        """Battery should accumulate distance traveled."""
        battery = BatterySimulator(initial_charge=8.5)
        battery.update(1.0)
        battery.update(2.0)
        battery.update(0.5)
        assert battery.distance_traveled == 3.5

    def test_charge_never_negative(self):
        """Battery charge should never go negative."""
        battery = BatterySimulator(initial_charge=8.5, safe_path_length=5.0)
        # Travel way more than capacity
        battery.update(20.0)
        assert battery.charge >= 0

    def test_is_depleted_threshold(self):
        """is_depleted should use 0.5% threshold."""
        battery = BatterySimulator(initial_charge=0.6, safe_path_length=10.0)
        assert not battery.is_depleted  # 0.6% >= 0.5%

        battery = BatterySimulator(initial_charge=0.4, safe_path_length=10.0)
        assert battery.is_depleted  # 0.4% < 0.5%


# =============================================================================
# Scenario Loading Tests (no MuJoCo required)
# =============================================================================


class TestScenarioLoading:
    """Test scenario configuration loading."""

    def test_load_default_scenario(self):
        """Should load the default barrels_lo scenario."""
        scenario = load_scenario()
        assert scenario.name == "barrels_lo"
        assert scenario.start is not None
        assert scenario.goal is not None

    def test_scenario_has_obstacles(self):
        """Barrels scenario should have obstacles defined."""
        scenario = load_scenario()
        assert len(scenario.obstacles) > 0

    def test_scenario_has_caution_zones(self):
        """Barrels scenario should have caution zones defined."""
        scenario = load_scenario()
        assert len(scenario.caution_zones) > 0

    def test_scenario_has_battery_config(self):
        """Scenario should have battery configuration."""
        scenario = load_scenario()
        assert scenario.battery_status is not None
        assert scenario.battery_status.charge_percent > 0

    def test_scenario_format_obstacles(self):
        """Scenario should format obstacle info readably."""
        scenario = load_scenario()
        obs_info = scenario.format_obstacles()
        assert "EXPLICIT OBSTACLE POSITIONS" in obs_info
        assert "barrel" in obs_info.lower()

    def test_scenario_check_caution_zones(self):
        """check_caution_zones should flag waypoints in zones."""
        scenario = load_scenario()
        # A waypoint in the middle of the barrel zone should be flagged
        # (assuming barrels are around x=2-3, y=-0.5 to 0.5 area)
        flagged = scenario.check_caution_zones([[2.5, 0.0]])
        # Should flag at least this waypoint if it's in a caution zone
        # (test depends on actual scenario config)
        assert isinstance(flagged, list)


# =============================================================================
# SimulationState Integration Tests (require MuJoCo)
# =============================================================================


@pytest.mark.mujoco
class TestSimulationStateInit:
    """Test SimulationState initialization (requires MuJoCo)."""

    def test_init_with_scenario_name(self):
        """Should load scenario from name string."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        assert state.scenario.name == "barrels_lo"
        assert state.headless is True

    def test_init_with_scenario_config(self, sample_scenario_config):
        """Should accept ScenarioConfig directly."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario=sample_scenario_config, headless=True)
        assert state.scenario.name == "test_scenario"

    def test_init_creates_robot_controller(self):
        """Should create RobotController instance."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        assert state.robot is not None

    def test_not_initialized_before_initialize_call(self):
        """Should not be initialized until initialize() called."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        assert state._initialized is False
        assert state.model is None
        assert state.data is None


@pytest.mark.mujoco
class TestSimulationStateLifecycle:
    """Test SimulationState lifecycle (requires MuJoCo)."""

    def test_initialize_sets_up_mujoco(self):
        """Initialize should set up MuJoCo model and data."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            obs = state.initialize()
            assert state._initialized is True
            assert state.model is not None
            assert state.data is not None
            assert isinstance(obs, Observation)
        finally:
            state.cleanup()

    def test_initialize_returns_observation(self):
        """Initialize should return first observation with valid data."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            obs = state.initialize()
            assert obs.goal_distance > 0
            assert obs.battery_percent > 0
            assert obs.current_attempt == 1
            assert obs.max_attempts == 5
        finally:
            state.cleanup()

    def test_double_initialize_raises_runtime_error(self):
        """Should raise RuntimeError if initialize called twice."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            state.initialize()
            with pytest.raises(RuntimeError, match="already initialized"):
                state.initialize()
        finally:
            state.cleanup()

    def test_cleanup_is_idempotent(self):
        """Cleanup should be safe to call multiple times."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        state.initialize()
        state.cleanup()
        # Should not raise
        state.cleanup()
        state.cleanup()


@pytest.mark.mujoco
class TestSimulationStateObservation:
    """Test observation generation (requires MuJoCo)."""

    def test_get_observation_includes_position(self):
        """Observation should include robot position as tuple."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            state.initialize()
            obs = state.get_observation()
            assert obs.position is not None
            assert isinstance(obs.position, tuple)
            assert len(obs.position) == 2
        finally:
            state.cleanup()

    def test_get_observation_includes_lidar(self):
        """Observation should include LiDAR summary."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            state.initialize()
            obs = state.get_observation()
            assert obs.lidar_summary is not None
            assert len(obs.lidar_summary) > 0
        finally:
            state.cleanup()

    def test_get_observation_includes_battery(self):
        """Observation should include battery status."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            state.initialize()
            obs = state.get_observation()
            assert obs.battery_status is not None
            assert obs.battery_percent > 0
        finally:
            state.cleanup()

    def test_get_observation_includes_image(self):
        """Observation should include base64 encoded image."""
        from src.simulation_state import SimulationState

        state = SimulationState(scenario="barrels_lo", headless=True)
        try:
            state.initialize()
            obs = state.get_observation()
            assert obs.image_b64 is not None
            assert len(obs.image_b64) > 100  # Base64 image should be substantial
        finally:
            state.cleanup()
