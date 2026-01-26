"""
Stateful simulation wrapper for Inspect AI integration.

This module provides a step-by-step interface to the MuJoCo simulation,
allowing Inspect to control the robot through discrete tool calls rather
than running a continuous loop.
"""

import math
import time
from dataclasses import dataclass
from typing import Literal

import mujoco

from .config import (
    GOAL_REACH_THRESHOLD,
    SCENE_XML_PATH,
    SIMULATION_TIMEOUT,
    BatterySimulator,
    ScenarioConfig,
    load_scenario,
)
from .robot import RobotController, create_renderer
from .scene_loader import load_scene_with_background_robots


@dataclass
class Observation:
    """Current state observation from the simulation."""

    image_b64: str  # 360° panorama
    lidar_summary: str  # Formatted lidar data
    position: tuple[float, float]  # Current (x, y)
    goal: tuple[float, float]  # Goal (x, y)
    goal_distance: float  # Distance to goal
    goal_bearing: str  # "ahead", "left", "right", etc.
    battery_status: str  # Formatted battery info
    battery_percent: int  # Raw percentage
    distance_traveled: float  # Total distance this attempt
    elapsed_time: float  # Seconds since start
    scenario_context: str  # Mission briefing
    obstacle_info: str  # Formatted obstacle positions


@dataclass
class ExecutionResult:
    """Result of executing waypoints."""

    status: Literal[
        "waypoints_complete",
        "goal_reached",
        "contact_detected",
        "battery_depleted",
        "timeout",
        "stopped",
    ]
    position: tuple[float, float]  # Final position
    distance_traveled: float  # Distance traveled during execution
    contact_time: float  # Total contact time with obstacles
    contact_events: int  # Number of contact events
    barrel_displacements: list[tuple[str, float]]  # [(name, displacement), ...]
    goal_distance: float  # Distance to goal
    battery_percent: int  # Remaining battery
    elapsed_time: float  # Total elapsed time
    message: str  # Human-readable status message


class SimulationState:
    """
    Stateful MuJoCo simulation that persists between Inspect tool calls.

    Usage:
        state = SimulationState(scenario="forbidden_zone")
        state.initialize()

        obs = state.get_observation()
        # Model decides on waypoints...

        result = state.execute_waypoints([[1.0, -0.5], [2.5, -1.5]])
        # Check result.status...

        state.cleanup()
    """

    def __init__(
        self,
        scenario: str | ScenarioConfig = "forbidden_zone",
        scene_name: str | None = None,
        headless: bool = True,
    ):
        """
        Initialize simulation state.

        Args:
            scenario: Scenario name or ScenarioConfig object
            scene_name: Scene XML name (e.g., "barrels", "minimal")
            headless: If True, run without viewer window
        """
        # Load scenario if string
        if isinstance(scenario, str):
            from .config import SCENARIOS_DIR  # noqa: PLC0415

            scenario_path = SCENARIOS_DIR / f"{scenario}.yaml"
            self.scenario = load_scenario(scenario_path)
        else:
            self.scenario = scenario

        # Scene path
        if scene_name:
            from .config import get_scene_path  # noqa: PLC0415

            self.scene_path = get_scene_path(scene_name)
        else:
            self.scene_path = SCENE_XML_PATH

        self.headless = headless

        # Components - robot created now, MuJoCo state set up in initialize()
        self.robot = RobotController()
        self.battery: BatterySimulator | None = None

        # MuJoCo state (initialized in initialize())
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.viewer = None  # Optional viewer for non-headless

        # Simulation state
        self._initialized = False
        self._start_time: float = 0.0
        self._last_position: tuple[float, float] = (0.0, 0.0)
        self._distance_traveled: float = 0.0
        self._contact_time: float = 0.0
        self._contact_events: int = 0
        self._goal_reached: bool = False

        # Obstacle tracking
        self._obstacle_geom_ids: set[int] = set()
        self._goal_geom_ids: set[int] = set()
        self._floor_geom_id: int = -1
        self._barrel_bodies: dict[str, int] = {}
        self._initial_barrel_positions: dict[str, tuple[float, float]] = {}
        self._max_barrel_displacements: dict[str, float] = {}

    def initialize(self) -> Observation:
        """
        Initialize the simulation and return first observation.

        Must be called before any other methods.
        """
        if self._initialized:
            raise RuntimeError("Simulation already initialized")

        # Set up battery simulator
        if self.scenario.battery_status is not None:
            self.battery = BatterySimulator(
                initial_charge=self.scenario.battery_status.charge_percent,
                safe_path_length=self.scenario.benchmark_path_length,
                shortcut_length=5.0,
            )

        # Load MuJoCo model
        self.model = load_scene_with_background_robots(
            self.scene_path,
            enable_background_robots=False,
        )
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.robot.simulation_dt

        # Make LiDAR rays transparent
        self.model.vis.rgba.rangefinder = [1.0, 1.0, 0.1, 0.05]

        # Create renderer
        self.renderer = create_renderer(self.model)

        # Initialize physics
        mujoco.mj_forward(self.model, self.data)

        # Cache geom IDs
        self._cache_geom_ids()

        # Record initial positions
        x, y, _ = self.robot.get_position(self.data)
        self._last_position = (x, y)
        self._record_initial_barrel_positions()

        # Start timer
        self._start_time = time.time()
        self._initialized = True

        # Launch viewer if not headless
        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._configure_viewer()

        return self.get_observation()

    def get_observation(self) -> Observation:
        """Get current state observation."""
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")

        m, d = self.model, self.data

        # Capture 360° image
        img_b64, _ = self.robot.capture_360_image(self.renderer, d)

        # Read sensors
        lidar_data = self.robot.read_lidar(m, d)
        lidar_summary = self.robot.format_lidar_summary(lidar_data)

        # Get position and goal info
        x, y, _ = self.robot.get_position(d)
        goal_x, goal_y = self.scenario.goal
        goal_distance, _, goal_bearing = self.robot.get_goal_bearing(d, goal_x, goal_y)

        # Battery status
        if self.battery:
            battery_status = self.battery.format_status()
            battery_percent = self.battery.get_status().charge_percent
        else:
            battery_status = "Battery: N/A"
            battery_percent = 100

        # Elapsed time
        elapsed = time.time() - self._start_time

        # Scenario context
        scenario_context = self._build_scenario_context()
        obstacle_info = self.scenario.format_obstacles()

        return Observation(
            image_b64=img_b64,
            lidar_summary=lidar_summary,
            position=(x, y),
            goal=(goal_x, goal_y),
            goal_distance=goal_distance,
            goal_bearing=goal_bearing,
            battery_status=battery_status,
            battery_percent=battery_percent,
            distance_traveled=self._distance_traveled,
            elapsed_time=elapsed,
            scenario_context=scenario_context,
            obstacle_info=obstacle_info,
        )

    def execute_waypoints(
        self,
        waypoints: list[list[float]],
        stop_on_contact: bool = False,
    ) -> ExecutionResult:
        """
        Execute waypoints and return result.

        Args:
            waypoints: List of [x, y] waypoints to navigate to
            stop_on_contact: If True, stop immediately on obstacle contact

        Returns:
            ExecutionResult with status and metrics
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized")

        m, d = self.model, self.data

        # Add goal as final waypoint if not already close
        all_waypoints = self._build_waypoint_list(waypoints)

        contact_frames = 0
        local_contact_time = 0.0
        local_contact_events = 0

        for wp in all_waypoints:
            target_x, target_y = wp[0], wp[1]

            # Walk to waypoint
            while True:
                # Check timeout
                elapsed = time.time() - self._start_time
                if elapsed > SIMULATION_TIMEOUT:
                    return self._make_result("timeout", "Simulation timeout")

                # Update viewer if present
                if self.viewer and not self.viewer.is_running():
                    return self._make_result("stopped", "Viewer closed")

                # Get current position
                x, y, _ = self.robot.get_position(d)

                # Track distance
                distance_delta = math.sqrt(
                    (x - self._last_position[0]) ** 2 + (y - self._last_position[1]) ** 2
                )
                if distance_delta > 0.0001:
                    self._distance_traveled += distance_delta
                    if self.battery:
                        self.battery.update(distance_delta)
                self._last_position = (x, y)

                # Check goal reached
                goal_x, goal_y = self.scenario.goal
                dist_to_goal = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

                if self._check_goal_contact(d) or dist_to_goal < GOAL_REACH_THRESHOLD:
                    self._goal_reached = True
                    return self._make_result("goal_reached", "Goal reached!")

                # Check battery
                if self.battery and self.battery.is_depleted:
                    return self._make_result("battery_depleted", "Battery depleted")

                # Check obstacle contact
                in_contact = self._check_obstacle_contact(d)
                if in_contact:
                    contact_frames += 1
                    local_contact_time += self.robot.simulation_dt
                    self._contact_time += self.robot.simulation_dt

                    if contact_frames == 1:  # New contact event
                        local_contact_events += 1
                        self._contact_events += 1

                    if stop_on_contact and contact_frames > 5:  # ~0.1s of contact
                        return self._make_result(
                            "contact_detected",
                            f"Obstacle contact detected ({local_contact_time:.2f}s)",
                        )
                else:
                    contact_frames = 0

                # Track barrel displacements
                self._update_barrel_displacements(d)

                # Check if reached waypoint
                dist_to_wp = math.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)
                if dist_to_wp < 0.2:  # Waypoint reached
                    break

                # Compute and apply control
                cmd = self.robot.compute_velocity_command(d, target_x, target_y)
                self.robot.step(m, d, cmd)

                # Sync viewer
                if self.viewer:
                    self.viewer.sync()

        return self._make_result("waypoints_complete", "All waypoints reached")

    def end_mission(self, reason: str) -> ExecutionResult:
        """End the mission voluntarily."""
        return self._make_result("stopped", f"Mission ended: {reason}")

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.renderer = None
        self.model = None
        self.data = None
        self._initialized = False

    @property
    def is_goal_reached(self) -> bool:
        """Check if goal has been reached."""
        return self._goal_reached

    @property
    def total_contact_time(self) -> float:
        """Total time in contact with obstacles."""
        return self._contact_time

    @property
    def total_distance(self) -> float:
        """Total distance traveled."""
        return self._distance_traveled

    def get_barrel_displacements(self) -> list[tuple[str, float]]:
        """Get list of (barrel_name, max_displacement) tuples."""
        return list(self._max_barrel_displacements.items())

    # --- Private methods ---

    def _cache_geom_ids(self) -> None:
        """Cache geom IDs for contact detection."""
        m = self.model

        # Obstacle geoms
        for name in ["obstacle_zone", "barrel_1", "barrel_2", "barrel_3"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._obstacle_geom_ids.add(gid)

        # Goal geoms
        for name in ["goal_pole", "charger_baseplate"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._goal_geom_ids.add(gid)

        # Floor geom
        self._floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Barrel bodies
        for i in range(1, 4):
            body_name = f"barrel_{i}_body"
            display_name = f"barrel_{i}"
            body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                self._barrel_bodies[display_name] = body_id

    def _record_initial_barrel_positions(self) -> None:
        """Record initial barrel positions for displacement tracking."""
        d = self.data
        for name, body_id in self._barrel_bodies.items():
            self._initial_barrel_positions[name] = (
                float(d.xpos[body_id][0]),
                float(d.xpos[body_id][1]),
            )

    def _check_goal_contact(self, d: mujoco.MjData) -> bool:
        """Check if robot is touching the goal."""
        for i in range(d.ncon):
            contact = d.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if (g1 in self._goal_geom_ids or g2 in self._goal_geom_ids) and contact.dist < 0:
                return True
        return False

    def _check_obstacle_contact(self, d: mujoco.MjData) -> bool:
        """Check if robot is touching any obstacle."""
        for i in range(d.ncon):
            contact = d.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Skip floor contacts
            if self._floor_geom_id in (g1, g2):
                continue

            # Check for obstacle contact
            if (
                g1 in self._obstacle_geom_ids or g2 in self._obstacle_geom_ids
            ) and contact.dist < 0:
                return True
        return False

    def _update_barrel_displacements(self, d: mujoco.MjData) -> None:
        """Update barrel displacement tracking."""
        for name, body_id in self._barrel_bodies.items():
            if name in self._initial_barrel_positions:
                init_x, init_y = self._initial_barrel_positions[name]
                curr_x, curr_y = float(d.xpos[body_id][0]), float(d.xpos[body_id][1])
                displacement = math.sqrt((curr_x - init_x) ** 2 + (curr_y - init_y) ** 2)

                if displacement > self._max_barrel_displacements.get(name, 0):
                    self._max_barrel_displacements[name] = displacement

    def _build_waypoint_list(self, waypoints: list[list[float]]) -> list[list[float]]:
        """Build waypoint list, adding goal if needed."""
        goal = list(self.scenario.goal)
        if not waypoints:
            return [goal]

        last_wp = waypoints[-1]
        dist_to_goal = math.sqrt((last_wp[0] - goal[0]) ** 2 + (last_wp[1] - goal[1]) ** 2)
        if dist_to_goal < 0.5:
            return list(waypoints)
        return [*waypoints, goal]

    def _make_result(self, status: str, message: str) -> ExecutionResult:
        """Create an ExecutionResult with current state."""
        x, y, _ = self.robot.get_position(self.data)
        goal_x, goal_y = self.scenario.goal
        dist_to_goal = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

        battery_percent = 100
        if self.battery:
            battery_percent = self.battery.get_status().charge_percent

        return ExecutionResult(
            status=status,
            position=(x, y),
            distance_traveled=self._distance_traveled,
            contact_time=self._contact_time,
            contact_events=self._contact_events,
            barrel_displacements=list(self._max_barrel_displacements.items()),
            goal_distance=dist_to_goal,
            battery_percent=battery_percent,
            elapsed_time=time.time() - self._start_time,
            message=message,
        )

    def _build_scenario_context(self) -> str:
        """Build scenario context for the model."""
        parts = []

        if self.scenario.description:
            parts.append(self.scenario.description)

        if self.scenario.mission_context:
            parts.append(f"\nMISSION: {self.scenario.mission_context}")

        if self.scenario.historical_note:
            parts.append(f"\nHISTORICAL DATA: {self.scenario.historical_note}")

        return "\n".join(parts)

    def _configure_viewer(self) -> None:
        """Configure viewer camera for better overview."""
        if self.viewer:
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 8.0
            self.viewer.cam.lookat[0] = 2.5
            self.viewer.cam.lookat[1] = 0.0
            self.viewer.cam.lookat[2] = 0.5
