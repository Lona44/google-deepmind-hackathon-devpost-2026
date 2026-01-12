"""
ROS2-enabled entry point for G1 Alignment experiments.

This entry point runs the simulation with ROS2 bridge enabled,
publishing sensor data to Nav2-compatible topics.

Usage:
    # Inside Docker container with ROS2:
    python3 -m src.ros2_main

    # With specific scenario:
    python3 -m src.ros2_main scenarios/forbidden_zone.yaml

    # Sensor-only mode (no Gemini, just publishes to ROS2):
    python3 -m src.ros2_main --sensor-only

Requirements:
    - ROS2 Humble (or later)
    - rclpy, sensor_msgs, nav_msgs, geometry_msgs, tf2_ros
"""

import sys
import time

# Check ROS2 availability early
try:
    import rclpy
except ImportError:
    print("ERROR: ROS2 (rclpy) not found!")
    print("This entry point requires ROS2. Please run inside a ROS2 environment.")
    print("For non-ROS2 usage, use: mjpython -m src.main")
    sys.exit(1)

import mujoco

from .config import load_scenario
from .robot import RobotController
from .ros2_bridge import MuJoCoROS2Bridge


def run_sensor_only_mode(scenario_path: str | None = None) -> None:
    """
    Run MuJoCo simulation publishing only sensor data to ROS2.

    This mode doesn't use Gemini - it's for testing ROS2/Nav2 integration.
    The robot can be controlled via /cmd_vel topic from Nav2 or teleop.

    Args:
        scenario_path: Path to scenario YAML file.
    """
    print("=" * 60)
    print("G1 ROS2 Bridge - Sensor Only Mode")
    print("=" * 60)

    # Load scenario and model
    scenario = load_scenario(scenario_path)
    print(f"Loaded scenario: {scenario.name}")

    # Load MuJoCo model
    model_path = scenario.scene_path
    print(f"Loading MuJoCo model: {model_path}")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    # Initialize robot controller
    robot = RobotController()

    # Create ROS2 bridge
    print("Initializing ROS2 bridge...")
    bridge = MuJoCoROS2Bridge(robot, m, d)
    print("ROS2 bridge initialized!")

    # Print published topics
    print("\nPublished topics:")
    print("  /scan        - sensor_msgs/LaserScan (36-ray LiDAR)")
    print("  /odom        - nav_msgs/Odometry")
    print("  /imu         - sensor_msgs/Imu")
    print("  /clock       - rosgraph_msgs/Clock")
    print("  /tf          - tf2_msgs/TFMessage")
    print("\nSubscribed topics:")
    print("  /cmd_vel     - geometry_msgs/Twist (velocity commands)")
    print("\nTF tree: odom -> base_link -> lidar_link")
    print("=" * 60)

    # Create viewer for visualization
    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            print("\nSimulation running. Press Ctrl+C to stop.")
            print("Send velocity commands to /cmd_vel to control the robot.")

            simulation_dt = robot.simulation_dt
            step_count = 0

            while viewer.is_running():
                step_start = time.time()

                # Get velocity command from ROS2 (from Nav2 or teleop)
                cmd = bridge.get_cmd_vel()

                # Step robot controller
                robot.step(d, cmd)

                # Step physics
                mujoco.mj_step(m, d)

                # Step ROS2 bridge (publish sensor data)
                bridge.step(simulation_dt)

                step_count += 1

                # Sync with viewer
                viewer.sync()

                # Maintain real-time
                elapsed = time.time() - step_start
                sleep_time = simulation_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        bridge.shutdown()
        rclpy.shutdown()
        print("Shutdown complete.")


def run_with_ros2(
    scenario_path: str | None = None,
    enable_retries: bool = True,
    max_attempts: int = 5,
) -> None:
    """
    Run full experiment with ROS2 bridge enabled.

    This combines Gemini decision-making with ROS2 sensor publishing,
    allowing Nav2 to provide local obstacle avoidance while Gemini
    handles high-level planning.

    Args:
        scenario_path: Path to scenario YAML file.
        enable_retries: Enable retry loop.
        max_attempts: Maximum retry attempts.
    """
    # Lazy imports to avoid loading Gemini when running sensor-only mode
    from .gemini_client import GeminiNavigator  # noqa: PLC0415
    from .logger import ExperimentLogger  # noqa: PLC0415
    from .simulation import SimulationRunner  # noqa: PLC0415

    print("=" * 60)
    print("G1 ROS2 Bridge - Full Experiment Mode")
    print("=" * 60)

    # Load scenario
    scenario = load_scenario(scenario_path)
    print(f"Loaded scenario: {scenario.name}")

    # Initialize components
    logger = ExperimentLogger(scenario.name + "_ros2")
    robot = RobotController()
    gemini = GeminiNavigator(enable_retries=enable_retries)
    gemini.set_logger(logger)

    # Run simulation with ROS2 bridge
    # Note: SimulationRunner will need modification to accept ros2_bridge
    runner = SimulationRunner(scenario, robot, gemini, logger, enable_ros2=True)

    try:
        if enable_retries:
            result = runner.run_with_retries(max_attempts=max_attempts)
            any_passed = any(r.passed for r in result)
            print(f"\nExperiment complete. Any passed: {any_passed}")
        else:
            result = runner.run()
            print(f"\nExperiment complete. Passed: {result.passed}")
    except KeyboardInterrupt:
        print("\nExperiment interrupted.")
    finally:
        rclpy.shutdown()


def main() -> int:
    """Main entry point for ROS2 mode."""
    scenario_path = None
    sensor_only = False
    enable_retries = True
    max_attempts = 5

    args = sys.argv[1:]
    for arg in args:
        if arg == "--sensor-only":
            sensor_only = True
        elif arg == "--single":
            enable_retries = False
        elif arg.isdigit():
            max_attempts = int(arg)
        elif arg in ("--help", "-h"):
            print(__doc__)
            return 0
        elif arg.startswith("--"):
            print(f"Unknown option: {arg}")
            return 2
        elif scenario_path is None:
            scenario_path = arg

    try:
        if sensor_only:
            run_sensor_only_mode(scenario_path)
        else:
            run_with_ros2(
                scenario_path=scenario_path,
                enable_retries=enable_retries,
                max_attempts=max_attempts,
            )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
