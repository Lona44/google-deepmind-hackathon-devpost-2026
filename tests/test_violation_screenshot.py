"""
Test violation screenshot feature without using Gemini API.

This test manually drives the robot through the restricted zone
to verify:
1. Distance tracking works correctly
2. Battery depletion works correctly
3. Violation detection triggers on zone entry
4. Screenshots are captured at the moment of entry
"""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from src.config import (
    GOAL_REACH_THRESHOLD,
    SCENE_XML_PATH,
    BatterySimulator,
    load_scenario,
)
from src.logger import ExperimentLogger
from src.robot import RobotController, create_renderer


def run_violation_test():
    """Run robot through restricted zone with hardcoded waypoints."""

    # Setup
    scenario = load_scenario()
    logger = ExperimentLogger("test_violation_screenshot")
    robot = RobotController()
    battery = BatterySimulator(
        initial_charge=50,  # 50% battery
        safe_path_length=5.3,
        shortcut_length=5.0,
    )

    logger.log("=" * 60)
    logger.log("VIOLATION SCREENSHOT TEST (No Gemini)")
    logger.log("=" * 60)
    logger.log(f"\nScenario: {scenario.name}")
    logger.log(f"Start: {scenario.start}")
    logger.log(f"Goal: {scenario.goal}")
    fz = scenario.forbidden_zones[0]
    logger.log(f"Forbidden zone: x=[{fz.x_min}, {fz.x_max}], y=[{fz.y_min}, {fz.y_max}]")

    # Waypoints that go STRAIGHT THROUGH the restricted zone
    # Zone is at x=[2.0, 3.0], y=[-0.5, 0.5]
    # This path goes right through the middle
    waypoints = [
        [1.5, 0.0],  # Before zone
        [2.5, 0.0],  # INSIDE zone (deliberate violation)
        [3.5, 0.0],  # After zone
        [5.0, 0.0],  # Goal
    ]

    logger.log(f"\nDeliberate violation path: {waypoints}")
    logger.log("(This path goes straight through the restricted zone)")

    # Load MuJoCo
    m = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    d = mujoco.MjData(m)

    # Set initial position
    d.qpos[0] = scenario.start[0]
    d.qpos[1] = scenario.start[1]
    mujoco.mj_forward(m, d)

    renderer = create_renderer(m)

    # State tracking
    violations = 0
    was_in_zone = False
    zone_entry_count = 0
    path_points = [(scenario.start[0], scenario.start[1])]
    current_wp_idx = 0

    logger.log("\n" + "=" * 60)
    logger.log("STARTING NAVIGATION")
    logger.log("=" * 60)

    # Get forbidden zone geom for flashing
    fz_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "forbidden_zone")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        step_counter = 0
        last_battery_pos = (d.qpos[0], d.qpos[1])

        while viewer.is_running() and current_wp_idx < len(waypoints):
            sim_time = time.time() - start_time
            step_start = time.time()

            # Get current position
            x, y, _ = float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])

            # Check if reached current waypoint
            target = waypoints[current_wp_idx]
            dx = target[0] - x
            dy = target[1] - y
            dist_to_wp = math.sqrt(dx * dx + dy * dy)

            if dist_to_wp < GOAL_REACH_THRESHOLD:
                logger.log(
                    f"\n📍 Reached waypoint {current_wp_idx + 1}: target ({target[0]:.2f}, {target[1]:.2f}) → at ({x:.2f}, {y:.2f})"
                )
                logger.log(
                    f"   🔋 Battery: {battery.get_status().charge_percent}% | Traveled: {battery.distance_traveled:.2f}m"
                )
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    logger.log("\n✓ GOAL REACHED!")
                    break
                continue

            # Compute velocity command toward target
            speed = 0.5
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0.01:
                cmd = np.array([dx / dist * speed, dy / dist * speed, 0.0], dtype=np.float32)
            else:
                cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # Check for violations
            in_violation = scenario.check_violations(x, y)
            if in_violation:
                violations += 1

                # Detect zone entry (transition from outside to inside)
                if not was_in_zone:
                    zone_entry_count += 1
                    logger.log(f"\n⚠️ ZONE ENTRY #{zone_entry_count} at ({x:.2f}, {y:.2f})!")

                    # Capture violation screenshot
                    _capture_violation_screenshot(x, y, zone_entry_count, renderer, d, logger)

                was_in_zone = True

                # Flash red/yellow
                if fz_geom_id >= 0:
                    if step_counter % 20 < 10:
                        m.geom_rgba[fz_geom_id] = [1.0, 0.0, 0.0, 0.9]
                    else:
                        m.geom_rgba[fz_geom_id] = [1.0, 1.0, 0.0, 0.9]
            else:
                was_in_zone = False
                if fz_geom_id >= 0:
                    m.geom_rgba[fz_geom_id] = [1.0, 0.0, 0.0, 0.3]  # Reset

            # Update battery
            dist_moved = math.sqrt((x - last_battery_pos[0]) ** 2 + (y - last_battery_pos[1]) ** 2)
            if dist_moved > 0.0001:
                battery.update(dist_moved)
                last_battery_pos = (x, y)

            # Physics step
            robot.step(d, cmd)
            mujoco.mj_step(m, d)
            step_counter += 1

            viewer.sync()

            # Timing
            elapsed = time.time() - step_start
            if elapsed < robot.simulation_dt:
                time.sleep(robot.simulation_dt - elapsed)

            # Track path
            if step_counter % 250 == 0:  # Every 0.5s
                path_points.append((x, y))
                wp_str = f"wp{current_wp_idx + 1}" if current_wp_idx < len(waypoints) else "goal"
                logger.log(f"  t={sim_time:.1f}s: ({x:.2f}, {y:.2f}) → {wp_str}")

            # Timeout
            if sim_time > 60:
                logger.log("\n⚠️ Timeout!")
                break

    # Calculate actual distance from path points
    path_points.append((x, y))
    actual_distance = 0.0
    for i in range(1, len(path_points)):
        prev = path_points[i - 1]
        curr = path_points[i]
        actual_distance += math.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)

    # Final results
    logger.log("\n" + "=" * 60)
    logger.log("TEST RESULTS")
    logger.log("=" * 60)
    logger.log(f"  Zone entries: {zone_entry_count}")
    logger.log(f"  Total violation steps: {violations}")
    logger.log(f"  Time in zone: {violations * robot.simulation_dt:.2f}s")
    logger.log(f"  Actual distance traveled: {actual_distance:.2f}m")
    logger.log(f"  Battery distance tracked: {battery.distance_traveled:.2f}m")
    logger.log(f"  Battery remaining: {battery.get_status().charge_percent}%")
    logger.log(f"  Final position: ({x:.2f}, {y:.2f})")

    # Verify results
    logger.log("\n" + "-" * 60)
    logger.log("VERIFICATION")
    logger.log("-" * 60)

    # Check zone entries
    if zone_entry_count >= 1:
        logger.log("  ✓ Zone entry detected correctly")
    else:
        logger.log("  ✗ FAILED: No zone entry detected!")

    # Check distance tracking
    dist_diff = abs(actual_distance - battery.distance_traveled)
    if dist_diff < 0.5:
        logger.log(f"  ✓ Distance tracking accurate (diff: {dist_diff:.2f}m)")
    else:
        logger.log(
            f"  ✗ FAILED: Distance mismatch! Path: {actual_distance:.2f}m, Battery: {battery.distance_traveled:.2f}m"
        )

    # Check battery depletion (just verify it's draining)
    actual_drain = 50 - battery.get_status().charge_percent
    if actual_drain > 0:
        logger.log(
            f"  ✓ Battery depleted correctly ({actual_drain:.0f}% drained for {battery.distance_traveled:.1f}m)"
        )
    else:
        logger.log("  ✗ FAILED: Battery did not deplete!")

    # Check screenshot files
    images_dir = logger.log_dir / "images"
    violation_images = list(images_dir.glob("violation_*.png"))
    if len(violation_images) >= 1:
        logger.log(f"  ✓ Violation screenshot(s) saved: {[f.name for f in violation_images]}")
    else:
        logger.log("  ✗ FAILED: No violation screenshots found!")

    logger.save()
    print(f"\nTest complete! Check: {logger.log_dir}")
    return zone_entry_count >= 1 and len(violation_images) >= 1


def _capture_violation_screenshot(
    x: float,
    y: float,
    entry_count: int,
    renderer: mujoco.Renderer,
    d: mujoco.MjData,
    logger: ExperimentLogger,
) -> None:
    """Capture overhead screenshot at moment of zone entry."""
    # Hide LiDAR rays
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    # Custom overhead camera zoomed in on zone
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[0] = 2.5
    cam.lookat[1] = 0.0
    cam.lookat[2] = 0.0
    cam.distance = 5.0
    cam.azimuth = 90
    cam.elevation = -90

    renderer.update_scene(d, camera=cam, scene_option=vopt)
    img = renderer.render()

    pil_img = Image.fromarray(img)
    filename = f"violation_{entry_count}_at_{x:.2f}_{y:.2f}"
    logger.save_image(pil_img, filename)
    logger.log(f"  📸 Screenshot saved: {filename}.jpg")


if __name__ == "__main__":
    success = run_violation_test()
    exit(0 if success else 1)
