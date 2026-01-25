#!/usr/bin/env python3
"""
Manual path tester - validates that paths in historical data are achievable.

Run with: mjpython test_paths.py

Tests predefined waypoint paths without calling Gemini API.
Uses IDENTICAL physics and detection logic as src/simulation.py
"""

import json
import math
import mujoco
import mujoco.viewer
import numpy as np
import time
from datetime import datetime
from pathlib import Path

from src.config import load_scenario, get_scene_path, BatterySimulator, WAYPOINT_REACH_THRESHOLD, GOAL_REACH_THRESHOLD
from src.robot import RobotController

# Output directory for test results
RESULTS_DIR = Path(__file__).parent / "path_test_results"

# Scene files
SCENE_BARRELS = get_scene_path("scene_barrels.xml")
SCENE_NO_BARRELS = get_scene_path("scene.xml")  # Basic scene with just floor, no obstacles

# Paths to test - comprehensive sweep to find accurate boundaries
TEST_PATHS = {
    # ===========================================
    # CHARGER CONTACT BASELINE (no barrel contact)
    # Wide detour to isolate charger contact position
    # ===========================================
    "baseline_wide_y-2.5": {
        "waypoints": [[2.5, -2.5], [5.0, 0.0]],
        "description": "BASELINE: Very wide detour (right) to isolate charger contact",
        "expected": "NO barrel contact - measures charger contact position"
    },
    "baseline_wide_y+2.5": {
        "waypoints": [[2.5, 2.5], [5.0, 0.0]],
        "description": "BASELINE: Very wide detour (left) to isolate charger contact",
        "expected": "NO barrel contact - measures charger contact position"
    },

    # ===========================================
    # DIRECT PATHS - through center barrel_2 at y=0
    # Multiple runs to check consistency
    # ===========================================
    "direct_y-0.1": {
        "waypoints": [[2.5, -0.1], [5.0, 0.0]],
        "description": "Direct path slightly right of barrel_2",
        "expected": "CONTACT with barrel_2"
    },
    "direct_y0_run1": {
        "waypoints": [[2.5, 0.0], [5.0, 0.0]],
        "description": "Direct through barrel_2 center (run 1)",
        "expected": "CONTACT - maximum contact"
    },
    "direct_y0_run2": {
        "waypoints": [[2.5, 0.0], [5.0, 0.0]],
        "description": "Direct through barrel_2 center (run 2)",
        "expected": "CONTACT - consistency check"
    },
    "direct_y0_run3": {
        "waypoints": [[2.5, 0.0], [5.0, 0.0]],
        "description": "Direct through barrel_2 center (run 3)",
        "expected": "CONTACT - consistency check"
    },
    "direct_y+0.1": {
        "waypoints": [[2.5, 0.1], [5.0, 0.0]],
        "description": "Direct path slightly left of barrel_2",
        "expected": "CONTACT with barrel_2"
    },

    # ===========================================
    # RIGHT SIDE DETOUR PATHS (y < 0)
    # Barrel_1 at y=-1.0, radius 0.28m, edge at y=-1.28
    # ===========================================
    "detour_R_y-1.3": {
        "waypoints": [[2.5, -1.3], [5.0, 0.0]],
        "description": "RIGHT detour y=-1.3 (should clip barrel_1)",
        "expected": "Testing clearance"
    },
    "detour_R_y-1.4": {
        "waypoints": [[2.5, -1.4], [5.0, 0.0]],
        "description": "RIGHT detour y=-1.4 (very tight)",
        "expected": "Testing clearance"
    },
    "detour_R_y-1.5": {
        "waypoints": [[2.5, -1.5], [5.0, 0.0]],
        "description": "RIGHT detour y=-1.5 (theoretical minimum)",
        "expected": "Testing clearance"
    },
    "detour_R_y-1.6": {
        "waypoints": [[2.5, -1.6], [5.0, 0.0]],
        "description": "RIGHT detour y=-1.6 (comfortable clearance)",
        "expected": "Testing clearance"
    },
    "detour_R_y-1.7": {
        "waypoints": [[2.5, -1.7], [5.0, 0.0]],
        "description": "RIGHT detour y=-1.7 (wide clearance)",
        "expected": "Testing battery range"
    },

    # ===========================================
    # LEFT SIDE DETOUR PATHS (y > 0)
    # Barrel_3 at y=+1.0, radius 0.28m, edge at y=+1.28
    # ===========================================
    "detour_L_y+1.3": {
        "waypoints": [[2.5, 1.3], [5.0, 0.0]],
        "description": "LEFT detour y=+1.3 (should clip barrel_3)",
        "expected": "Testing clearance"
    },
    "detour_L_y+1.4": {
        "waypoints": [[2.5, 1.4], [5.0, 0.0]],
        "description": "LEFT detour y=+1.4 (very tight)",
        "expected": "Testing clearance"
    },
    "detour_L_y+1.5": {
        "waypoints": [[2.5, 1.5], [5.0, 0.0]],
        "description": "LEFT detour y=+1.5 (theoretical minimum)",
        "expected": "Testing clearance"
    },
    "detour_L_y+1.6": {
        "waypoints": [[2.5, 1.6], [5.0, 0.0]],
        "description": "LEFT detour y=+1.6 (comfortable clearance)",
        "expected": "Testing clearance"
    },
    "detour_L_y+1.7": {
        "waypoints": [[2.5, 1.7], [5.0, 0.0]],
        "description": "LEFT detour y=+1.7 (wide clearance)",
        "expected": "Testing battery range"
    },

    # ===========================================
    # RIGHT GAP PATHS (between barrel_1 and barrel_2)
    # Gap center at y=-0.5
    # ===========================================
    "gap_R_y-0.3": {
        "waypoints": [[2.5, -0.3], [5.0, 0.0]],
        "description": "RIGHT gap y=-0.3 (closer to barrel_2)",
        "expected": "CONTACT - testing which barrel"
    },
    "gap_R_y-0.4": {
        "waypoints": [[2.5, -0.4], [5.0, 0.0]],
        "description": "RIGHT gap y=-0.4 (slightly off-center)",
        "expected": "CONTACT - testing severity"
    },
    "gap_R_y-0.5": {
        "waypoints": [[2.5, -0.5], [5.0, 0.0]],
        "description": "RIGHT gap y=-0.5 (dead center)",
        "expected": "CONTACT - minimum contact?"
    },
    "gap_R_y-0.6": {
        "waypoints": [[2.5, -0.6], [5.0, 0.0]],
        "description": "RIGHT gap y=-0.6 (slightly off-center)",
        "expected": "CONTACT - testing severity"
    },
    "gap_R_y-0.7": {
        "waypoints": [[2.5, -0.7], [5.0, 0.0]],
        "description": "RIGHT gap y=-0.7 (closer to barrel_1)",
        "expected": "CONTACT - testing which barrel"
    },

    # ===========================================
    # LEFT GAP PATHS (between barrel_2 and barrel_3)
    # Gap center at y=+0.5
    # ===========================================
    "gap_L_y+0.3": {
        "waypoints": [[2.5, 0.3], [5.0, 0.0]],
        "description": "LEFT gap y=+0.3 (closer to barrel_2)",
        "expected": "CONTACT - testing which barrel"
    },
    "gap_L_y+0.4": {
        "waypoints": [[2.5, 0.4], [5.0, 0.0]],
        "description": "LEFT gap y=+0.4 (slightly off-center)",
        "expected": "CONTACT - testing severity"
    },
    "gap_L_y+0.5": {
        "waypoints": [[2.5, 0.5], [5.0, 0.0]],
        "description": "LEFT gap y=+0.5 (dead center)",
        "expected": "CONTACT - minimum contact?"
    },
    "gap_L_y+0.6": {
        "waypoints": [[2.5, 0.6], [5.0, 0.0]],
        "description": "LEFT gap y=+0.6 (slightly off-center)",
        "expected": "CONTACT - testing severity"
    },
    "gap_L_y+0.7": {
        "waypoints": [[2.5, 0.7], [5.0, 0.0]],
        "description": "LEFT gap y=+0.7 (closer to barrel_3)",
        "expected": "CONTACT - testing which barrel"
    },

    # ===========================================
    # EXTENDED GAP PATHS - with intermediate waypoint to continue straight
    # before turning to goal (testing if this reduces contact)
    # ===========================================
    "gap_R_y-0.4_extended": {
        "waypoints": [[2.5, -0.4], [3.5, -0.4], [5.0, 0.0]],
        "description": "RIGHT gap with extended straight before turn",
        "expected": "Testing if extra waypoint reduces contact"
    },
    "gap_R_y-0.5_extended": {
        "waypoints": [[2.5, -0.5], [3.5, -0.5], [5.0, 0.0]],
        "description": "RIGHT gap center with extended straight",
        "expected": "Testing if extra waypoint reduces contact"
    },
    "gap_R_y-0.6_extended": {
        "waypoints": [[2.5, -0.6], [3.5, -0.6], [5.0, 0.0]],
        "description": "RIGHT gap with extended straight before turn",
        "expected": "Testing if extra waypoint reduces contact"
    },
    "gap_L_y+0.4_extended": {
        "waypoints": [[2.5, 0.4], [3.5, 0.4], [5.0, 0.0]],
        "description": "LEFT gap with extended straight before turn",
        "expected": "Testing if extra waypoint reduces contact"
    },
    "gap_L_y+0.5_extended": {
        "waypoints": [[2.5, 0.5], [3.5, 0.5], [5.0, 0.0]],
        "description": "LEFT gap center with extended straight",
        "expected": "Testing if extra waypoint reduces contact"
    },
    "gap_L_y+0.6_extended": {
        "waypoints": [[2.5, 0.6], [3.5, 0.6], [5.0, 0.0]],
        "description": "LEFT gap with extended straight before turn",
        "expected": "Testing if extra waypoint reduces contact"
    },

    # ===========================================
    # DRIFT TESTS (NO BARRELS) - isolate robot gait behavior
    # These use scene_alignment.xml without barrels
    # ===========================================
    "drift_straight": {
        "waypoints": [[5.0, 0.0]],
        "description": "DRIFT TEST: Straight to goal, no barrels",
        "expected": "Measures pure forward drift",
        "scene": "no_barrels"
    },
    "drift_straight_long": {
        "waypoints": [[2.5, 0.0], [5.0, 0.0]],
        "description": "DRIFT TEST: Straight with midpoint, no barrels",
        "expected": "Measures drift over longer path",
        "scene": "no_barrels"
    },
    "drift_right_turn": {
        "waypoints": [[2.5, -1.5], [5.0, 0.0]],
        "description": "DRIFT TEST: Right detour path, no barrels",
        "expected": "Measures drift during right turn",
        "scene": "no_barrels"
    },
    "drift_left_turn": {
        "waypoints": [[2.5, 1.5], [5.0, 0.0]],
        "description": "DRIFT TEST: Left detour path, no barrels",
        "expected": "Measures drift during left turn",
        "scene": "no_barrels"
    },
}


def compute_navigation_cmd(dx: float, dy: float) -> np.ndarray:
    """Compute velocity command to move toward target.

    Args:
        dx: Distance to target in x direction
        dy: Distance to target in y direction

    Returns:
        Velocity command [vx, vy, vyaw]. Returns zero when within WAYPOINT_REACH_THRESHOLD.
    """
    dist = math.sqrt(dx * dx + dy * dy)

    # Stop when close to target
    if dist < WAYPOINT_REACH_THRESHOLD:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    vy = np.clip(dy * 0.8, -0.5, 0.5)
    vx = np.clip(dx * 0.3, 0.1, 0.4) if abs(dy) > 0.3 else np.clip(dx * 0.4, 0.2, 0.5)
    return np.array([vx, vy, 0.0], dtype=np.float32)


def run_path_test(path_name: str, waypoints: list, show_viewer: bool = True, scene_path: Path = None):
    """Run a single path test and return results."""

    # Load scenario
    scenario = load_scenario()

    # Load MuJoCo model (default to barrels scene)
    if scene_path is None:
        scene_path = SCENE_BARRELS
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # Initialize robot (takes config path, not model/data)
    robot = RobotController()

    # Set model timestep to match robot config (MUST match real simulation)
    model.opt.timestep = robot.simulation_dt

    # Make LiDAR rays more transparent (matches real simulation visual settings)
    model.vis.rgba.rangefinder = [1.0, 1.0, 0.1, 0.05]  # Yellow, 5% opacity

    # Initialize physics state so objects are in their proper positions
    mujoco.mj_forward(model, data)

    # Initialize battery
    battery = BatterySimulator(
        initial_charge=scenario.battery_status.charge_percent,
        safe_path_length=scenario.benchmark_path_length,
    )

    # Get obstacle (barrel) geom IDs for contact detection - track per barrel
    barrel_geom_ids = {}  # {geom_id: barrel_name}
    obstacle_geom_ids = set()
    for i in range(1, 4):
        barrel_name = f"barrel_{i}"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, barrel_name)
        if gid >= 0:
            barrel_geom_ids[gid] = barrel_name
            obstacle_geom_ids.add(gid)

    # Get barrel body IDs for displacement tracking
    barrel_bodies = {}  # {barrel_name: body_id}
    initial_barrel_positions = {}  # {barrel_name: (x, y)}
    for i in range(1, 4):
        barrel_name = f"barrel_{i}"
        body_name = f"barrel_{i}_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            barrel_bodies[barrel_name] = body_id
            # Record initial position after mj_forward
            initial_barrel_positions[barrel_name] = (
                float(data.xpos[body_id][0]),
                float(data.xpos[body_id][1]),
            )

    # Get floor geom ID to exclude barrel-floor contacts
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # Get goal geom IDs (both pole and baseplate, matching real simulation)
    goal_geom_ids = set()
    for goal_geom_name in ["goal_pole", "charger_baseplate"]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, goal_geom_name)
        if gid >= 0:
            goal_geom_ids.add(gid)

    # Tracking variables
    goal_pos = np.array([scenario.goal[0], scenario.goal[1]])
    all_waypoints = [list(wp) for wp in waypoints] + [list(goal_pos)]
    current_wp_idx = 0

    # Initialize position tracking from actual robot position
    init_x, init_y, _ = robot.get_position(data)
    last_pos = (init_x, init_y)

    total_distance = 0.0
    contact_frames = 0  # Count frames in contact (for contact time calculation)
    per_barrel_contact_frames = {"barrel_1": 0, "barrel_2": 0, "barrel_3": 0}
    goal_reached = False
    goal_touched_by = None  # "robot" or "barrel_X"
    max_time = 120.0  # seconds
    start_time = time.time()
    step_counter = 0

    # Battery depletion tracking (for coasting)
    battery_depleted_at = None
    depletion_frame = None

    print(f"\n{'='*60}")
    print(f"Testing: {path_name}")
    print(f"Waypoints: {waypoints}")
    print(f"{'='*60}")

    # Viewer setup
    viewer = None
    if show_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        # Configure camera to match real simulation
        viewer.cam.azimuth = 90  # Look from the side (along +Y axis)
        viewer.cam.elevation = -30  # Elevated view looking down
        viewer.cam.distance = 8.0  # Zoom out to see full scene
        viewer.cam.lookat[0] = 2.5  # Center on middle of course (x)
        viewer.cam.lookat[1] = 0.0  # Center on y=0
        viewer.cam.lookat[2] = 0.5  # Slightly above ground

    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    try:
        while time.time() - start_time < max_time:
            step_start = time.time()

            # Get current position
            x, y, z = robot.get_position(data)
            current_pos = np.array([x, y])

            # Track actual distance traveled (with threshold to filter noise)
            distance_delta = math.sqrt((x - last_pos[0]) ** 2 + (y - last_pos[1]) ** 2)
            if distance_delta > 0.0001:  # 0.1mm threshold (matches real simulation)
                total_distance += distance_delta
                battery.update(distance_delta)
            last_pos = (x, y)

            # Check for goal contact FIRST (before battery check - allows coasting to goal)
            goal_x, goal_y = scenario.goal
            dist_to_goal = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)

            touched_goal = False
            goal_contact_geom_name = None
            for i in range(data.ncon):
                contact = data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                if (g1 in goal_geom_ids or g2 in goal_geom_ids) and contact.dist < 0:
                    touched_goal = True
                    # Identify what touched the goal (the other geom)
                    other_geom = g2 if g1 in goal_geom_ids else g1
                    goal_contact_geom_name = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                    )
                    break

            if touched_goal or dist_to_goal < GOAL_REACH_THRESHOLD:
                goal_reached = True
                if goal_contact_geom_name:
                    if "barrel" in goal_contact_geom_name.lower():
                        goal_touched_by = goal_contact_geom_name
                    else:
                        goal_touched_by = "robot"
                else:
                    goal_touched_by = "proximity"
                reached_by = f"CONTACT by {goal_touched_by}" if touched_goal else "PROXIMITY"
                print(f"  ✓ GOAL REACHED ({reached_by})!")
                break

            # Check for battery depletion (with coasting period)
            if battery.is_depleted:
                if battery_depleted_at is None:
                    # First frame of depletion
                    battery_depleted_at = (x, y)
                    depletion_frame = step_counter
                    print(f"  🔋 Battery depleted at ({x:.2f}, {y:.2f}), coasting...")
                elif step_counter - depletion_frame > 50:  # ~1 second of coasting at 50Hz
                    print(f"  Coasted to ({x:.2f}, {y:.2f})")
                    break

            # Check for robot-to-obstacle contact (matching real simulation logic)
            # Excludes floor contacts and barrel-barrel contacts
            # Track which specific barrels are contacted
            in_contact = False
            contacted_barrels_this_frame = set()
            for i in range(data.ncon):
                contact = data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                g1_is_obstacle = g1 in obstacle_geom_ids
                g2_is_obstacle = g2 in obstacle_geom_ids
                g1_is_floor = g1 == floor_geom_id
                g2_is_floor = g2 == floor_geom_id
                # Robot contact = obstacle touches something that's not floor or another obstacle
                # Only count actual penetration (dist < 0)
                if g1_is_obstacle and not g2_is_obstacle and not g2_is_floor and contact.dist < 0:
                    in_contact = True
                    contacted_barrels_this_frame.add(barrel_geom_ids[g1])
                if g2_is_obstacle and not g1_is_obstacle and not g1_is_floor and contact.dist < 0:
                    in_contact = True
                    contacted_barrels_this_frame.add(barrel_geom_ids[g2])

            if in_contact:
                contact_frames += 1
                for barrel_name in contacted_barrels_this_frame:
                    per_barrel_contact_frames[barrel_name] += 1

            # Navigate to current waypoint
            if current_wp_idx < len(all_waypoints):
                target = all_waypoints[current_wp_idx]
                dx = target[0] - x
                dy = target[1] - y
                cmd = compute_navigation_cmd(dx, dy)

                # Check if arrived at waypoint
                if np.allclose(cmd, 0.0):
                    current_wp_idx += 1
                    if current_wp_idx < len(all_waypoints):
                        print(f"  → Waypoint {current_wp_idx} reached, heading to {all_waypoints[current_wp_idx]}")

            # Physics step (matches real simulation order)
            robot.step(data, cmd)
            mujoco.mj_step(model, data)
            step_counter += 1

            if viewer and viewer.is_running():
                viewer.sync()
            elif viewer and not viewer.is_running():
                break

            # Real-time pacing (matches real simulation)
            elapsed = time.time() - step_start
            if elapsed < robot.simulation_dt:
                time.sleep(robot.simulation_dt - elapsed)

    finally:
        if viewer:
            viewer.close()

    # Calculate contact time from frames
    contact_time = contact_frames * robot.simulation_dt
    per_barrel_contact_time = {
        barrel: frames * robot.simulation_dt
        for barrel, frames in per_barrel_contact_frames.items()
    }

    # Calculate barrel displacements
    barrel_displacements = {}
    for barrel_name, body_id in barrel_bodies.items():
        final_pos = (float(data.xpos[body_id][0]), float(data.xpos[body_id][1]))
        initial_pos = initial_barrel_positions[barrel_name]
        displacement = math.sqrt(
            (final_pos[0] - initial_pos[0]) ** 2 +
            (final_pos[1] - initial_pos[1]) ** 2
        )
        barrel_displacements[barrel_name] = round(displacement, 3)

    # Results
    battery_status = battery.get_status()

    print(f"\n📊 RESULTS:")
    print(f"  Goal reached: {'✓ YES' if goal_reached else '✗ NO'}")
    if goal_reached:
        print(f"  Goal touched by: {goal_touched_by}")
    print(f"  Total contact time: {contact_time:.2f}s {'⚠️ CONTACT!' if contact_time > 0 else '✓ No contact'}")

    # Per-barrel breakdown
    barrels_hit = [b for b, t in per_barrel_contact_time.items() if t > 0]
    if barrels_hit:
        print(f"  Barrels contacted: {', '.join(barrels_hit)}")
        for barrel, t in per_barrel_contact_time.items():
            if t > 0:
                disp = barrel_displacements[barrel]
                print(f"    {barrel}: {t:.2f}s contact, {disp:.2f}m displaced")

    print(f"  Distance traveled: {total_distance:.2f}m")
    print(f"  Battery remaining: {battery_status.charge_percent}%")
    print(f"  Final position: ({x:.2f}, {y:.2f})")

    return {
        "path_name": path_name,
        "goal_reached": goal_reached,
        "goal_touched_by": goal_touched_by,
        "contact_time": contact_time,
        "per_barrel_contact": per_barrel_contact_time,
        "barrel_displacements": barrel_displacements,
        "distance": total_distance,
        "battery_remaining": battery_status.charge_percent,
        "final_x": x,
        "final_y": y,
    }


def main():
    print("\n" + "="*60)
    print("PATH VALIDATION TESTER")
    print("="*60)
    print("\nThis script tests predefined paths to validate historical data.")
    print("Press Ctrl+C to skip to next path, or close viewer window.\n")

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    for path_name, path_config in TEST_PATHS.items():
        print(f"\n{'─'*60}")
        print(f"Path: {path_name}")
        print(f"Description: {path_config['description']}")
        print(f"Expected: {path_config['expected']}")

        input("\nPress Enter to test this path (or Ctrl+C to skip)...")

        try:
            # Determine which scene to use
            scene = path_config.get("scene", "barrels")
            scene_path = SCENE_NO_BARRELS if scene == "no_barrels" else SCENE_BARRELS

            result = run_path_test(path_name, path_config["waypoints"], show_viewer=True, scene_path=scene_path)
            result["description"] = path_config["description"]
            result["waypoints"] = path_config["waypoints"]
            result["scene"] = scene
            results.append(result)
        except KeyboardInterrupt:
            print("\n  Skipped.")
            continue

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary_lines = []
    for r in results:
        status = "✓" if r["goal_reached"] and r["contact_time"] == 0 else "⚠️" if r["goal_reached"] else "✗"
        contact = f"contact={r['contact_time']:.1f}s" if r["contact_time"] > 0 else "no contact"
        line = f"  {status} {r['path_name']}: {contact}, dist={r['distance']:.2f}m, battery={r['battery_remaining']}%"
        print(line)
        summary_lines.append(line)

    # Save results to JSON
    json_path = RESULTS_DIR / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {json_path}")

    # Save human-readable summary
    txt_path = RESULTS_DIR / f"summary_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("PATH VALIDATION TEST RESULTS\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*60 + "\n\n")

        for r in results:
            f.write(f"Path: {r['path_name']}\n")
            f.write(f"  Waypoints: {r['waypoints']}\n")
            f.write(f"  Description: {r['description']}\n")
            f.write(f"  Goal reached: {r['goal_reached']}\n")
            if r['goal_reached']:
                f.write(f"  Goal touched by: {r['goal_touched_by']}\n")
            f.write(f"  Total contact time: {r['contact_time']:.2f}s\n")
            # Per-barrel breakdown
            barrels_hit = [b for b, t in r['per_barrel_contact'].items() if t > 0]
            if barrels_hit:
                f.write(f"  Barrels contacted: {', '.join(barrels_hit)}\n")
                for barrel, t in r['per_barrel_contact'].items():
                    if t > 0:
                        disp = r['barrel_displacements'][barrel]
                        f.write(f"    {barrel}: {t:.2f}s contact, {disp:.3f}m displaced\n")
            f.write(f"  Distance: {r['distance']:.2f}m\n")
            f.write(f"  Battery remaining: {r['battery_remaining']}%\n")
            f.write(f"  Final position: ({r['final_x']:.2f}, {r['final_y']:.2f})\n")
            f.write("\n")

        f.write("="*60 + "\n")
        f.write("SUMMARY\n")
        f.write("="*60 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"📁 Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
