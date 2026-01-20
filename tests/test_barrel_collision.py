#!/usr/bin/env python3
"""Test script to see if robot can physically push through barrels.

This bypasses Gemini and forces the robot to walk directly toward the goal,
colliding with the barrels. Used to verify the physics simulation.

Run with: mjpython tests/test_barrel_collision.py
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ROBOT_CONFIG_PATH
from src.robot import RobotController


def main():
    # Load scene with barrels
    scene_path = (
        Path(__file__).parent.parent
        / "unitree_rl_gym/resources/robots/g1_description/scene_barrels.xml"
    )

    print(f"Loading scene: {scene_path}")
    m = mujoco.MjModel.from_xml_path(str(scene_path))
    d = mujoco.MjData(m)

    # Initialize robot controller
    robot = RobotController(ROBOT_CONFIG_PATH)
    robot.reset()

    # Target: walk straight to goal at (5, 0) - directly through the barrels
    goal = np.array([5.0, 0.0])

    print("\n" + "=" * 60)
    print("BARREL COLLISION TEST")
    print("=" * 60)
    print(f"Goal: {goal}")
    print("Robot will attempt to walk STRAIGHT through the barrels.")
    print("Watch to see if it pushes them aside or gets stuck.")
    print("=" * 60 + "\n")

    # Track barrel positions
    barrel_body_ids = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ["barrel_1_body", "barrel_2_body", "barrel_3_body"]
    ]

    def get_barrel_positions():
        positions = []
        for bid in barrel_body_ids:
            pos = d.xpos[bid].copy()
            positions.append(pos)
        return positions

    # Step physics once to initialize positions
    mujoco.mj_forward(m, d)
    initial_barrel_pos = get_barrel_positions()
    print("Initial barrel positions:")
    for i, pos in enumerate(initial_barrel_pos):
        print(f"  Barrel {i + 1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    start_time = time.time()
    step_count = 0
    contact_count = 0
    max_time = 60  # Run for up to 60 seconds

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Hide LiDAR rays for cleaner visualization
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        # Set camera to side view for better observation
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -20
        viewer.cam.distance = 6
        viewer.cam.lookat[:] = [2.5, 0, 0.5]

        # Physics steps per control update (for smooth walking)
        n_steps = 10
        control_dt = m.opt.timestep * n_steps  # Time per control loop

        while viewer.is_running() and (time.time() - start_time) < max_time:
            step_start = time.time()

            # Get robot position
            x, y, _z = robot.get_position(d)

            # Calculate command to walk toward goal
            dx = goal[0] - x
            dy = goal[1] - y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 0.3:
                print(f"\n✅ GOAL REACHED at ({x:.2f}, {y:.2f})!")
                break

            # Command format: [vx, vy, vyaw] - same as main simulation
            # vx = forward speed, vy = lateral speed, vyaw = turn rate
            vy = np.clip(dy * 0.8, -0.5, 0.5)
            vx = np.clip(dx * 0.3, 0.1, 0.4) if abs(dy) > 0.3 else np.clip(dx * 0.4, 0.2, 0.5)
            cmd = np.array([vx, vy, 0.0], dtype=np.float32)

            # Multiple physics steps per control update
            # robot.step() must be called every physics step (it handles its own timing)
            for _ in range(n_steps):
                robot.step(d, cmd)
                mujoco.mj_step(m, d)
            step_count += 1

            # Check for contacts with barrels
            for i in range(d.ncon):
                contact = d.contact[i]
                geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

                if geom1_name and geom2_name:
                    is_barrel = "barrel" in geom1_name or "barrel" in geom2_name

                    if is_barrel and contact.dist < 0:
                        contact_count += 1

            # Print status every 50 control steps (~1 second)
            if step_count % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{elapsed:.1f}s] Robot at ({x:.2f}, {y:.2f}), "
                    f"distance to goal: {distance:.2f}m, contacts: {contact_count}"
                )

            # Sync viewer
            viewer.sync()

            # Real-time sync
            elapsed_step = time.time() - step_start
            if elapsed_step < control_dt:
                time.sleep(control_dt - elapsed_step)

    # Final report
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    final_pos = robot.get_position(d)
    print(f"Robot final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
    print(f"Total contacts with barrels: {contact_count}")

    final_barrel_pos = get_barrel_positions()
    print("\nBarrel movement:")
    for i, (init, final) in enumerate(zip(initial_barrel_pos, final_barrel_pos, strict=True)):
        dx = final[0] - init[0]
        dy = final[1] - init[1]
        dz = final[2] - init[2]
        total_movement = np.sqrt(dx**2 + dy**2 + dz**2)
        if total_movement > 0.05:
            print(
                f"  Barrel {i + 1}: MOVED {total_movement:.2f}m "
                f"(from ({init[0]:.2f}, {init[1]:.2f}) to ({final[0]:.2f}, {final[1]:.2f}))"
            )
        else:
            print(f"  Barrel {i + 1}: unchanged")

    distance_to_goal = np.sqrt((final_pos[0] - goal[0]) ** 2 + (final_pos[1] - goal[1]) ** 2)
    if distance_to_goal < 0.5:
        print("\n✅ Robot successfully pushed through barrels!")
    else:
        print(f"\n⚠️ Robot stopped {distance_to_goal:.2f}m from goal")


if __name__ == "__main__":
    main()
