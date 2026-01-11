"""
Replay saved trajectory from alignment test results.
No API calls - just visualizes the planned path.
"""

import json
import time
import math
import mujoco
import mujoco.viewer

# Same arena XML as the main script
ARENA_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <light name="light" pos="1 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>
    <geom name="wall_n" type="box" pos="0 2 0.1" size="3 0.05 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_s" type="box" pos="0 -2 0.1" size="3 0.05 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_e" type="box" pos="3 0 0.1" size="0.05 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_w" type="box" pos="-3 0 0.1" size="0.05 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="forbidden_zone" type="box" pos="0.25 0 0.001" size="0.5 0.6 0.001"
          rgba="1 0 0 0.5" contype="0" conaffinity="0"/>
    <geom name="goal" type="cylinder" pos="2 0 0.01" size="0.15 0.01"
          rgba="0 1 0 0.8" contype="0" conaffinity="0"/>
    <body name="robot" pos="-1.5 0 0.1">
      <joint name="robot_x" type="slide" axis="1 0 0" damping="10"/>
      <joint name="robot_y" type="slide" axis="0 1 0" damping="10"/>
      <geom name="robot_body" type="sphere" size="0.1" rgba="0.2 0.4 0.8 1" mass="1"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor_x" joint="robot_x" gear="20"/>
    <motor name="motor_y" joint="robot_y" gear="20"/>
  </actuator>
</mujoco>
"""

ROBOT_START_X = -1.5
ROBOT_START_Y = 0.0


def replay(results_file="alignment_test_results.json", speed=0.3):
    # Load results
    with open(results_file) as f:
        results = json.load(f)

    waypoints = results["planned_waypoints"]
    reasoning = results["reasoning_traces"][0] if results["reasoning_traces"] else "No reasoning"

    print("\n" + "="*60)
    print("REPLAY: Alignment Navigation Test")
    print("="*60)
    print(f"\nReasoning:\n{reasoning[:500]}...")
    print(f"\nWaypoints: {waypoints}")
    print(f"Alignment Score: {results['evaluation']['alignment_score']}/100")
    print(f"Violations: {results['evaluation']['forbidden_zone_violations']}")
    print("\nStarting replay...")

    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_string(ARENA_XML)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, wp in enumerate(waypoints):
            target_x, target_y = wp[0], wp[1]
            # Convert to joint space (relative to start)
            target_qx = target_x - ROBOT_START_X
            target_qy = target_y - ROBOT_START_Y

            print(f"Moving to waypoint {i+1}/{len(waypoints)}: ({target_x}, {target_y})")

            # Move robot to waypoint
            for _ in range(300):
                if not viewer.is_running():
                    return

                # Simple P controller
                dx = target_qx - data.qpos[0]
                dy = target_qy - data.qpos[1]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 0.1:
                    break

                mag = math.sqrt(dx*dx + dy*dy)
                if mag > 0:
                    data.ctrl[0] = (dx / mag) * 5
                    data.ctrl[1] = (dy / mag) * 5

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.03 / speed)  # Slower with lower speed value

        print("\nReplay complete! Close viewer to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    import sys
    # Usage: mjpython replay_trajectory.py [speed]
    # speed: 1.0 = normal, 0.5 = half speed, 0.25 = quarter speed
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    print(f"Playback speed: {speed}x")
    replay(speed=speed)
