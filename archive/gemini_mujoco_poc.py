"""
Minimal Proof of Concept: Gemini 3 controlling a robot arm in MuJoCo

To run with visualization on macOS:
    venv/bin/mjpython gemini_mujoco_poc.py --gui

To run headless (text output only):
    python gemini_mujoco_poc.py
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
import mujoco
from google import genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(api_key=api_key)
MODEL = "gemini-3-flash-preview"

# Simple robot arm XML (2-joint arm)
ARM_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <worldbody>
    <light name="light" pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.1 0.05" rgba="0.3 0.3 0.3 1"/>
      <body name="upper_arm" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.4" rgba="0.8 0.2 0.2 1"/>
        <body name="forearm" pos="0 0 0.4">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-120 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.3" rgba="0.2 0.2 0.8 1"/>
          <body name="hand" pos="0 0 0.3">
            <geom type="sphere" size="0.05" rgba="0.2 0.8 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="shoulder_motor" joint="shoulder" gear="50"/>
    <motor name="elbow_motor" joint="elbow" gear="50"/>
  </actuator>
</mujoco>
"""


def ask_gemini_for_action(shoulder_angle, elbow_angle, goal, step):
    """Ask Gemini 3 to decide what joint angles to set."""
    prompt = f"""You are controlling a 2-joint robot arm in a physics simulation.

Current state:
- Shoulder angle: {shoulder_angle:.1f} degrees (range: -90 to 90)
- Elbow angle: {elbow_angle:.1f} degrees (range: -120 to 0)
- Step: {step} of 10

Goal: {goal}

Respond with ONLY a JSON object with the target angles:
{{"shoulder": <float>, "elbow": <float>}}

Make smooth movements - change angles by at most 15 degrees per step.
For a wave motion: move shoulder back and forth, keep elbow slightly bent.
"""

    response = client.models.generate_content(model=MODEL, contents=prompt)

    # Parse JSON from response
    text = response.text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    return json.loads(text)


def run_headless():
    """Run simulation without GUI - just text output."""
    print("\n=== Gemini 3 + MuJoCo Proof of Concept (Headless) ===\n")

    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    goal = "Wave the robot arm in a friendly greeting motion"
    print(f"Goal: {goal}\n")

    api_times = []
    total_start = time.time()

    for step in range(10):
        # Get current joint angles in degrees
        shoulder_deg = float(data.qpos[0]) * 180 / 3.14159
        elbow_deg = float(data.qpos[1]) * 180 / 3.14159

        print(f"Step {step + 1}: shoulder={shoulder_deg:.1f}°, elbow={elbow_deg:.1f}°")
        print("  Asking Gemini 3...")

        try:
            api_start = time.time()
            action = ask_gemini_for_action(shoulder_deg, elbow_deg, goal, step + 1)
            api_elapsed = time.time() - api_start
            api_times.append(api_elapsed)
            print(f"  Gemini says: shoulder={action['shoulder']:.1f}°, elbow={action['elbow']:.1f}° ({api_elapsed:.2f}s)")

            # Convert to radians and set control
            target_shoulder = action["shoulder"] * 3.14159 / 180
            target_elbow = action["elbow"] * 3.14159 / 180

            # Apply control and simulate for 1 second
            for _ in range(500):
                data.ctrl[0] = (target_shoulder - data.qpos[0]) * 10
                data.ctrl[1] = (target_elbow - data.qpos[1]) * 10
                mujoco.mj_step(model, data)

            # Show result
            new_shoulder = float(data.qpos[0]) * 180 / 3.14159
            new_elbow = float(data.qpos[1]) * 180 / 3.14159
            print(f"  Result: shoulder={new_shoulder:.1f}°, elbow={new_elbow:.1f}°\n")

        except Exception as e:
            print(f"  Error: {e}\n")

    total_elapsed = time.time() - total_start
    print("=" * 50)
    print("TIMING SUMMARY")
    print("=" * 50)
    print(f"Total time:      {total_elapsed:.2f}s")
    print(f"API calls:       {len(api_times)}")
    print(f"Avg API latency: {sum(api_times)/len(api_times):.2f}s")
    print(f"Min API latency: {min(api_times):.2f}s")
    print(f"Max API latency: {max(api_times):.2f}s")
    print(f"Total API time:  {sum(api_times):.2f}s ({100*sum(api_times)/total_elapsed:.0f}% of total)")
    print("=" * 50)


def run_with_gui():
    """Run simulation with MuJoCo viewer."""
    import mujoco.viewer

    print("\n=== Gemini 3 + MuJoCo Proof of Concept (GUI) ===\n")

    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    goal = "Wave the robot arm in a friendly greeting motion"
    print(f"Goal: {goal}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        last_query_time = 0

        print("Simulation running. Close the viewer window to exit.\n")

        while viewer.is_running() and step < 10:
            current_time = time.time()

            if current_time - last_query_time > 2.0:
                shoulder_deg = float(data.qpos[0]) * 180 / 3.14159
                elbow_deg = float(data.qpos[1]) * 180 / 3.14159

                print(f"Step {step + 1}: shoulder={shoulder_deg:.1f}°, elbow={elbow_deg:.1f}°")
                print("  Asking Gemini 3...")

                try:
                    action = ask_gemini_for_action(shoulder_deg, elbow_deg, goal, step + 1)
                    print(f"  Gemini says: shoulder={action['shoulder']:.1f}°, elbow={action['elbow']:.1f}°\n")

                    target_shoulder = action["shoulder"] * 3.14159 / 180
                    target_elbow = action["elbow"] * 3.14159 / 180
                    data.ctrl[0] = (target_shoulder - data.qpos[0]) * 10
                    data.ctrl[1] = (target_elbow - data.qpos[1]) * 10

                except Exception as e:
                    print(f"  Error: {e}\n")

                step += 1
                last_query_time = current_time

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    print("Demo complete!")


if __name__ == "__main__":
    if "--gui" in sys.argv:
        run_with_gui()
    else:
        run_headless()
