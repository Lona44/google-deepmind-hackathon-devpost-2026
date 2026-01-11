"""
Minimal Proof of Concept: Gemini 3 controlling a robot arm in PyBullet
"""

import os
import json
import time
from dotenv import load_dotenv
import pybullet as p
import pybullet_data
from google import genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini client
client = genai.Client(api_key=api_key)
MODEL = "gemini-3-flash-preview"

# Initialize PyBullet
physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load ground plane and Kuka robot arm
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# Get number of joints
num_joints = p.getNumJoints(robot_id)
print(f"Robot loaded with {num_joints} joints")


def get_robot_state():
    """Get current joint positions of the robot."""
    joint_states = []
    for i in range(num_joints):
        state = p.getJointState(robot_id, i)
        joint_states.append(round(state[0], 3))  # Joint position
    return joint_states


def set_robot_joints(positions):
    """Set robot joint positions."""
    for i, pos in enumerate(positions):
        if i < num_joints:
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL, targetPosition=pos
            )


def ask_gemini_for_action(current_state, goal):
    """Ask Gemini 3 to decide what joint positions to set."""
    prompt = f"""You are controlling a Kuka robot arm with {num_joints} joints in a physics simulation.

Current joint positions (radians): {current_state}
Joint limits: approximately -2.9 to 2.9 radians for each joint

Goal: {goal}

Respond with ONLY a JSON array of {num_joints} float values representing the new target joint positions.
Example response: [0.0, 0.5, -0.3, 0.8, 0.0, -0.5, 0.0]

Keep movements small and smooth (change each joint by at most 0.3 radians from current position).
"""

    response = client.models.generate_content(model=MODEL, contents=prompt)

    # Parse JSON from response
    text = response.text.strip()
    # Handle potential markdown code blocks
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    return json.loads(text)


def main():
    print("\n=== Gemini 3 + PyBullet Proof of Concept ===\n")

    goal = "Wave the robot arm in a friendly greeting motion"
    print(f"Goal: {goal}\n")

    # Run for a few iterations
    for step in range(5):
        current_state = get_robot_state()
        print(f"Step {step + 1}: Current joints = {current_state}")

        # Ask Gemini for next action
        print("  Asking Gemini 3...")
        new_positions = ask_gemini_for_action(current_state, goal)
        print(f"  Gemini suggests: {new_positions}")

        # Apply the action
        set_robot_joints(new_positions)

        # Step simulation
        for _ in range(240):  # ~1 second at 240Hz
            p.stepSimulation()
            time.sleep(1/240)

        print()

    print("Demo complete! Press Enter to exit...")
    input()
    p.disconnect()


if __name__ == "__main__":
    main()
