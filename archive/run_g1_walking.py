"""
G1 Humanoid Walking Demo - Pre-trained Policy
Uses Unitree's official pre-trained walking policy.
"""

import time
import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
import yaml

# Configuration
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/unitree_rl_gym"

def get_gravity_orientation(quaternion):
    """Convert quaternion to gravity vector in body frame."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD controller for joint torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_g1_walking(cmd=None, duration=60.0, headless=False):
    """
    Run G1 walking with pre-trained policy.

    Args:
        cmd: [vx, vy, yaw_rate] velocity command. Default [0.5, 0, 0] = walk forward
        duration: simulation duration in seconds
        headless: if True, run without viewer (for integration)

    Returns:
        trajectory: list of (x, y, z, time) positions
    """

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    # Use provided command or default
    if cmd is None:
        cmd = np.array(config["cmd_init"], dtype=np.float32)
    else:
        cmd = np.array(cmd, dtype=np.float32)

    print(f"G1 Walking Demo")
    print(f"  Command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, yaw={cmd[2]:.2f}")
    print(f"  Duration: {duration}s")

    # Initialize state
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"  Policy loaded from {policy_path}")

    # Track trajectory
    trajectory = []
    start_time = time.time()

    def step_simulation():
        nonlocal counter, action, target_dof_pos, obs

        # PD control
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        counter += 1
        if counter % control_decimation == 0:
            # Create observation
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]

            qj_normalized = (qj - default_angles) * dof_pos_scale
            dqj_normalized = dqj * dof_vel_scale
            gravity_orientation = get_gravity_orientation(quat)
            omega_normalized = omega * ang_vel_scale

            period = 0.8
            count = counter * simulation_dt
            phase = count % period / period
            sin_phase = np.sin(2 * np.pi * phase)
            cos_phase = np.cos(2 * np.pi * phase)

            obs[:3] = omega_normalized
            obs[3:6] = gravity_orientation
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + num_actions] = qj_normalized
            obs[9 + num_actions:9 + 2 * num_actions] = dqj_normalized
            obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
            obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = policy(obs_tensor).detach().numpy().squeeze()
            target_dof_pos = action * action_scale + default_angles

        # Record position
        x, y, z = d.qpos[0], d.qpos[1], d.qpos[2]
        trajectory.append((x, y, z, time.time() - start_time))

        return x, y, z

    if headless:
        # Run without viewer
        steps = int(duration / simulation_dt)
        for i in range(steps):
            x, y, z = step_simulation()
            if i % 1000 == 0:
                print(f"  Step {i}: pos=({x:.2f}, {y:.2f}, {z:.2f})")
    else:
        # Run with viewer
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and time.time() - start_time < duration:
                step_start = time.time()
                x, y, z = step_simulation()
                viewer.sync()

                # Maintain real-time
                time_until_next_step = simulation_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    print(f"\nSimulation complete!")
    print(f"  Final position: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f})")
    print(f"  Distance traveled: {np.sqrt(trajectory[-1][0]**2 + trajectory[-1][1]**2):.2f}m")

    return trajectory


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    args = parser.parse_args()

    trajectory = run_g1_walking(
        cmd=[args.vx, args.vy, args.yaw],
        duration=args.duration,
        headless=args.headless
    )
