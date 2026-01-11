"""
Watch G1 Walk - Visual Demo with MuJoCo Viewer

This opens a window showing the G1 humanoid walking in real-time.
You can rotate the camera with mouse drag.
"""

import time
import os
import numpy as np
import torch
import mujoco
import mujoco.viewer
import yaml

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/unitree_rl_gym"


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def watch_g1_walk():
    """Open viewer and watch G1 walk."""

    print("="*50)
    print("G1 HUMANOID WALKING DEMO")
    print("="*50)
    print("\nControls:")
    print("  - Left mouse drag: Rotate camera")
    print("  - Right mouse drag: Pan camera")
    print("  - Scroll: Zoom")
    print("  - Double-click: Track robot")
    print("  - ESC or close window: Exit")
    print("\nStarting simulation...")

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

    # Command: walk forward at 0.5 m/s
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)

    # State
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    policy = torch.jit.load(policy_path)
    print(f"Policy loaded!")

    # Launch viewer
    print("\nOpening viewer window...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running():
            step_start = time.time()

            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                           np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Build observation
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

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < simulation_dt:
                time.sleep(simulation_dt - elapsed)

            # Print position occasionally
            if counter % 2500 == 0:
                x, y, z = d.qpos[0], d.qpos[1], d.qpos[2]
                t = time.time() - start
                print(f"  t={t:.1f}s: pos=({x:.2f}, {y:.2f}), height={z:.2f}m")

    print("\nViewer closed.")


if __name__ == "__main__":
    watch_g1_walk()
