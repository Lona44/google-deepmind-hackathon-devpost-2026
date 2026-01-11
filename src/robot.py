"""
Robot controller for G1 humanoid in MuJoCo.
"""

import base64
from io import BytesIO
from typing import Optional

import numpy as np
import torch
import yaml
import mujoco
from PIL import Image

from .config import (
    ROBOT_CONFIG_PATH,
    LEGGED_GYM_ROOT,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_NAME,
)

# LiDAR configuration
LIDAR_NUM_RAYS = 36
LIDAR_ANGLE_INCREMENT = 10.0  # degrees between each ray
LIDAR_MIN_RANGE = 0.1  # meters
LIDAR_MAX_RANGE = 10.0  # meters (cutoff)


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Compute gravity orientation from quaternion."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray
) -> np.ndarray:
    """PD controller for joint torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


class RobotController:
    """Controls the G1 humanoid robot."""

    def __init__(self, config_path: str = ROBOT_CONFIG_PATH):
        self.config_path = config_path
        self._load_config()
        self._load_policy()
        self._init_state()

    def _load_config(self) -> None:
        """Load robot configuration from YAML."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Timing
        self.simulation_dt = config["simulation_dt"]
        self.control_decimation = config["control_decimation"]

        # Control gains
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)

        # Scaling
        self.ang_vel_scale = config["ang_vel_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # Dimensions
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]

        # Policy path
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT)
        self.policy_path = policy_path

    def _load_policy(self) -> None:
        """Load the pre-trained walking policy."""
        self.policy = torch.jit.load(self.policy_path)

    def _init_state(self) -> None:
        """Initialize state variables."""
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0

    def reset(self) -> None:
        """Reset robot state."""
        self._init_state()

    def get_position(self, d: mujoco.MjData) -> tuple[float, float, float]:
        """Get robot base position (x, y, z)."""
        return float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])

    def read_sensors(self, m: mujoco.MjModel, d: mujoco.MjData) -> dict:
        """Read all sensor data from the robot."""
        sensors = {}

        # IMU Gyro
        gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id >= 0:
            gyro_adr = m.sensor_adr[gyro_id]
            sensors["gyro"] = d.sensordata[gyro_adr:gyro_adr + 3].tolist()

        # IMU Accelerometer
        accel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if accel_id >= 0:
            accel_adr = m.sensor_adr[accel_id]
            sensors["accel"] = d.sensordata[accel_adr:accel_adr + 3].tolist()

        # Foot contact
        left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_touch")
        right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_touch")
        if left_id >= 0:
            sensors["left_foot_contact"] = float(d.sensordata[m.sensor_adr[left_id]])
        if right_id >= 0:
            sensors["right_foot_contact"] = float(d.sensordata[m.sensor_adr[right_id]])

        return sensors

    def read_lidar(self, m: mujoco.MjModel, d: mujoco.MjData) -> dict:
        """Read LiDAR rangefinder data.

        Returns:
            dict with:
                - ranges: list of 36 distance values (meters), -1 means no hit within cutoff
                - angles: list of 36 angles (degrees) from robot forward direction
                - min_range: closest obstacle distance
                - min_angle: angle to closest obstacle
                - obstacles_detected: number of rays that hit something
        """
        ranges = []
        angles = []

        for i in range(LIDAR_NUM_RAYS):
            sensor_name = f"lidar-{i:02d}"
            sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

            if sensor_id >= 0:
                sensor_adr = m.sensor_adr[sensor_id]
                distance = float(d.sensordata[sensor_adr])
                ranges.append(distance)
                angles.append(i * LIDAR_ANGLE_INCREMENT)
            else:
                ranges.append(-1)
                angles.append(i * LIDAR_ANGLE_INCREMENT)

        # Find closest obstacle (ignoring -1 which means no hit)
        valid_ranges = [(r, a) for r, a in zip(ranges, angles) if r > 0]
        if valid_ranges:
            min_range, min_angle = min(valid_ranges, key=lambda x: x[0])
        else:
            min_range = -1
            min_angle = 0

        return {
            "ranges": ranges,
            "angles": angles,
            "min_range": min_range,
            "min_angle": min_angle,
            "obstacles_detected": len(valid_ranges),
        }

    def format_lidar_summary(self, lidar_data: dict) -> str:
        """Format LiDAR data as a human-readable summary for Gemini.

        Groups readings into 8 sectors (45° each) for easier understanding.
        """
        ranges = lidar_data["ranges"]

        # Group into 8 sectors: Front, Front-Right, Right, Back-Right, Back, Back-Left, Left, Front-Left
        sectors = {
            "FRONT (350°-10°)": [],
            "FRONT-RIGHT (10°-80°)": [],
            "RIGHT (80°-100°)": [],
            "BACK-RIGHT (100°-170°)": [],
            "BACK (170°-190°)": [],
            "BACK-LEFT (190°-260°)": [],
            "LEFT (260°-280°)": [],
            "FRONT-LEFT (280°-350°)": [],
        }

        for i, distance in enumerate(ranges):
            angle = i * LIDAR_ANGLE_INCREMENT
            if angle >= 350 or angle < 10:
                sectors["FRONT (350°-10°)"].append(distance)
            elif angle < 80:
                sectors["FRONT-RIGHT (10°-80°)"].append(distance)
            elif angle < 100:
                sectors["RIGHT (80°-100°)"].append(distance)
            elif angle < 170:
                sectors["BACK-RIGHT (100°-170°)"].append(distance)
            elif angle < 190:
                sectors["BACK (170°-190°)"].append(distance)
            elif angle < 260:
                sectors["BACK-LEFT (190°-260°)"].append(distance)
            elif angle < 280:
                sectors["LEFT (260°-280°)"].append(distance)
            else:
                sectors["FRONT-LEFT (280°-350°)"].append(distance)

        # Build summary
        lines = ["LiDAR SCAN (360°, 36 rays):"]
        for sector_name, distances in sectors.items():
            valid = [d for d in distances if d > 0]
            if valid:
                min_d = min(valid)
                if min_d < 0.5:
                    status = "⚠️ OBSTACLE CLOSE"
                elif min_d < 1.0:
                    status = "obstacle nearby"
                else:
                    status = "clear"
                lines.append(f"  {sector_name}: {min_d:.2f}m ({status})")
            else:
                lines.append(f"  {sector_name}: >10m (clear)")

        if lidar_data["min_range"] > 0:
            lines.append(f"  CLOSEST: {lidar_data['min_range']:.2f}m at {lidar_data['min_angle']:.0f}°")

        return "\n".join(lines)

    def capture_image(
        self,
        renderer: mujoco.Renderer,
        d: mujoco.MjData,
        camera_name: str = CAMERA_NAME
    ) -> tuple[str, Image.Image]:
        """Capture image from robot's camera and return as base64 + PIL Image."""
        # Create visualization options that hide LiDAR rays from camera
        # This gives Gemini a realistic robot view (no debug visualization)
        # while the viewer still shows the LiDAR rays for human observation
        vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(vopt)  # Initialize with defaults

        # Disable rangefinder ray visualization (the yellow lines)
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

        # Also disable site groups 3-5 just in case
        vopt.sitegroup[3] = False
        vopt.sitegroup[4] = False
        vopt.sitegroup[5] = False

        renderer.update_scene(d, camera=camera_name, scene_option=vopt)
        img = renderer.render()

        # Convert to PIL and then base64
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return img_base64, pil_img

    def compute_observation(self, d: mujoco.MjData, cmd: np.ndarray) -> np.ndarray:
        """Compute observation vector for policy."""
        qj = d.qpos[7:]
        dqj = d.qvel[6:]
        quat = d.qpos[3:7]
        omega = d.qvel[3:6]

        qj_normalized = (qj - self.default_angles) * self.dof_pos_scale
        dqj_normalized = dqj * self.dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_normalized = omega * self.ang_vel_scale

        # Phase for gait
        period = 0.8
        count = self.counter * self.simulation_dt
        ph = count % period / period
        sin_phase = np.sin(2 * np.pi * ph)
        cos_phase = np.cos(2 * np.pi * ph)

        # Build observation
        self.obs[:3] = omega_normalized
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = cmd * self.cmd_scale
        self.obs[9:9 + self.num_actions] = qj_normalized
        self.obs[9 + self.num_actions:9 + 2 * self.num_actions] = dqj_normalized
        self.obs[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = self.action
        self.obs[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = np.array([sin_phase, cos_phase])

        return self.obs

    def compute_action(self, d: mujoco.MjData, cmd: np.ndarray) -> np.ndarray:
        """Compute action from policy given command."""
        self.compute_observation(d, cmd)
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
        return self.action

    def apply_control(self, d: mujoco.MjData) -> None:
        """Apply PD control to robot joints."""
        tau = pd_control(
            self.target_dof_pos,
            d.qpos[7:],
            self.kps,
            np.zeros_like(self.kds),
            d.qvel[6:],
            self.kds
        )
        d.ctrl[:] = tau

    def step(self, d: mujoco.MjData, cmd: np.ndarray) -> None:
        """Perform one control step (call every control_decimation physics steps)."""
        self.counter += 1
        if self.counter % self.control_decimation == 0:
            self.compute_action(d, cmd)
        self.apply_control(d)


def create_renderer(m: mujoco.MjModel) -> mujoco.Renderer:
    """Create a MuJoCo renderer for camera capture."""
    return mujoco.Renderer(m, CAMERA_HEIGHT, CAMERA_WIDTH)
