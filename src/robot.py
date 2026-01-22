"""
Robot controller for G1 humanoid in MuJoCo.
"""

import base64
import math
from io import BytesIO
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from .config import (
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    LEGGED_GYM_ROOT,
    ROBOT_CONFIG_PATH,
)

# LiDAR configuration
LIDAR_NUM_RAYS = 180
LIDAR_ANGLE_INCREMENT = 2.0  # degrees between each ray (360/180)
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
    kd: np.ndarray,
) -> np.ndarray:
    """PD controller for joint torques."""
    result: np.ndarray = (target_q - q) * kp + (target_dq - dq) * kd
    return result


class RobotController:
    """Controls the G1 humanoid robot."""

    def __init__(self, config_path: str | Path = ROBOT_CONFIG_PATH):
        self.config_path = config_path
        self._load_config()
        self._load_policy()
        self._init_state()

    def _load_config(self) -> None:
        """Load robot configuration from YAML."""
        with Path(self.config_path).open() as f:
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
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(LEGGED_GYM_ROOT))
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

    def get_heading(self, d: mujoco.MjData) -> float:
        """Get robot heading (yaw) in radians. 0 = facing +X direction."""
        qw, qx, qy, qz = d.qpos[3:7]
        # Extract yaw from quaternion (rotation around Z axis)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return float(yaw)

    def get_goal_bearing(
        self, d: mujoco.MjData, goal_x: float, goal_y: float
    ) -> tuple[float, float, str]:
        """Calculate bearing and distance to goal.

        Returns:
            (distance_m, bearing_deg, direction_str)
            bearing_deg: positive = goal is to the left, negative = right
            direction_str: human-readable like "30° left" or "straight ahead"
        """
        x, y, _ = self.get_position(d)
        heading = self.get_heading(d)

        # Calculate angle to goal in world frame
        dx = goal_x - x
        dy = goal_y - y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)

        # Relative bearing (how much robot needs to turn)
        relative_bearing = angle_to_goal - heading

        # Normalize to -pi to pi
        while relative_bearing > np.pi:
            relative_bearing -= 2 * np.pi
        while relative_bearing < -np.pi:
            relative_bearing += 2 * np.pi

        bearing_deg = float(np.degrees(relative_bearing))

        # Format as human-readable direction
        if abs(bearing_deg) < 10:
            direction = "straight ahead"
        elif bearing_deg > 0:
            direction = f"{abs(bearing_deg):.0f}° left"
        else:
            direction = f"{abs(bearing_deg):.0f}° right"

        return float(distance), bearing_deg, direction

    def read_sensors(self, m: mujoco.MjModel, d: mujoco.MjData) -> dict:
        """Read all sensor data from the robot."""
        sensors = {}

        # IMU Gyro
        gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id >= 0:
            gyro_adr = m.sensor_adr[gyro_id]
            sensors["gyro"] = d.sensordata[gyro_adr : gyro_adr + 3].tolist()

        # IMU Accelerometer
        accel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if accel_id >= 0:
            accel_adr = m.sensor_adr[accel_id]
            sensors["accel"] = d.sensordata[accel_adr : accel_adr + 3].tolist()

        # Foot contact
        left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_touch")
        right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_touch")
        if left_id >= 0:
            sensors["left_foot_contact"] = float(d.sensordata[m.sensor_adr[left_id]])
        if right_id >= 0:
            sensors["right_foot_contact"] = float(d.sensordata[m.sensor_adr[right_id]])

        return sensors

    def _read_lidar_ring(self, m: mujoco.MjModel, d: mujoco.MjData, prefix: str) -> dict:
        """Read a single LiDAR ring.

        Args:
            prefix: Sensor name prefix (e.g., "lidar_gnd" or "lidar_mid")

        Returns:
            dict with ranges, angles, min_range, min_angle, obstacles_detected
        """
        ranges = []
        angles = []

        for i in range(LIDAR_NUM_RAYS):
            sensor_name = f"{prefix}-{i:03d}"
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
        valid_ranges = [(r, a) for r, a in zip(ranges, angles, strict=True) if r > 0]
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

    def read_lidar(self, m: mujoco.MjModel, d: mujoco.MjData) -> dict:
        """Read waist-ring LiDAR rangefinder data.

        Note: Head ring was removed because it was hitting the robot's shoulders.
        Only the waist ring (~0.75m, horizontal) is used for obstacle detection.

        Returns:
            dict with:
                - waist_ring: data from waist level (~0.75m), horizontal
                - closest_obstacle: closest detection from waist ring
        """
        waist_ring = self._read_lidar_ring(m, d, "lidar_gnd")

        # Closest obstacle from waist ring
        closest_range = waist_ring["min_range"] if waist_ring["min_range"] > 0 else -1
        closest_angle = waist_ring["min_angle"] if waist_ring["min_range"] > 0 else 0

        return {
            "waist_ring": waist_ring,
            "closest_obstacle": {
                "range": closest_range,
                "angle": closest_angle,
            },
        }

    def compute_obstacle_cartesian(
        self, lidar_data: dict, robot_x: float, robot_y: float, robot_heading: float = 0.0
    ) -> list[tuple[float, float]]:
        """Convert LiDAR hits to world cartesian coordinates.

        Args:
            lidar_data: Output from read_lidar()
            robot_x, robot_y: Robot's current position
            robot_heading: Robot's heading in radians (0 = facing +X)

        Returns:
            List of (x, y) world coordinates where obstacles were detected.
        """
        obstacle_points = []

        ring = lidar_data["waist_ring"]
        for dist, angle_deg in zip(ring["ranges"], ring["angles"], strict=True):
            if dist > 0 and dist < LIDAR_MAX_RANGE:
                # Convert to radians (LiDAR angle is relative to robot front)
                # MuJoCo: +Y is left, angles increase counter-clockwise
                angle_rad = math.radians(angle_deg) + robot_heading

                # Polar to cartesian (relative to robot)
                rel_x = dist * math.cos(angle_rad)
                rel_y = dist * math.sin(angle_rad)

                # Transform to world coordinates
                world_x = robot_x + rel_x
                world_y = robot_y + rel_y

                obstacle_points.append((world_x, world_y))

        return obstacle_points

    def compute_obstacle_boundaries(
        self, obstacle_points: list[tuple[float, float]]
    ) -> dict | None:
        """Estimate obstacle cluster boundaries from cartesian points.

        Returns:
            Dict with x_min, x_max, y_min, y_max, or None if no obstacles.
        """
        if not obstacle_points:
            return None

        x_coords = [p[0] for p in obstacle_points]
        y_coords = [p[1] for p in obstacle_points]

        return {
            "x_min": min(x_coords),
            "x_max": max(x_coords),
            "y_min": min(y_coords),
            "y_max": max(y_coords),
            "num_hits": len(obstacle_points),
        }

    def format_obstacle_scan(
        self,
        lidar_data: dict,
        robot_x: float,
        robot_y: float,
        robot_heading: float = 0.0,
    ) -> str:
        """Format detailed obstacle scan for Gemini using raw polar data.

        Reports actual LiDAR detections as distance @ angle pairs.
        This is the most accurate representation of what the sensor detects.

        Note: robot_x, robot_y, robot_heading are kept for API compatibility
        but not used since we're reporting raw polar data.
        """
        _ = (robot_x, robot_y, robot_heading)  # Suppress unused warnings

        ring = lidar_data["waist_ring"]

        # Collect all hits as (distance, angle) pairs
        hits: list[tuple[float, float]] = []
        for dist, angle_deg in zip(ring["ranges"], ring["angles"], strict=True):
            if dist > 0 and dist < LIDAR_MAX_RANGE:
                hits.append((dist, angle_deg))

        if not hits:
            return "OBSTACLE DETECTIONS: None within 10m range."

        # Sort by angle for readability (front first, then around)
        hits.sort(key=lambda h: h[1])

        lines = [
            f"OBSTACLE DETECTIONS ({len(hits)} hits):",
            "(Format: distance @ angle, where 0°=front, 90°=left, 180°=back, 270°=right)",
            "",
        ]

        # Group hits by sector for cleaner output
        # Angles are 0-360: 0=front, 90=left, 180=back, 270=right
        sectors: dict[str, list[tuple[float, float]]] = {
            "FRONT (0°±22°)": [],
            "FRONT-LEFT (23° to 67°)": [],
            "LEFT (68° to 112°)": [],
            "BACK-LEFT (113° to 157°)": [],
            "BACK (158° to 202°)": [],
            "BACK-RIGHT (203° to 247°)": [],
            "RIGHT (248° to 292°)": [],
            "FRONT-RIGHT (293° to 337°)": [],
        }

        for dist, angle in hits:
            # Normalize angle to 0-360 range
            norm_angle = angle % 360
            if norm_angle <= 22 or norm_angle >= 338:
                sectors["FRONT (0°±22°)"].append((dist, norm_angle))
            elif 23 <= norm_angle <= 67:
                sectors["FRONT-LEFT (23° to 67°)"].append((dist, norm_angle))
            elif 68 <= norm_angle <= 112:
                sectors["LEFT (68° to 112°)"].append((dist, norm_angle))
            elif 113 <= norm_angle <= 157:
                sectors["BACK-LEFT (113° to 157°)"].append((dist, norm_angle))
            elif 158 <= norm_angle <= 202:
                sectors["BACK (158° to 202°)"].append((dist, norm_angle))
            elif 203 <= norm_angle <= 247:
                sectors["BACK-RIGHT (203° to 247°)"].append((dist, norm_angle))
            elif 248 <= norm_angle <= 292:
                sectors["RIGHT (248° to 292°)"].append((dist, norm_angle))
            elif 293 <= norm_angle <= 337:
                sectors["FRONT-RIGHT (293° to 337°)"].append((dist, norm_angle))

        # Output each sector with hits
        for sector_name, sector_hits in sectors.items():
            if sector_hits:
                lines.append(f"  {sector_name}:")
                # Show all hits in sector (typically just a few per sector)
                for dist, angle in sector_hits:
                    lines.append(f"    {dist:.2f}m @ {angle:.0f}°")

        return "\n".join(lines)

    def _format_ring_sectors(self, ring_data: dict) -> dict[str, float]:
        """Group a LiDAR ring's readings into 8 sectors.

        Returns dict mapping sector name to minimum distance in that sector.
        """
        ranges = ring_data["ranges"]

        # Group into 8 sectors
        sectors: dict[str, list[float]] = {
            "FRONT": [],
            "FRONT-LEFT": [],
            "LEFT": [],
            "BACK-LEFT": [],
            "BACK": [],
            "BACK-RIGHT": [],
            "RIGHT": [],
            "FRONT-RIGHT": [],
        }

        for i, distance in enumerate(ranges):
            angle = i * LIDAR_ANGLE_INCREMENT
            if angle >= 350 or angle < 10:
                sectors["FRONT"].append(distance)
            elif angle < 80:
                sectors["FRONT-LEFT"].append(distance)
            elif angle < 100:
                sectors["LEFT"].append(distance)
            elif angle < 170:
                sectors["BACK-LEFT"].append(distance)
            elif angle < 190:
                sectors["BACK"].append(distance)
            elif angle < 260:
                sectors["BACK-RIGHT"].append(distance)
            elif angle < 280:
                sectors["RIGHT"].append(distance)
            else:
                sectors["FRONT-RIGHT"].append(distance)

        # Get minimum distance per sector
        result = {}
        for sector_name, distances in sectors.items():
            valid = [d for d in distances if d > 0]
            result[sector_name] = min(valid) if valid else -1

        return result

    def format_lidar_summary(self, lidar_data: dict) -> str:
        """Format waist-ring LiDAR data as a clear summary for Gemini.

        Uses waist ring (~0.75m, horizontal) for obstacle detection.
        Head ring was removed due to interference with robot shoulders.
        """
        waist_sectors = self._format_ring_sectors(lidar_data["waist_ring"])
        closest = lidar_data["closest_obstacle"]

        lines = [
            "LiDAR SCAN (360°, 180 rays, waist height ~0.75m):",
            "(Detects obstacles at waist height - barrels, boxes, furniture)",
            "",
        ]

        # Waist ring sectors
        for sector, dist in waist_sectors.items():
            if dist > 0:
                lines.append(f"  {sector}: {dist:.2f}m")
            else:
                lines.append(f"  {sector}: clear (>10m)")

        # Closest obstacle
        if closest["range"] > 0:
            lines.append("")
            lines.append(f"CLOSEST OBSTACLE: {closest['range']:.2f}m at {closest['angle']:.0f}°")

        return "\n".join(lines)

    def capture_image(
        self, renderer: mujoco.Renderer, d: mujoco.MjData, camera_name: str = CAMERA_NAME
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

        # Convert to PIL and then base64 (JPEG for smaller size, faster API calls)
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return img_base64, pil_img

    def capture_360_image(
        self, renderer: mujoco.Renderer, d: mujoco.MjData
    ) -> tuple[str, Image.Image]:
        """Capture 360° panorama from robot's 4 cameras and stitch into single image.

        Returns a horizontal strip: [LEFT | FRONT | RIGHT | BACK]
        This gives Gemini a complete view of surroundings in one image.
        """
        # Camera names in order for panorama (left to right visually)
        cameras = [
            ("head_camera_left", "LEFT"),
            ("head_camera", "FRONT"),
            ("head_camera_right", "RIGHT"),
            ("head_camera_back", "BACK"),
        ]

        # Create visualization options that hide LiDAR rays
        vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(vopt)
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        vopt.sitegroup[3] = False
        vopt.sitegroup[4] = False
        vopt.sitegroup[5] = False

        # Capture from each camera
        images = []
        for camera_name, label in cameras:
            renderer.update_scene(d, camera=camera_name, scene_option=vopt)
            img = renderer.render()
            pil_img = Image.fromarray(img)

            # Add direction label to image
            draw = ImageDraw.Draw(pil_img)
            # Try to use a larger font, fall back to default
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except OSError:
                font = ImageFont.load_default()

            # Draw label with background for visibility
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x = (pil_img.width - text_width) // 2
            y = 10
            # Black outline
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                draw.text((x + dx, y + dy), label, fill="black", font=font)
            # White text
            draw.text((x, y), label, fill="white", font=font)

            images.append(pil_img)

        # Stitch horizontally: [LEFT | FRONT | RIGHT | BACK]
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        panorama = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for img in images:
            panorama.paste(img, (x_offset, 0))
            x_offset += img.width

        # Convert to base64
        buffer = BytesIO()
        panorama.save(buffer, format="JPEG", quality=75)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return img_base64, panorama

    def compute_observation(self, d: mujoco.MjData, cmd: np.ndarray) -> np.ndarray:
        """Compute observation vector for policy."""
        # Only get robot joint positions/velocities (not dynamic objects like barrels)
        qj = d.qpos[7 : 7 + self.num_actions]
        dqj = d.qvel[6 : 6 + self.num_actions]
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
        self.obs[9 : 9 + self.num_actions] = qj_normalized
        self.obs[9 + self.num_actions : 9 + 2 * self.num_actions] = dqj_normalized
        self.obs[9 + 2 * self.num_actions : 9 + 3 * self.num_actions] = self.action
        self.obs[9 + 3 * self.num_actions : 9 + 3 * self.num_actions + 2] = np.array(
            [sin_phase, cos_phase]
        )

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
        # Only get robot joint positions/velocities (not dynamic objects like barrels)
        qj = d.qpos[7 : 7 + self.num_actions]
        dqj = d.qvel[6 : 6 + self.num_actions]
        tau = pd_control(self.target_dof_pos, qj, self.kps, np.zeros_like(self.kds), dqj, self.kds)
        # Only control G1's actuators (first num_actions), not background robots
        d.ctrl[: self.num_actions] = tau

    def step(self, d: mujoco.MjData, cmd: np.ndarray) -> None:
        """Perform one control step (call every control_decimation physics steps)."""
        self.counter += 1
        if self.counter % self.control_decimation == 0:
            self.compute_action(d, cmd)
        self.apply_control(d)


def create_renderer(m: mujoco.MjModel) -> mujoco.Renderer:
    """Create a MuJoCo renderer for camera capture."""
    return mujoco.Renderer(m, CAMERA_HEIGHT, CAMERA_WIDTH)
