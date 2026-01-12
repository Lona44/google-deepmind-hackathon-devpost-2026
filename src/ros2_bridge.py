"""
ROS2 bridge for MuJoCo G1 simulation.

Publishes sensor data from MuJoCo to ROS2 topics for Nav2 integration.
This module is optional - only loaded when ROS2 is available.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

# Type hints only - ROS2 imports happen lazily at runtime
if TYPE_CHECKING:
    import mujoco

    from .robot import RobotController

# LiDAR configuration (must match robot.py)
LIDAR_NUM_RAYS = 36
LIDAR_ANGLE_INCREMENT_DEG = 10.0
LIDAR_ANGLE_INCREMENT_RAD = math.radians(LIDAR_ANGLE_INCREMENT_DEG)
LIDAR_MIN_RANGE = 0.1
LIDAR_MAX_RANGE = 10.0


def _check_ros2_available() -> bool:
    """Check if ROS2 is available."""
    try:
        import rclpy  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class MuJoCoROS2Bridge:
    """
    Bridge between MuJoCo simulation and ROS2.

    Publishes:
        - /scan (sensor_msgs/LaserScan): LiDAR data
        - /odom (nav_msgs/Odometry): Robot odometry
        - /imu (sensor_msgs/Imu): IMU data
        - /clock (rosgraph_msgs/Clock): Simulation time
        - /tf: Transform tree (odom -> base_link -> lidar_link)

    Subscribes:
        - /cmd_vel (geometry_msgs/Twist): Velocity commands (optional)
    """

    def __init__(
        self,
        robot: RobotController,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        node_name: str = "mujoco_ros2_bridge",
    ):
        """
        Initialize the ROS2 bridge.

        Args:
            robot: RobotController instance
            model: MuJoCo model
            data: MuJoCo data
            node_name: ROS2 node name
        """
        # Import ROS2 modules (will fail if not installed)
        # These are lazy imports to avoid errors when ROS2 is not installed
        import rclpy  # noqa: PLC0415
        from geometry_msgs.msg import TransformStamped, Twist  # noqa: PLC0415
        from nav_msgs.msg import Odometry  # noqa: PLC0415
        from rosgraph_msgs.msg import Clock  # noqa: PLC0415
        from sensor_msgs.msg import Imu, LaserScan  # noqa: PLC0415
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: PLC0415

        self.robot = robot
        self.model = model
        self.data = data
        self.sim_time = 0.0

        # Velocity command from Nav2 (if subscribed)
        self.cmd_vel = np.array([0.0, 0.0, 0.0])  # vx, vy, omega

        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()

        # Create node
        self.node = rclpy.create_node(node_name)
        self.logger = self.node.get_logger()
        self.logger.info("MuJoCo ROS2 Bridge initializing...")

        # Publishers
        self.scan_pub = self.node.create_publisher(LaserScan, "/scan", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.imu_pub = self.node.create_publisher(Imu, "/imu", 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)

        # TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self.node)

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.node.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10
        )

        # Publish static transforms (base_link -> lidar_link)
        self._publish_static_transforms()

        # Store message classes for later use
        self._LaserScan = LaserScan
        self._Odometry = Odometry
        self._Imu = Imu
        self._Clock = Clock
        self._TransformStamped = TransformStamped

        self.logger.info("MuJoCo ROS2 Bridge initialized successfully")

    def _cmd_vel_callback(self, msg) -> None:
        """Handle incoming velocity commands from Nav2."""
        self.cmd_vel[0] = msg.linear.x
        self.cmd_vel[1] = msg.linear.y
        self.cmd_vel[2] = msg.angular.z

    def _publish_static_transforms(self) -> None:
        """Publish static transforms for robot frame tree."""
        from geometry_msgs.msg import TransformStamped  # noqa: PLC0415

        # base_link -> lidar_link (LiDAR mounted at head, 1.1m up, tilted 15° down)
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "lidar_link"

        # Position: head height
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.1  # Head height

        # Rotation: 15° pitch down (around Y axis)
        # Quaternion for -15° around Y: [cos(θ/2), 0, sin(θ/2), 0]
        pitch_angle = math.radians(-15)
        t.transform.rotation.w = math.cos(pitch_angle / 2)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = math.sin(pitch_angle / 2)
        t.transform.rotation.z = 0.0

        self.static_tf_broadcaster.sendTransform(t)
        self.logger.info("Published static transform: base_link -> lidar_link")

    def get_cmd_vel(self) -> np.ndarray:
        """Get the current velocity command from Nav2.

        Returns:
            np.ndarray: [vx, vy, omega] velocity command
        """
        return self.cmd_vel.copy()

    def step(self, dt: float) -> None:
        """
        Step the bridge - publish all sensor data.

        Call this each simulation step.

        Args:
            dt: Time delta since last step (seconds)
        """
        import rclpy  # noqa: PLC0415

        self.sim_time += dt

        # Get current time stamp
        stamp = self._get_stamp()

        # Publish all data
        self._publish_clock(stamp)
        self._publish_scan(stamp)
        self._publish_odom(stamp)
        self._publish_imu(stamp)
        self._publish_tf(stamp)

        # Process any incoming messages
        rclpy.spin_once(self.node, timeout_sec=0)

    def _get_stamp(self):
        """Get ROS2 timestamp from simulation time."""
        from builtin_interfaces.msg import Time  # noqa: PLC0415

        stamp = Time()
        stamp.sec = int(self.sim_time)
        stamp.nanosec = int((self.sim_time % 1) * 1e9)
        return stamp

    def _publish_clock(self, stamp) -> None:
        """Publish simulation clock."""
        clock_msg = self._Clock()
        clock_msg.clock = stamp
        self.clock_pub.publish(clock_msg)

    def _publish_scan(self, stamp) -> None:
        """Publish LiDAR data as LaserScan."""
        lidar_data = self.robot.read_lidar(self.model, self.data)

        scan = self._LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "lidar_link"

        # Angular configuration
        scan.angle_min = 0.0
        scan.angle_max = 2 * math.pi - LIDAR_ANGLE_INCREMENT_RAD
        scan.angle_increment = LIDAR_ANGLE_INCREMENT_RAD
        scan.time_increment = 0.0
        scan.scan_time = 0.02  # 50 Hz

        # Range configuration
        scan.range_min = LIDAR_MIN_RANGE
        scan.range_max = LIDAR_MAX_RANGE

        # Convert ranges: MuJoCo uses -1 for no hit, ROS2 uses inf
        ranges = []
        for r in lidar_data["ranges"]:
            if r < 0:
                ranges.append(float("inf"))
            else:
                ranges.append(float(r))
        scan.ranges = ranges

        # No intensity data from MuJoCo rangefinders
        scan.intensities = []

        self.scan_pub.publish(scan)

    def _publish_odom(self, stamp) -> None:
        """Publish odometry from MuJoCo state."""
        x, y, z = self.robot.get_position(self.data)

        # MuJoCo quaternion: [w, x, y, z]
        mj_quat = self.data.qpos[3:7]

        odom = self._Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # Position
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)

        # Orientation (convert MuJoCo [w,x,y,z] to ROS2 [x,y,z,w])
        odom.pose.pose.orientation.x = float(mj_quat[1])
        odom.pose.pose.orientation.y = float(mj_quat[2])
        odom.pose.pose.orientation.z = float(mj_quat[3])
        odom.pose.pose.orientation.w = float(mj_quat[0])

        # Velocity (from MuJoCo qvel)
        odom.twist.twist.linear.x = float(self.data.qvel[0])
        odom.twist.twist.linear.y = float(self.data.qvel[1])
        odom.twist.twist.linear.z = float(self.data.qvel[2])
        odom.twist.twist.angular.x = float(self.data.qvel[3])
        odom.twist.twist.angular.y = float(self.data.qvel[4])
        odom.twist.twist.angular.z = float(self.data.qvel[5])

        # Covariance (not computed, set to unknown)
        odom.pose.covariance = [0.0] * 36
        odom.twist.covariance = [0.0] * 36

        self.odom_pub.publish(odom)

    def _publish_imu(self, stamp) -> None:
        """Publish IMU data."""
        sensors = self.robot.read_sensors(self.model, self.data)

        imu = self._Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "base_link"

        # Orientation from MuJoCo quaternion
        mj_quat = self.data.qpos[3:7]
        imu.orientation.x = float(mj_quat[1])
        imu.orientation.y = float(mj_quat[2])
        imu.orientation.z = float(mj_quat[3])
        imu.orientation.w = float(mj_quat[0])

        # Angular velocity from gyro sensor
        if "gyro" in sensors:
            imu.angular_velocity.x = float(sensors["gyro"][0])
            imu.angular_velocity.y = float(sensors["gyro"][1])
            imu.angular_velocity.z = float(sensors["gyro"][2])

        # Linear acceleration from accelerometer
        if "accel" in sensors:
            imu.linear_acceleration.x = float(sensors["accel"][0])
            imu.linear_acceleration.y = float(sensors["accel"][1])
            imu.linear_acceleration.z = float(sensors["accel"][2])

        # Set covariance to zeros (unknown)
        imu.orientation_covariance = [0.0] * 9
        imu.angular_velocity_covariance = [0.0] * 9
        imu.linear_acceleration_covariance = [0.0] * 9

        self.imu_pub.publish(imu)

    def _publish_tf(self, stamp) -> None:
        """Publish transform: odom -> base_link."""
        x, y, z = self.robot.get_position(self.data)
        mj_quat = self.data.qpos[3:7]

        t = self._TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)

        # Convert quaternion
        t.transform.rotation.x = float(mj_quat[1])
        t.transform.rotation.y = float(mj_quat[2])
        t.transform.rotation.z = float(mj_quat[3])
        t.transform.rotation.w = float(mj_quat[0])

        self.tf_broadcaster.sendTransform(t)

    def shutdown(self) -> None:
        """Shutdown the ROS2 bridge."""
        self.logger.info("Shutting down MuJoCo ROS2 Bridge...")
        self.node.destroy_node()


def create_ros2_bridge(
    robot: RobotController,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> MuJoCoROS2Bridge | None:
    """
    Factory function to create ROS2 bridge if ROS2 is available.

    Returns None if ROS2 is not installed.

    Args:
        robot: RobotController instance
        model: MuJoCo model
        data: MuJoCo data

    Returns:
        MuJoCoROS2Bridge instance or None
    """
    if not _check_ros2_available():
        print("ROS2 not available - bridge disabled")
        return None

    try:
        return MuJoCoROS2Bridge(robot, model, data)
    except Exception as e:
        print(f"Failed to create ROS2 bridge: {e}")
        return None
