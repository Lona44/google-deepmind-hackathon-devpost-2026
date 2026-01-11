"""
Test G1 Sensor Capabilities

Shows what sensor data is available from the G1 simulation:
- Camera (vision)
- IMU (gyro + accelerometer)
- Foot contact sensors
"""

from pathlib import Path

import mujoco
import mujoco.viewer
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGGED_GYM_ROOT_DIR = PROJECT_ROOT / "unitree_rl_gym"


def test_sensors():
    """Test available G1 sensors."""

    print("=" * 60)
    print("G1 SENSOR TEST")
    print("=" * 60)

    # Load model
    xml_path = LEGGED_GYM_ROOT_DIR / "resources/robots/g1_description/scene_alignment.xml"
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)

    # Print available sensors
    print("\n📊 AVAILABLE SENSORS:")
    print("-" * 40)
    for i in range(m.nsensor):
        sensor_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_type = m.sensor_type[i]
        sensor_dim = m.sensor_dim[i]
        print(f"  {i}: {sensor_name} (type={sensor_type}, dim={sensor_dim})")

    # Print available cameras
    print("\n📷 AVAILABLE CAMERAS:")
    print("-" * 40)
    for i in range(m.ncam):
        cam_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"  {i}: {cam_name}")

    # Step simulation a few times
    for _ in range(100):
        mujoco.mj_step(m, d)

    # Read sensor data
    print("\n📈 SENSOR READINGS:")
    print("-" * 40)

    if m.nsensor > 0:
        # IMU Gyro (angular velocity)
        gyro_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id >= 0:
            gyro_adr = m.sensor_adr[gyro_id]
            gyro_data = d.sensordata[gyro_adr : gyro_adr + 3]
            print(
                f"  IMU Gyro (rad/s): [{gyro_data[0]:.4f}, {gyro_data[1]:.4f}, {gyro_data[2]:.4f}]"
            )

        # IMU Accelerometer
        accel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if accel_id >= 0:
            accel_adr = m.sensor_adr[accel_id]
            accel_data = d.sensordata[accel_adr : accel_adr + 3]
            print(
                f"  IMU Accel (m/s²): [{accel_data[0]:.4f}, {accel_data[1]:.4f}, {accel_data[2]:.4f}]"
            )

        # Foot touch sensors
        left_touch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_touch")
        if left_touch_id >= 0:
            left_touch = d.sensordata[m.sensor_adr[left_touch_id]]
            print(f"  Left foot contact: {left_touch:.4f}")

        right_touch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_touch")
        if right_touch_id >= 0:
            right_touch = d.sensordata[m.sensor_adr[right_touch_id]]
            print(f"  Right foot contact: {right_touch:.4f}")

    # Render camera image
    print("\n📷 CAMERA CAPTURE:")
    print("-" * 40)

    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if cam_id >= 0:
        # Create renderer
        width, height = 640, 480
        renderer = mujoco.Renderer(m, height, width)

        # Render from head camera
        renderer.update_scene(d, camera="head_camera")
        img = renderer.render()

        # Save image
        img_path = PROJECT_ROOT / "experiments" / "g1_camera_view.png"
        Image.fromarray(img).save(img_path)
        print(f"  Camera image saved to: {img_path}")
        print(f"  Resolution: {width}x{height}")

        # Also render a third-person view
        renderer.update_scene(d, camera=-1)  # Default free camera
        img_third = renderer.render()
        img_third_path = PROJECT_ROOT / "experiments" / "g1_third_person.png"
        Image.fromarray(img_third).save(img_third_path)
        print(f"  Third-person view saved to: {img_third_path}")
    else:
        print("  No head_camera found!")

    # Summary
    print("\n" + "=" * 60)
    print("SENSOR SUMMARY")
    print("=" * 60)
    print("""
Available sensors for Gemini integration:

1. 📷 VISION (head_camera)
   - RGB images from robot's POV
   - Can be sent to Gemini for visual understanding
   - Resolution: configurable (640x480 default)

2. 🔄 IMU (imu_gyro, imu_accel)
   - Angular velocity (3-axis)
   - Linear acceleration (3-axis)
   - Useful for balance/orientation feedback

3. 👣 TOUCH (left_foot_touch, right_foot_touch)
   - Contact force on each foot
   - Useful for gait detection

4. 📍 POSITION (from d.qpos)
   - Robot base position (x, y, z)
   - Robot orientation (quaternion)
   - Joint angles (12 DOF)

All these can be fed to Gemini for decision-making!
""")


if __name__ == "__main__":
    test_sensors()
