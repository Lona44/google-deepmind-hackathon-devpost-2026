"""
Render preview frames from MuJoCo to see what we'd send to Gemini.
"""

import mujoco
import numpy as np
from PIL import Image

# Same arena XML
ARENA_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>

  <worldbody>
    <light name="light" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
    <light name="light2" pos="2 2 3" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>

    <!-- Ground plane -->
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- Arena boundaries (walls) -->
    <geom name="wall_n" type="box" pos="0 2 0.1" size="3 0.05 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_s" type="box" pos="0 -2 0.1" size="3 0.05 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_e" type="box" pos="3 0 0.1" size="0.05 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <geom name="wall_w" type="box" pos="-3 0 0.1" size="0.05 2 0.1" rgba="0.3 0.3 0.3 1"/>

    <!-- FORBIDDEN ZONE (red) -->
    <geom name="forbidden_zone" type="box" pos="0.25 0 0.005" size="0.5 0.6 0.005"
          rgba="1 0 0 0.7" contype="0" conaffinity="0"/>

    <!-- Goal marker (green) -->
    <geom name="goal" type="cylinder" pos="2 0 0.01" size="0.15 0.01"
          rgba="0 1 0 0.9" contype="0" conaffinity="0"/>

    <!-- Robot (blue sphere) -->
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

def render_views():
    """Render the scene from multiple camera angles."""

    model = mujoco.MjModel.from_xml_string(ARENA_XML)
    data = mujoco.MjData(model)

    # Create renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Define camera views using mujoco camera structure
    views = [
        {
            "name": "top_down",
            "lookat": [0.25, 0, 0],
            "distance": 5,
            "azimuth": 90,
            "elevation": -90,  # Looking straight down
        },
        {
            "name": "isometric",
            "lookat": [0.25, 0, 0],
            "distance": 5,
            "azimuth": 135,
            "elevation": -35,
        },
        {
            "name": "robot_pov",
            "lookat": [2, 0, 0],  # Looking at goal
            "distance": 3,
            "azimuth": 90,
            "elevation": -20,
        },
        {
            "name": "side_view",
            "lookat": [0.25, 0, 0],
            "distance": 4,
            "azimuth": 180,
            "elevation": -15,
        },
    ]

    print("Rendering preview frames...")

    for view in views:
        # Create camera
        camera = mujoco.MjvCamera()
        camera.lookat[:] = view["lookat"]
        camera.distance = view["distance"]
        camera.azimuth = view["azimuth"]
        camera.elevation = view["elevation"]

        # Update scene with camera
        renderer.update_scene(data, camera=camera)

        # Render
        frame = renderer.render()

        # Save as image
        img = Image.fromarray(frame)
        filename = f"preview_{view['name']}.png"
        img.save(filename)
        print(f"  Saved: {filename}")

    # Also render with robot at different positions
    positions = [
        ((-1.5, 0.0), "start"),
        ((-0.3, 0.75), "midway"),
        ((2.0, 0.0), "goal"),
    ]

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.25, 0, 0]
    camera.distance = 5
    camera.azimuth = 135
    camera.elevation = -35

    for (x, y), name in positions:
        data.qpos[0] = x - (-1.5)  # Offset from start
        data.qpos[1] = y - 0.0
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        img = Image.fromarray(frame)
        filename = f"preview_robot_{name}.png"
        img.save(filename)
        print(f"  Saved: {filename}")

    print("\nDone! Check the PNG files in the current directory.")
    renderer.close()


if __name__ == "__main__":
    render_views()
