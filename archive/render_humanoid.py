"""
Render the MuJoCo humanoid model to show what's available.
"""

import mujoco
import numpy as np
from PIL import Image

def render_humanoid():
    """Render the DeepMind humanoid model."""

    # Load the humanoid model
    model = mujoco.MjModel.from_xml_path("humanoid.xml")
    data = mujoco.MjData(model)

    # Step forward to get a natural pose
    mujoco.mj_forward(model, data)

    # Create renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)

    # Camera setup
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.0]  # Look at torso height
    camera.distance = 4
    camera.azimuth = 135
    camera.elevation = -20

    # Render
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()

    # Save
    img = Image.fromarray(frame)
    img.save("preview_humanoid.png")
    print("Saved: preview_humanoid.png")

    # Also render from front
    camera.azimuth = 180
    renderer.update_scene(data, camera=camera)
    frame = renderer.render()
    img = Image.fromarray(frame)
    img.save("preview_humanoid_front.png")
    print("Saved: preview_humanoid_front.png")

    renderer.close()


if __name__ == "__main__":
    render_humanoid()
