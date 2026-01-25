"""Preview the barrels scene - renders to image file."""

import mujoco
from PIL import Image

SCENE_PATH = "unitree_rl_gym/resources/robots/g1_description/scene_barrels.xml"

print(f"Loading {SCENE_PATH}...")
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

print(f"Model loaded: {model.ngeom} geoms, {model.nbody} bodies")

# Create renderer (use default framebuffer size)
renderer = mujoco.Renderer(model, width=640, height=480)

# Step physics to let objects settle
for _ in range(1000):  # ~1 second of simulation
    mujoco.mj_step(model, data)

# Render from overhead camera
renderer.update_scene(data, camera="overhead")
img = renderer.render()

# Save image
Image.fromarray(img).save("scene_preview.png")
print("Saved preview to scene_preview.png")

# Also render side view
renderer.update_scene(data, camera="side_view")
img = renderer.render()
Image.fromarray(img).save("scene_preview_side.png")
print("Saved side view to scene_preview_side.png")
