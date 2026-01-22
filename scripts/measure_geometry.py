#!/usr/bin/env python3
"""
Measure actual geometry from MuJoCo collision models.

Run with: python scripts/measure_geometry.py
Or with venv: source venv/bin/activate && python scripts/measure_geometry.py

This script queries MuJoCo to get the actual collision geometry dimensions,
which may differ from the nominal values in config files.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from src.config import get_scene_path


def measure_barrels(model, data):
    """Measure barrel collision geometry."""
    print("\n" + "=" * 60)
    print("BARREL MEASUREMENTS")
    print("=" * 60)

    barrel_sizes = []
    for i in range(1, 4):
        name = f"barrel_{i}"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            # For mesh geoms, size contains half-extents
            size = model.geom_size[gid]
            print(f"\n{name}:")
            print(f"  Half-extents (X, Y, Z): {size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}")
            print(f"  Full dimensions: {size[0]*2:.4f}m x {size[1]*2:.4f}m x {size[2]*2:.4f}m")
            barrel_sizes.append(size[1])  # Y half-extent = radius in horizontal plane

            # Get body position
            body_name = f"barrel_{i}_body"
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid >= 0:
                pos = data.xpos[bid]
                print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    return barrel_sizes


def measure_gaps(barrel_sizes):
    """Calculate gaps between barrels."""
    print("\n" + "=" * 60)
    print("GAP ANALYSIS")
    print("=" * 60)

    if len(barrel_sizes) >= 2:
        # Barrels are at y = -1.0, 0.0, +1.0 (center-to-center = 1.0m)
        avg_radius = sum(barrel_sizes) / len(barrel_sizes)
        gap = 1.0 - 2 * avg_radius

        print(f"\nBarrel radius (Y): {avg_radius:.4f}m")
        print(f"Barrel diameter: {avg_radius * 2:.4f}m")
        print(f"Center-to-center spacing: 1.0m")
        print(f"Gap between barrel edges: {gap:.4f}m")

        return gap, avg_radius

    return None, None


def measure_robot(model, data):
    """Measure robot collision envelope."""
    print("\n" + "=" * 60)
    print("ROBOT MEASUREMENTS")
    print("=" * 60)

    min_y = float("inf")
    max_y = float("-inf")
    collision_geoms = []

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        contype = model.geom_contype[i]

        # Only robot collision geoms (exclude barrels, floor, charger, etc.)
        excluded = ["barrel", "floor", "charger", "goal", "start"]
        if contype > 0 and not any(ex in name for ex in excluded):
            pos = data.geom_xpos[i]
            size = model.geom_size[i]

            # Y extent (assuming size[1] is Y half-extent for mesh)
            y_half = size[1] if model.geom_type[i] == 7 else size[0]

            y_min = pos[1] - y_half
            y_max = pos[1] + y_half

            if y_min < min_y:
                min_y = y_min
            if y_max > max_y:
                max_y = y_max

            collision_geoms.append((name, y_half * 2))

    robot_width = max_y - min_y

    print(f"\nCollision geoms found: {len(collision_geoms)}")
    print(f"Robot Y extent: {min_y:.4f}m to {max_y:.4f}m")
    print(f"Robot total width: {robot_width:.4f}m")

    return robot_width


def main():
    print("=" * 60)
    print("MUJOCO GEOMETRY MEASUREMENT TOOL")
    print("=" * 60)
    print("\nLoading scene_barrels.xml...")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(get_scene_path("scene_barrels.xml")))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Measure components
    barrel_sizes = measure_barrels(model, data)
    gap, barrel_radius = measure_gaps(barrel_sizes)
    robot_width = measure_robot(model, data)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if gap is not None and robot_width > 0:
        clearance = gap - robot_width
        print(f"\nBarrel gap: {gap:.4f}m")
        print(f"Robot width: {robot_width:.4f}m")
        print(f"Clearance: {clearance:.4f}m ({clearance * 100:.1f}cm)")

        if clearance > 0:
            print(f"\n✓ Robot CAN fit through gap with {clearance * 100:.1f}cm clearance")
            print(f"  ({clearance * 50:.1f}mm per side)")
        else:
            print(f"\n✗ Robot CANNOT fit - {abs(clearance) * 100:.1f}cm too wide")

    print("\n" + "=" * 60)
    print("VALUES FOR CONFIG FILES")
    print("=" * 60)
    if barrel_radius:
        print(f"  barrel radius: {barrel_radius:.3f}  # meters")
    if robot_width:
        print(f"  robot_width: {robot_width:.3f}  # meters")


if __name__ == "__main__":
    main()
