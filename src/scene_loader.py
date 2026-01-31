"""
Scene loader with support for background robots from mujoco_menagerie.
"""

from pathlib import Path

import mujoco

from src.paths import PROJECT_ROOT

# Paths to menagerie models (relative to project root)
MENAGERIE_PATH = PROJECT_ROOT / "mujoco_menagerie"


def _get_menagerie_model(name: str) -> Path:
    """Get path to a menagerie robot model."""
    model_map = {
        "spot": MENAGERIE_PATH / "boston_dynamics_spot" / "spot.xml",
        "go2": MENAGERIE_PATH / "unitree_go2" / "go2.xml",
        "panda": MENAGERIE_PATH / "franka_emika_panda" / "panda.xml",
        "barkour": MENAGERIE_PATH / "google_barkour_vb" / "barkour_vb.xml",
        "google_robot": MENAGERIE_PATH / "google_robot" / "robot.xml",
    }
    if name not in model_map:
        raise ValueError(f"Unknown menagerie model: {name}")
    return model_map[name]


def load_scene_with_background_robots(
    base_scene_path: str | Path,
    enable_background_robots: bool = True,
) -> mujoco.MjModel:
    """
    Load a MuJoCo scene with optional background robots from mujoco_menagerie.

    Args:
        base_scene_path: Path to the base scene XML file
        enable_background_robots: Whether to add background robots (Spot, Go2, Panda)

    Returns:
        Combined MjModel with all robots attached
    """
    if not enable_background_robots or not MENAGERIE_PATH.exists():
        # Fall back to standard loading
        return mujoco.MjModel.from_xml_path(str(base_scene_path))

    # Load base scene as spec
    base_spec = mujoco.MjSpec.from_file(str(base_scene_path))

    # Add background robots
    _add_spot(base_spec)
    _add_go2(base_spec)
    _add_panda_workstation(base_spec)
    _add_barkour(base_spec)
    _add_google_robot(base_spec)

    # Compile and return
    return base_spec.compile()


def _add_spot(spec: mujoco.MjSpec) -> None:
    """Add Boston Dynamics Spot to the scene."""
    try:
        spot_spec = mujoco.MjSpec.from_file(str(_get_menagerie_model("spot")))

        # Position Spot in back-right area, facing toward center
        frame = spec.worldbody.add_frame()
        frame.pos = [7, -3, 0]
        frame.quat = [0.92, 0, 0, 0.38]  # Rotated to face center

        spot_body = spot_spec.worldbody.first_body()
        frame.attach_body(spot_body, prefix="spot_")
    except Exception:
        pass  # Silently skip if model unavailable


def _add_go2(spec: mujoco.MjSpec) -> None:
    """Add Unitree Go2 to the scene."""
    try:
        go2_spec = mujoco.MjSpec.from_file(str(_get_menagerie_model("go2")))

        # Position Go2 in back-left area
        frame = spec.worldbody.add_frame()
        frame.pos = [6, 4, 0]
        frame.quat = [0.92, 0, 0, -0.38]  # Rotated to face center

        go2_body = go2_spec.worldbody.first_body()
        frame.attach_body(go2_body, prefix="go2_")
    except Exception:
        pass


def _add_panda_workstation(spec: mujoco.MjSpec) -> None:
    """Add Franka Panda arm on a workbench."""
    try:
        panda_spec = mujoco.MjSpec.from_file(str(_get_menagerie_model("panda")))

        # Add workbench
        workbench = spec.worldbody.add_geom()
        workbench.name = "workbench"
        workbench.type = mujoco.mjtGeom.mjGEOM_BOX
        workbench.size = [0.4, 0.6, 0.4]
        workbench.pos = [-1.5, -4, 0.4]
        workbench.rgba = [0.45, 0.4, 0.35, 1]
        workbench.contype = 0  # No collision
        workbench.conaffinity = 0

        # Add Panda on workbench
        frame = spec.worldbody.add_frame()
        frame.pos = [-1.5, -4, 0.8]
        frame.quat = [0.707, 0, 0, 0.707]  # 90 degree rotation

        panda_body = panda_spec.worldbody.first_body()
        frame.attach_body(panda_body, prefix="panda_")
    except Exception:
        pass


def _add_barkour(spec: mujoco.MjSpec) -> None:
    """Add Google Barkour quadruped to the scene."""
    try:
        barkour_spec = mujoco.MjSpec.from_file(str(_get_menagerie_model("barkour")))

        # Position Barkour near the front-left area
        frame = spec.worldbody.add_frame()
        frame.pos = [-2, 3, 0]
        frame.quat = [0.85, 0, 0, -0.53]  # Facing toward barrels

        barkour_body = barkour_spec.worldbody.first_body()
        frame.attach_body(barkour_body, prefix="barkour_")
    except Exception:
        pass


def _add_google_robot(spec: mujoco.MjSpec) -> None:
    """Add Google Robot (mobile manipulator) to the scene."""
    try:
        grobot_spec = mujoco.MjSpec.from_file(str(_get_menagerie_model("google_robot")))

        # Position near the back, as if observing
        frame = spec.worldbody.add_frame()
        frame.pos = [8, 0, 0]
        frame.quat = [0.707, 0, 0, 0.707]  # Facing toward the scene

        grobot_body = grobot_spec.worldbody.first_body()
        frame.attach_body(grobot_body, prefix="grobot_")
    except Exception:
        pass


BACKGROUND_PREFIXES = ("spot_", "go2_", "panda_", "barkour_", "grobot_")

# Cache for background robot indices (computed once per model)
_bg_joint_cache: dict[int, list[tuple[int, int]]] = {}  # model_ptr -> [(vel_addr, num_dof), ...]
_bg_actuator_cache: dict[int, list[int]] = {}  # model_ptr -> [actuator_indices]


def _get_background_indices(m: mujoco.MjModel) -> tuple[list[tuple[int, int]], list[int]]:
    """Get cached indices for background robot joints and actuators."""
    model_id = id(m)

    if model_id not in _bg_joint_cache:
        joint_indices = []
        for i in range(m.njnt):
            jnt = m.joint(i)
            jname = jnt.name
            if any(jname.startswith(prefix) for prefix in BACKGROUND_PREFIXES):
                vel_addr = jnt.dofadr[0]
                num_dof = 6 if jnt.type == mujoco.mjtJoint.mjJNT_FREE else 1
                joint_indices.append((vel_addr, num_dof))
        _bg_joint_cache[model_id] = joint_indices

        actuator_indices = []
        for i in range(m.nu):
            act_name = m.actuator(i).name
            if any(act_name.startswith(prefix) for prefix in BACKGROUND_PREFIXES):
                actuator_indices.append(i)
        _bg_actuator_cache[model_id] = actuator_indices

    return _bg_joint_cache[model_id], _bg_actuator_cache[model_id]


def freeze_background_robots(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Freeze all background robots by zeroing their velocities and controls.

    Call this every physics step to keep background robots static.
    """
    joint_indices, actuator_indices = _get_background_indices(m)

    # Zero velocities
    for vel_addr, num_dof in joint_indices:
        d.qvel[vel_addr : vel_addr + num_dof] = 0

    # Zero actuator controls
    for act_idx in actuator_indices:
        d.ctrl[act_idx] = 0


def set_background_robot_poses(m: mujoco.MjModel, d: mujoco.MjData) -> None:
    """
    Set standing/resting poses for all background robots.

    Call this after creating MjData to pose the background robots naturally.
    """
    for i in range(m.njnt):
        jnt = m.joint(i)
        jname = jnt.name
        addr = jnt.qposadr[0]

        # Spot standing pose
        if jname == "spot_freejoint":
            d.qpos[addr : addr + 3] = [7, -3, 0.52]
            d.qpos[addr + 3 : addr + 7] = [0.92, 0, 0, 0.38]
        elif "spot_" in jname:
            if "_hy" in jname:
                d.qpos[addr] = 0.7
            elif "_kn" in jname:
                d.qpos[addr] = -1.4

        # Go2 standing pose
        if jname == "go2_freejoint":
            d.qpos[addr : addr + 3] = [6, 4, 0.35]
            d.qpos[addr + 3 : addr + 7] = [0.92, 0, 0, -0.38]
        elif "go2_" in jname:
            if "thigh" in jname:
                d.qpos[addr] = 0.8
            elif "calf" in jname:
                d.qpos[addr] = -1.5

        # Panda resting pose
        if "panda_joint" in jname:
            poses = {
                "joint1": 0,
                "joint2": -0.5,
                "joint3": 0,
                "joint4": -2.0,
                "joint5": 0,
                "joint6": 1.5,
                "joint7": 0.7,
            }
            for joint_name, pose in poses.items():
                if joint_name in jname:
                    d.qpos[addr] = pose
                    break
        elif "panda_finger" in jname:
            d.qpos[addr] = 0.04  # Gripper open

        # Barkour standing pose
        if jname == "barkour_freejoint":
            d.qpos[addr : addr + 3] = [-2, 3, 0.28]
            d.qpos[addr + 3 : addr + 7] = [0.85, 0, 0, -0.53]
        elif "barkour_" in jname:
            if "hip" in jname:
                d.qpos[addr] = 0.0
            elif "thigh" in jname:
                d.qpos[addr] = 0.7
            elif "calf" in jname or "shin" in jname:
                d.qpos[addr] = -1.4

        # Google Robot pose
        if jname == "grobot_freejoint":
            d.qpos[addr : addr + 3] = [8, 0, 0.05]
            d.qpos[addr + 3 : addr + 7] = [0.707, 0, 0, 0.707]
        elif "grobot_" in jname:
            # Arm in a natural resting position
            if "shoulder_pan" in jname:
                d.qpos[addr] = 0.0
            elif "shoulder_lift" in jname:
                d.qpos[addr] = -0.5
            elif "upperarm_roll" in jname:
                d.qpos[addr] = 0.0
            elif "elbow_flex" in jname:
                d.qpos[addr] = -1.5
            elif "forearm_roll" in jname:
                d.qpos[addr] = 0.0
            elif "wrist_flex" in jname:
                d.qpos[addr] = -0.5
            elif "wrist_roll" in jname:
                d.qpos[addr] = 0.0
