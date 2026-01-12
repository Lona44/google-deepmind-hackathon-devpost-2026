"""
Genesis G1 Humanoid Environment for Alignment Research.

This environment trains a G1 humanoid to walk while respecting safety constraints.
It can use either standard rewards or Gemini-based RLAIF rewards.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import genesis as gs
except ImportError:
    gs = None  # Allow import for type checking without genesis installed

# Project root for finding robot models
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class G1EnvConfig:
    """Configuration for G1 environment."""

    # Simulation
    dt: float = 0.01  # 100Hz simulation
    substeps: int = 4
    episode_length: int = 1000  # 10 seconds per episode

    # Robot
    robot_path: str = "unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml"
    initial_height: float = 1.0  # Starting height above ground

    # Observation space (simplified)
    # - Base orientation (4: quaternion)
    # - Base angular velocity (3)
    # - Joint positions (12)
    # - Joint velocities (12)
    # - Previous actions (12)
    # Total: 43
    obs_dim: int = 43

    # Action space: 12 DOF joint position targets
    action_dim: int = 12

    # Reward weights (standard locomotion)
    reward_forward_vel: float = 1.0
    reward_lateral_vel: float = -0.5
    reward_angular_vel: float = -0.5
    reward_joint_torque: float = -0.0001
    reward_alive: float = 0.1

    # Safety constraints (for alignment research)
    forbidden_zone_penalty: float = -10.0
    forbidden_zones: list = None  # List of (x_min, x_max, y_min, y_max)

    # Termination
    terminate_on_fall: bool = True
    fall_height_threshold: float = 0.5

    # Rendering
    show_viewer: bool = False

    def __post_init__(self):
        if self.forbidden_zones is None:
            self.forbidden_zones = []


class G1Env:
    """
    Genesis-based G1 humanoid environment.

    Supports both standard RL rewards and Gemini-based RLAIF rewards.
    """

    def __init__(self, config: G1EnvConfig | None = None, num_envs: int = 1):
        """
        Initialize the G1 environment.

        Args:
            config: Environment configuration.
            num_envs: Number of parallel environments (for batch training).
        """
        if gs is None:
            raise ImportError(
                "Genesis not installed. Install with: pip install genesis-world"
            )

        self.config = config or G1EnvConfig()
        self.num_envs = num_envs

        # State tracking
        self.step_count = 0
        self.episode_rewards = np.zeros(num_envs)
        self.prev_actions = np.zeros((num_envs, self.config.action_dim))

        # Gemini reward model (optional, set via set_reward_model)
        self.gemini_reward_model = None
        self.use_gemini_rewards = False

        # Initialize Genesis
        self._init_genesis()

    def _init_genesis(self):
        """Initialize Genesis simulation."""
        # Initialize Genesis with appropriate backend
        gs.init(backend=gs.cpu)  # Use gs.gpu or gs.metal for acceleration

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.dt,
                substeps=self.config.substeps,
            ),
            show_viewer=self.config.show_viewer,
        )

        # Add ground plane
        self.ground = self.scene.add_entity(gs.morphs.Plane())

        # Load G1 robot
        robot_path = PROJECT_ROOT / self.config.robot_path
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(robot_path),
                pos=(0, 0, self.config.initial_height),
            ),
        )

        # Build scene
        self.scene.build()

        # Get joint info
        self.joint_names = []  # TODO: Extract from robot
        self.n_joints = self.config.action_dim

    def set_reward_model(self, gemini_reward_model):
        """
        Set Gemini as the reward model for RLAIF training.

        Args:
            gemini_reward_model: GeminiRewardModel instance.
        """
        self.gemini_reward_model = gemini_reward_model
        self.use_gemini_rewards = True

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        Returns:
            Initial observation.
        """
        self.step_count = 0
        self.episode_rewards = np.zeros(self.num_envs)
        self.prev_actions = np.zeros((self.num_envs, self.config.action_dim))

        # Reset robot position
        # TODO: Implement proper reset in Genesis

        return self._get_obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Step the environment.

        Args:
            actions: Joint position targets (num_envs, action_dim).

        Returns:
            obs: Observations (num_envs, obs_dim).
            rewards: Rewards (num_envs,).
            dones: Episode termination flags (num_envs,).
            info: Additional information.
        """
        # Apply actions (joint position control)
        # TODO: Implement proper control in Genesis
        self.prev_actions = actions.copy()

        # Step simulation
        self.scene.step()
        self.step_count += 1

        # Get observations
        obs = self._get_obs()

        # Compute rewards
        if self.use_gemini_rewards and self.gemini_reward_model is not None:
            rewards = self._compute_gemini_rewards(obs, actions)
        else:
            rewards = self._compute_standard_rewards(obs, actions)

        self.episode_rewards += rewards

        # Check termination
        dones = self._check_termination(obs)

        # Episode timeout
        if self.step_count >= self.config.episode_length:
            dones = np.ones(self.num_envs, dtype=bool)

        info = {
            "episode_rewards": self.episode_rewards.copy(),
            "step_count": self.step_count,
        }

        return obs, rewards, dones, info

    def _get_obs(self) -> np.ndarray:
        """Get current observations."""
        # TODO: Implement proper observation extraction from Genesis
        # For now, return zeros as placeholder
        return np.zeros((self.num_envs, self.config.obs_dim))

    def _compute_standard_rewards(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """
        Compute standard locomotion rewards.

        Args:
            obs: Current observations.
            actions: Actions taken.

        Returns:
            Rewards for each environment.
        """
        rewards = np.zeros(self.num_envs)

        # TODO: Extract velocities from observations
        # forward_vel = ...
        # lateral_vel = ...
        # angular_vel = ...

        # Alive bonus
        rewards += self.config.reward_alive

        # Torque penalty (encourage smooth movements)
        torque_penalty = np.sum(actions**2, axis=-1) * self.config.reward_joint_torque
        rewards += torque_penalty

        # Forbidden zone penalties (for alignment research)
        # TODO: Check robot position against forbidden zones
        # rewards += self._compute_forbidden_zone_penalty()

        return rewards

    def _compute_gemini_rewards(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """
        Compute rewards using Gemini as the judge (RLAIF).

        Args:
            obs: Current observations.
            actions: Actions taken.

        Returns:
            Rewards from Gemini evaluation.
        """
        if self.gemini_reward_model is None:
            return self._compute_standard_rewards(obs, actions)

        # Get Gemini's evaluation of the robot's behavior
        # This is called periodically, not every step (too expensive)
        return self.gemini_reward_model.evaluate(obs, actions)

    def _check_termination(self, obs: np.ndarray) -> np.ndarray:
        """Check if episodes should terminate."""
        dones = np.zeros(self.num_envs, dtype=bool)

        if self.config.terminate_on_fall:
            # TODO: Check robot height from observations
            # heights = ...
            # dones = heights < self.config.fall_height_threshold
            pass

        return dones

    def render(self):
        """Render the environment."""
        if self.config.show_viewer:
            # Genesis handles rendering automatically with viewer
            pass

    def close(self):
        """Clean up resources."""
        pass
