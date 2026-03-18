"""Gymnasium environment for the Push task with the 3-DOF robot.

The end effector must reach the cube and push it (displace it from its
initial position). No target goal — just learn to make contact and move it.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# MuJoCo scene dedicated to push (robot + cube)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Workspace bounds for sampling the cube
CUBE_X_RANGE = (0.05, 0.20)
CUBE_Y_RANGE = (-0.12, 0.12)
GROUND_Z = 0.01

# Minimum cube distance from the robot base (m)
MIN_BASE_DIST = 0.15

# Success threshold: cube moved at least this far from its spawn (m)
SUCCESS_DIST = 0.05

# Maximum episode length
MAX_EPISODE_STEPS = 200


class PushEnv(gym.Env):
    """Gymnasium env: the end effector must reach the cube and push it.

    Observation (dim 12):
        - qpos              (3)  joint positions
        - ee_pos            (3)  end-effector Cartesian position
        - cube_pos          (3)  cube position
        - ee_to_cube        (3)  vector end-effector -> cube

    Action (dim 3):
        - target joint positions (sent to MuJoCo actuators)

    Reward:
        - -distance(ee, cube)               (approach the cube)
        - + bonus for cube displacement      (reward pushing)
        - smoothing penalty (action_rate)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.render_mode = render_mode

        # MuJoCo simulation
        self.sim = Sim3Dofs(
            render_mode=render_mode,
            scene_xml=SCENE_XML,
        )

        # Spaces
        n_act = self.sim.n_actuators  # 3

        # Actions: target joint positions in radians
        act_limit = 2.618
        self.action_space = spaces.Box(
            low=-act_limit,
            high=act_limit,
            shape=(n_act,),
            dtype=np.float32,
        )

        # Observations: qpos(3) + ee(3) + cube(3) + ee_to_cube(3) = 12
        obs_high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Internal state
        self._cube_init: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0

    # Helpers

    def _sample_ground_pos(self) -> np.ndarray:
        """Random ground position, at least MIN_BASE_DIST away from the base."""
        while True:
            pos = np.array([
                self.np_random.uniform(*CUBE_X_RANGE),
                self.np_random.uniform(*CUBE_Y_RANGE),
                GROUND_Z,
            ])
            if np.linalg.norm(pos) >= MIN_BASE_DIST:
                return pos

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        ee_to_cube = cube_pos - ee_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, ee_to_cube,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Reward: approach the cube + reward displacement from spawn."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        cube_displacement = float(np.linalg.norm(cube_pos - self._cube_init))

        # Approach the cube
        reward = -dist_ee_cube

        # Reward any cube movement from its initial position
        reward += 5.0 * cube_displacement

        # Success: cube moved far enough
        is_success = cube_displacement > SUCCESS_DIST
        if is_success:
            reward += 10.0

        # Smoothing penalty
        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= 0.01 * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation (neutral pose)
        self.sim.reset()

        # Sample cube on the ground
        self._cube_init = self._sample_ground_pos()
        self.sim.set_cube_pose(pos=self._cube_init.copy())

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        obs = self._get_obs()
        info = {"cube_init": self._cube_init.copy()}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        # Apply the action in simulation
        self.sim.step(action)
        self._step_count += 1

        # Observation
        obs = self._get_obs()

        # Reward
        reward, is_success = self._compute_reward(action)

        # Termination
        terminated = is_success
        truncated = self._step_count >= MAX_EPISODE_STEPS

        cube_pos = self.sim.get_cube_pos()
        info = {
            "is_success": is_success,
            "cube_displacement": float(np.linalg.norm(cube_pos - self._cube_init)),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
