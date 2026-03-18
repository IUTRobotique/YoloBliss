"""Gymnasium environment for the Push task with the 3-DOF robot.

The end effector must push a cube toward a target position on the ground.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_3dofs import Sim3Dofs

# MuJoCo scene dedicated to push (robot + cube + goal marker)
SCENE_XML = os.path.join(os.path.dirname(__file__), "scene_push.xml")

# Workspace bounds for sampling the goal and cube
GOAL_X_RANGE = (0.05, 0.20)
GOAL_Y_RANGE = (-0.12, 0.12)
GROUND_Z = 0.01

# Minimum goal/cube distance from the robot base (m)
MIN_GOAL_DIST = 0.15

# Minimum distance between cube and goal to avoid overlap (m)
MIN_CUBE_GOAL_DIST = 0.1

# Success threshold (m)
SUCCESS_THRESHOLD = 0.01  # 1 cm

# Maximum episode length
MAX_EPISODE_STEPS = 200


class PushEnv(gym.Env):
    """Gymnasium env: the end effector must push a cube toward a ground goal.

    Observation (dim 15):
        - qpos              (3)  joint positions
        - ee_pos            (3)  end-effector Cartesian position
        - cube_pos          (3)  cube position
        - ee_to_cube        (3)  vector end-effector -> cube
        - cube_to_goal      (3)  vector cube -> goal

    Action (dim 3):
        - target joint positions (sent to MuJoCo actuators)

    Reward:
        - -distance(cube, goal)        (dense, main objective)
        - -0.5 * distance(ee, cube)    (approach the cube)
        - + bonus if cube on goal
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

        # Observations: qpos(3) + ee(3) + cube(3) + ee_to_cube(3) + cube_to_goal(3) = 15
        obs_high = np.full(15, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

        # Internal state
        self._goal: np.ndarray = np.zeros(3)
        self._prev_action: np.ndarray = np.zeros(n_act)
        self._step_count: int = 0

    # Helpers

    def _sample_ground_pos(self) -> np.ndarray:
        """Random ground position, at least MIN_GOAL_DIST away from the base."""
        while True:
            pos = np.array([
                self.np_random.uniform(*GOAL_X_RANGE),
                self.np_random.uniform(*GOAL_Y_RANGE),
                GROUND_Z,
            ])
            if np.linalg.norm(pos) >= MIN_GOAL_DIST:
                return pos

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector including the cube."""
        qpos = self.sim.get_qpos()
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()
        ee_to_cube = cube_pos - ee_pos
        cube_to_goal = self._goal - cube_pos
        return np.concatenate([
            qpos, ee_pos, cube_pos, ee_to_cube, cube_to_goal,
        ]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, bool]:
        """Compute the pushing-oriented reward."""
        ee_pos = self.sim.get_end_effector_pos()
        cube_pos = self.sim.get_cube_pos()

        dist_cube_goal = float(np.linalg.norm(cube_pos - self._goal))

        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))

        reward = -dist_cube_goal - 0.5 * dist_ee_cube

        is_success = dist_cube_goal < SUCCESS_THRESHOLD
        if is_success:
            reward += 25.0

        action_rate = float(np.sum((action - self._prev_action) ** 2))
        reward -= 0.01 * action_rate

        return reward, is_success

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation (neutral pose)
        self.sim.reset()

        # Sample goal and cube on the ground, ensuring they do not overlap
        self._goal = self._sample_ground_pos()
        while True:
            cube_pos = self._sample_ground_pos()
            if np.linalg.norm(cube_pos - self._goal) >= MIN_CUBE_GOAL_DIST:
                break

        self.sim.set_cube_pose(pos=cube_pos)
        self.sim.set_goal_marker(self._goal)

        self._prev_action = np.zeros(self.sim.n_actuators)
        self._step_count = 0

        obs = self._get_obs()
        info = {"goal": self._goal.copy()}
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
            "dist_cube_goal": float(np.linalg.norm(cube_pos - self._goal)),
            "goal": self._goal.copy(),
        }

        self._prev_action = action.copy()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()
