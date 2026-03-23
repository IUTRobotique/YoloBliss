"""Surcouche HER (Hindsight Experience Replay) pour SortingEnv.

HER (Andrychowicz et al., 2017) reetiquette les transitions echouees en succes
potentiels : apres un episode ou le but g n'est pas atteint, certaines
transitions (s_t, a_t, s_{t+1}) sont relabellisees avec le but g' = achieved_goal
d'une transition ulterieure du meme episode (strategie "future").

GoalEnv : HerReplayBuffer exige que l'environnement expose des observations
dict avec les cles ``observation``, ``achieved_goal`` et ``desired_goal``,
et implemente ``compute_reward(achieved_goal, desired_goal, info)``.
``SortingGoalEnv`` adapte ``SortingEnv`` a ce contrat.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from robot_env.sorting_env import SortingEnv, SUCCESS_THRESHOLD

TOTAL_TIMESTEPS: int = 1_000_000_000_000_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

SUCCESS_RATE_TARGET: float = 0.90
MIN_EVAL_EPISODES_FOR_SUCCESS: int = 10
EVAL_FREQ_FOR_SUCCESS_CHECK: int = 5_000

N_SAMPLED_GOAL: int = 4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.ReLU,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "her_sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "her_sac")


class _RenderCallback(BaseCallback):
    """Appelle training_env.render() a chaque pas de collecte."""

    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


class _SuccessStoppingCallback(BaseCallback):
    """Arrete l'entrainement quand la tache est maitrisee (success_rate >= seuil)."""

    def __init__(self, success_rate_target: float = 0.90, verbose: int = 0):
        super().__init__()
        self.success_rate_target = success_rate_target
        self.verbose = verbose
        self.best_success_rate = 0.0
        self.last_check_timestep = 0

    def _on_step(self) -> bool:
        current_timesteps = self.model.num_timesteps

        if current_timesteps - self.last_check_timestep < EVAL_FREQ_FOR_SUCCESS_CHECK:
            return True

        self.last_check_timestep = current_timesteps

        if current_timesteps > LEARNING_STARTS and len(self.model.ep_info_buffer) > 0:
            recent_episodes = list(self.model.ep_info_buffer)

            if len(recent_episodes) >= MIN_EVAL_EPISODES_FOR_SUCCESS:
                sample_size = min(MIN_EVAL_EPISODES_FOR_SUCCESS * 2, len(recent_episodes))
                recent_episodes = recent_episodes[-sample_size:]

                successes = 0
                for ep_info in recent_episodes:
                    if "is_success" in ep_info and ep_info["is_success"]:
                        successes += 1

                success_rate = successes / len(recent_episodes)

                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    if self.verbose > 0:
                        print(f"\nTimestep {current_timesteps:,} | Success: {success_rate:.1%} | Best: {self.best_success_rate:.1%}")

                if success_rate >= self.success_rate_target:
                    if self.verbose > 0:
                        print(f"\nOBJECTIF ATTEINT! Success rate: {success_rate:.1%}")
                        print(f"   Entrainement arrete apres {current_timesteps:,} timesteps")
                    return False

        return True


class SortingGoalEnv(gym.Env):
    """Adaptateur GoalEnv de SortingEnv pour HerReplayBuffer.

    Transforme l'observation de SortingEnv en dictionnaire GoalEnv :

    ``observation``   (6)  : etat robot [qpos(3) | ee_pos(3)]
    ``achieved_goal`` (6)  : positions xy des 2 objets [cube_xy(3) | cylinder_xy(3)]
    ``desired_goal``  (6)  : positions des 2 cibles [goal_cube(3) | goal_cylinder(3)]
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode: str | None = render_mode
        self._inner: SortingEnv = SortingEnv(render_mode=render_mode)

        obs_dim: int = 6   # qpos(3) + ee_pos(3)
        goal_dim: int = 6  # cube_pos(3) + cylinder_pos(3)

        obs_high: np.ndarray = np.full(obs_dim, np.inf, dtype=np.float32)
        goal_high: np.ndarray = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space: spaces.Dict = spaces.Dict({
            "observation":   spaces.Box(-obs_high, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(-goal_high, goal_high, dtype=np.float32),
        })
        self.action_space: spaces.Box = self._inner.action_space

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Construit l'observation GoalEnv depuis l'etat courant de la simulation."""
        qpos: np.ndarray = self._inner.sim.get_qpos()
        ee_pos: np.ndarray = self._inner.sim.get_end_effector_pos()
        cube_pos: np.ndarray = self._inner.sim.get_cube_pos()
        cylinder_pos: np.ndarray = self._inner.sim.get_cylinder_pos()
        return {
            "observation":   np.concatenate([qpos, ee_pos]).astype(np.float32),
            "achieved_goal": np.concatenate([cube_pos, cylinder_pos]).astype(np.float32),
            "desired_goal":  np.concatenate([
                self._inner._goal_cube, self._inner._goal_cylinder,
            ]).astype(np.float32),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Recompense relabellisable pour HER.

        achieved_goal (6) : [cube_pos(3), cylinder_pos(3)]
        desired_goal  (6) : [goal_cube(3), goal_cylinder(3)]

        Recompense = -dist_xy(cube, goal_cube) - dist_xy(cylinder, goal_cylinder)
                     + bonus si chaque objet est trie + bonus si les deux le sont.
        """
        # Distances xy objet -> cible
        dist_cube: np.ndarray = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)
        dist_cyl: np.ndarray = np.linalg.norm(
            achieved_goal[..., 3:5] - desired_goal[..., 3:5], axis=-1
        ).astype(np.float32)

        reward = -dist_cube - dist_cyl

        # Bonus par objet trie
        cube_sorted = (dist_cube < SUCCESS_THRESHOLD).astype(np.float32)
        cyl_sorted = (dist_cyl < SUCCESS_THRESHOLD).astype(np.float32)
        reward += 20.0 * cube_sorted
        reward += 20.0 * cyl_sorted
        reward += 50.0 * (cube_sorted * cyl_sorted)

        return reward

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._inner.reset(seed=seed, options=options)
        obs: dict[str, np.ndarray] = self._build_obs()
        return obs, {
            "goal_cube": self._inner._goal_cube.copy(),
            "goal_cylinder": self._inner._goal_cylinder.copy(),
        }

    def step(self, action: np.ndarray):
        _, _, terminated, truncated, inner_info = self._inner.step(action)

        obs: dict[str, np.ndarray] = self._build_obs()
        ee_pos: np.ndarray = self._inner.sim.get_end_effector_pos()
        cube_pos: np.ndarray = self._inner.sim.get_cube_pos()
        cylinder_pos: np.ndarray = self._inner.sim.get_cylinder_pos()

        # Recompense relabellisable depuis compute_reward()
        achieved = np.concatenate([cube_pos, cylinder_pos]).astype(np.float32)
        desired = np.concatenate([
            self._inner._goal_cube, self._inner._goal_cylinder,
        ]).astype(np.float32)
        goal_reward: float = float(self.compute_reward(achieved, desired, {}))

        # Terme d'approche NON relabellisable (ee -> objet le plus eloigne)
        dist_cube_goal = float(np.linalg.norm(cube_pos[:2] - self._inner._goal_cube[:2]))
        dist_cyl_goal = float(np.linalg.norm(cylinder_pos[:2] - self._inner._goal_cylinder[:2]))
        target_obj = cube_pos if dist_cube_goal >= dist_cyl_goal else cylinder_pos
        dist_ee_target = float(np.linalg.norm(ee_pos - target_obj))
        approach_reward = -2.0 * dist_ee_target

        reward = goal_reward + approach_reward

        info: dict[str, Any] = {
            "is_success": inner_info["is_success"],
            "cube_sorted": inner_info["cube_sorted"],
            "cylinder_sorted": inner_info["cylinder_sorted"],
            "dist_cube_goal": inner_info["dist_cube_goal"],
            "dist_cylinder_goal": inner_info["dist_cylinder_goal"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()


def make_her_sac(
    env: SortingGoalEnv,
    log_dir: str = LOG_DIR,
) -> SAC:
    """Construit un SAC avec HerReplayBuffer sur SortingGoalEnv."""
    return SAC(
        "MultiInputPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=GRADIENT_STEPS,
        ent_coef="auto",
        target_entropy="auto",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": N_SAMPLED_GOAL,
            "goal_selection_strategy": "future",
        },
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )


def make_env(render_mode: str | None = None) -> SortingGoalEnv:
    """Cree une instance fraiche de SortingGoalEnv."""
    return SortingGoalEnv(render_mode=render_mode)


def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    model_dir: str = MODEL_DIR,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> SAC:
    """Entraine un agent SAC+HER sur la tache de sorting."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*70)
    print("ENTRAINEMENT SAC+HER - Sorting")
    print("="*70)
    print(f"Limite timesteps: {total_timesteps:,}")
    print(f"Objectif succes: {SUCCESS_RATE_TARGET:.0%} de reussite")
    print(f"Evaluation tous les: {EVAL_FREQ_FOR_SUCCESS_CHECK:,} pas")
    print("="*70 + "\n")

    render_mode: str | None = "human" if render else None
    env: SortingGoalEnv = make_env(render_mode=render_mode)
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: SAC = make_her_sac(env, log_dir=log_dir)

    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ_FOR_SUCCESS_CHECK,
        n_eval_episodes=MIN_EVAL_EPISODES_FOR_SUCCESS,
        deterministic=True,
    )

    success_callback: _SuccessStoppingCallback = _SuccessStoppingCallback(
        success_rate_target=SUCCESS_RATE_TARGET,
        verbose=1,
    )

    callbacks: list[BaseCallback] = [eval_callback, success_callback]
    if render:
        callbacks.append(_RenderCallback())

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\nEntrainement interrompu par l'utilisateur")

    model.save(os.path.join(model_dir, "her_sac_final"))

    print("\n" + "="*70)
    print("ENTRAINEMENT TERMINE")
    print(f"   Model sauvegarde: {model_dir}/her_sac_final.zip")
    print(f"   Logs TensorBoard: {log_dir}")
    print("="*70 + "\n")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entrainement SAC+HER sur Sorting (MuJoCo)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Nombre de pas d'environnement (defaut : {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Affiche la simulation MuJoCo en temps reel pendant l'entrainement"
    )
    args: argparse.Namespace = parser.parse_args()
    train(total_timesteps=args.timesteps, render=args.render)
