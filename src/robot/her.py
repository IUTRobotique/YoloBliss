"""Surcouche HER (Hindsight Experience Replay) pour ReachingEnv.

HER (Andrychowicz et al., 2017) réétiquette les transitions échouées en succès
potentiels : après un épisode où le but g n'est pas atteint, certaines
transitions (s_t, a_t, s_{t+1}) sont relabellisées avec le but g' = achieved_goal
d'une transition ultérieure du même épisode (stratégie "future").
Ainsi, l'agent apprend que s_{t+1} était « un succès pour g' »,
même si g n'a pas été atteint.

Pertinence pour le reaching robotique :
  - Sans HER, la récompense dense (-distance) permet quand même d'apprendre.
  - Avec HER, la courbe « décolle » bien plus vite car chaque épisode fournit
    n_sampled_goal × épisode transitions relabellisées réussies supplémentaires.
  - HER est surtout crucial pour les récompenses éparses (0/1) où sans relabelling
    l'agent ne verrait presque jamais de signal positif.

GoalEnv : HerReplayBuffer exige que l'environnement expose des observations
dict avec les clés ``observation``, ``achieved_goal`` et ``desired_goal``,
et implémente ``compute_reward(achieved_goal, desired_goal, info)``.
``ReachingGoalEnv`` adapte ``ReachingEnv`` à ce contrat.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from reaching_env import ReachingEnv, SUCCESS_THRESHOLD, MAX_EPISODE_STEPS

TOTAL_TIMESTEPS: int = 300_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 5_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

#nombre de buts virtuels relabellisés par transition réelle
N_SAMPLED_GOAL: int = 4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "her_sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "her_sac")


class ReachingGoalEnv(gym.Env):
    """Adaptateur GoalEnv de ReachingEnv pour HerReplayBuffer.

    Transforme l'observation vectorielle de ReachingEnv (dim 9) en un
    dictionnaire GoalEnv avec la décomposition :

      ``observation``   (6) : état interne [qpos(3) | ee_pos(3)]
      ``achieved_goal`` (3) : position cartésienne courante de l'effecteur
      ``desired_goal``  (3) : position cible à atteindre

    Cette décomposition est indispensable pour que HerReplayBuffer puisse
    substituer ``desired_goal`` par l'``achieved_goal`` d'une transition future
    lors du relabelling, puis appeler ``compute_reward`` pour recalculer la
    récompense de la transition relabellisée.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None =None) -> None:
        super().__init__()
        self.render_mode: str | None = render_mode
        self._inner: ReachingEnv = ReachingEnv(render_mode=render_mode)

        obs_dim: int = 6   #qpos(3) + ee_pos(3) : pas le vecteur goal-ee pour éviter la fuite
        goal_dim: int = 3

        obs_high: np.ndarray = np.full(obs_dim, np.inf, dtype=np.float32)
        goal_high: np.ndarray = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space: spaces.Dict = spaces.Dict({
            "observation":   spaces.Box(-obs_high, obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(-goal_high, goal_high, dtype=np.float32),
            "desired_goal":  spaces.Box(-goal_high, goal_high, dtype=np.float32),
        })
        self.action_space: spaces.Box = self._inner.action_space

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Construit l'observation GoalEnv depuis l'état courant de la simulation.
        Returns:
            dict[str, np.ndarray]: dictionnaire avec les trois clés GoalEnv.
        """
        qpos: np.ndarray = self._inner.sim.get_qpos()
        ee_pos: np.ndarray = self._inner.sim.get_end_effector_pos()
        return {
            "observation":   np.concatenate([qpos, ee_pos]).astype(np.float32),
            "achieved_goal": ee_pos.astype(np.float32),
            "desired_goal":  self._inner._goal.astype(np.float32),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Récompense relabellisable : appelée par HerReplayBuffer lors du relabelling.

        Doit accepter des batchs : achieved_goal et desired_goal peuvent être
        de forme (batch_size, goal_dim) ou (goal_dim,).
        Parameters:
            achieved_goal (np.ndarray): position(s) effecteur atteinte(s)
            desired_goal (np.ndarray): but(s) cible(s) substitué(s) par HER
            info (dict): ignoré ici, présent pour respecter l'interface GoalEnv
        Returns:
            np.ndarray: récompenses de forme (batch_size,) ou scalaire float.
        """
        #axis=-1 supporte à la fois les batchs (B, 3) et les vecteurs (3,)
        distance: np.ndarray = np.linalg.norm(
            achieved_goal - desired_goal, axis=-1
        ).astype(np.float32)
        reward: np.ndarray = -distance
        reward += (distance < SUCCESS_THRESHOLD).astype(np.float32)
        return reward

    def reset(self, *, seed: int | None =None, options: dict | None =None):
        super().reset(seed=seed)
        self._inner.reset(seed=seed, options=options)
        obs: dict[str, np.ndarray] = self._build_obs()
        return obs, {"goal": self._inner._goal.copy()}

    def step(self, action: np.ndarray):
        #on délègue l'avance de simulation à l'env interne, on récupère son info
        _, _, terminated, truncated, inner_info = self._inner.step(action)

        obs: dict[str, np.ndarray] = self._build_obs()
        ee_pos: np.ndarray = self._inner.sim.get_end_effector_pos()
        reward: float = float(self.compute_reward(ee_pos, self._inner._goal, {}))

        info: dict[str, Any] = {
            "is_success": inner_info["is_success"],
            "distance": inner_info["distance"],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()


def make_her_sac(
    env: ReachingGoalEnv,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Construit un SAC avec HerReplayBuffer sur ReachingGoalEnv.

    La politique ``MultiInputPolicy`` traite automatiquement le dict d'observation
    en concaténant ``observation``, ``achieved_goal`` et ``desired_goal`` via
    un extracteur de features dédié.
    Parameters:
        env (ReachingGoalEnv): environnement GoalEnv (non vectorisé)
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: modèle SAC+HER configuré, prêt pour model.learn().
    """
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
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": N_SAMPLED_GOAL,
            #"future" : les buts relabellisés sont tirés parmi les transitions futures du même épisode
            "goal_selection_strategy": "future",
        },
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )


def make_env() -> ReachingGoalEnv:
    """Crée une instance fraîche de ReachingGoalEnv.
    Returns:
        ReachingGoalEnv: GoalEnv prête pour HER.
    """
    return ReachingGoalEnv()


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Entraîne un agent SAC+HER sur la tâche de reaching.

    Avec N_SAMPLED_GOAL=4, chaque transition génère 4 transitions relabellisées
    supplémentaires, soit un buffer effectif 5× plus dense en signal utile.
    L'effet est surtout visible dans les 50 premiers épisodes où la courbe
    « décolle » bien plus tôt que SAC sans HER.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: agent SAC+HER entraîné.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env: ReachingGoalEnv = make_env()
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: SAC = make_her_sac(env, log_dir=log_dir)

    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(model_dir, "her_sac_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
