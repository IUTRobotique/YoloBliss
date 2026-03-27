"""Entraînement TD3 (Twin Delayed Deep Deterministic Policy Gradient).

TD3 (Fujimoto et al., 2018) améliore DDPG par trois mécanismes correctifs :

  1. Twin critics (Double Q-learning) : deux réseaux Q indépendants,
     on prend le minimum lors du calcul des cibles de Bellman pour éviter
     la surestimation systématique des valeurs d'action.

  2. Delayed policy update : l'acteur et les réseaux cibles sont mis à jour
     moins fréquemment que les critiques (tous les ``policy_delay`` steps).
     Ceci laisse les critiques converger avant de mettre à jour la politique.

  3. Target policy smoothing : du bruit gaussien clipé est ajouté aux actions
     cibles lors du calcul des cibles de Bellman :
         a'(s') = clip(π_target(s') + clip(ε, -c, c), a_low, a_high)
     Ceci régularise les critiques et réduit leur variance.

La politique est déterministe ; l'exploration se fait en ajoutant un bruit
gaussien N(0, σ²) à l'action lors de la collecte (pas lors de l'évaluation).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv

from robot_env.push_in_hole_env import PushInHoleEnv as ReachingEnv

TOTAL_TIMESTEPS: int = 500_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 10_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

#l'acteur est mis à jour 1 fois pour 2 mises à jour des critiques
POLICY_DELAY: int = 2

#écart-type du bruit gaussien ajouté à l'action pendant la collecte
ACTION_NOISE_STD: float = 0.1

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "td3")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "td3")


class _RenderCallback(BaseCallback):
    """Appelle training_env.render() à chaque pas de collecte.

    SB3 ne déclenche pas le rendu automatiquement pendant learn() ;
    ce callback est le seul moyen d'afficher la simulation en temps réel.
    """

    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


def make_env(render_mode: str | None =None) -> ReachingEnv:
    """Crée une instance de ReachingEnv (factory pour make_vec_env).
    Parameters:
        render_mode (str | None): ``"human"`` pour afficher MuJoCo en temps réel,
            ``None`` pour l'entraînement headless (plus rapide).
    Returns:
        ReachingEnv: environnement gymnasium initialisé.
    """
    return ReachingEnv(render_mode=render_mode)


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
    render: bool =False,
) -> TD3:
    """Entraîne un agent TD3 sur la tâche de reaching.

    TD3 est plus stable que DDPG mais sa courbe d'apprentissage est plus
    erratique que SAC : la qualité de la convergence dépend fortement du
    bruit d'exploration. Si ACTION_NOISE_STD est trop faible, l'agent ne
    s'échappe pas des minima locaux ; trop élevé, le signal d'apprentissage
    se dégrade. 0.1 (10% de l'amplitude articulaire) est un bon point de départ.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle (EvalCallback)
        log_dir (str): répertoire TensorBoard pour les courbes d'apprentissage
    Returns:
        TD3: agent entraîné (modèle final, pas nécessairement le meilleur).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode: str | None = "human" if render else None
    env: ReachingEnv = make_env(render_mode=render_mode)
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    #bruit gaussien indépendant par dimension d'action pour l'exploration
    n_actions: int = env.action_space.shape[0]
    action_noise: NormalActionNoise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=ACTION_NOISE_STD * np.ones(n_actions),
    )

    model: TD3 = TD3(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=GRADIENT_STEPS,
        policy_delay=POLICY_DELAY,
        action_noise=action_noise,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )

    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(os.path.join(model_dir, "td3_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entraînement TD3 sur le robot 3-DDL (MuJoCo)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Nombre de pas d'environnement (défaut : {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Affiche la simulation MuJoCo en temps réel pendant l'entraînement"
    )
    args: argparse.Namespace = parser.parse_args()
    train(total_timesteps=args.timesteps, render=args.render)
