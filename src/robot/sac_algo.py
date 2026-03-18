"""Entraînement SAC (Soft Actor-Critic) sur ReachingEnv.

SAC est un algorithme off-policy qui maximise conjointement la récompense
et l'entropie de la politique (Maximum Entropy RL) :

    π* = argmax_π E[Σ_t r_t + α H(π(·|s_t))]

La politique est stochastique via le reparametrization trick Tanh-Gaussian :

    a = tanh(μ_θ(s) + σ_θ(s) ⊙ ε),  ε ~ N(0, I)

Ce trick permet de rétropropager à travers l'échantillonnage. Deux réseaux Q
(twin critics) réduisent la surestimation : on utilise le minimum des deux
lors du calcul des cibles de Bellman (Soft Bellman Residual).
Le coefficient d'entropie α est ajusté automatiquement pour maintenir
une entropie cible H_target ≈ -dim(A).
"""

from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from reaching_env import ReachingEnv

TOTAL_TIMESTEPS: int = 500_000
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 10_000  #phase d'exploration aléatoire avant le premier gradient step
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005             #taux de mise à jour douce (Polyak) des réseaux cibles
LEARNING_RATE: float = 3e-4

#UTD = 1 : 1 gradient step par pas d'env (voir cross_q.py pour un UTD élevé)
GRADIENT_STEPS: int = 1

ENT_COEF: str = "auto"         #ajustement automatique de α
TARGET_ENTROPY: str = "auto"   #H_target = -dim(A) = -3 pour ce robot

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "sac")


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
) -> SAC:
    """Entraîne un agent SAC sur la tâche de reaching.

    Contrairement à PPO, SAC réutilise toutes les transitions stockées dans
    le replay buffer. L'ascension de la courbe d'apprentissage est très rapide
    au début (environ 50k steps), puis oscille légèrement à cause de
    l'ajustement automatique de α. Résultat typique : plateau en ~200-300k steps
    contre >1M pour PPO sur la même tâche.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle (EvalCallback)
        log_dir (str): répertoire TensorBoard pour les courbes d'apprentissage
    Returns:
        SAC: agent entraîné (modèle final, pas nécessairement le meilleur).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode: str | None = "human" if render else None
    env: ReachingEnv = make_env(render_mode=render_mode)
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: SAC = SAC(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=GRADIENT_STEPS,
        ent_coef=ENT_COEF,
        target_entropy=TARGET_ENTROPY,
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

    #SB3 n'appelle jamais env.render() dans la boucle learn() : callback nécessaire
    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(os.path.join(model_dir, "sac_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entraînement SAC sur le robot 3-DDL (MuJoCo)"
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
