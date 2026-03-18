"""Entraînement PPO (Proximal Policy Optimization) sur ReachingEnv.

PPO est un algorithme on-policy : chaque lot d'expériences collecté est
consommé puis jeté. La stabilité est assurée par le clipping de l'objectif
surrogate, qui empêche les mises à jour de politique trop agressives :

    L_CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

Où r_t(θ) = π_θ(a|s) / π_θ_old(a|s) est le ratio des probabilités.
L'avantage A_t est estimé par GAE (Generalized Advantage Estimation).
"""

from __future__ import annotations

import argparse
import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from reaching_env import ReachingEnv

#collecte on-policy sur plusieurs envs en parallèle pour diversifier les données
N_ENVS: int = 4
TOTAL_TIMESTEPS: int = 1_000_000

#rollout de N_STEPS par env avant chaque update : N_STEPS * N_ENVS transitions consommées
N_STEPS: int = 2048
BATCH_SIZE: int = 64

#N_EPOCHS passes sur le même rollout : compromis efficacité / déviation off-policy
N_EPOCHS: int = 10

GAMMA: float = 0.99
GAE_LAMBDA: float = 0.95    #λ de GAE : trade-off biais/variance de l'avantage
CLIP_RANGE: float = 0.2     #ε du Clipped Surrogate Objective
ENT_COEF: float = 0.0       #pas de bonus d'entropie : la stabilité prime sur l'exploration
VF_COEF: float = 0.5        #poids de la perte de la fonction de valeur
MAX_GRAD_NORM: float = 0.5  #gradient clipping pour la stabilité numérique
LEARNING_RATE: float = 3e-4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    "activation_fn": torch.nn.Tanh,
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "ppo")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "ppo")


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
) -> PPO:
    """Entraîne un agent PPO sur la tâche de reaching.

    N_ENVS environnements parallèles collectent des trajectoires on-policy.
    Chaque rollout de N_STEPS × N_ENVS transitions est ensuite consommé
    en N_EPOCHS passes sur des mini-batchs de taille BATCH_SIZE.
    En pratique PPO converge très lentement sur des tâches continues :
    plusieurs millions de steps sont souvent nécessaires pour atteindre
    le plateau final, contre quelques centaines de milliers pour SAC/TD3.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle (EvalCallback)
        log_dir (str): répertoire TensorBoard pour les courbes d'apprentissage
    Returns:
        PPO: agent entraîné (modèle final, pas nécessairement le meilleur).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    render_mode: str | None = "human" if render else None
    #en mode render : 1 seul env pour éviter N_ENVS fenêtres MuJoCo simultanées
    n_envs_train: int = 1 if render else N_ENVS
    vec_env: VecEnv = make_vec_env(
        make_env,
        n_envs=n_envs_train,
        env_kwargs={"render_mode": render_mode},
    )
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: PPO = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        learning_rate=LEARNING_RATE,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=log_dir,
        verbose=1,
    )

    #sauvegarde automatique du meilleur modèle selon la récompense moyenne d'évaluation
    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    model.save(os.path.join(model_dir, "ppo_final"))

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entraînement PPO sur le robot 3-DDL (MuJoCo)"
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
