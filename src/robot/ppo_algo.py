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
from typing import Literal

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from robot_env.push_in_hole_env import PushInHoleEnv as PushEnv


#collecte on-policy sur plusieurs envs en parallèle pour diversifier les données
N_ENVS: int = 16
TOTAL_TIMESTEPS: int = 300_000

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

#récompense moyenne par épisode au-delà de laquelle l'entraînement s'arrête
#push : +30 succès + récompenses d'approche ≈ 35 max ; 35 = succès systématique
REWARD_THRESHOLD: float = 30.0  # FIX: abaisser le seuil d'arret PPO selon la nouvelle cible


def _setup_torch_for_cuda(device: str) -> None:
    """Active des optimisations CUDA sûres pour accélérer les calculs PPO."""
    if device.startswith("cuda") and torch.cuda.is_available():
        # Sur RTX récentes, TF32 améliore nettement le débit pour peu de perte numérique.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _resolve_device(device: Literal["auto", "cpu", "cuda"]) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA demandé mais non disponible. Bascule sur CPU.")
        return "cpu"
    return device


class _RenderCallback(BaseCallback):
    """Appelle training_env.render() à chaque pas de collecte.

    SB3 ne déclenche pas le rendu automatiquement pendant learn() ;
    ce callback est le seul moyen d'afficher la simulation en temps réel.
    """

    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


def make_env(render_mode: str | None =None) -> PushEnv:
    """Crée une instance de PushEnv (factory pour make_vec_env).
    Parameters:
        render_mode (str | None): ``"human"`` pour afficher MuJoCo en temps réel,
            ``None`` pour l'entraînement headless (plus rapide).
    Returns:
        PushEnv: environnement gymnasium initialisé.
    """
    return PushEnv(render_mode=render_mode)


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
    render: bool =False,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    n_envs: int = N_ENVS,
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

    resolved_device: str = _resolve_device(device)
    _setup_torch_for_cuda(resolved_device)

    render_mode: str | None = "human" if render else None
    #en mode render : 1 seul env pour éviter N_ENVS fenêtres MuJoCo simultanées
    n_envs_train: int = 1 if render else max(1, n_envs)
    vec_env_cls = None if render or n_envs_train == 1 else SubprocVecEnv
    vec_env: VecEnv = make_vec_env(
        make_env,
        n_envs=n_envs_train,
        env_kwargs={"render_mode": render_mode},
        vec_env_cls=vec_env_cls,
        monitor_kwargs={"info_keywords": ("is_success", "cube_displacement")}
    )
    eval_env: VecEnv = make_vec_env(
        make_env,
        n_envs=1,
        env_kwargs={"render_mode": None},
        monitor_kwargs={"info_keywords": ("is_success", "cube_displacement")}
    )
    model: PPO = PPO(
        "MlpPolicy",
        vec_env,
        device=resolved_device,
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
    n_params: int = sum(p.numel() for p in model.policy.parameters())
    print(f"Device: {resolved_device} | n_envs_train: {n_envs_train} | vec_env: {type(vec_env).__name__} | Paramètres : {n_params:,}")

    stop_callback: StopTrainingOnRewardThreshold = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1,
    )
    #sauvegarde automatique du meilleur modèle selon la récompense moyenne d'évaluation
    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks: list[BaseCallback] = [eval_callback]
    if render:
        callbacks.append(_RenderCallback())

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
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
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Périphérique de calcul (défaut : auto)"
    )
    parser.add_argument(
        "--envs", type=int, default=N_ENVS,
        help=f"Nombre d'environnements parallèles hors rendu (défaut : {N_ENVS})"
    )
    args: argparse.Namespace = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        render=args.render,
        device=args.device,
        n_envs=args.envs,
    )
