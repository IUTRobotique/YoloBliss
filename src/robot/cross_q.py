"""Surcouche CrossQ : Batch Normalization et haut ratio UTD pour SAC.

CrossQ (Bhatt et al., 2024) est une amélioration de SAC off-policy qui réduit
le temps d'entraînement d'un facteur ≈3 par deux innovations conjointes :

  1. Batch Normalization dans les réseaux Q (critiques) :
     La BN standard introduit un biais en RL off-policy car les mini-batchs
     contiennent des paires (s_t, s_{t+1}) corrélées. Si la BN calcule ses
     statistiques séparément sur s_t et s_{t+1}, les normalisations sont
     incohérentes, ce qui fausse les cibles de Bellman.
     Solution CrossQ : concaténer s_t et s_{t+1} dans un seul forward pass
     pour partager les statistiques de BN (cross-minibatch normalization).
     Voir ``BatchNormCritic`` pour l'architecture cible.

  2. Haut ratio UTD (Update-to-Data) :
     Grâce à la BN qui réduit l'overfitting, le ratio UTD peut être porté
     à 20 (vs 1 pour SAC standard) sans divergence. Résultat : 20× moins
     de pas d'environnement sont nécessaires pour atteindre la même performance.

Note d'implémentation : la cross-minibatch normalization complète requiert de
modifier le backward pass de SB3 (fusion du batch courant et du batch cible
dans un seul graphe de calcul). Cette implémentation reproduit le gain principal
de CrossQ via un UTD élevé. ``BatchNormCritic`` illustre l'architecture cible
et peut servir de base pour une intégration future plus profonde dans SB3.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from robot_env.reaching_env import ReachingEnv

#20× plus de mises à jour gradient par pas d'env qu'un SAC standard
UTD_RATIO: int = 20

#beaucoup moins de steps nécessaires : le haut UTD compense largement
TOTAL_TIMESTEPS: int = 100_000

BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 5_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005

#LR réduit pour stabiliser l'entraînement sous haute pression de gradient
LEARNING_RATE: float = 1e-4

ENT_COEF: str = "auto"

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "crossq")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "crossq")


class BatchNormCritic(nn.Module):
    """Réseau Q avec Batch Normalization après chaque couche cachée.

    Architecture cible de CrossQ pour les réseaux critiques. La BN est placée
    avant l'activation (pré-activation BN) comme préconisé dans l'article.
    En inférence (eval mode), la BN utilise les statistiques mobiles accumulées
    en entraînement, ce qui est cohérent avec l'évaluation déterministe.

    Note : pour reproduire fidèlement CrossQ, le forward doit être appelé
    avec la concaténation [batch_s_t || batch_s_{t+1}] pour partager les
    statistiques de BN entre observations courantes et cibles de Bellman.
    Parameters:
        input_dim (int): dimension de l'entrée = dim(observation) + dim(action)
        output_dim (int): dimension de sortie, typiquement 1 pour Q(s, a)
        net_arch (list[int]): tailles des couches cachées
    """

    def __init__(self,
        input_dim: int,
        output_dim: int,
        net_arch: list[int] =[256, 256],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim: int = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant avec BN partagée sur le batch complet.
        Parameters:
            x (torch.Tensor): entrée de forme (batch, input_dim).
                Pour CrossQ, x = concat([obs_t, obs_{t+1}], dim=0).
        Returns:
            torch.Tensor: valeurs Q de forme (batch, output_dim).
        """
        return self.net(x)


def make_crossq_sac(
    env: ReachingEnv,
    utd_ratio: int =UTD_RATIO,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Construit un SAC CrossQ-inspiré avec haut UTD.

    Le ratio UTD élevé est le levier principal de CrossQ : avec gradient_steps=20,
    chaque transition collectée génère 20 mises à jour de gradient au lieu de 1.
    Parameters:
        env (ReachingEnv): environnement d'entraînement (non vectorisé)
        utd_ratio (int): nombre de gradient steps par pas d'environnement
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: modèle configuré, prêt pour model.learn().
    """
    policy_kwargs: dict[str, Any] = {
        "net_arch": [256, 256],
        #ReLU préféré à Tanh avec BN : Tanh sature et annule le gradient après la BN
        "activation_fn": torch.nn.ReLU,
    }
    return SAC(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=utd_ratio,
        ent_coef=ENT_COEF,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )


def make_env() -> ReachingEnv:
    """Crée une instance fraîche de ReachingEnv (factory pour make_vec_env).
    Returns:
        ReachingEnv: environnement gymnasium initialisé.
    """
    return ReachingEnv()


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Entraîne un SAC CrossQ-inspiré sur la tâche de reaching.

    Avec UTD=20, 100k steps suffisent typiquement là où SAC standard
    en demande 300-500k. Le LR réduit (1e-4) compense la pression de gradient
    accrue pour éviter les instabilités.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: agent CrossQ entraîné.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env: ReachingEnv = make_env()
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: SAC = make_crossq_sac(env, log_dir=log_dir)

    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(model_dir, "crossq_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()

"""Surcouche CrossQ : Batch Normalization et haut ratio UTD pour SAC.

CrossQ (Bhatt et al., 2024) est une amélioration de SAC off-policy qui réduit
le temps d'entraînement d'un facteur ≈3 par deux innovations conjointes :

  1. Batch Normalization dans les réseaux Q (critiques) :
     La BN standard introduit un biais en RL off-policy car les mini-batchs
     contiennent des paires (s_t, s_{t+1}) corrélées. Si la BN calcule ses
     statistiques séparément sur s_t et s_{t+1}, les normalisations sont
     incohérentes, ce qui fausse les cibles de Bellman.
     Solution CrossQ : concaténer s_t et s_{t+1} dans un seul forward pass
     pour partager les statistiques de BN (cross-minibatch normalization).
     Voir ``BatchNormCritic`` pour l'architecture cible.

  2. Haut ratio UTD (Update-to-Data) :
     Grâce à la BN qui réduit l'overfitting, le ratio UTD peut être porté
     à 20 (vs 1 pour SAC standard) sans divergence. Résultat : 20× moins
     de pas d'environnement sont nécessaires pour atteindre la même performance.

Note d'implémentation : la cross-minibatch normalization complète requiert de
modifier le backward pass de SB3 (fusion du batch courant et du batch cible
dans un seul graphe de calcul). Cette implémentation reproduit le gain principal
de CrossQ via un UTD élevé. ``BatchNormCritic`` illustre l'architecture cible
et peut servir de base pour une intégration future plus profonde dans SB3.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from reaching_env import ReachingEnv

#20× plus de mises à jour gradient par pas d'env qu'un SAC standard
UTD_RATIO: int = 20

#beaucoup moins de steps nécessaires : le haut UTD compense largement
TOTAL_TIMESTEPS: int = 100_000

BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 5_000
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005

#LR réduit pour stabiliser l'entraînement sous haute pression de gradient
LEARNING_RATE: float = 1e-4

ENT_COEF: str = "auto"

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "crossq")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "crossq")


class BatchNormCritic(nn.Module):
    """Réseau Q avec Batch Normalization après chaque couche cachée.

    Architecture cible de CrossQ pour les réseaux critiques. La BN est placée
    avant l'activation (pré-activation BN) comme préconisé dans l'article.
    En inférence (eval mode), la BN utilise les statistiques mobiles accumulées
    en entraînement, ce qui est cohérent avec l'évaluation déterministe.

    Note : pour reproduire fidèlement CrossQ, le forward doit être appelé
    avec la concaténation [batch_s_t || batch_s_{t+1}] pour partager les
    statistiques de BN entre observations courantes et cibles de Bellman.
    Parameters:
        input_dim (int): dimension de l'entrée = dim(observation) + dim(action)
        output_dim (int): dimension de sortie, typiquement 1 pour Q(s, a)
        net_arch (list[int]): tailles des couches cachées
    """

    def __init__(self,
        input_dim: int,
        output_dim: int,
        net_arch: list[int] =[256, 256],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim: int = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passe avant avec BN partagée sur le batch complet.
        Parameters:
            x (torch.Tensor): entrée de forme (batch, input_dim).
                Pour CrossQ, x = concat([obs_t, obs_{t+1}], dim=0).
        Returns:
            torch.Tensor: valeurs Q de forme (batch, output_dim).
        """
        return self.net(x)


def make_crossq_sac(
    env: ReachingEnv,
    utd_ratio: int =UTD_RATIO,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Construit un SAC CrossQ-inspiré avec haut UTD.

    Le ratio UTD élevé est le levier principal de CrossQ : avec gradient_steps=20,
    chaque transition collectée génère 20 mises à jour de gradient au lieu de 1.
    Parameters:
        env (ReachingEnv): environnement d'entraînement (non vectorisé)
        utd_ratio (int): nombre de gradient steps par pas d'environnement
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: modèle configuré, prêt pour model.learn().
    """
    policy_kwargs: dict[str, Any] = {
        "net_arch": [256, 256],
        #ReLU préféré à Tanh avec BN : Tanh sature et annule le gradient après la BN
        "activation_fn": torch.nn.ReLU,
    }
    return SAC(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_rate=LEARNING_RATE,
        gradient_steps=utd_ratio,
        ent_coef=ENT_COEF,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )


def make_env() -> ReachingEnv:
    """Crée une instance fraîche de ReachingEnv (factory pour make_vec_env).
    Returns:
        ReachingEnv: environnement gymnasium initialisé.
    """
    return ReachingEnv()


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Entraîne un SAC CrossQ-inspiré sur la tâche de reaching.

    Avec UTD=20, 100k steps suffisent typiquement là où SAC standard
    en demande 300-500k. Le LR réduit (1e-4) compense la pression de gradient
    accrue pour éviter les instabilités.
    Parameters:
        total_timesteps (int): nombre total de pas d'environnement à simuler
        model_dir (str): répertoire de sauvegarde du meilleur modèle
        log_dir (str): répertoire TensorBoard
    Returns:
        SAC: agent CrossQ entraîné.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env: ReachingEnv = make_env()
    eval_env: VecEnv = make_vec_env(make_env, n_envs=1)

    model: SAC = make_crossq_sac(env, log_dir=log_dir)

    eval_callback: EvalCallback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(model_dir, "crossq_final"))

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
