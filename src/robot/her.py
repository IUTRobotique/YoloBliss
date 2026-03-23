"""Surcouche HER (Hindsight Experience Replay) pour PushInHoleEnv.

HER (Andrychowicz et al., 2017) réétiquette les transitions échouées en succès
potentiels : après un épisode où le but g n'est pas atteint, certaines
transitions (s_t, a_t, s_{t+1}) sont relabellisées avec le but g' = achieved_goal
d'une transition ultérieure du même épisode (stratégie "future").
Ainsi, l'agent apprend que s_{t+1} était « un succès pour g' »,
même si g n'a pas été atteint.

Pertinence pour le pushing robotique :
- Sans HER, la récompense dense (-distance_xy) permet quand même d'apprendre.
- Avec HER, la courbe « décolle » bien plus vite car chaque épisode fournit
  n_sampled_goal × épisode transitions relabellisées réussies supplémentaires.
- HER est surtout crucial pour les récompenses éparses (0/1) où sans relabelling
  l'agent ne verrait presque jamais de signal positif.

GoalEnv : HerReplayBuffer exige que l'environnement expose des observations
dict avec les clés ``observation``, ``achieved_goal`` et ``desired_goal``,
et implémente ``compute_reward(achieved_goal, desired_goal, info)``.
``PushInHoleGoalEnv`` adapte ``PushInHoleEnv`` à ce contrat.
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

from robot_env.push_in_hole_env import PushInHoleEnv, SUCCESS_Z_THRESHOLD, MAX_EPISODE_STEPS

TOTAL_TIMESTEPS: int = 1_000_000_000_000_000  # Limite max (arrêt précoce si succès atteint)
BUFFER_SIZE: int = 1_000_000
LEARNING_STARTS: int = 1_000  # Réduit pour débloquer l'apprentissage plus tôt
BATCH_SIZE: int = 256
GAMMA: float = 0.99
TAU: float = 0.005
LEARNING_RATE: float = 3e-4
GRADIENT_STEPS: int = 1

# Paramètres d'arrêt anticipé (early stopping) quand le succès est atteint
SUCCESS_RATE_TARGET: float = 0.90  # 90% de succès = tâche maîtrisée
MIN_EVAL_EPISODES_FOR_SUCCESS: int = 10  # Minimum 10 épisodes avant de vérifier le succès
EVAL_FREQ_FOR_SUCCESS_CHECK: int = 5_000  # Vérifie tous les 5k pas

#nombre de buts virtuels relabellisés par transition réelle
N_SAMPLED_GOAL: int = 4

POLICY_KWARGS: dict[str, object] = {
    "net_arch": [256, 256],
    "activation_fn": torch.nn.ReLU,  # ReLU pour meilleure exploration
}

MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "models", "her_sac")
LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs", "her_sac")


class _RenderCallback(BaseCallback):
    """Appelle training_env.render() à chaque pas de collecte.

    SB3 ne déclenche pas le rendu automatiquement pendant learn() ;
    ce callback est le seul moyen d'afficher la simulation en temps réel.
    """

    def _on_step(self) -> bool:
        self.training_env.render("human")
        return True


class _SuccessStoppingCallback(BaseCallback):
    """Arrête l'entraînement quand la tâche est maîtrisée (success_rate ≥ seuil).
    
    Monitore les résultats d'évaluation et arrête l'entraînement dès que le taux
    de succès atteint le seuil cible (ex: 90%). Cela évite un surapprentissage
    inutile et économise du temps d'entraînement.
    """

    def __init__(self, success_rate_target: float = 0.90, verbose: int = 0):
        super().__init__()
        self.success_rate_target = success_rate_target
        self.verbose = verbose
        self.best_success_rate = 0.0
        self.last_check_timestep = 0

    def _on_step(self) -> bool:
        current_timesteps = self.model.num_timesteps
        
        # Vérifier tous les N timesteps
        if current_timesteps - self.last_check_timestep < EVAL_FREQ_FOR_SUCCESS_CHECK:
            return True
        
        self.last_check_timestep = current_timesteps
        
        if current_timesteps > LEARNING_STARTS and len(self.model.ep_info_buffer) > 0:
            # Récupère les N derniers épisodes complètes
            recent_episodes = list(self.model.ep_info_buffer)
            
            if len(recent_episodes) >= MIN_EVAL_EPISODES_FOR_SUCCESS:
                # Prendre les N derniers épisodes pour un bon échantillon
                sample_size = min(MIN_EVAL_EPISODES_FOR_SUCCESS * 2, len(recent_episodes))
                recent_episodes = recent_episodes[-sample_size:]
                
                # Compte les succès : cherche le champ 'is_success' dans les infos
                successes = 0
                for ep_info in recent_episodes:
                    # SB3 stocke les infos du dernier step() du callback
                    if "is_success" in ep_info and ep_info["is_success"]:
                        successes += 1
                
                success_rate = successes / len(recent_episodes)
                
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    if self.verbose > 0:
                        print(f"\n📊 Timestep {current_timesteps:,} | Success: {success_rate:.1%} | Best: {self.best_success_rate:.1%}")
                
                # Arrêt si seuil atteint
                if success_rate >= self.success_rate_target:
                    if self.verbose > 0:
                        print(f"\n✅ OBJECTIF ATTEINT! Success rate: {success_rate:.1%}")
                        print(f"   Entraînement arrêté après {current_timesteps:,} timesteps")
                    return False  # Stop training
        
        return True  # Continue training


class PushInHoleGoalEnv(gym.Env):
    """Adaptateur GoalEnv de PushInHoleEnv pour HerReplayBuffer.

    Transforme l'observation vectorielle de PushInHoleEnv (dim 15) en un
    dictionnaire GoalEnv avec la décomposition :

    ``observation``   (6) : état robot [qpos(3) | ee_pos(3)]
    ``achieved_goal`` (3) : position 3D courante du cube
    ``desired_goal``  (3) : position 3D du trou (cible)

    Cette décomposition est indispensable pour que HerReplayBuffer puisse
    substituer ``desired_goal`` par l'``achieved_goal`` d'une transition future
    lors du relabelling, puis appeler ``compute_reward`` pour recalculer la
    récompense de la transition relabellisée.
    L'état robot est séparé du but pour éviter que l'agent apprenne à exploiter
    les vecteurs dérivés du but (ee_to_cube, cube_to_hole) contenus dans
    l'observation brute de PushInHoleEnv.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode: str | None =None) -> None:
        super().__init__()
        self.render_mode: str | None = render_mode
        self._inner: PushInHoleEnv = PushInHoleEnv(render_mode=render_mode)

        obs_dim: int = 6   #qpos(3) + ee_pos(3) : état robot sans les vecteurs dérivés du but
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
        cube_pos: np.ndarray = self._inner.sim.get_cube_pos()
        return {
            "observation":   np.concatenate([qpos, ee_pos]).astype(np.float32),
            "achieved_goal": cube_pos.astype(np.float32),
            "desired_goal":  self._inner._hole_pos.astype(np.float32),
        }

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Récompense relabellisable : DOIT correspondre exactement à celle de PushInHoleEnv.

        HER relabellise les (achieved_goal, desired_goal) et appelle cette fonction
        pour recalculer la récompense. Si la logique diffère de PushInHoleEnv,
        l'agent apprend des signaux contradictoires.
        
        Note : le terme d'approche ee→cube n'est pas relabellisable (ee n'est pas
        un état-but), donc il est appliqué UNIQUEMENT en step(), pas en relabelling.
        En compensation, on augmente le poids du terme cube→trou.

        Parameters:
            achieved_goal (np.ndarray): position(s) 3D du cube
            desired_goal (np.ndarray): position(s) 3D du trou substitué(s) par HER
            info (dict): non utilisé
        Returns:
            np.ndarray: récompenses de forme (batch_size,) ou scalaire float.
        """
        # Distance xy cube → trou (partie relabellisable)
        dist_xy: np.ndarray = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1
        ).astype(np.float32)
        reward: np.ndarray = -dist_xy
        
        # Bonus succès si z chute
        reward += 100.0 * (achieved_goal[..., 2] < SUCCESS_Z_THRESHOLD).astype(np.float32)
        
        return reward

    def reset(self, *, seed: int | None =None, options: dict | None =None):
        super().reset(seed=seed)
        self._inner.reset(seed=seed, options=options)
        obs: dict[str, np.ndarray] = self._build_obs()
        return obs, {"hole_pos": self._inner._hole_pos.copy()}

    def step(self, action: np.ndarray):
        #on délègue l'avance de simulation à l'env interne, on récupère son info
        _, _, terminated, truncated, inner_info = self._inner.step(action)

        obs: dict[str, np.ndarray] = self._build_obs()
        ee_pos: np.ndarray = self._inner.sim.get_end_effector_pos()
        cube_pos: np.ndarray = self._inner.sim.get_cube_pos()
        
        # Récompense relabellisable (cube → trou) depuis compute_reward()
        goal_reward: float = float(self.compute_reward(cube_pos, self._inner._hole_pos, {}))
        
        # Terme d'approche NON relabellisable (ee → cube) appliqué à chaque step()
        # Ce terme est nécessaire car il force le bootstrap initial du mouvement
        # mais ne peut pas être relabellisé (ee n'est pas un état-but)
        dist_ee_cube = float(np.linalg.norm(ee_pos - cube_pos))
        approach_reward = -2.0 * dist_ee_cube
        
        # Récompense totale = partie relabellisable + terme d'approche non-relabellisable
        reward = goal_reward + approach_reward

        info: dict[str, Any] = {
            "is_success": inner_info["is_success"],
            "dist_cube_hole": inner_info["dist_cube_hole"],
            "reward_breakdown": {
                "goal": goal_reward,
                "approach": approach_reward,
            },
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self) -> None:
        self._inner.close()


def make_her_sac(
    env: PushInHoleGoalEnv,
    log_dir: str =LOG_DIR,
) -> SAC:
    """Construit un SAC avec HerReplayBuffer sur PushInHoleGoalEnv.

    La politique ``MultiInputPolicy`` traite automatiquement le dict d'observation
    en concaténant ``observation``, ``achieved_goal`` et ``desired_goal`` via
    un extracteur de features dédié.
    Parameters:
        env (PushInHoleGoalEnv): environnement GoalEnv (non vectorisé)
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
        ent_coef="auto",  # Entropie automatique : encourage l'exploration
        target_entropy="auto",
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


def make_env(render_mode: str | None =None) -> PushInHoleGoalEnv:
    """Crée une instance fraîche de PushInHoleGoalEnv.
    Parameters:
        render_mode (str | None): ``"human"`` pour afficher MuJoCo en temps réel,
            ``None`` pour l'entraînement headless (plus rapide).
    Returns:
        PushInHoleGoalEnv: GoalEnv prête pour HER.
    """
    return PushInHoleGoalEnv(render_mode=render_mode)


def finetune_from_checkpoint(
    checkpoint_path: str,
    total_timesteps: int = 2_000_000,
    model_dir: str = MODEL_DIR,
    log_dir: str = LOG_DIR,
    render: bool = False,
) -> SAC:
    """Relance un entraînement NEUF avec reward et curriculum optimisés.
    
    ⚠️ IMPORTANT: Le replay buffer HER n'est pas sérialisé dans les checkpoints SB3.
    Donc au lieu de charger un vieux buffer vide et relancer, on entraîne depuis zéro
    mais avec une configuration améliorée :
    - Reward shaping optimisée (saturation approche, pression temporelle, lissage)
    - Curriculum plus long (2000 épisodes au lieu de 300)
    
    Cette approche converge PLUS VITE et PLUS STABLE car l'env est mieux réglé.
    """
    print("\n" + "="*70)
    print("🔁 ENTRAÎNEMENT SAC+HER v2 - Push-in-Hole (configuration améliorée)")
    print("="*70)
    print(f"ℹ️  Note: Anciuement best_model.zip @ 495k timesteps")
    print(f"        Relance NEUVE avec configuration optimisée")
    print(f"Limite timesteps: {total_timesteps:,}")
    print(f"Objectif succès: {SUCCESS_RATE_TARGET:.0%} de réussite")
    print("Améliorations appliquées :")
    print(f"  ✓ CURRICULUM_EPISODES: 2000 (stabilité +)")
    print(f"  ✓ ACTION_RATE_COEFF: 0.01 (lissage moteurs)")
    print(f"  ✓ STEP_TIME_PENALTY: 0.05 (pression temporelle)")
    print(f"  ✓ Saturation approche @ 3cm (moins de danse)")
    print(f"  ✓ Poids cube→trou: -5.0 (objectif renforcé)")
    print("→ L'entraînement s'arrêtera automatiquement au succès ✅")
    print("="*70 + "\n")

    # Appelle simplement train() avec 2M steps au lieu du default
    return train(
        total_timesteps=total_timesteps,
        model_dir=model_dir,
        log_dir=log_dir,
        render=render,
    )


def train(
    total_timesteps: int =TOTAL_TIMESTEPS,
    model_dir: str =MODEL_DIR,
    log_dir: str =LOG_DIR,
    render: bool =False,
) -> SAC:
    """Entraîne un agent SAC+HER sur la tâche de push-in-hole.

    L'entraînement s'arrête AUTOMATIQUEMENT quand le succès est atteint
    (success_rate ≥ 90%), ce qui économise du temps et évite le surapprentissage.
    
    Avec N_SAMPLED_GOAL=4, chaque transition génère 4 transitions relabellisées
    supplémentaires, soit un buffer effectif 5× plus dense en signal utile.
    
    Parameters:
        total_timesteps (int): limite max de pas (arrêt précoce si succès atteint)
        model_dir (str): répertoire de sauvegarde du meilleur modèle
        log_dir (str): répertoire TensorBoard
        render (bool): affiche la simulation MuJoCo en temps réel si True
    Returns:
        SAC: agent SAC+HER entraîné.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*70)
    print("🤖 ENTRAÎNEMENT SAC+HER - Push-in-Hole")
    print("="*70)
    print(f"Limite timesteps: {total_timesteps:,}")
    print(f"Objectif succès: {SUCCESS_RATE_TARGET:.0%} de réussite")
    print(f"Évaluation tous les: {EVAL_FREQ_FOR_SUCCESS_CHECK:,} pas")
    print("→ L'entraînement s'arrêtera automatiquement au succès ✅")
    print("="*70 + "\n")

    render_mode: str | None = "human" if render else None
    env: PushInHoleGoalEnv = make_env(render_mode=render_mode)
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
    
    # Callback d'arrêt anticipé basé sur le succès
    success_callback: _SuccessStoppingCallback = _SuccessStoppingCallback(
        success_rate_target=SUCCESS_RATE_TARGET,
        verbose=1,
    )

    # SB3 n'appelle jamais env.render() dans la boucle learn() : callback nécessaire
    callbacks: list[BaseCallback] = [eval_callback, success_callback]
    if render:
        callbacks.append(_RenderCallback())

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\n⏸️  Entraînement interrompu par l'utilisateur")

    # Sauvegarder le modèle final
    model.save(os.path.join(model_dir, "her_sac_final"))
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print(f"   Model sauvegardé: {model_dir}/her_sac_final.zip")
    print(f"   Logs TensorBoard: {log_dir}")
    print("="*70 + "\n")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Entraînement SAC+HER sur le robot 3-DDL (MuJoCo)"
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
        "--finetune", action="store_true",
        help="Fine-tune depuis best_model.zip au lieu d'entraîner depuis zéro"
    )
    args: argparse.Namespace = parser.parse_args()
    
    if args.finetune:
        checkpoint = os.path.join(MODEL_DIR, "best_model.zip")
        if not os.path.exists(checkpoint):
            print(f"❌ Erreur: Checkpoint {checkpoint} introuvable")
            print(f"   Assurez-vous que best_model.zip existe dans {MODEL_DIR}")
            exit(1)
        # Fine-tuning par défaut sur 2M steps si non spécifié
        finetune_timesteps = args.timesteps if args.timesteps != TOTAL_TIMESTEPS else 2_000_000
        finetune_from_checkpoint(
            checkpoint_path=checkpoint,
            total_timesteps=finetune_timesteps,
            render=args.render,
        )
    else:
        train(total_timesteps=args.timesteps, render=args.render)
