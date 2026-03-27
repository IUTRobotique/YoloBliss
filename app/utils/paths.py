"""Chemins centralises pour l'application YoloBliss."""
import os

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROBOT_SRC  = os.path.join(ROOT, "src", "robot")
MODELS_DIR = os.path.join(ROBOT_SRC, "models")
LOGS_DIR   = os.path.join(ROBOT_SRC, "logs")
YOLO_MODEL = os.path.join(ROOT, "yolo11n.pt")
RUN_EPISODE_SCRIPT = os.path.join(os.path.dirname(__file__), "run_sim_episode.py")

# Environnements disponibles
ENV_NAMES = {
    "reaching":     "Reaching — Atteindre une cible 3D",
    "push":         "Push — Pousser un objet vers une cible",
    "push_in_hole": "Push-in-Hole — Pousser dans un trou",
    "sorting":      "Sorting — Trier les objets",
}

# Algorithmes disponibles  (model_file peut etre best_model.zip ou *_final.zip)
ALGO_INFO = {
    "SAC": {
        "env":         "reaching",
        "model_dir":   "sac",
        "model_file":  "best_model.zip",
        "algo_class":  "SAC",
        "main_algo":   "sac",
        "log_dir":     "sac",
        "color":       "sac",
        "description": "Soft Actor-Critic. Off-policy, maximise recompense + entropie. Converge rapidement sur des espaces continus.",
    },
    "PPO": {
        "env":         "push_in_hole",
        "model_dir":   "ppo",
        "model_file":  "best_model.zip",
        "algo_class":  "PPO",
        "main_algo":   "ppo",
        "log_dir":     "ppo",
        "color":       "ppo",
        "description": "Proximal Policy Optimization. On-policy, stable, clipping du surrogate loss.",
    },    "TD3": {
        "env":         "reaching",
        "model_dir":   "td3",
        "model_file":  "best_model.zip",
        "algo_class":  "TD3",
        "main_algo":   "td3",
        "log_dir":     "td3",
        "color":       "td3",
        "description": "Twin Delayed DDPG. Off-policy, twin critics + delayed policy update + target smoothing.",
    },
    "CrossQ": {
        "env":         "reaching",
        "model_dir":   "crossq",
        "model_file":  "best_model.zip",
        "algo_class":  "CrossQ",
        "main_algo":   "crossq",
        "log_dir":     "crossq",
        "color":       "crossq",
        "description": "CrossQ : SAC + Batch Normalisation + UTD ratio eleve. Jusqu'a 3x plus efficace.",
    },
    "SAC+HER — Push-in-Hole": {
        "env":         "push_in_hole",
        "model_dir":   "her_sac",
        "model_file":  "best_model.zip",
        "algo_class":  "SAC",
        "main_algo":   "her",
        "log_dir":     "her_sac",
        "color":       "her",
        "description": "SAC + Hindsight Experience Replay. But : pousser l'objet dans le trou cible.",
    },
    "SAC+HER — Sorting": {
        "env":         "sorting",
        "model_dir":   "her_sac_sorting",
        "model_file":  "her_sac_final.zip",
        "algo_class":  "SAC",
        "main_algo":   "her",
        "log_dir":     "her_sac_sorting",
        "color":       "her",
        "description": "SAC+HER pour la tache Sorting : trier les objets par couleur.",
    },
}


def model_path(algo_key: str) -> str:
    info = ALGO_INFO.get(algo_key, {})
    return os.path.join(MODELS_DIR, info.get("model_dir", ""), info.get("model_file", ""))


def log_path(algo_key: str) -> str:
    info = ALGO_INFO.get(algo_key, {})
    return os.path.join(LOGS_DIR, info.get("log_dir", ""))
