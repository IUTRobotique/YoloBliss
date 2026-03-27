"""Script autonome : lance un episode de simulation MuJoCo.

Reprend la meme logique que src/robot/main.py (resolution model, envs HER, etc.)

Usage:
    python run_sim_episode.py <env_name> <main_algo> <output_dir> [max_steps]

    main_algo : sac | ppo | crossq | her
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROBOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "robot"))
sys.path.insert(0, ROBOT_SRC)
sys.path.insert(0, os.path.join(ROBOT_SRC, "robot_env"))


# -- Classes SB3 (meme mapping que main.py) --
ALGO_CLS = {
    "sac":    "SAC",
    "ppo":    "PPO",
    "td3":    "TD3",
    "crossq": "SAC",   # CrossQ checkpoint charge via SAC
    "her":    "SAC",
}

# -- Resolution de modele (meme logique que main.py) --
_MODELS_DIR = Path(ROBOT_SRC) / "models"
# Mapping HER env -> dossier (main.py a des conflits git sur ce point)
_HER_MODEL_DIRS = {
    "push_in_hole": "her_sac_1st_working_push_in_hole",
    "sorting":      "her_sac_sorting",
}


def _resolve_model_path(env_name: str, algo: str) -> Path | None:
    """Resout le chemin du modele comme le fait main.py."""
    if algo == "her":
        if env_name in _HER_MODEL_DIRS:
            model_dir = _MODELS_DIR / _HER_MODEL_DIRS[env_name]
        else:
            her_dir = _MODELS_DIR / f"her_sac_{env_name}"
            model_dir = her_dir if her_dir.exists() else _MODELS_DIR / algo
    else:
        specific = _MODELS_DIR / f"{algo}_{env_name}"
        model_dir = specific if specific.exists() else _MODELS_DIR / algo

    candidates = [
        model_dir / "best_model.zip",
        model_dir / "best_model",
        model_dir / f"{algo}_{env_name}_final.zip",
        model_dir / f"her_sac_{env_name}_final.zip",
        model_dir / f"{algo}_final.zip",
        model_dir / "her_sac_final.zip",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _write(output_dir: str, data: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(data, f, indent=2)


def _make_env(env_name: str, algo: str):
    """Cree l'env. Pour HER, utilise les GoalEnv comme dans main.py."""
    if algo == "her":
        if env_name == "push_in_hole":
            from her_push_in_hole import PushInHoleGoalEnv
            return PushInHoleGoalEnv(render_mode="rgb_array")
        elif env_name == "sorting":
            from her_sorting import SortingGoalEnv
            return SortingGoalEnv(render_mode="rgb_array")
        else:
            return None

    mapping = {
        "reaching":     ("robot_env.reaching_env",    "ReachingEnv"),
        "push":         ("robot_env.push_env",         "PushEnv"),
        "sliding":      ("robot_env.sliding_env",      "SlidingEnv"),
        "push_in_hole": ("robot_env.push_in_hole_env", "PushInHoleEnv"),
        "sorting":      ("robot_env.sorting_env",      "SortingEnv"),
    }
    if env_name not in mapping:
        return None
    mod_name, cls_name = mapping[env_name]
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        return cls(render_mode="rgb_array")
    except Exception as e:
        print(f"[run_sim] Env error: {e}", flush=True)
        return None


def _load_model(env_name: str, algo: str, env):
    """Resout et charge le modele avec la classe SB3 correspondante (comme main.py)."""
    model_path = _resolve_model_path(env_name, algo)
    if model_path is None:
        print(f"[run_sim] Aucun modele trouve pour env={env_name} algo={algo}", flush=True)
        return None
    cls_name = ALGO_CLS.get(algo, "SAC")
    try:
        from stable_baselines3 import SAC, PPO, TD3
        cls_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
        if cls_name in cls_map:
            return cls_map[cls_name].load(str(model_path), env=env)
    except Exception as e:
        print(f"[run_sim] Model load error: {e}", flush=True)
    return None


def _extract_distance(info: dict) -> float:
    """Recupere une metrique de distance (meme logique que main.py)."""
    for key in ("distance", "cube_displacement", "dist_cube_hole",
                "dist_cube_goal", "dist_cylinder_goal"):
        if key in info:
            return float(info[key])
    return float("nan")


def run_episode(env_name: str, algo: str,
                output_dir: str, max_steps: int = 300) -> dict:
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    env = _make_env(env_name, algo)
    if env is None:
        result = {"error": f"Environnement '{env_name}' non supporte (algo={algo})"}
        _write(output_dir, result)
        return result

    model = _load_model(env_name, algo, env)
    obs, _ = env.reset()
    total_reward, step = 0.0, 0
    frames: list[np.ndarray] = []
    info: dict = {}

    for step in range(max_steps):
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame)
        if terminated or truncated:
            break

    dist = _extract_distance(info)
    env.close()

    # Sauvegarde video
    video_path = ""
    if frames:
        try:
            import cv2
            h, w = frames[0].shape[:2]

            # Essayer H.264 via ffmpeg si disponible (meilleure compatibilite navigateur)
            import shutil
            if shutil.which("ffmpeg"):
                import subprocess
                raw_path = os.path.join(output_dir, "episode_raw.mp4")
                final_path = os.path.join(output_dir, "episode.mp4")
                out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                ret = subprocess.run(
                    ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264",
                     "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                     final_path],
                    capture_output=True, timeout=60,
                )
                if ret.returncode == 0:
                    os.remove(raw_path)
                    video_path = final_path
                else:
                    os.rename(raw_path, final_path)
                    video_path = final_path
            else:
                # Fallback : WebM VP8 — lisible par tous les navigateurs sans ffmpeg
                video_path = os.path.join(output_dir, "episode.webm")
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"VP80"), 25, (w, h))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
        except Exception as e:
            print(f"[run_sim] Video error: {e}", flush=True)
            video_path = ""

    metrics = {
        "n_steps":      step + 1,
        "total_reward": round(total_reward, 4),
        "is_success":   bool(info.get("is_success", False)),
        "distance":     round(dist, 4) if not (dist != dist) else None,  # NaN check
        "n_frames":     len(frames),
        "video_path":   video_path,
        "env":          env_name,
        "algo":         algo,
        "model":        str(_resolve_model_path(env_name, algo) or "random"),
    }
    _write(output_dir, metrics)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run_sim_episode.py <env> <main_algo> <output_dir> [max_steps]")
        sys.exit(1)
    env_arg   = sys.argv[1]
    algo_arg  = sys.argv[2]
    out_arg   = sys.argv[3]
    steps_arg = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    result = run_episode(env_arg, algo_arg, out_arg, steps_arg)
    print(json.dumps(result, indent=2))
