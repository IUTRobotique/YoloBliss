"""Script autonome : lance un episode de simulation MuJoCo.

Reprend la meme logique que src/robot/main.py (resolution model, envs HER, etc.)

Usage:
    python run_sim_episode.py <env_name> <main_algo> <model_path|none> <output_dir> [max_steps]

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


def _load_model(model_path_str: str | None, algo: str, env):
    """Charge le modele avec la classe SB3 correspondante (comme main.py)."""
    if not model_path_str or model_path_str == "none" or not os.path.exists(model_path_str):
        return None
    cls_name = ALGO_CLS.get(algo, "SAC")
    try:
        from stable_baselines3 import SAC, PPO, TD3
        cls_map = {"SAC": SAC, "PPO": PPO, "TD3": TD3}
        if cls_name in cls_map:
            return cls_map[cls_name].load(model_path_str, env=env)
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


def run_episode(env_name: str, algo: str, model_path_str: str | None,
                output_dir: str, max_steps: int = 300) -> dict:
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    env = _make_env(env_name, algo)
    if env is None:
        result = {"error": f"Environnement '{env_name}' non supporte (algo={algo})"}
        _write(output_dir, result)
        return result

    model = _load_model(model_path_str, algo, env)
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
        "model":        model_path_str or "random",
    }
    _write(output_dir, metrics)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: run_sim_episode.py <env> <main_algo> <model|none> <output_dir> [max_steps]")
        sys.exit(1)
    env_arg   = sys.argv[1]
    algo_arg  = sys.argv[2]
    model_arg = sys.argv[3]
    out_arg   = sys.argv[4]
    steps_arg = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    result = run_episode(env_arg, algo_arg, model_arg, out_arg, steps_arg)
    print(json.dumps(result, indent=2))
