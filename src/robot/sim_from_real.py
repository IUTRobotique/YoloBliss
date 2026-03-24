"""sim_from_real.py — Boucle principale sim-to-real piloté par caméra.

Ce module expose une seule fonction `run()`, appelée par main.py
quand l'option --real est activée.

Dépendances :
  - real_perception.py  : vision (RealSense + ArUco + YOLO)
  - sim_to_real.py      : interface moteurs Dynamixel
"""

from __future__ import annotations

import time
import numpy as np

from real_perception import RealPerception, inject_into_sim, get_obs
import sim_to_real


def run(env, model, env_name: str, perception: RealPerception) -> None:
    """Boucle caméra → simulation → RL → moteurs réels.

    Args:
        env:          Environnement MuJoCo (GoalEnv ou Env standard)
        model:        Modèle SB3 déjà chargé
        env_name:     Nom de l'env ("reaching", "push_in_hole", ...)
        perception:   Instance RealPerception déjà initialisée
    """
    obs, _ = env.reset()
    sim_to_real.init_real_robot()

    print("\n[RUN] Bougez le cube devant la caméra — le robot suit en temps réel !")
    print("Ctrl+C pour quitter.\n")

    try:
        while True:
            # Perception : position 3D de l'objet dans le repère base
            pos_real = perception.get_object_position()

            if pos_real is not None:
                inject_into_sim(env, pos_real, env_name)
                obs = get_obs(env)
                print(f"  Objet → X={pos_real[0]:.3f}  Y={pos_real[1]:.3f}  Z={pos_real[2]:.3f}", end="\r")

            # Prédiction RL + pas de simulation
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            env.render()

            # Envoyer la commande aux vrais moteurs
            inner = env._inner if hasattr(env, "_inner") else env
            sim_to_real.update_real_robot_position(inner.sim.get_qpos())

            if terminated or truncated:
                print("\n[INFO] Episode terminé — reset de la simulation.")
                obs, _ = env.reset()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[STOP] Arrêt demandé.")
    finally:
        sim_to_real.close_real_robot()
        perception.stop()
        env.close()
