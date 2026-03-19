#!/usr/bin/env python3
"""
Test des environnements Gymnasium du robot 3-DDL.

Ce script instancie les 3 environnements et exécute quelques steps avec des
actions aléatoires pour vérifier que tout fonctionne sans crash.

Tests effectués pour chaque env :
  1. Instanciation (chargement de la scène MuJoCo)
  2. Vérification des espaces d'observation et d'action
  3. reset() → vérification de la forme de l'observation
  4. N steps avec actions aléatoires → vérification de (obs, reward, done, info)
  5. Fermeture propre (close)

Usage :
  python src/robot/test_env.py                    # teste les 3 envs
  python src/robot/test_env.py --env reaching     # teste uniquement ReachingEnv
  python src/robot/test_env.py --env push
  python src/robot/test_env.py --env push_in_hole
  python src/robot/test_env.py --steps 50         # nombre de steps par env
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Ajoute le dossier src/robot au path pour trouver sim_3dofs.py et les envs
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── Utilitaire d'affichage ─────────────────────────────────────────────────────

def titre(texte: str) -> None:
    print(f"\n{'='*56}")
    print(f"  {texte}")
    print(f"{'='*56}")


def ok(msg: str)   -> None: print(f"   {msg}")
def err(msg: str)  -> None: print(f"   {msg}")
def info(msg: str) -> None: print(f"   {msg}")


# ── Test générique ─────────────────────────────────────────────────────────────

def tester_env(nom: str, env_class, n_steps: int) -> bool:
    titre(nom)
    try:
        # ── 1. Instanciation ──────────────────────────────────────────────────
        t0  = time.time()
        env = env_class(render_mode=None)
        ok(f"Instanciation OK ({time.time() - t0:.2f}s)")

        # ── 2. Espaces ────────────────────────────────────────────────────────
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        ok(f"Observation space : {env.observation_space.__class__.__name__}  dim={obs_shape}")
        ok(f"Action space      : {env.action_space.__class__.__name__}  dim={act_shape}")

        # ── 3. reset ──────────────────────────────────────────────────────────
        obs, info_reset = env.reset()
        assert obs.shape == obs_shape, f"Forme obs incorrecte : {obs.shape} ≠ {obs_shape}"
        assert np.all(np.isfinite(obs)), "Observation contient NaN/Inf après reset"
        ok(f"reset()  OK  obs={obs.shape}  infos={list(info_reset.keys())}")

        # ── 4. Steps aléatoires ───────────────────────────────────────────────
        n_done   = 0
        rewards  = []
        t0       = time.time()

        for step in range(n_steps):
            action                          = env.action_space.sample()
            obs, reward, terminated, truncated, info_step = env.step(action)

            assert obs.shape == obs_shape, f"[step {step}] Forme obs incorrecte : {obs.shape}"
            assert np.isfinite(reward),    f"[step {step}] Reward non fini : {reward}"
            assert np.all(np.isfinite(obs)), f"[step {step}] Obs contient NaN/Inf"

            rewards.append(reward)
            if terminated or truncated:
                n_done += 1
                obs, _ = env.reset()

        elapsed = time.time() - t0
        ok(f"{n_steps} steps  — {n_steps/elapsed:.0f} steps/s  — "
           f"{n_done} épisode(s) terminé(s)")
        info(f"Reward : min={min(rewards):.4f}  max={max(rewards):.4f}  "
             f"moy={np.mean(rewards):.4f}")

        # ── 5. Fermeture ──────────────────────────────────────────────────────
        env.close()
        ok("close() OK")
        return True

    except Exception as exc:
        err(f"ERREUR : {exc}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test des environnements Gymnasium")
    parser.add_argument("--env", choices=["reaching", "push", "push_in_hole", "all"],
                        default="all", help="Environnement à tester (défaut : all)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Nombre de steps par environnement (défaut : 30)")
    args = parser.parse_args()

    tests = {
        "reaching":    ("ReachingEnv",    "robot_env.reaching_env",    "ReachingEnv"),
        "push":        ("PushEnv",         "robot_env.push_env",        "PushEnv"),
        "push_in_hole":("PushInHoleEnv",   "robot_env.push_in_hole_env","PushInHoleEnv"),
    }

    a_tester = list(tests.keys()) if args.env == "all" else [args.env]
    resultats = {}

    for key in a_tester:
        label, module_path, class_name = tests[key]
        try:
            import importlib
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)
        except ImportError as exc:
            titre(label)
            err(f"Import impossible ({module_path}) : {exc}")
            resultats[key] = False
            continue

        resultats[key] = tester_env(label, env_class, args.steps)

    # ── Récapitulatif ──────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("  RÉCAPITULATIF")
    print(f"{'='*56}")
    for key, res in resultats.items():
        symbole = "yes" if res else "no"
        print(f"  [{symbole}] {tests[key][0]}")

    total_ok  = sum(resultats.values())
    total     = len(resultats)
    print(f"\n  {total_ok}/{total} environnement(s) OK\n")

    sys.exit(0 if all(resultats.values()) else 1)


if __name__ == "__main__":
    main()
