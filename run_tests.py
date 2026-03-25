#!/usr/bin/env python3
"""Lanceur de tests du projet YoloBliss / Robot 3-DDL.

Trois catégories :

  PYTEST       — tests unitaires automatiques (sans matériel)
                 tests/robot/test_cross_q.py, test_her.py,
                 test_robot_envs.py, tests/test_aruco_map.py
  SCRIPTS      — scripts autonomes exécutables sans matériel
                 tests/robot/test_mirror.py (conversions Dynamixel + liste ports)
  MANUELS      — scripts interactifs (caméra, MuJoCo, servos…)
                 listés avec leurs commandes, non exécutés automatiquement

Usage :
  python3 run_tests.py           # tout : pytest + scripts + liste manuels
  python3 run_tests.py --auto    # pytest uniquement (CI)
  python3 run_tests.py --manual  # affiche uniquement la liste des manuels
  python3 run_tests.py -v        # mode verbeux pour pytest
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 1. Tests pytest (unitaires, sans matériel) ────────────────────────────────
PYTEST_TESTS = [
    os.path.join(ROOT, "tests", "robot", "test_cross_q.py"),
    os.path.join(ROOT, "tests", "robot", "test_her.py"),
    os.path.join(ROOT, "tests", "robot", "test_robot_envs.py"),
    os.path.join(ROOT, "tests", "test_aruco_map.py"),
]

# ── 2. Scripts autonomes (exécutables sans matériel, stdlib only) ─────────────
SCRIPT_TESTS = [
    {
        "label": "Conversions Dynamixel (stdlib only, pas de matériel)",
        "file":  "tests/robot/test_mirror.py",
        "args":  [],   # mode par défaut = tests logiciels seuls
    },
]

# ── 3. Tests manuels (matériel / interface graphique requis) ──────────────────
MANUAL_TESTS = [
    {
        "label": "Détection ArUco (caméra ou image)",
        "file":  "tests/test_aruco_detection.py",
        "commandes": [
            "python3 tests/test_aruco_detection.py --generer           # sans caméra",
            "python3 tests/test_aruco_detection.py --camera 0          # flux live",
            "python3 tests/test_aruco_detection.py --image photo.jpg   # image statique",
        ],
    },
    {
        "label": "Détection YOLO (modèle entraîné ou base)",
        "file":  "tests/test_yolo_detection.py",
        "commandes": [
            "python3 tests/test_yolo_detection.py --dataset            # première image du dataset",
            "python3 tests/test_yolo_detection.py --camera 0           # flux live",
            "python3 tests/test_yolo_detection.py --image photo.jpg    # image statique",
        ],
    },
    {
        "label": "Correction de distorsion (calibration)",
        "file":  "tests/calibration/test_undistort.py",
        "commandes": [
            "python3 tests/calibration/test_undistort.py --comparer",
            "python3 tests/calibration/test_undistort.py --camera 0",
            "python3 tests/calibration/test_undistort.py --image photo.jpg",
        ],
    },
    {
        "label": "Conversions Dynamixel + connexion réelle",
        "file":  "tests/robot/test_mirror.py",
        "commandes": [
            "python3 tests/robot/test_mirror.py --lister-ports",
            "python3 tests/robot/test_mirror.py --port /dev/ttyACM0  # avec matériel branché",
        ],
    },
    {
        "label": "Environnements Gymnasium du robot 3-DDL (MuJoCo)",
        "file":  "tests/robot/test_env.py",
        "commandes": [
            "python3 tests/robot/test_env.py                       # teste les 3 envs",
            "python3 tests/robot/test_env.py --env reaching",
            "python3 tests/robot/test_env.py --env push",
            "python3 tests/robot/test_env.py --env push_in_hole",
        ],
    },
    {
        "label": "Viewer MuJoCo interactif (scène push)",
        "file":  "tests/robot/test_sim.py",
        "commandes": [
            "python3 tests/robot/test_sim.py",
        ],
    },
    {
        "label": "Dessin de formes sur robot réel ou Meshcat",
        "file":  "tests/robot/test_real_draw_forms.py",
        "commandes": [
            "python3 tests/robot/test_real_draw_forms.py --shape circle",
            "python3 tests/robot/test_real_draw_forms.py --real --shape rect",
        ],
    },
    {
        "label": "Caméra simple (OpenCV)",
        "file":  "tests/test_camera_simple.py",
        "commandes": [
            "python3 tests/test_camera_simple.py",
        ],
    },
    {
        "label": "Caméra (diagnostic complet)",
        "file":  "tests/test_camera.py",
        "commandes": [
            "python3 tests/test_camera.py",
        ],
    },
    {
        "label": "Profondeur RealSense",
        "file":  "tests/test_depth_realsense.py",
        "commandes": [
            "python3 tests/test_depth_realsense.py",
        ],
    },
]


# ── Helpers d'affichage ───────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def titre(texte: str) -> None:
    print(f"\n{_BOLD}{'─' * 60}{_RESET}")
    print(f"{_BOLD}  {texte}{_RESET}")
    print(f"{_BOLD}{'─' * 60}{_RESET}")


def afficher_tests_manuels() -> None:
    titre("Tests manuels (matériel / GUI requis — non exécutés automatiquement)")
    for t in MANUAL_TESTS:
        exists = os.path.isfile(os.path.join(ROOT, t["file"]))
        status = f"{_GREEN}[OK]{_RESET}" if exists else f"{_RED}[ABSENT]{_RESET}"
        print(f"\n  {status}  {_BOLD}{t['label']}{_RESET}")
        print(f"         {_CYAN}{t['file']}{_RESET}")
        for cmd in t["commandes"]:
            print(f"    $ {cmd}")


# ── Lancement pytest ──────────────────────────────────────────────────────────

def lancer_pytest(verbose: bool = False) -> int:
    """Lance pytest sur PYTEST_TESTS. Retourne le code de sortie."""
    titre("1/2  Tests unitaires (pytest)")

    existants = [f for f in PYTEST_TESTS if os.path.isfile(f)]
    absents   = [f for f in PYTEST_TESTS if not os.path.isfile(f)]

    for f in absents:
        print(f"  {_YELLOW}[ABSENT]{_RESET}  {os.path.relpath(f, ROOT)}")

    if not existants:
        print(f"  {_RED}Aucun fichier de test pytest trouvé.{_RESET}")
        return 1

    cmd = [sys.executable, "-m", "pytest"] + existants + (["-v"] if verbose else ["--tb=short"])
    print(f"  $ {' '.join(os.path.relpath(a, ROOT) if os.path.isabs(a) else a for a in cmd)}\n")
    return subprocess.run(cmd, cwd=ROOT).returncode


# ── Lancement scripts autonomes ───────────────────────────────────────────────

def lancer_scripts(verbose: bool = False) -> int:
    """Exécute chaque SCRIPT_TESTS et rapporte le résultat."""
    titre("2/2  Scripts autonomes (sans matériel)")

    global_ok = True
    for t in SCRIPT_TESTS:
        path = os.path.join(ROOT, t["file"])
        if not os.path.isfile(path):
            print(f"  {_YELLOW}[ABSENT]{_RESET}  {t['file']}")
            global_ok = False
            continue

        cmd = [sys.executable, path] + t["args"]
        label = t["label"]
        print(f"\n  {_CYAN}▶ {label}{_RESET}")
        print(f"  $ python3 {t['file']}")
        result = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose)
        if result.returncode == 0:
            print(f"  {_GREEN}[PASS]{_RESET}")
            if verbose and result.stdout:
                print(result.stdout.decode(errors="replace"))
        else:
            global_ok = False
            print(f"  {_RED}[FAIL]  code={result.returncode}{_RESET}")
            if result.stdout:
                print(result.stdout.decode(errors="replace"))
            if result.stderr:
                print(result.stderr.decode(errors="replace"))

    return 0 if global_ok else 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lanceur de tests YoloBliss / Robot 3-DDL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemples :
              python3 run_tests.py           # tout : pytest + scripts + liste manuels
              python3 run_tests.py --auto    # CI : pytest + scripts seulement
              python3 run_tests.py --manual  # afficher uniquement les tests manuels
              python3 run_tests.py -v        # mode verbeux
        """),
    )
    parser.add_argument("--auto",   action="store_true", help="Pytest + scripts uniquement (pas d'affichage manuels)")
    parser.add_argument("--manual", action="store_true", help="Affiche uniquement les tests manuels")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mode verbeux")
    args = parser.parse_args()

    if args.manual:
        afficher_tests_manuels()
        return

    code_pytest  = lancer_pytest(verbose=args.verbose)
    code_scripts = lancer_scripts(verbose=args.verbose)
    exit_code    = 0 if (code_pytest == 0 and code_scripts == 0) else 1

    if not args.auto:
        afficher_tests_manuels()

    titre("Résumé global")
    nb_pytest  = len(PYTEST_TESTS)
    nb_scripts = len(SCRIPT_TESTS)
    nb_manuels = len(MANUAL_TESTS)

    sym_p = _GREEN + "PASS" + _RESET if code_pytest  == 0 else _RED + "FAIL" + _RESET
    sym_s = _GREEN + "PASS" + _RESET if code_scripts == 0 else _RED + "FAIL" + _RESET

    print(f"  Tests pytest   ({nb_pytest:2d} fichier(s))  : {sym_p}")
    print(f"  Scripts auto   ({nb_scripts:2d} fichier(s))  : {sym_s}")
    print(f"  Tests manuels  ({nb_manuels:2d} fichier(s))  : non exécutés (matériel requis)")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
