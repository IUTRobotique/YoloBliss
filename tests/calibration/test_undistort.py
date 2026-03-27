#!/usr/bin/env python3
"""
Test de correction de distorsion de la caméra.

Charge la calibration intrinsèque (calibration_intrinseque.pkl) et la compare
à la calibration d'usine RealSense (realsense_calibration.json).

Modes :
  1. Caméra live        : python test_undistort.py --camera <id>
  2. Image statique     : python test_undistort.py --image <chemin>
  3. Comparaison seule  : python test_undistort.py --comparer
     (affiche uniquement les différences numériques entre les deux calibrations)

Pré-requis :
  - calibration_intrinseque.pkl  (produit par calibration_intrinsique.py)
  - realsense_calibration.json   (produit par get_realsense_intrinsics.py)

Usage :
  python src/calibration/test_undistort.py --camera 0
  python src/calibration/test_undistort.py --image ma_photo.jpg
  python src/calibration/test_undistort.py --comparer
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

# ── Chemins ────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(_HERE, "..", ".."))

PKL_PATH   = os.path.join(ROOT, "calibration_intrinseque.pkl")
JSON_PATH  = os.path.join(ROOT, "realsense_calibration.json")


# ── Chargement des calibrations ────────────────────────────────────────────────

def charger_calibration_custom() -> dict | None:
    """Charge calibration_intrinseque.pkl."""
    if not os.path.isfile(PKL_PATH):
        print(f" calibration_intrinseque.pkl introuvable : {PKL_PATH}")
        print("    Lance d'abord : python src/calibration/calibration_intrinsique.py")
        return None
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f" Calibration custom chargée depuis {PKL_PATH}")
    return data


def charger_calibration_realsense() -> dict | None:
    """Charge realsense_calibration.json."""
    if not os.path.isfile(JSON_PATH):
        print(f" realsense_calibration.json introuvable : {JSON_PATH}")
        print(" Lance d'abord : python src/calibration/get_realsense_intrinsics.py")
        return None
    with open(JSON_PATH) as f:
        data = json.load(f)
    print(f" Calibration RealSense chargée depuis {JSON_PATH}")
    return data


def extraire_K_D_realsense(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extrait K et D depuis realsense_calibration.json (flux color)."""
    color = data.get("color", data)
    K = np.array(color["camera_matrix"], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    return K, D


# ── Comparaison numérique ──────────────────────────────────────────────────────

def comparer_calibrations(custom: dict, rs: dict) -> None:
    K_c = custom["camera_matrix"]
    D_c = custom["dist_coeffs"].flatten()
    K_r, _ = extraire_K_D_realsense(rs)

    print("\n" + "=" * 56)
    print("  COMPARAISON DES CALIBRATIONS")
    print("=" * 56)
    print(f"{'Paramètre':<12} {'Custom':>12} {'RealSense':>12} {'abs':>10} {'%':>8}")
    print("-" * 56)

    params = [
        ("fx", K_c[0, 0], K_r[0, 0]),
        ("fy", K_c[1, 1], K_r[1, 1]),
        ("cx", K_c[0, 2], K_r[0, 2]),
        ("cy", K_c[1, 2], K_r[1, 2]),
    ]
    for name, vc, vr in params:
        delta_abs = abs(vc - vr)
        delta_pct = delta_abs / abs(vr) * 100 if vr != 0 else float("inf")
        print(f"  {name:<10} {vc:>12.3f} {vr:>12.3f} {delta_abs:>10.3f} {delta_pct:>7.2f}%")

    print("-" * 56)
    # Norme de la différence
    norm_diff = np.linalg.norm(K_c - K_r)
    print(f"  Norme |K_custom - K_RS|  : {norm_diff:.4f}")
    print(f"\n  Distorsion custom (k1..p2) : {D_c[:5]}")
    print(f"  Distorsion RealSense      : [0. 0. 0. 0. 0.] (d'usine)")
    print("=" * 56 + "\n")


# ── Affichage correction ───────────────────────────────────────────────────────

def construire_maps(K, D, size=(640, 480)):
    """Précalcule les maps de correction (plus rapide dans une boucle vidéo)."""
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=1)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)
    return map1, map2, roi


def annoter_panneau(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 60), 2)
    return out


def afficher_comparaison(frame: np.ndarray,
                         map1_c, map2_c,
                         map1_r, map2_r) -> None:
    """Affiche 3 panneaux : original | custom | realsense."""
    corrigee_c = cv2.remap(frame, map1_c, map2_c, cv2.INTER_LINEAR)
    corrigee_r = cv2.remap(frame, map1_r, map2_r, cv2.INTER_LINEAR)

    panneau = np.hstack([
        annoter_panneau(frame,      "Original"),
        annoter_panneau(corrigee_c, "Custom"),
        annoter_panneau(corrigee_r, "RealSense"),
    ])
    cv2.imshow("Comparaison corrections distorsion", panneau)


def tester_sur_image(image_path: str, custom: dict | None, rs: dict | None) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f" Impossible de lire l'image : {image_path}")
        sys.exit(1)

    h, w = frame.shape[:2]

    if custom:
        map1_c, map2_c, _ = construire_maps(
            custom["camera_matrix"], custom["dist_coeffs"], (w, h)
        )
    else:
        map1_c, map2_c = None, None

    if rs:
        K_r, D_r = extraire_K_D_realsense(rs)
        map1_r, map2_r, _ = construire_maps(K_r, D_r, (w, h))
    else:
        map1_r, map2_r = None, None

    if map1_c is not None and map1_r is not None:
        afficher_comparaison(frame, map1_c, map2_c, map1_r, map2_r)
    elif map1_c is not None:
        corrigee = cv2.remap(frame, map1_c, map2_c, cv2.INTER_LINEAR)
        cv2.imshow("Original | Custom", np.hstack([frame, corrigee]))
    elif map1_r is not None:
        corrigee = cv2.remap(frame, map1_r, map2_r, cv2.INTER_LINEAR)
        cv2.imshow("Original | RealSense", np.hstack([frame, corrigee]))
    else:
        print(" Aucune calibration disponible.")
        sys.exit(1)

    print(" Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tester_camera(camera_id: int, custom: dict | None, rs: dict | None) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f" Impossible d'ouvrir la caméra {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    map1_c = map2_c = map1_r = map2_r = None
    if custom:
        map1_c, map2_c, _ = construire_maps(
            custom["camera_matrix"], custom["dist_coeffs"]
        )
    if rs:
        K_r, D_r = extraire_K_D_realsense(rs)
        map1_r, map2_r, _ = construire_maps(K_r, D_r)

    if map1_c is None and map1_r is None:
        print(" Aucune calibration disponible — impossible d'afficher la correction.")
        cap.release()
        sys.exit(1)

    print(f" Caméra {camera_id} — 'q' pour quitter, 's' pour capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if map1_c is not None and map1_r is not None:
            afficher_comparaison(frame, map1_c, map2_c, map1_r, map2_r)
        elif map1_c is not None:
            corrigee = cv2.remap(frame, map1_c, map2_c, cv2.INTER_LINEAR)
            cv2.imshow("Original | Custom", np.hstack([frame, corrigee]))
        else:
            corrigee = cv2.remap(frame, map1_r, map2_r, cv2.INTER_LINEAR)
            cv2.imshow("Original | RealSense", np.hstack([frame, corrigee]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("undistort_capture.jpg", frame)
            print(" Capture sauvegardée : undistort_capture.jpg")

    cap.release()
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test correction de distorsion")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera",   type=int, help="ID caméra")
    group.add_argument("--image",    type=str, help="Chemin vers une image")
    group.add_argument("--comparer", action="store_true",
                       help="Affiche uniquement la comparaison numérique des calibrations")
    args = parser.parse_args()

    custom = charger_calibration_custom()
    rs     = charger_calibration_realsense()

    if custom is None and rs is None:
        print(" Aucune calibration trouvée. Impossible de continuer.")
        sys.exit(1)

    if custom and rs:
        comparer_calibrations(custom, rs)

    if args.comparer:
        return

    if args.image:
        tester_sur_image(args.image, custom, rs)
    elif args.camera is not None:
        tester_camera(args.camera, custom, rs)
    else:
        print(" Aucun argument. Comparaison numérique seule (--camera ou --image pour visualiser).")


if __name__ == "__main__":
    main()
