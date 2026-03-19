#!/usr/bin/env python3
"""
Test de détection des marqueurs ArUco.


Vérifie que la détection ArUco fonctionne AVANT d'utiliser RealSense ou YOLO.

Modes :
  1. Flux caméra live   : python test_aruco_detection.py --camera <id>
  2. Image statique     : python test_aruco_detection.py --image <chemin>
  3. Image générée      : python test_aruco_detection.py --generer
     (génère un marqueur de test sans avoir besoin de la planche imprimée)

Ce script utilise uniquement OpenCV (pas de RealSense ni YOLO).
La planche A4 utilise DICT_4X4_50, IDs 3, 4, 5, 6.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ── Dictionnaire ArUco attendu ─────────────────────────────────────────────────
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_IDS    = [3, 4, 5, 6]       # IDs de la planche A4
MARKER_SIZE   = 0.06               # taille réelle en mètres (6 cm)

# Couleurs par ID
ID_COLORS = {
    3: (0,   255,   0),    # vert   — sup gauche
    4: (255, 128,   0),    # bleu   — sup droit
    5: (0,   0,   255),    # rouge  — inf gauche
    6: (128, 0,   255),    # violet — inf droit
}

# Intrinsèques approximatives (640×480) — suffisant pour estimer la pose
# (affinage avec realsense_calibration.json si besoin)
DEFAULT_CAMERA_MATRIX = np.array([
    [606.28,    0.0,  318.37],
    [  0.0,  606.13,  246.42],
    [  0.0,    0.0,     1.0 ]
], dtype=np.float64)
DEFAULT_DIST_COEFFS = np.zeros(5, dtype=np.float64)

# Points 3D du marqueur pour solvePnP
_h = MARKER_SIZE / 2
OBJ_POINTS = np.array([
    [-_h,  _h, 0],
    [ _h,  _h, 0],
    [ _h, -_h, 0],
    [-_h, -_h, 0],
], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def creer_detecor():
    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def detecter_et_annoter(frame: np.ndarray, detector, camera_matrix, dist_coeffs) -> tuple[np.ndarray, list[int]]:
    """
    Détecte les marqueurs ArUco et annote l'image.

    Retourne (image_annotee, liste_ids_detectes).
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    annotated  = frame.copy()
    ids_trouves = []

    if ids is not None:
        ids_flat = ids.flatten()
        ids_trouves = ids_flat.tolist()

        for i, marker_id in enumerate(ids_flat):
            color  = ID_COLORS.get(int(marker_id), (200, 200, 200))
            pts    = corners[i][0].astype(np.int32)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Centre du marqueur
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            cv2.putText(annotated, f"ID {marker_id}", (cx - 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Estimation de pose (rvec / tvec)
            ok, rvec, tvec = cv2.solvePnP(
                OBJ_POINTS, corners[i][0],
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                dist_cm = float(np.linalg.norm(tvec)) * 100
                cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE * 0.6)
                cv2.putText(annotated, f"{dist_cm:.1f} cm",
                            (cx - 12, cy + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Compteur de marqueurs attendus
    attendus = [mid for mid in MARKER_IDS if mid in [int(x) for x in ids_trouves]]
    cv2.putText(annotated,
                f"Marqueurs A4 : {len(attendus)}/4  {attendus}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if ids is None:
        cv2.putText(annotated, "Aucun marqueur détecté",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return annotated, ids_trouves


def charger_intrinsiques_realsense() -> tuple[np.ndarray, np.ndarray]:
    """Charge les intrinsèques depuis realsense_calibration.json si disponible."""
    root    = os.path.dirname(os.path.abspath(__file__))
    calib_f = os.path.join(root, "realsense_calibration.json")
    if not os.path.isfile(calib_f):
        print(" realsense_calibration.json absent — utilisation des intrinsèques approx.")
        return DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS

    import json
    with open(calib_f) as f:
        data = json.load(f)
    intr = data.get("color", data)
    K = np.array(intr["camera_matrix"], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    print(f" Intrinsèques RealSense chargées depuis {calib_f}")
    return K, D


# ── Modes ──────────────────────────────────────────────────────────────────────

def tester_image(image_path: str) -> None:
    detector = creer_detecor()
    K, D     = charger_intrinsiques_realsense()

    if not os.path.isfile(image_path):
        print(f" Image introuvable : {image_path}")
        sys.exit(1)

    frame = cv2.imread(image_path)
    annotated, ids = detecter_et_annoter(frame, detector, K, D)

    print(f" Image : {image_path}")
    print(f" IDs détectés : {ids}")
    marqueurs_a4 = [mid for mid in ids if mid in MARKER_IDS]
    if len(marqueurs_a4) >= 3:
        print(f" {len(marqueurs_a4)}/4 marqueurs A4 visibles — calibration possible")
    else:
        print(f" Seulement {len(marqueurs_a4)}/4 marqueurs A4 — il en faut au moins 3")

    cv2.imshow("Test ArUco — image statique", annotated)
    print(" Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tester_camera(camera_id: int) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f" Impossible d'ouvrir la caméra {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = creer_detecor()
    K, D     = charger_intrinsiques_realsense()

    print(f" Caméra {camera_id} ouverte — appuyez sur 'q' pour quitter, 's' pour capture")
    print(f" Présentez la planche A4 avec les marqueurs IDs {MARKER_IDS}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Impossible de lire la frame.")
            break

        annotated, ids = detecter_et_annoter(frame, detector, K, D)
        cv2.imshow("Test ArUco — flux live", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = "aruco_capture.jpg"
            cv2.imwrite(fname, annotated)
            print(f" Capture sauvegardée : {fname}")

    cap.release()
    cv2.destroyAllWindows()


def tester_avec_marqueur_genere() -> None:
    """
    Génère un marqueur ArUco de test et vérifie que la détection fonctionne.
    Aucun matériel requis.
    """
    print(" Génération d'une image de test avec le marqueur ID 6...")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector   = creer_detecor()
    K, D       = charger_intrinsiques_realsense()

    # Créer une image blanche 640×480 avec le marqueur ID 6 centré
    img = np.ones((480, 640, 3), dtype=np.uint8) * 230

    # Générer le marqueur
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 6, 200)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    # Coller au centre
    y0, x0 = 140, 220
    img[y0:y0+200, x0:x0+200] = marker_bgr

    cv2.putText(img, "Image de test — marqueur ID 6 (DICT_4X4_50)",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    annotated, ids = detecter_et_annoter(img, detector, K, D)

    if 6 in ids:
        print(" Test OK : marqueur ID 6 détecté correctement")
    else:
        print(" Test ÉCHOUÉ : marqueur ID 6 non détecté sur l'image générée")

    cv2.imshow("Test ArUco — marqueur généré", annotated)
    print(" Appuyez sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test de détection ArUco")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera",   type=int, help="ID caméra (ex: 0, 8, 37)")
    group.add_argument("--image",    type=str, help="Chemin vers une image")
    group.add_argument("--generer",  action="store_true",
                       help="Génère un marqueur de test (pas de caméra ni d'image requise)")
    args = parser.parse_args()

    if args.camera is not None:
        tester_camera(args.camera)
    elif args.image:
        tester_image(args.image)
    elif args.generer:
        tester_avec_marqueur_genere()
    else:
        print(" Aucun argument. Mode génération par défaut (--camera, --image ou --generer disponibles)")
        tester_avec_marqueur_genere()


if __name__ == "__main__":
    main()
