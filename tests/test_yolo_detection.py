#!/usr/bin/env python3
"""
Test du modèle YOLO : chargement, inférence et affichage des résultats.

Modes :
  1. Image statique         : python test_yolo_detection.py --image <chemin>
  2. Flux caméra live       : python test_yolo_detection.py --camera <id>
  3. Image du dataset       : python test_yolo_detection.py --dataset  (prend la première image trouvée)

Le modèle utilisé est 'best.pt' (entraîné) si disponible, sinon 'yolo11n.pt' (base).
"""

import argparse
import os
import sys

import cv2
import numpy as np

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_MODEL   = os.path.join(ROOT, "runs", "detect", "detection_objets", "weights", "best.pt")
BASE_MODEL   = os.path.join(ROOT, "yolo11n.pt")
DATASET_IMGS = os.path.join(ROOT, "dataset_localisation", "dataset_yolo", "images")

CLASS_NAMES = ["cube", "cylindre"]
COLORS = {
    "cube":     (0,  255,   0),   # vert
    "cylindre": (0,  128, 255),   # orange
}


def choisir_modele() -> str:
    """Utilise best.pt si disponible, sinon yolo11n.pt."""
    if os.path.isfile(BEST_MODEL):
        print(f" Modèle entraîné trouvé : {BEST_MODEL}")
        return BEST_MODEL
    elif os.path.isfile(BASE_MODEL):
        print(f" best.pt absent. Utilisation du modèle de base : {BASE_MODEL}")
        return BASE_MODEL
    else:
        print(" Aucun modèle YOLO trouvé. Vérifiez les chemins :")
        print(f"     {BEST_MODEL}")
        print(f"     {BASE_MODEL}")
        sys.exit(1)


def charger_modele(model_path: str):
    from ultralytics import YOLO
    print(f" Chargement du modèle : {os.path.basename(model_path)}")
    model = YOLO(model_path)
    print(f" Modèle chargé — tâche : {model.task}, classes : {model.names}")
    return model


def dessiner_detections(image: np.ndarray, results) -> np.ndarray:
    """Dessine les boîtes et labels sur l'image."""
    annotated = image.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            color = COLORS.get(label, (200, 200, 200))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(annotated, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    nb = sum(len(r.boxes) for r in results)
    cv2.putText(annotated, f"Détections : {nb}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


# ── Modes de test ──────────────────────────────────────────────────────────────

def tester_image(model, image_path: str) -> None:
    """Inférence sur une image statique."""
    if not os.path.isfile(image_path):
        print(f" Image introuvable : {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    print(f" Inférence sur : {image_path}  ({image.shape[1]}×{image.shape[0]})")

    results = model.predict(image, conf=0.25, verbose=False)
    annotated = dessiner_detections(image, results)

    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            print(f"  [{label}]  conf={conf:.2f}  bbox=({x1},{y1})-({x2},{y2})")

    nb = sum(len(r.boxes) for r in results)
    print(f" {nb} objet(s) détecté(s) — appuyez sur une touche pour fermer")

    cv2.imshow("Test YOLO — image statique", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tester_dataset(model) -> None:
    """Teste YOLO sur la première image du dataset."""
    if not os.path.isdir(DATASET_IMGS):
        print(f" Dossier dataset introuvable : {DATASET_IMGS}")
        sys.exit(1)

    images = [f for f in sorted(os.listdir(DATASET_IMGS))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print(" Aucune image trouvée dans le dataset.")
        sys.exit(1)

    print(f" {len(images)} image(s) dans le dataset.")
    tester_image(model, os.path.join(DATASET_IMGS, images[0]))


def tester_camera(model, camera_id: int) -> None:
    """Inférence en temps réel sur le flux caméra."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f" Impossible d'ouvrir la caméra {camera_id}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f" Caméra {camera_id} ouverte — appuyez sur 'q' pour quitter, 's' pour capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Impossible de lire la frame.")
            break

        results  = model.predict(frame, conf=0.25, verbose=False)
        annotated = dessiner_detections(frame, results)

        cv2.imshow("Test YOLO — flux live", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = "yolo_capture.jpg"
            cv2.imwrite(fname, annotated)
            print(f" Capture sauvegardée : {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test du modèle YOLO")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image",   type=str, help="Chemin vers une image à analyser")
    group.add_argument("--camera",  type=int, help="ID de la caméra (ex: 0, 8, 37)")
    group.add_argument("--dataset", action="store_true",
                       help="Utilise la première image du dataset YOLO")
    args = parser.parse_args()

    model_path = choisir_modele()
    model      = charger_modele(model_path)

    if args.image:
        tester_image(model, args.image)
    elif args.camera is not None:
        tester_camera(model, args.camera)
    elif args.dataset:
        tester_dataset(model)
    else:
        # Si aucun argument, lance le mode dataset par défaut
        print(" Aucun argument. Mode dataset par défaut (--image, --camera ou --dataset disponibles)")
        tester_dataset(model)


if __name__ == "__main__":
    main()
