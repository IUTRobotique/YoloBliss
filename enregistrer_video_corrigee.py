"""
Enregistrement de vidéo avec correction de distorsion en temps réel
Basé sur test_camera.py mais applique la calibration intrinsèque
"""
import cv2
import numpy as np
import pickle
import os
from datetime import datetime

def charger_calibration(fichier_pkl='calibration_intrinseque.pkl'):
    """Charge les paramètres de calibration intrinsèque"""
    if not os.path.exists(fichier_pkl):
        raise FileNotFoundError(f"Fichier {fichier_pkl} introuvable. Lancez d'abord calibration_intrinsique.py")
    
    with open(fichier_pkl, 'rb') as f:
        calibration_data = pickle.load(f)
    
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']
    
    print(f"✅ Calibration chargée depuis {fichier_pkl}")
    print(f"   Focale fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}")
    print(f"   Distorsion k1={dist_coeffs[0][0]:.4f}, k2={dist_coeffs[0][1]:.4f}")
    
    return camera_matrix, dist_coeffs


def main():
    # Charger la calibration intrinsèque
    try:
        camera_matrix, dist_coeffs = charger_calibration()
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("   Exécutez d'abord: python calibration_intrinsique.py")
        return
    
    # Demander l'ID de la caméra à l'utilisateur
    camera_id = int(input("Entrez l'ID de la caméra (ex: 37 pour RealSense): "))
    
    # Initialiser la caméra
    cap = cv2.VideoCapture(camera_id)
    
    # Configuration résolution 640x480 (harmonisée avec calibration)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\n📹 Caméra initialisée: {actual_width}x{actual_height} @ {fps} fps")
    
    # Calculer la nouvelle matrice caméra pour l'image corrigée
    # alpha=1 conserve tous les pixels (bords noirs possibles)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, 
        (actual_width, actual_height), 
        alpha=1, 
        newImgSize=(actual_width, actual_height)
    )
    
    # Créer le dossier de sortie si nécessaire
    output_dir = "dataset_localisation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Variable pour l'enregistrement
    is_recording = False
    video_writer = None
    output_file = None
    frame_count = 0
    
    print("\n📺 Aperçu vidéo CORRIGÉE (distorsion supprimée)")
    print("   'r' : Démarrer/arrêter l'enregistrement")
    print("   's' : Sauvegarder une image")
    print("   'q' : Quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erreur lecture caméra")
            break
        
        # Appliquer la correction de distorsion
        frame_corrected = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # Enregistrer la frame sans texte
        if is_recording:
            video_writer.write(frame_corrected)
            frame_count += 1
        
        # Créer une copie pour l'affichage avec le statut
        frame_display = frame_corrected.copy()
        
        # Afficher le statut d'enregistrement seulement dans la prévisualisation
        if is_recording:
            cv2.circle(frame_display, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame_display, f"REC - Frame {frame_count}", (40, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame_display, "PRET - Appuyez sur 'r' pour enregistrer", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Ajouter texte "CORRIGEE" seulement dans la prévisualisation
        cv2.putText(frame_display, "VIDEO CORRIGEE", (10, actual_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Camera RealSense - Corrigee", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            if not is_recording:
                # Démarrer l'enregistrement
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"video_corrigee_{timestamp}.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file, fourcc, fps, 
                                              (actual_width, actual_height))
                
                is_recording = True
                frame_count = 0
                print(f"\n🔴 Enregistrement démarré: {output_file}")
            else:
                # Arrêter l'enregistrement
                is_recording = False
                if video_writer:
                    video_writer.release()
                print(f"⏹️  Enregistrement arrêté: {frame_count} frames sauvegardées dans {output_file}")
                video_writer = None
        
        elif key == ord('s'):
            # Sauvegarder une image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_file = os.path.join(output_dir, f"image_corrigee_{timestamp}.jpg")
            cv2.imwrite(image_file, frame_corrected)
            print(f"📸 Image corrigée sauvegardée: {image_file}")
    
    # Nettoyage
    if is_recording and video_writer:
        video_writer.release()
        print(f"⏹️  Enregistrement final: {frame_count} frames dans {output_file}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Programme terminé")

if __name__ == "__main__":
    main()
