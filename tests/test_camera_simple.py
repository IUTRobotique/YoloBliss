#!/usr/bin/env python3
"""Test simple de caméra"""

import cv2
import sys

camera_id = 8  # Changez à 5 pour tester la RealSense

print(f"Ouverture de la caméra {camera_id}...")

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
    sys.exit(1)

print("✅ Caméra ouverte")
print("Appuyez sur 'q' pour quitter")

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Erreur de lecture")
            break
        
        # Afficher les informations sur la frame
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Camera {camera_id} - {w}x{h}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera {camera_id}', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\n⚠️ Interrompu par l'utilisateur")
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Caméra fermée")
