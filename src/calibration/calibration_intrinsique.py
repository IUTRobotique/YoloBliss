"""
Calibration intrinsèque de la caméra
=====================================

Ce script permet de calibrer les paramètres intrinsèques d'une caméra (matrice de calibration, 
coefficients de distorsion) en utilisant un damier (chessboard pattern).

Étapes:
1. Capturer plusieurs images d'un damier sous différents angles
2. Détecter les coins du damier sur chaque image
3. Calculer les paramètres intrinsèques de la caméra
4. Sauvegarder les paramètres pour utilisation ultérieure

Utilisation:
- Imprimez un damier (exemple: 9x6 coins intérieurs)
- Lancez le script et capturez ~20 images du damier sous différents angles
- Appuyez sur 's' pour sauvegarder une image, 'q' pour terminer la capture
- Le script calculera automatiquement les paramètres de calibration
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime

class CameraCalibration:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Initialisation de la calibration de caméra.
        
        Args:
            chessboard_size: Tuple (largeur, hauteur) du nombre de coins intérieurs du damier
            square_size: Taille d'un carré du damier en mm
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Préparer les points 3D du damier (0,0,0), (1,0,0), (2,0,0) ...
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays pour stocker les points 3D et 2D de toutes les images
        self.objpoints = []  # Points 3D dans l'espace réel
        self.imgpoints = []  # Points 2D dans le plan image
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def capture_calibration_images(self, camera_id=5, num_images=20, save_dir='calibration_images'):
        """
        Capture des images du damier pour la calibration.
        
        Args:
            camera_id: ID de la caméra (par défaut 5)
            num_images: Nombre d'images recommandé à capturer
            save_dir: Dossier où sauvegarder les images
        
        Returns:
            Liste des chemins des images capturées
        """
        # Créer le dossier de sauvegarde
        os.makedirs(save_dir, exist_ok=True)
        
        # Ouvrir la caméra
        cam = cv2.VideoCapture(camera_id)
        
        # Configurer la résolution
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Vérifier que la caméra est ouverte
        if not cam.isOpened():
            print(f"❌ Erreur : impossible d'ouvrir la caméra {camera_id}")
            return []
        
        print(f"\n📸 CAPTURE D'IMAGES POUR CALIBRATION")
        print(f"="*60)
        print(f"Instructions:")
        print(f"  - Placez le damier ({self.chessboard_size[0]}x{self.chessboard_size[1]} coins) devant la caméra")
        print(f"  - Appuyez sur 's' pour sauvegarder une image")
        print(f"  - Appuyez sur 'q' pour terminer")
        print(f"  - Objectif: capturer ~{num_images} images sous différents angles")
        print(f"="*60)
        
        captured_images = []
        count = 0
        
        while count < num_images:
            ret, frame = cam.read()
            
            if not ret:
                print("❌ Erreur de lecture de la caméra")
                break
            
            # Convertir en niveaux de gris pour la détection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Chercher les coins du damier
            ret_corners, corners = cv2.findChessboardCorners(
                gray, 
                self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Copie pour l'affichage
            display_frame = frame.copy()
            
            # Si le damier est détecté, dessiner les coins
            if ret_corners:
                cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret_corners)
                status_text = f"Damier detecte! Appuyez sur 's' pour sauvegarder ({count}/{num_images})"
                color = (0, 255, 0)
            else:
                status_text = f"Damier non detecte. Ajustez la position ({count}/{num_images})"
                color = (0, 0, 255)
            
            # Afficher le statut
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Afficher l'image
            cv2.imshow('Calibration Camera - Appuyez sur s pour sauvegarder, q pour quitter', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Sauvegarder l'image si 's' est pressé
            if key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f'calib_{count:03d}_{timestamp}.jpg'
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, frame)
                captured_images.append(filepath)
                count += 1
                if ret_corners:
                    print(f"✅ Image {count}/{num_images} sauvegardée: {filename}")
                else:
                    print(f"⚠️  Image {count}/{num_images} sauvegardée: {filename} (damier non détecté - sera ignorée si toujours invalide)")
            
            # Quitter si 'q' est pressé
            elif key == ord('q'):
                print(f"\n⚠️  Capture interrompue. {count} images capturées.")
                break
        
        # Libérer les ressources
        cam.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Capture terminée: {len(captured_images)} images sauvegardées")
        return captured_images
    
    def calibrate_from_images(self, image_paths):
        """
        Calibre la caméra à partir d'une liste d'images du damier.
        
        Args:
            image_paths: Liste des chemins vers les images de calibration
            
        Returns:
            True si la calibration a réussi, False sinon
        """
        if len(image_paths) == 0:
            print("❌ Aucune image fournie pour la calibration")
            return False
        
        print(f"\n🔧 CALIBRATION EN COURS")
        print(f"="*60)
        print(f"Analyse de {len(image_paths)} images...")
        
        valid_images = 0
        img_shape = None
        
        for img_path in image_paths:
            # Lire l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Impossible de lire: {img_path}")
                continue
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_shape = gray.shape[::-1]
            
            # Trouver les coins du damier
            ret, corners = cv2.findChessboardCorners(
                gray, 
                self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Affiner la détection des coins
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners_refined)
                valid_images += 1
                print(f"  ✅ {os.path.basename(img_path)} - Coins détectés")
            else:
                print(f"  ❌ {os.path.basename(img_path)} - Échec de détection")
        
        if valid_images < 3:
            print(f"\n❌ Pas assez d'images valides ({valid_images}/3 minimum)")
            return False
        
        print(f"\n📊 {valid_images}/{len(image_paths)} images utilisées pour la calibration")
        print(f"⏳ Calcul des paramètres de calibration...")
        
        # Calibrer la caméra
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            img_shape, 
            None, 
            None
        )
        
        if not ret:
            print("❌ Échec de la calibration")
            return False
        
        # Calculer l'erreur de reprojection
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], 
                self.rvecs[i], 
                self.tvecs[i], 
                self.camera_matrix, 
                self.dist_coeffs
            )
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        print(f"\n✅ CALIBRATION RÉUSSIE!")
        print(f"="*60)
        print(f"📐 Matrice de calibration (K):")
        print(self.camera_matrix)
        print(f"\n🔍 Coefficients de distorsion:")
        print(self.dist_coeffs)
        print(f"\n📊 Erreur de reprojection moyenne: {mean_error:.4f} pixels")
        print(f"   (Une erreur < 0.5 pixels est excellente)")
        print(f"="*60)
        
        return True
    
    def save_calibration(self, filename='camera_calibration.pkl'):
        """
        Sauvegarde les paramètres de calibration dans un fichier.
        
        Args:
            filename: Nom du fichier de sauvegarde
        """
        if self.camera_matrix is None:
            print("❌ Aucune calibration à sauvegarder")
            return False
        
        calibration_data = {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'chessboard_size': self.chessboard_size,
            'square_size': self.square_size,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(calibration_data, f)
        
        print(f"\n✅ Calibration sauvegardée dans: {filename}")
        return True
    
    def load_calibration(self, filename='camera_calibration.pkl'):
        """
        Charge les paramètres de calibration depuis un fichier.
        
        Args:
            filename: Nom du fichier de calibration
        """
        if not os.path.exists(filename):
            print(f"❌ Fichier {filename} non trouvé")
            return False
        
        with open(filename, 'rb') as f:
            calibration_data = pickle.load(f)
        
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.chessboard_size = calibration_data['chessboard_size']
        self.square_size = calibration_data['square_size']
        
        print(f"\n✅ Calibration chargée depuis: {filename}")
        print(f"📅 Date de calibration: {calibration_data.get('date', 'Inconnue')}")
        return True
    
    def test_distortion_level(self):
        """
        Évalue numériquement le niveau de distorsion de la caméra.
        
        Returns:
            True si la distorsion est acceptable, False sinon
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("❌ Aucune calibration chargée. Effectuez d'abord la calibration.")
            return False
        
        print(f"\n🔍 ANALYSE DE LA DISTORSION")
        print(f"="*60)
        
        # Extraire les coefficients
        k1, k2, p1, p2, k3 = self.dist_coeffs[0]
        
        print(f"Coefficients de distorsion radiale:")
        print(f"  k1 = {k1:+.6f}")
        print(f"  k2 = {k2:+.6f}")
        print(f"  k3 = {k3:+.6f}")
        print(f"\nCoefficients de distorsion tangentielle:")
        print(f"  p1 = {p1:+.6f}")
        print(f"  p2 = {p2:+.6f}")
        
        # Calculer les métriques
        radial_distortion = np.sqrt(k1**2 + k2**2 + k3**2)
        tangential_distortion = np.sqrt(p1**2 + p2**2)
        max_distortion = np.max(np.abs(self.dist_coeffs))
        
        print(f"\n📊 Métriques:")
        print(f"  Distorsion radiale (RMS)     : {radial_distortion:.6f}")
        print(f"  Distorsion tangentielle (RMS): {tangential_distortion:.6f}")
        print(f"  Distorsion maximale (abs)    : {max_distortion:.6f}")
        
        # Évaluation
        print(f"\n📈 ÉVALUATION:")
        print(f"="*60)
        
        if max_distortion < 0.05:
            print(f"✅ DISTORSION TRÈS FAIBLE - EXCELLENTE")
            print(f"   Impact visuel: négligeable")
            print(f"   Correction: optionnelle")
            verdict = "ACCEPTABLE"
        elif max_distortion < 0.15:
            print(f"✅ DISTORSION FAIBLE - BONNE")
            print(f"   Impact visuel: minime (visible aux bords)")
            print(f"   Correction: recommandée pour mesures précises")
            verdict = "ACCEPTABLE"
        elif max_distortion < 0.3:
            print(f"⚠️  DISTORSION MODÉRÉE")
            print(f"   Impact visuel: visible")
            print(f"   Correction: nécessaire")
            verdict = "ACCEPTABLE avec correction"
        else:
            print(f"❌ DISTORSION ÉLEVÉE")
            print(f"   Impact visuel: important")
            print(f"   Correction: indispensable")
            verdict = "CORRECTION OBLIGATOIRE"
        
        print(f"\n🎯 VERDICT: {verdict}")
        print(f"="*60)
        
        return max_distortion < 0.3  # Acceptable si < 0.3


def main():
    """
    Fonction principale pour effectuer la calibration complète.
    """
    print("\n" + "="*60)
    print("CALIBRATION INTRINSÈQUE DE LA CAMÉRA")
    print("="*60)
    
    # Paramètres du damier
    # IMPORTANT: Ajustez ces valeurs selon votre damier imprimé
    chessboard_size = (9, 6)  # Nombre de coins INTÉRIEURS (largeur, hauteur)
    square_size = 23.0  # Taille d'un carré en mm
    
    print(f"\n📋 Configuration du damier:")
    print(f"   - Taille: {chessboard_size[0]}x{chessboard_size[1]} coins intérieurs")
    print(f"   - Taille des carrés: {square_size} mm")
    print(f"\n💡 Assurez-vous d'avoir imprimé un damier avec ces dimensions!")
    
    # Créer l'objet de calibration
    calib = CameraCalibration(chessboard_size=chessboard_size, square_size=square_size)
    
    # Étape 1: Capturer les images
    print(f"\n{'='*60}")
    print("ÉTAPE 1: CAPTURE DES IMAGES")
    print(f"{'='*60}")
    
    image_paths = calib.capture_calibration_images(
        camera_id=37,
        num_images=20,
        save_dir='./calibration_images/calibration_intrinsique/'
    )
    
    if len(image_paths) < 3:
        print("\n❌ Pas assez d'images capturées. Relancez le script.")
        return
    
    # Étape 2: Calibrer
    print(f"\n{'='*60}")
    print("ÉTAPE 2: CALCUL DE LA CALIBRATION")
    print(f"{'='*60}")
    
    success = calib.calibrate_from_images(image_paths)
    
    if not success:
        print("\n❌ La calibration a échoué.")
        return
    
    # Étape 3: Sauvegarder
    calib.save_calibration('calibration_intrinseque.pkl')
    
    # Étape 4: Test numérique de la distorsion
    print(f"\n{'='*60}")
    print("ÉTAPE 3: ÉVALUATION DE LA DISTORSION")
    print(f"{'='*60}")
    
    calib.test_distortion_level()
    
    print(f"\n{'='*60}")
    print("✅ CALIBRATION TERMINÉE!")
    print(f"{'='*60}")
    print("\nFichiers générés:")
    print("  - calibration_images/ : Images capturées")
    print("  - calibration_intrinseque.pkl : Paramètres de calibration")
    print("\nPour utiliser cette calibration dans vos projets:")
    print("  >>> from calibration_intrinsique import CameraCalibration")
    print("  >>> calib = CameraCalibration()")
    print("  >>> calib.load_calibration('calibration_intrinseque.pkl')")


if __name__ == "__main__":
    main()
