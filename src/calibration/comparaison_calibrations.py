"""
Comparaison Calibration Custom vs Calibration RealSense
========================================================

Ce script compare les paramètres intrinsèques obtenus par:
1. Calibration CUSTOM avec damier (calibration_intrinseque.pkl)
2. Calibration USINE de la RealSense (realsense_calibration.json)

Objectifs pédagogiques:
- Comprendre les différences entre calibration custom et fabricant
- Évaluer la qualité de sa propre calibration
- Visualiser l'impact des différences sur une image réelle
"""

import cv2
import numpy as np
import pickle
import json
import pyrealsense2 as rs


class ComparaisonCalibrations:
    def __init__(self, custom_calib_file='calibration_intrinseque.pkl',
                 realsense_calib_file='realsense_calibration.json'):
        """
        Initialise la comparaison entre les deux calibrations.
        
        Args:
            custom_calib_file: Fichier de calibration custom (damier)
            realsense_calib_file: Fichier de calibration RealSense
        """
        self.custom_calib_file = custom_calib_file
        self.realsense_calib_file = realsense_calib_file
        
        # Charger les deux calibrations
        self.load_calibrations()
        
    def load_calibrations(self):
        """Charge les deux fichiers de calibration."""
        print("📂 Chargement des calibrations...")
        print("="*70)
        
        # 1. Calibration CUSTOM (damier)
        try:
            with open(self.custom_calib_file, 'rb') as f:
                custom_data = pickle.load(f)
            
            self.custom_K = custom_data['camera_matrix']
            self.custom_dist = custom_data['dist_coeffs']
            
            print(f"✅ Calibration CUSTOM chargée: {self.custom_calib_file}")
            print(f"   Date: {custom_data.get('date', 'Inconnue')}")
        except FileNotFoundError:
            print(f"❌ Fichier {self.custom_calib_file} non trouvé!")
            print(f"   Lancez d'abord: python calibration_intrinsique.py")
            return False
        
        # 2. Calibration REALSENSE (usine)
        try:
            with open(self.realsense_calib_file, 'r') as f:
                realsense_data = json.load(f)
            
            # Extraire les paramètres RGB
            rgb_data = realsense_data['color']
            
            # Construire la matrice K
            self.realsense_K = np.array(rgb_data['matrix_K'], dtype=np.float64)
            self.realsense_dist = np.array([rgb_data['coeffs']], dtype=np.float64)
            
            print(f"✅ Calibration REALSENSE chargée: {self.realsense_calib_file}")
            print(f"   Modèle: {rgb_data['model']}")
        except FileNotFoundError:
            print(f"❌ Fichier {self.realsense_calib_file} non trouvé!")
            print(f"   Lancez d'abord: python get_realsense_intrinsics.py")
            return False
        
        print("="*70)
        return True
    
    def compare_matrices(self):
        """Compare les matrices de calibration K."""
        print("\n📊 COMPARAISON DES MATRICES DE CALIBRATION K")
        print("="*70)
        
        print("\n🔧 CALIBRATION CUSTOM (Damier):")
        print(self.custom_K)
        
        print("\n🏭 CALIBRATION REALSENSE (Usine):")
        print(self.realsense_K)
        
        # Différence absolue
        diff_K = np.abs(self.custom_K - self.realsense_K)
        print("\n📉 DIFFÉRENCE ABSOLUE:")
        print(diff_K)
        
        # Extraire les paramètres importants
        custom_fx, custom_fy = self.custom_K[0, 0], self.custom_K[1, 1]
        custom_cx, custom_cy = self.custom_K[0, 2], self.custom_K[1, 2]
        
        realsense_fx, realsense_fy = self.realsense_K[0, 0], self.realsense_K[1, 1]
        realsense_cx, realsense_cy = self.realsense_K[0, 2], self.realsense_K[1, 2]
        
        # Calculer les différences en pourcentage
        print("\n📈 DIFFÉRENCES RELATIVES (en %):")
        print(f"   Focale fx : {abs(custom_fx - realsense_fx) / realsense_fx * 100:6.2f}%  "
              f"(Custom: {custom_fx:.2f} | RealSense: {realsense_fx:.2f})")
        print(f"   Focale fy : {abs(custom_fy - realsense_fy) / realsense_fy * 100:6.2f}%  "
              f"(Custom: {custom_fy:.2f} | RealSense: {realsense_fy:.2f})")
        print(f"   Centre cx : {abs(custom_cx - realsense_cx) / realsense_cx * 100:6.2f}%  "
              f"(Custom: {custom_cx:.2f} | RealSense: {realsense_cx:.2f})")
        print(f"   Centre cy : {abs(custom_cy - realsense_cy) / realsense_cy * 100:6.2f}%  "
              f"(Custom: {custom_cy:.2f} | RealSense: {realsense_cy:.2f})")
        
        # Norme de la différence
        diff_norm = np.linalg.norm(diff_K)
        print(f"\n📏 Norme de la différence: {diff_norm:.2f}")
        
        return diff_K
    
    def compare_distortion(self):
        """Compare les coefficients de distorsion."""
        print("\n🔍 COMPARAISON DES COEFFICIENTS DE DISTORSION")
        print("="*70)
        
        print("\n🔧 CALIBRATION CUSTOM (Damier):")
        print(f"   {self.custom_dist}")
        
        print("\n🏭 CALIBRATION REALSENSE (Usine):")
        print(f"   {self.realsense_dist}")
        
        # Différence
        diff_dist = np.abs(self.custom_dist - self.realsense_dist)
        print("\n📉 DIFFÉRENCE ABSOLUE:")
        print(f"   {diff_dist}")
        
        # Analyse des coefficients
        custom_k1, custom_k2, custom_p1, custom_p2, custom_k3 = self.custom_dist[0]
        realsense_k1, realsense_k2, realsense_p1, realsense_p2, realsense_k3 = self.realsense_dist[0]
        
        print("\n📊 ANALYSE DÉTAILLÉE:")
        print(f"   k1 (radial 1)      : Custom={custom_k1:+.6f} | RealSense={realsense_k1:+.6f} | Diff={abs(custom_k1-realsense_k1):.6f}")
        print(f"   k2 (radial 2)      : Custom={custom_k2:+.6f} | RealSense={realsense_k2:+.6f} | Diff={abs(custom_k2-realsense_k2):.6f}")
        print(f"   k3 (radial 3)      : Custom={custom_k3:+.6f} | RealSense={realsense_k3:+.6f} | Diff={abs(custom_k3-realsense_k3):.6f}")
        print(f"   p1 (tangentiel 1)  : Custom={custom_p1:+.6f} | RealSense={realsense_p1:+.6f} | Diff={abs(custom_p1-realsense_p1):.6f}")
        print(f"   p2 (tangentiel 2)  : Custom={custom_p2:+.6f} | RealSense={realsense_p2:+.6f} | Diff={abs(custom_p2-realsense_p2):.6f}")
        
        # Distorsion maximale
        max_custom = np.max(np.abs(self.custom_dist))
        max_realsense = np.max(np.abs(self.realsense_dist))
        
        print(f"\n📏 DISTORSION MAXIMALE:")
        print(f"   Custom    : {max_custom:.6f}")
        print(f"   RealSense : {max_realsense:.6f}")
        
        return diff_dist
    
    def test_visual_comparison(self, camera_id=37):
        """
        Compare visuellement les deux corrections sur une image en temps réel.
        
        Args:
            camera_id: ID de la caméra
        """
        print("\n🎥 TEST VISUEL DE COMPARAISON")
        print("="*70)
        print("Ouverture de la caméra pour comparaison visuelle...")
        print("Touches:")
        print("  'q' = quitter")
        print("  's' = sauvegarder une capture de comparaison")
        print("="*70)
        
        # Ouvrir la caméra
        cam = cv2.VideoCapture(camera_id)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        if not cam.isOpened():
            print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
            return
        
        # Lire une frame pour obtenir la résolution
        ret, frame = cam.read()
        if not ret:
            print("❌ Erreur de lecture caméra")
            return
        
        h, w = frame.shape[:2]
        
        # Préparer les corrections
        print(f"⏳ Calcul des maps de correction pour résolution {w}x{h}...")
        
        # Custom
        new_K_custom, roi_custom = cv2.getOptimalNewCameraMatrix(
            self.custom_K, self.custom_dist, (w, h), 1, (w, h)
        )
        mapx_custom, mapy_custom = cv2.initUndistortRectifyMap(
            self.custom_K, self.custom_dist, None, new_K_custom, (w, h), cv2.CV_32FC1
        )
        
        # RealSense
        new_K_realsense, roi_realsense = cv2.getOptimalNewCameraMatrix(
            self.realsense_K, self.realsense_dist, (w, h), 1, (w, h)
        )
        mapx_realsense, mapy_realsense = cv2.initUndistortRectifyMap(
            self.realsense_K, self.realsense_dist, None, new_K_realsense, (w, h), cv2.CV_32FC1
        )
        
        print("✅ Maps de correction calculées")
        
        capture_count = 0
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            # Appliquer les deux corrections
            corrected_custom = cv2.remap(frame, mapx_custom, mapy_custom, cv2.INTER_LINEAR)
            corrected_realsense = cv2.remap(frame, mapx_realsense, mapy_realsense, cv2.INTER_LINEAR)
            
            # Calculer la différence visuelle
            diff_image = cv2.absdiff(corrected_custom, corrected_realsense)
            diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
            diff_amplified = np.clip(diff_gray * 10, 0, 255).astype(np.uint8)
            diff_color = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_JET)
            
            # Affichage en grille 2x2
            top_row = np.hstack([frame, corrected_custom])
            bottom_row = np.hstack([corrected_realsense, diff_color])
            comparison = np.vstack([top_row, bottom_row])
            
            # Redimensionner pour affichage
            comparison = cv2.resize(comparison, (1280, 720))
            
            # Ajouter les labels
            cv2.putText(comparison, "ORIGINAL", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "CUSTOM (Damier)", (650, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, "REALSENSE (Usine)", (10, 390), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(comparison, "DIFFERENCE x10", (650, 390), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow('Comparaison Calibrations - q pour quitter, s pour sauvegarder', comparison)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'comparaison_calibration_{capture_count:03d}.jpg'
                cv2.imwrite(filename, comparison)
                print(f"📸 Capture sauvegardée: {filename}")
                capture_count += 1
        
        cam.release()
        cv2.destroyAllWindows()
    
    def generate_report(self):
        """Génère un rapport de comparaison complet."""
        print("\n" + "="*70)
        print("📋 RAPPORT DE COMPARAISON")
        print("="*70)
        
        # Calculer les métriques
        diff_K = np.linalg.norm(self.custom_K - self.realsense_K)
        diff_dist = np.linalg.norm(self.custom_dist - self.realsense_dist)
        
        custom_fx = self.custom_K[0, 0]
        realsense_fx = self.realsense_K[0, 0]
        focal_diff_percent = abs(custom_fx - realsense_fx) / realsense_fx * 100
        
        max_custom_dist = np.max(np.abs(self.custom_dist))
        max_realsense_dist = np.max(np.abs(self.realsense_dist))
        
        print("\n🎯 CONCLUSIONS:")
        print("-" * 70)
        
        # Évaluation de la différence de focale
        if focal_diff_percent < 5:
            print(f"✅ Focales très proches ({focal_diff_percent:.2f}% de différence)")
            print(f"   → La calibration custom est cohérente avec l'usine")
        elif focal_diff_percent < 15:
            print(f"⚠️  Focales modérément différentes ({focal_diff_percent:.2f}% de différence)")
            print(f"   → Vérifiez la qualité de votre calibration custom")
        else:
            print(f"❌ Focales très différentes ({focal_diff_percent:.2f}% de différence)")
            print(f"   → Refaites la calibration custom avec plus d'images")
        
        # Évaluation de la distorsion
        print(f"\n🔍 Distorsion détectée:")
        print(f"   Custom    : {max_custom_dist:.6f}")
        print(f"   RealSense : {max_realsense_dist:.6f}")
        
        if max_realsense_dist < 0.01 and max_custom_dist > 0.1:
            print(f"   💡 RealSense corrige la distorsion en interne!")
            print(f"      Votre calibration custom mesure la distorsion réelle de l'objectif")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        print("-" * 70)
        
        if max_custom_dist > 0.2:
            print("   ✓ Utilisez la calibration CUSTOM pour des mesures précises")
            print("   ✓ La correction de distorsion améliorera la précision")
        else:
            print("   ✓ La calibration REALSENSE est suffisante pour la plupart des cas")
            print("   ✓ La calibration CUSTOM peut apporter une légère amélioration")
        
        print("="*70)


def main():
    """Fonction principale."""
    print("\n" + "="*70)
    print("COMPARAISON CALIBRATION CUSTOM vs REALSENSE")
    print("="*70)
    
    # Créer l'objet de comparaison
    comp = ComparaisonCalibrations(
        custom_calib_file='calibration_intrinseque.pkl',
        realsense_calib_file='realsense_calibration.json'
    )
    
    # Comparer les matrices
    comp.compare_matrices()
    
    # Comparer la distorsion
    comp.compare_distortion()
    
    # Générer le rapport
    comp.generate_report()
    
    # Test visuel
    print("\n" + "="*70)
    print("Voulez-vous voir la comparaison visuelle en temps réel? (o/n): ", end='')
    try:
        response = input().lower()
        if response == 'o':
            comp.test_visual_comparison(camera_id=37)
    except:
        print("Mode non-interactif, test visuel ignoré.")
    
    print("\n✅ Comparaison terminée!")


if __name__ == "__main__":
    main()
