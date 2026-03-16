"""
Récupération des paramètres intrinsèques de la caméra RealSense
================================================================

Ce script montre comment accéder aux paramètres intrinsèques (matrice K, 
coefficients de distorsion) directement depuis la caméra RealSense.

Pas besoin de calibration manuelle avec une mire - RealSense est pré-calibré !
"""

import pyrealsense2 as rs
import numpy as np
import json

def get_realsense_intrinsics():
    """
    Récupère les paramètres intrinsèques de la caméra RealSense.
    
    Returns:
        dict: Paramètres intrinsèques pour RGB et Depth
    """
    # Créer le pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configurer les flux RGB et Depth
    # Note: 640x480 @ 30fps (résolution native RealSense)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Démarrer le streaming
    print("🔧 Démarrage de la caméra RealSense...")
    profile = pipeline.start(config)
    
    # Attendre quelques frames pour stabiliser
    for _ in range(5):
        pipeline.wait_for_frames()
    
    # Récupérer les profils de streaming
    color_profile = profile.get_stream(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)
    
    # Récupérer les intrinsèques
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    
    # Arrêter le pipeline
    pipeline.stop()
    
    print("\n✅ Paramètres intrinsèques récupérés avec succès!\n")
    
    return color_intrinsics, depth_intrinsics


def display_intrinsics(intrinsics, stream_name="Camera"):
    """
    Affiche les paramètres intrinsèques de manière lisible.
    
    Args:
        intrinsics: Objet intrinsics de pyrealsense2
        stream_name: Nom du flux (RGB ou Depth)
    """
    print(f"{'='*60}")
    print(f"PARAMÈTRES INTRINSÈQUES - {stream_name}")
    print(f"{'='*60}")
    
    print(f"\n📐 Résolution:")
    print(f"   Largeur  : {intrinsics.width} pixels")
    print(f"   Hauteur  : {intrinsics.height} pixels")
    
    print(f"\n🎯 Focales (en pixels):")
    print(f"   fx : {intrinsics.fx:.2f} pixels")
    print(f"   fy : {intrinsics.fy:.2f} pixels")
    
    print(f"\n📍 Centre optique (point principal):")
    print(f"   cx : {intrinsics.ppx:.2f} pixels")
    print(f"   cy : {intrinsics.ppy:.2f} pixels")
    
    print(f"\n🔍 Modèle de distorsion: {intrinsics.model}")
    print(f"   Coefficients: {intrinsics.coeffs}")
    
    # Construire la matrice K (matrice de calibration)
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    
    print(f"\n📊 Matrice de calibration K:")
    print(K)
    print()
    
    return K


def save_intrinsics_to_file(color_intr, depth_intr, filename="realsense_calibration.json"):
    """
    Sauvegarde les paramètres intrinsèques dans un fichier JSON.
    
    Args:
        color_intr: Intrinsèques de la caméra RGB
        depth_intr: Intrinsèques de la caméra Depth
        filename: Nom du fichier de sauvegarde
    """
    calibration_data = {
        'color': {
            'width': color_intr.width,
            'height': color_intr.height,
            'fx': color_intr.fx,
            'fy': color_intr.fy,
            'ppx': color_intr.ppx,
            'ppy': color_intr.ppy,
            'model': str(color_intr.model),
            'coeffs': color_intr.coeffs,
            'matrix_K': [
                [color_intr.fx, 0, color_intr.ppx],
                [0, color_intr.fy, color_intr.ppy],
                [0, 0, 1]
            ]
        },
        'depth': {
            'width': depth_intr.width,
            'height': depth_intr.height,
            'fx': depth_intr.fx,
            'fy': depth_intr.fy,
            'ppx': depth_intr.ppx,
            'ppy': depth_intr.ppy,
            'model': str(depth_intr.model),
            'coeffs': depth_intr.coeffs,
            'matrix_K': [
                [depth_intr.fx, 0, depth_intr.ppx],
                [0, depth_intr.fy, depth_intr.ppy],
                [0, 0, 1]
            ]
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"💾 Paramètres sauvegardés dans: {filename}")


def demo_pixel_to_3d(color_intrinsics):
    """
    Démonstration : Comment convertir un pixel en coordonnées 3D.
    
    Args:
        color_intrinsics: Paramètres intrinsèques de la caméra RGB
    """
    print(f"\n{'='*60}")
    print("DÉMONSTRATION : PIXEL → COORDONNÉES 3D")
    print(f"{'='*60}\n")
    
    print("Pour un pixel (u, v) avec une profondeur Z (en mètres):")
    print("Les coordonnées 3D (X, Y, Z) dans le repère caméra sont:\n")
    print("  X = (u - cx) * Z / fx")
    print("  Y = (v - cy) * Z / fy")
    print("  Z = profondeur_en_metres\n")
    
    # Exemple concret
    u, v = 320, 240  # Centre de l'image (exemple)
    Z = 1.5  # 1.5 mètres de profondeur
    
    X = (u - color_intrinsics.ppx) * Z / color_intrinsics.fx
    Y = (v - color_intrinsics.ppy) * Z / color_intrinsics.fy
    
    print(f"📌 Exemple:")
    print(f"   Pixel (u={u}, v={v}) avec profondeur Z={Z}m")
    print(f"   → Coordonnées 3D: X={X:.3f}m, Y={Y:.3f}m, Z={Z}m")
    print(f"\n💡 C'est ce principe que vous utiliserez avec les bounding boxes YOLO!")


def main():
    """
    Fonction principale.
    """
    print("\n" + "="*60)
    print("RÉCUPÉRATION DES PARAMÈTRES INTRINSÈQUES REALSENSE")
    print("="*60 + "\n")
    
    try:
        # Récupérer les paramètres
        color_intr, depth_intr = get_realsense_intrinsics()
        
        # Afficher les paramètres RGB
        K_color = display_intrinsics(color_intr, "CAMÉRA RGB")
        
        # Afficher les paramètres Depth
        K_depth = display_intrinsics(depth_intr, "CAMÉRA DEPTH")
        
        # Sauvegarder dans un fichier
        save_intrinsics_to_file(color_intr, depth_intr)
        
        # Démonstration de conversion pixel → 3D
        demo_pixel_to_3d(color_intr)
        
        print(f"\n{'='*60}")
        print("✅ TERMINÉ!")
        print(f"{'='*60}\n")
        
        print("📝 NOTES IMPORTANTES:")
        print("   1. Ces paramètres sont pré-calibrés en usine par Intel")
        print("   2. Pas besoin de calibration manuelle avec une mire")
        print("   3. Utilisez rs.align() pour synchroniser RGB et Depth")
        print("   4. Pour YOLO: détectez sur RGB, puis utilisez Depth aligné\n")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Vérifiez que:")
        print("   - La caméra RealSense est bien connectée")
        print("   - pyrealsense2 est installé (pip install pyrealsense2)")


if __name__ == "__main__":
    main()
