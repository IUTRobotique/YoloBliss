"""
Génération des marqueurs ArUco pour la détection 3D
====================================================

Génère 4 marqueurs ArUco (IDs 3, 4, 5, 6) disposés aux 4 coins d'une feuille A4 paysage.
Le marqueur 6 sert de repère monde dans detection_avec_repere_aruco.py.
"""

import cv2
import numpy as np

def generate_aruco_marker(aruco_id, size_pixels=200):
    """
    Génère un marqueur ArUco.
    
    Args:
        aruco_id: ID du marqueur
        size_pixels: Taille en pixels
    
    Returns:
        Image du marqueur avec marge
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, aruco_id, size_pixels)
    
    # Ajouter une marge blanche
    margin = 20
    marker_with_margin = np.ones((size_pixels + 2*margin, size_pixels + 2*margin), dtype=np.uint8) * 255
    marker_with_margin[margin:margin+size_pixels, margin:margin+size_pixels] = marker_image
    
    # Ajouter le texte ID
    text = f"ID {aruco_id}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (marker_with_margin.shape[1] - text_size[0]) // 2
    text_y = size_pixels + margin + 25
    
    cv2.putText(marker_with_margin, text, 
               (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    
    return marker_with_margin


def create_a4_marker_sheet():
    """
    Crée une planche A4 paysage avec 4 marqueurs ArUco aux coins.
    """
    print("\n" + "="*70)
    print("GÉNÉRATION DES MARQUEURS ARUCO - PLANCHE A4")
    print("="*70)
    
    print("\n📋 Ce script génère 4 marqueurs ArUco :")
    print("  - ID=3, 4, 5, 6 disposés aux 4 coins d'une A4 paysage")
    print("  - Marqueur 6 = REPÈRE MONDE pour detection_avec_repere_aruco.py")
    
    # Dimensions A4 paysage à 300 DPI
    # A4 = 297mm x 210mm
    # 300 DPI = 11.8 pixels/mm
    a4_width_px = int(297 * 11.8)   # ~3508 pixels
    a4_height_px = int(210 * 11.8)  # ~2480 pixels
    
    # Créer une page blanche
    sheet = np.ones((a4_height_px, a4_width_px), dtype=np.uint8) * 255
    
    # Taille des marqueurs (environ 60mm = 700px)
    marker_size = 700
    
    # Générer les 4 marqueurs
    markers = {
        3: generate_aruco_marker(3, marker_size),
        4: generate_aruco_marker(4, marker_size),
        5: generate_aruco_marker(5, marker_size),
        6: generate_aruco_marker(6, marker_size)
    }
    
    # Marges depuis les bords
    margin_x = 150
    margin_y = 150
    
    # Positionner les marqueurs aux 4 coins
    # Coin supérieur gauche - ID 3
    y1, x1 = margin_y, margin_x
    sheet[y1:y1+markers[3].shape[0], x1:x1+markers[3].shape[1]] = markers[3]
    
    # Coin supérieur droit - ID 4
    y2, x2 = margin_y, a4_width_px - markers[4].shape[1] - margin_x
    sheet[y2:y2+markers[4].shape[0], x2:x2+markers[4].shape[1]] = markers[4]
    
    # Coin inférieur gauche - ID 5
    y3, x3 = a4_height_px - markers[5].shape[0] - margin_y, margin_x
    sheet[y3:y3+markers[5].shape[0], x3:x3+markers[5].shape[1]] = markers[5]
    
    # Coin inférieur droit - ID 6 (REPÈRE MONDE)
    y4, x4 = a4_height_px - markers[6].shape[0] - margin_y, a4_width_px - markers[6].shape[1] - margin_x
    sheet[y4:y4+markers[6].shape[0], x4:x4+markers[6].shape[1]] = markers[6]
    
    # Ajouter le titre au centre
    title = "Marqueurs ArUco - Detection 3D"
    subtitle = "Marqueur 6 = REPERE MONDE"
    instructions = "Imprimez en A4 paysage - Mesurez la taille du carre noir apres impression"
    
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    
    center_x = a4_width_px // 2
    center_y = a4_height_px // 2
    
    cv2.putText(sheet, title, 
               (center_x - title_size[0]//2, center_y - 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    cv2.putText(sheet, subtitle, 
               (center_x - subtitle_size[0]//2, center_y + 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
    cv2.putText(sheet, instructions, 
               (center_x - instr_size[0]//2, center_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    # Sauvegarder
    filename = 'aruco_markers_a4.png'
    cv2.imwrite(filename, sheet)
    
    print(f"\n✅ Fichier généré : {filename}")
    print(f"   Dimensions : {a4_width_px} x {a4_height_px} pixels (300 DPI)")
    print(f"   Format : A4 paysage (297mm x 210mm)")
    
    print(f"\n🖨️  IMPRESSION :")
    print(f"   1. Ouvrez {filename}")
    print(f"   2. Imprimez en mode PAYSAGE sur papier A4")
    print(f"   3. Vérifiez 'Taille réelle' ou '100%' (pas de mise à l'échelle)")
    print(f"   4. Mesurez le côté du carré NOIR d'un marqueur")
    
    print(f"\n📏 UTILISATION :")
    print(f"   Dans detection_avec_repere_aruco.py :")
    print(f"   - Le marqueur 6 définit le REPÈRE MONDE (origine)")
    print(f"   - Les autres marqueurs peuvent servir de références")
    print(f"   - Entrez la taille mesurée du carré noir (ex: 5.2 cm)")
    
    print(f"\n💡 AVANTAGES DES 4 MARQUEURS :")
    print(f"   ✓ Calibration plus robuste avec plusieurs points")
    print(f"   ✓ Redondance si un marqueur est caché")
    print(f"   ✓ Validation de la cohérence de la détection")
    
    print(f"\n🎯 MARQUEUR 6 = ORIGINE DU REPÈRE MONDE")
    print(f"   Placez le marqueur 6 à l'endroit que vous voulez comme origine (0,0,0)")
    print(f"   Les positions des objets seront données par rapport au centre du marqueur 6")
    
    print("\n✅ Prêt pour la détection 3D !")
    print("   Lancez : python detection_avec_repere_aruco.py\n")


if __name__ == "__main__":
    create_a4_marker_sheet()
