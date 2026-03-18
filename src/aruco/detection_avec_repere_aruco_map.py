#!/usr/bin/env python3
"""
Détection et localisation 3D avec repère monde basé sur feuille A4
===================================================================

Workflow:
1. Détecte les marqueurs ArUco 3, 4, 5, 6 avec cv2.solvePnP
2. Calibre la transformation caméra → repère monde (centre de la feuille A4)
3. Robuste : fonctionne avec minimum 3 marqueurs sur 4 visibles
4. Détecte les objets avec YOLO
5. Transforme les positions du repère caméra vers le repère monde (centre A4)

Le repère monde a son origine au CENTRE de la feuille A4.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import json


class DetectionAvecRepereA4:
    def __init__(self, yolo_model_path, aruco_marker_size, marker_width_distance, marker_height_distance):
        """
        Initialise le système de détection avec repère A4
        
        Args:
            yolo_model_path: Chemin vers le modèle YOLO
            aruco_marker_size: Taille réelle du marqueur ArUco en mètres (ex: 0.06 pour 6cm)
            marker_width_distance: Distance mesurée entre centres des marqueurs gauche-droite en mètres
            marker_height_distance: Distance mesurée entre centres des marqueurs haut-bas en mètres
        """
        # Configuration RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        
        # Démarrage du pipeline
        profile = self.pipeline.start(self.config)
        
        # Alignement depth sur color
        self.align = rs.align(rs.stream.color)
        
        # Récupération des intrinsèques
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Matrice de calibration et coefficients de distorsion pour OpenCV
        self.camera_matrix = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(self.intrinsics.coeffs)
        
        # Modèle YOLO
        self.model = YOLO(yolo_model_path)
        
        # Configuration ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_marker_size = aruco_marker_size
        
        # Points 3D d'un marqueur ArUco (dans son propre repère)
        half_size = aruco_marker_size / 2
        self.obj_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)
        
        # Positions 3D des marqueurs dans le repère A4 (centre = origine)
        # Calculées à partir des distances mesurées entre les centres des marqueurs
        half_width = marker_width_distance / 2
        half_height = marker_height_distance / 2
        
        self.a4_marker_positions = {
            3: np.array([-half_width,  half_height, 0]),  # Coin sup gauche
            4: np.array([ half_width,  half_height, 0]),  # Coin sup droit
            5: np.array([-half_width, -half_height, 0]),  # Coin inf gauche
            6: np.array([ half_width, -half_height, 0])   # Coin inf droit
        }
        
        # Transformation caméra → A4 (sera calculée)
        self.R_cam_to_world = None
        self.T_cam_to_world = None
        self.calibration_error = None
        
        print("✅ Système initialisé")
        print(f"   - Résolution: 640x480 @ 15fps")
        print(f"   - Taille marqueur ArUco: {aruco_marker_size*100} cm")
        print(f"   - Distance marqueurs (largeur): {marker_width_distance*100} cm")
        print(f"   - Distance marqueurs (hauteur): {marker_height_distance*100} cm")
        print(f"   - Repère monde: centre de la configuration des marqueurs")
        print(f"   - Intrinsèques: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")
    
    
    def get_aruco_pose(self, color_image):
        """
        Détecte les marqueurs ArUco et calcule leur pose avec cv2.solvePnP
        
        Returns:
            dict: {marker_id: {'rvec': rotation_vector, 'tvec': translation_vector, 'corners': corners}}
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        poses = {}
        
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                
                # Résout la pose avec solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    marker_corners,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    poses[marker_id] = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': marker_corners
                    }
        
        return poses
    
    
    def calibrate_camera_to_a4(self, aruco_poses):
        """
        Calibre la transformation caméra → repère A4 à partir des marqueurs détectés
        
        Args:
            aruco_poses: Dictionnaire des poses des marqueurs détectés
        
        Returns:
            bool: True si calibration réussie (au moins 3 marqueurs)
        """
        # Filtrer les marqueurs de la feuille A4 (IDs 3, 4, 5, 6)
        valid_markers = {mid: pose for mid, pose in aruco_poses.items() if mid in [3, 4, 5, 6]}
        
        if len(valid_markers) < 3:
            self.R_cam_to_world = None
            self.T_cam_to_world = None
            self.calibration_error = None
            return False
        
        # Collecter les correspondances
        points_a4 = []      # Points 3D dans le repère A4
        points_camera = []  # Points 3D dans le repère caméra
        
        for marker_id, pose_data in valid_markers.items():
            # Position du marqueur dans le repère A4 (connue)
            pos_a4 = self.a4_marker_positions[marker_id]
            points_a4.append(pos_a4)
            
            # Position du marqueur dans le repère caméra (obtenue par solvePnP)
            # tvec donne directement la position du centre du marqueur
            pos_cam = pose_data['tvec'].flatten()
            points_camera.append(pos_cam)
        
        points_a4 = np.array(points_a4, dtype=np.float32)
        points_camera = np.array(points_camera, dtype=np.float32)
        
        # Calculer la transformation avec la méthode de Kabsch (SVD)
        # Trouve R et T qui minimisent ||R * p_cam + T - p_a4||
        
        # Centrer les nuages de points
        centroid_cam = np.mean(points_camera, axis=0)
        centroid_a4 = np.mean(points_a4, axis=0)
        
        centered_cam = points_camera - centroid_cam
        centered_a4 = points_a4 - centroid_a4
        
        # SVD
        H = centered_cam.T @ centered_a4
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Assurer une rotation propre (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculer T
        T = centroid_a4 - R @ centroid_cam
        
        self.R_cam_to_world = R
        self.T_cam_to_world = T.reshape(3, 1)
        
        # Calculer l'erreur de reprojection
        errors = []
        for p_cam, p_a4 in zip(points_camera, points_a4):
            p_a4_estimated = R @ p_cam + T
            error = np.linalg.norm(p_a4_estimated - p_a4)
            errors.append(error)
        
        self.calibration_error = np.mean(errors) * 100  # en cm
        
        return True
    
    
    def get_3d_point(self, u, v, depth_frame):
        """
        Convertit un pixel (u,v) en coordonnées 3D dans le repère caméra
        """
        depth = depth_frame.get_distance(int(u), int(v))
        if depth == 0:
            return None
        
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
        return np.array(point_3d)
    
    
    def get_bbox_3d_center(self, bbox, depth_frame):
        """
        Calcule la position 3D du centre d'une bounding box
        Moyenne la profondeur sur une zone 5x5 pixels pour plus de robustesse
        """
        x1, y1, x2, y2 = map(int, bbox)
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2
        
        # Moyenner la profondeur sur une zone 5x5 autour du centre
        depths = []
        for du in range(-2, 3):
            for dv in range(-2, 3):
                u = center_u + du
                v = center_v + dv
                if 0 <= u < 640 and 0 <= v < 480:
                    depth = depth_frame.get_distance(u, v)
                    if depth > 0:  # Ignorer les valeurs nulles
                        depths.append(depth)
        
        if len(depths) == 0:
            return None
        
        # Utiliser la médiane pour être robuste aux outliers
        median_depth = np.median(depths)
        
        return self.get_3d_point(center_u, center_v, depth_frame) if median_depth > 0 else None
    
    
    def transform_camera_to_world(self, point_camera):
        """
        Transforme un point du repère caméra vers le repère monde (centre A4)
        
        Args:
            point_camera: Point dans le repère caméra (3,) [x, y, z]
        
        Returns:
            point_world: Point dans le repère monde (centre A4)
        """
        if self.R_cam_to_world is None:
            return None
        
        point_world = self.R_cam_to_world @ point_camera + self.T_cam_to_world.flatten()
        return point_world
    
    
    def detect_and_localize(self):
        """
        Boucle principale de détection
        
        Appuyez sur :
        - 'q' pour quitter
        """
        print("\n🎯 Détection en cours...")
        print("   - Marqueurs 3, 4, 5, 6 définissent le repère monde (centre A4)")
        print("   - Minimum 3 marqueurs nécessaires pour la calibration")
        print("   - Rouge : Objets YOLO")
        print("\nCommandes :")
        print("   - 'q' : Quitter")
        
        try:
            self.pipeline.stop()
            self.pipeline.start(self.config)

            while True:
                # Capture des frames
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                display_image = color_image.copy()
                
                # Détection des marqueurs ArUco
                aruco_poses = self.get_aruco_pose(color_image)
                
                # Calibration de la transformation caméra → A4
                calibration_success = self.calibrate_camera_to_a4(aruco_poses)
                
                info_text = []
                
                # Affichage des marqueurs détectés
                for marker_id, pose_data in aruco_poses.items():
                    if marker_id not in [3, 4, 5, 6]:
                        continue
                    
                    rvec = pose_data['rvec']
                    tvec = pose_data['tvec']
                    corners = pose_data['corners']
                    
                    # Couleur selon l'état
                    if calibration_success:
                        color = (0, 255, 0)  # Vert si calibration OK
                    else:
                        color = (0, 165, 255)  # Orange si calibration impossible
                    
                    # Dessine le contour
                    pts = corners.astype(int)
                    cv2.polylines(display_image, [pts], True, color, 2)
                    
                    # Dessine les axes du marqueur
                    cv2.drawFrameAxes(display_image, self.camera_matrix, self.dist_coeffs, 
                                     rvec, tvec, self.aruco_marker_size * 0.5)
                    
                    # Position dans le repère caméra
                    pos_cam = tvec.flatten()
                    
                    # Affiche les infos
                    center = tuple(pts.mean(axis=0).astype(int))
                    cv2.putText(display_image, f"ID {marker_id}", 
                               (center[0] - 30, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    info_text.append(f"Marqueur {marker_id}: Camera({pos_cam[0]*100:.1f}, {pos_cam[1]*100:.1f}, {pos_cam[2]*100:.1f}) cm")
                
                # Affichage du statut de calibration
                if calibration_success:
                    status_text = f"CALIBRATION OK - {len([m for m in aruco_poses.keys() if m in [3,4,5,6]])}/4 marqueurs - Erreur: {self.calibration_error:.2f}cm"
                    status_color = (0, 255, 0)
                else:
                    detected_count = len([m for m in aruco_poses.keys() if m in [3,4,5,6]])
                    status_text = f"CALIBRATION IMPOSSIBLE - {detected_count}/4 marqueurs (min 3 requis)"
                    status_color = (0, 0, 255)
                
                cv2.putText(display_image, status_text,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # Détection YOLO uniquement si calibration réussie
                if calibration_success:
                    results = self.model(color_image, verbose=False)
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Bounding box
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]
                            
                            # Position 3D dans le repère caméra
                            point_camera = self.get_bbox_3d_center([x1, y1, x2, y2], depth_frame)
                            
                            if point_camera is not None:
                                # Transformation vers le repère monde (centre A4)
                                point_monde = self.transform_camera_to_world(point_camera)
                                
                                if point_monde is not None:
                                    # Affichage
                                    cv2.rectangle(display_image, (int(x1), int(y1)), 
                                                (int(x2), int(y2)), (0, 0, 255), 2)
                                    
                                    # Texte avec positions
                                    label_cam = f"{class_name} {conf:.2f}"
                                    label_pos_cam = f"Cam: ({point_camera[0]*100:.1f}, {point_camera[1]*100:.1f}, {point_camera[2]*100:.1f}) cm"
                                    label_pos_monde = f"A4: ({point_monde[0]*100:.1f}, {point_monde[1]*100:.1f}, {point_monde[2]*100:.1f}) cm"
                                    
                                    cv2.putText(display_image, label_cam,
                                              (int(x1), int(y1) - 40),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    cv2.putText(display_image, label_pos_cam,
                                              (int(x1), int(y1) - 20),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                    cv2.putText(display_image, label_pos_monde,
                                              (int(x1), int(y1) - 5),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                                    
                                    info_text.append(f"{class_name}: Camera{tuple(point_camera*100)} cm -> A4{tuple(point_monde*100)} cm")
                
                # Affichage des infos
                y_offset = 60
                for text in info_text:
                    cv2.putText(display_image, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    y_offset += 20
                
                # Affichage
                cv2.imshow('Detection avec repere A4', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


def main():
    print("="*70)
    print("DÉTECTION 3D AVEC REPÈRE MONDE = CENTRE FEUILLE A4")
    print("="*70)
    
    # Configuration
    yolo_model = '../../runs/detect/detection_objets/weights/best.pt'
    
    print("\n📏 MESURES NÉCESSAIRES :")
    print("   1. Taille d'un marqueur ArUco (côté du carré NOIR)")
    print("   2. Distance entre centres des marqueurs GAUCHE ↔ DROITE (largeur)")
    print("   3. Distance entre centres des marqueurs HAUT ↔ BAS (hauteur)")
    
    # Taille des marqueurs
    marker_size_input = input("\n1️⃣ Taille d'un marqueur en cm (ex: 6): ").strip()
    try:
        marker_size_cm = float(marker_size_input)
        marker_size_m = marker_size_cm / 100.0
    except:
        print("❌ Taille invalide, utilisation de 6 cm par défaut")
        marker_size_m = 0.06
    
    # Distance largeur (gauche-droite)
    width_input = input("2️⃣ Distance entre centres GAUCHE ↔ DROITE en cm (ex: 24.6): ").strip()
    try:
        width_cm = float(width_input)
        width_m = width_cm / 100.0
    except:
        print("❌ Distance invalide, utilisation de 24.6 cm par défaut")
        width_m = 0.246
    
    # Distance hauteur (haut-bas)
    height_input = input("3️⃣ Distance entre centres HAUT ↔ BAS en cm (ex: 16): ").strip()
    try:
        height_cm = float(height_input)
        height_m = height_cm / 100.0
    except:
        print("❌ Distance invalide, utilisation de 16 cm par défaut")
        height_m = 0.16
    
    print(f"\n✅ Configuration:")
    print(f"   - Modèle YOLO: {yolo_model}")
    print(f"   - Taille marqueurs: {marker_size_m*100} cm")
    print(f"   - Distance largeur: {width_m*100} cm")
    print(f"   - Distance hauteur: {height_m*100} cm")
    print(f"   - Repère monde: CENTRE de la configuration des marqueurs")
    print(f"   - Marqueurs utilisés: 3, 4, 5, 6 (minimum 3 visibles)")
    print(f"   - Robustesse: fonctionne même si 1 marqueur est masqué")
    
    # Lancement
    detector = DetectionAvecRepereA4(yolo_model, marker_size_m, width_m, height_m)
    detector.detect_and_localize()


if __name__ == "__main__":
    main()
