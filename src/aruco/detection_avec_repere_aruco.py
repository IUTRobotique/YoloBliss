#!/usr/bin/env python3
"""
Détection et localisation 3D avec repère monde basé sur marqueur ArUco

Workflow:
1. Détecte le marqueur ArUco 6 avec cv2.solvePnP (vision pure)
2. Utilise le marqueur 6 comme origine du repère monde
3. Détecte le cylindre avec YOLO
4. Calcule position 3D du cylindre avec RealSense depth
5. Transforme la position du cylindre du repère caméra vers le repère du marqueur 6
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import json


class DetectionAvecRepereArUco:
    def __init__(self, yolo_model_path, aruco_marker_size):
        """
        Initialise le système de détection avec repère ArUco
        
        Args:
            yolo_model_path: Chemin vers le modèle YOLO
            aruco_marker_size: Taille réelle du marqueur ArUco en mètres (ex: 0.05 pour 5cm)
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
        
        # Points 3D du marqueur ArUco (dans son propre repère)
        # Origine au centre, coordonnées dans le sens horaire depuis coin sup-gauche
        half_size = aruco_marker_size / 2
        self.obj_points = np.array([
            [-half_size,  half_size, 0],  # Coin supérieur gauche
            [ half_size,  half_size, 0],  # Coin supérieur droit
            [ half_size, -half_size, 0],  # Coin inférieur droit
            [-half_size, -half_size, 0]   # Coin inférieur gauche
        ], dtype=np.float32)
        
        print("✅ Système initialisé")
        print(f"   - Résolution: 640x480 @ 15fps")
        print(f"   - Taille marqueur ArUco: {aruco_marker_size*100} cm")
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
                marker_corners = corners[i][0]  # Shape (4, 2)
                
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
        
        # Mesurer la profondeur du pixel central d'abord
        center_depth = depth_frame.get_distance(center_u, center_v)
        
        if center_depth == 0:
            return None
        
        # Collecter les profondeurs dans une zone 5x5 autour du centre
        depths = []
        for du in range(-2, 3):
            for dv in range(-2, 3):
                u = center_u + du
                v = center_v + dv
                if 0 <= u < 640 and 0 <= v < 480:
                    depth = depth_frame.get_distance(u, v)
                    if depth > 0:
                        depths.append(depth)
        
        if len(depths) < 5:  # Pas assez de mesures valides
            return self.get_3d_point(center_u, center_v, depth_frame)
        
        # Filtrer les valeurs aberrantes : garder seulement celles à ±20% du centre
        depths_filtered = [d for d in depths if abs(d - center_depth) / center_depth < 0.2]
        
        # Si trop de valeurs filtrées, utiliser juste le centre
        if len(depths_filtered) < 5:
            final_depth = center_depth
        else:
            # Utiliser la médiane des valeurs filtrées
            final_depth = np.median(depths_filtered)
        
        # Utiliser final_depth au lieu de la profondeur du pixel central
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_u, center_v], final_depth)
        return np.array(point_3d)
    
    
    
    def transform_camera_to_marker(self, point_camera, rvec_marker, tvec_marker):
        """
        Transforme un point du repère caméra vers le repère du marqueur
        
        Le marqueur définit le repère monde :
        - Origine = centre du marqueur
        - Axes = axes du marqueur (X droite, Y bas sur le marqueur, Z perpendiculaire sortant)
        
        Args:
            point_camera: Point dans le repère caméra (3,) [x, y, z]
            rvec_marker: Vecteur rotation du marqueur (obtenu par solvePnP)
            tvec_marker: Vecteur translation du marqueur (obtenu par solvePnP)
        
        Returns:
            point_marker: Point dans le repère du marqueur
        """
        # Conversion du vecteur rotation en matrice de rotation
        R_cam_to_marker, _ = cv2.Rodrigues(rvec_marker)
        t_cam_to_marker = tvec_marker.flatten()
        
        # La transformation donne : position du marqueur dans le repère caméra
        # On veut l'inverse : position dans le repère du marqueur
        
        # R_cam_to_marker : rotation du repère marqueur par rapport à la caméra
        # t_cam_to_marker : position du centre du marqueur dans le repère caméra
        
        # Pour transformer un point de la caméra vers le marqueur :
        # P_marker = R^T * (P_camera - t)
        R_marker_to_cam = R_cam_to_marker.T
        point_marker = R_marker_to_cam @ (point_camera - t_cam_to_marker)
        
        return point_marker
    
    
    def detect_and_localize(self):
        """
        Boucle principale de détection
        
        Appuyez sur :
        - 'q' pour quitter
        - 's' pour sauvegarder les détections
        """
        print("\n🎯 Détection en cours...")
        print("   - Vert : Marqueur 6 (repère monde)")
        print("   - Rouge : Objets YOLO")
        print("\nCommandes :")
        print("   - 'q' : Quitter")
        print("   - 's' : Sauvegarder la détection actuelle")
        
        detections_log = []
        self.pipeline.stop()
        self.pipeline.start(self.config)
        try:
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
                
                # Variables pour stocker les informations
                marker_6_detected = False
                marker_6_pose = None
                info_text = []
                
                # Affichage des marqueurs détectés
                for marker_id, pose_data in aruco_poses.items():
                    rvec = pose_data['rvec']
                    tvec = pose_data['tvec']
                    corners = pose_data['corners']
                    
                    # Couleur selon l'ID
                    if marker_id == 6:
                        color = (0, 255, 0)  # Vert pour le repère monde
                        marker_6_detected = True
                        marker_6_pose = pose_data
                        label = f"Marqueur 6 (REPERE MONDE)"
                    else:
                        color = (128, 128, 128)  # Gris pour autres
                        label = f"Marqueur {marker_id}"
                    
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
                    cv2.putText(display_image, label, 
                               (center[0] - 50, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    info_text.append(f"{label}: Camera({pos_cam[0]*100:.1f}, {pos_cam[1]*100:.1f}, {pos_cam[2]*100:.1f}) cm")
                
                # Détection YOLO uniquement si le marqueur 6 est détecté
                if marker_6_detected:
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
                                # Transformation vers le repère du marqueur 6
                                point_monde = self.transform_camera_to_marker(
                                    point_camera, 
                                    marker_6_pose['rvec'], 
                                    marker_6_pose['tvec']
                                )
                                
                                # Affichage
                                cv2.rectangle(display_image, (int(x1), int(y1)), 
                                            (int(x2), int(y2)), (0, 0, 255), 2)
                                
                                # Texte avec positions
                                label_cam = f"{class_name} {conf:.2f}"
                                label_pos_cam = f"Cam: ({point_camera[0]*100:.1f}, {point_camera[1]*100:.1f}, {point_camera[2]*100:.1f}) cm"
                                label_pos_monde = f"M6: ({point_monde[0]*100:.1f}, {point_monde[1]*100:.1f}, {point_monde[2]*100:.1f}) cm"
                                
                                cv2.putText(display_image, label_cam,
                                          (int(x1), int(y1) - 40),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.putText(display_image, label_pos_cam,
                                          (int(x1), int(y1) - 20),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                cv2.putText(display_image, label_pos_monde,
                                          (int(x1), int(y1) - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                                
                                info_text.append(f"{class_name}: Camera{tuple(point_camera*100)} cm -> Monde{tuple(point_monde*100)} cm")
                                
                                # Sauvegarde pour le log
                                detections_log.append({
                                    'class': class_name,
                                    'confidence': float(conf),
                                    'position_camera_cm': point_camera.tolist(),
                                    'position_monde_cm': point_monde.tolist()
                                })
                else:
                    # Message si marqueur 6 non détecté
                    cv2.putText(display_image, "MARQUEUR 6 NON DETECTE - Repere monde indisponible",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Affichage des infos en haut
                y_offset = 60
                for text in info_text:
                    cv2.putText(display_image, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    y_offset += 20
                
                # Affichage
                cv2.imshow('Detection avec repere ArUco', display_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and len(info_text) > 0:
                    print(f"\n📸 Sauvegarde de {len(detections_log)} détections")
                    for text in info_text:
                        print(f"   {text}")
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


def main():
    print("="*70)
    print("DÉTECTION 3D AVEC REPÈRE ARUCO")
    print("="*70)
    
    # Configuration
    yolo_model = '../../runs/detect/detection_objets/weights/best.pt'
    
    print("\n📏 Quelle est la TAILLE RÉELLE de vos marqueurs ArUco ?")
    print("   (Mesurez le côté du carré NOIR en cm)")
    marker_size_input = input("Taille en cm (ex: 5): ").strip()
    
    try:
        marker_size_cm = float(marker_size_input)
        marker_size_m = marker_size_cm / 100.0
    except:
        print("❌ Taille invalide, utilisation de 5 cm par défaut")
        marker_size_m = 0.05
    
    print(f"\n✅ Configuration:")
    print(f"   - Modèle YOLO: {yolo_model}")
    print(f"   - Taille marqueurs: {marker_size_m*100} cm")
    print(f"   - Marqueur 6 = REPÈRE MONDE (origine)")
    
    # Lancement
    detector = DetectionAvecRepereArUco(yolo_model, marker_size_m)
    detector.detect_and_localize()


if __name__ == "__main__":
    main()
