"""
Module de détection 3D avec repère monde basé sur marqueur ArUco.

Usage:
    from detection_module import DetectionModule

    detector = DetectionModule(
        yolo_model_path='path/to/best.pt',
        aruco_marker_size=0.05  # en mètres
    )

    # Retourne une liste de détections ou None si le marqueur 6 est absent
    detections = detector.get_positions()
    # [{'class': 'cylinder', 'confidence': 0.91, 'position_m': (x, y, z)}, ...]

    # Pour fermer proprement la caméra
    detector.close()
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Nombre de frames à jeter au démarrage pour laisser le capteur se stabiliser
_WARMUP_FRAMES = 30


class DetectionModule:
    def __init__(self, yolo_model_path: str, aruco_marker_size: float):
        """
        Args:
            yolo_model_path:    Chemin vers le modèle YOLO (.pt)
            aruco_marker_size:  Taille réelle du marqueur ArUco en mètres (ex: 0.05)
        """
        # --- RealSense ---
        self.pipeline = rs.pipeline()

        # On stocke la config pour pouvoir redémarrer le pipeline (comme l'original)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self._last_display_frame = None

        # Démarrage initial pour récupérer les intrinsèques
        profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intrinsics = intrinsics
        self.camera_matrix = np.array([
            [intrinsics.fx, 0,             intrinsics.ppx],
            [0,             intrinsics.fy, intrinsics.ppy],
            [0,             0,             1             ],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

        # --- YOLO ---
        self.model = YOLO(yolo_model_path)

        # --- ArUco ---
        self.aruco_dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params   = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.aruco_marker_size = aruco_marker_size


        # Defining the aruco marker's origin point
        # We decided to shift it by x - 2.5 cm so the robot itself will be the center.
        half = aruco_marker_size / 2
        offset_x = 0.025  # 2.5 cm

        self.obj_points = np.array([
            [-half - offset_x, half, 0],
            [half - offset_x, half, 0],
            [half - offset_x, -half, 0],
            [-half - offset_x, -half, 0],
        ], dtype=np.float32)

        # Redémarre le pipeline proprement (pattern du code original)
        # et jette les premières frames pour laisser l'auto-exposition se stabiliser
        self.pipeline.stop()
        self.pipeline.start(self.config)
        print(f"Warmup camera ({_WARMUP_FRAMES} frames)...")
        for _ in range(_WARMUP_FRAMES):
            self.pipeline.wait_for_frames()
        print("Camera prete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_positions(self, confidence_threshold=0.5, target_classes=None,
                      show_preview=True, window_name='Detection 3D'):
        color_image, depth_frame = self._capture_frame()
        display_image = color_image.copy()  # always build it

        aruco_poses = self._get_all_aruco_poses(color_image)
        marker_6_pose = aruco_poses.get(0, None)

        self._draw_aruco_markers(display_image, aruco_poses)  # always annotate

        if marker_6_pose is None:
            cv2.putText(display_image, "MARQUEUR 6 NON DETECTE - Repere monde indisponible",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self._last_display_frame = display_image
            return None

        results = self.model(color_image, verbose=False)
        detections = []
        info_lines = []

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]

                if conf < confidence_threshold:
                    continue
                if target_classes is not None and class_name not in target_classes:
                    continue

                bbox = box.xyxy[0].cpu().numpy()
                point_cam = self._bbox_depth_center(bbox, depth_frame)
                if point_cam is None:
                    continue

                point_world = self._camera_to_marker(
                    point_cam, marker_6_pose['rvec'], marker_6_pose['tvec'])

                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 3),
                    'position_m': tuple(point_world.tolist()),
                })
                info_lines.append(
                    f"{class_name}: "
                    f"Cam({point_cam[0] * 100:.1f}, {point_cam[1] * 100:.1f}, {point_cam[2] * 100:.1f}) cm"
                    f" -> M6({point_world[0] * 100:.1f}, {point_world[1] * 100:.1f}, {point_world[2] * 100:.1f}) cm"
                )
                self._draw_detection(display_image, bbox, class_name, conf, point_cam, point_world)

        y = 60
        for line in info_lines:
            cv2.putText(display_image, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 18

        self._last_display_frame = display_image  # always store
        return detections

    def close(self):
        """Libère la caméra RealSense et ferme les fenêtres OpenCV."""
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _capture_frame(self):
        """
        Attend une frame valide — réessaie indéfiniment comme l'original
        (équivalent du `continue` dans la boucle while de detect_and_localize).
        """
        while True:
            frames  = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame and depth_frame:
                return np.asanyarray(color_frame.get_data()), depth_frame
            # frame invalide → on réessaie silencieusement

    def _get_all_aruco_poses(self, color_image: np.ndarray) -> dict:
        """
        Détecte tous les marqueurs ArUco et retourne leurs poses.
        Identique à get_aruco_pose() de DetectionAvecRepereArUco.

        Returns:
            {marker_id: {'rvec': ..., 'tvec': ..., 'corners': ...}}
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        poses = {}
        if ids is None:
            return poses

        for i, marker_id in enumerate(ids.flatten()):
            success, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                corners[i][0],
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if success:
                poses[marker_id] = {
                    'rvec':    rvec,
                    'tvec':    tvec,
                    'corners': corners[i][0],
                }

        return poses

    def _draw_aruco_markers(self, img: np.ndarray, poses: dict):
        """Dessine les contours et axes de tous les marqueurs détectés."""
        for marker_id, data in poses.items():
            if marker_id == 6:
                color = (0, 255, 0)   # vert = repère monde
                label = "Marqueur 6 (REPERE MONDE)"
            else:
                color = (128, 128, 128)
                label = f"Marqueur {marker_id}"

            pts = data['corners'].astype(int)
            cv2.polylines(img, [pts], True, color, 2)

            cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs,
                              data['rvec'], data['tvec'],
                              self.aruco_marker_size * 0.5)

            pos_cam = data['tvec'].flatten()
            center  = tuple(pts.mean(axis=0).astype(int))
            cv2.putText(img, label,
                        (center[0] - 50, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(img,
                        f"Cam: ({pos_cam[0]*100:.1f}, {pos_cam[1]*100:.1f}, {pos_cam[2]*100:.1f}) cm",
                        (center[0] - 50, center[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_detection(self, img, bbox, class_name, conf, point_cam, point_world):
        """Dessine la bounding box et les labels — identique à l'original."""
        x1, y1, x2, y2 = map(int, bbox)
        RED   = (0,   0, 255)
        WHITE = (255, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)

        pc = point_cam   * 100
        pw = point_world * 100

        cv2.putText(img, f"{class_name} {conf:.2f}",
                    (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        cv2.putText(img,
                    f"Cam: ({pc[0]:.1f}, {pc[1]:.1f}, {pc[2]:.1f}) cm",
                    (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
        cv2.putText(img,
                    f"M6:  ({pw[0]:.1f}, {pw[1]:.1f}, {pw[2]:.1f}) cm",
                    (x1, y1 - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 2)

    def _show(self, image, window_name):
        self._last_display_frame = image

    def get_last_frame(self):
        return self._last_display_frame

    def _bbox_depth_center(self, bbox, depth_frame) -> np.ndarray | None:
        """
        Projette le centre d'une bounding box en 3D (repère caméra).
        Identique à get_bbox_3d_center() de l'original.
        """
        x1, y1, x2, y2 = map(int, bbox)
        cu, cv_ = (x1 + x2) // 2, (y1 + y2) // 2

        center_depth = depth_frame.get_distance(cu, cv_)
        if center_depth == 0:
            return None

        depths = [
            depth_frame.get_distance(cu + du, cv_ + dv)
            for du in range(-2, 3)
            for dv in range(-2, 3)
            if 0 <= cu + du < 640 and 0 <= cv_ + dv < 480
        ]
        depths = [d for d in depths if d > 0]

        if len(depths) < 5:
            final_depth = center_depth
        else:
            filtered = [d for d in depths if abs(d - center_depth) / center_depth < 0.2]
            final_depth = np.median(filtered) if len(filtered) >= 5 else center_depth

        pt = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cu, cv_], final_depth)
        return np.array(pt)

    def _camera_to_marker(self, point_cam, rvec, tvec) -> np.ndarray:
        """Transforme un point du repère caméra vers le repère du marqueur."""
        R, _ = cv2.Rodrigues(rvec)
        return R.T @ (point_cam - tvec.flatten())