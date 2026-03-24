from __future__ import annotations

import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Taille du marqueur ArUco en mètres (2 cm)
MARKER_SIZE_M = 0.02

# Coin 3D du marqueur dans son propre repère (plan Z=0)
_MARKER_OBJ_PTS = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
], dtype=np.float32)


class RealPerception:

    def __init__(self, yolo_model_path: str = "./best.pt", base_offset: np.ndarray | None = None):
      
        print("[PERCEPTION] Chargement de YOLO...")
        self.yolo = YOLO(yolo_model_path)

        self.base_offset = base_offset if base_offset is not None else np.array([0.04, 0.0, 0.0])

        print("[PERCEPTION] Démarrage de la caméra RealSense...")
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self._profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

        # Matrice intrinsèque
        intr = self._profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ], dtype=float)
        self.dist = np.zeros(5)

        # Détecteur ArUco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Warm-up caméra
        print("[PERCEPTION] Stabilisation du flux vidéo (max 10 s)...")
        for _ in range(10):
            try:
                self._pipeline.wait_for_frames(timeout_ms=1000)
                break
            except RuntimeError:
                time.sleep(0.5)

        print("[PERCEPTION] Prêt.")

    def get_object_position(self) -> np.ndarray | None:
       
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None

        img = np.asanyarray(color_frame.get_data())

        # 1. Localisation ArUco - repère base
        R_base_cam, t_base_cam = self._get_base_transform(img, depth_frame)
        if R_base_cam is None:
            cv2.imshow("Vue Camera", img)
            cv2.waitKey(1)
            return None

        # 2. Détection YOLO - position 3D dans repère base
        pos_base = self._detect_object(img, depth_frame, R_base_cam, t_base_cam)

        # 3. Affichage
        cv2.imshow("Vue Camera", img)
        cv2.waitKey(1)

        if pos_base is not None:
            # Clampage de Z pour ne jamais envoyer le robot sous la table
            pos_base[2] = max(pos_base[2], 0.01)

        return pos_base

    def stop(self):
        """Arrête proprement la caméra et ferme les fenêtres OpenCV."""
        self._pipeline.stop()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _get_base_transform(self, img: np.ndarray, depth_frame) -> tuple:
        """Estime R_base_cam et t_base_cam depuis l'ArUco détecté.

        Returns:
            (R_base_cam, t_base_cam) ou (None, None) si pas de marqueur.
        """
        corners, ids, _ = self._detector.detectMarkers(img)
        if ids is None or len(ids) == 0:
            return None, None

        _, rvec, tvec = cv2.solvePnP(_MARKER_OBJ_PTS, corners[0], self.K, self.dist)
        R_cam_aruco, _ = cv2.Rodrigues(rvec)

        # Transformation caméra → aruco
        T_cam_aruco = np.eye(4)
        T_cam_aruco[:3, :3] = R_cam_aruco
        T_cam_aruco[:3, 3] = tvec.flatten()

        # Décalage aruco → base robot
        T_aruco_base = np.eye(4)
        T_aruco_base[:3, 3] = self.base_offset

        T_cam_base = T_cam_aruco @ T_aruco_base
        T_base_cam = np.linalg.inv(T_cam_base)
        R_base_cam = T_base_cam[:3, :3]
        t_base_cam = T_base_cam[:3, 3]

        # Dessin de debug
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        cv2.drawFrameAxes(img, self.K, self.dist, rvec, tvec, 0.05)

        return R_base_cam, t_base_cam

    def _detect_object(self, img: np.ndarray, depth_frame, R_base_cam: np.ndarray, t_base_cam: np.ndarray) -> np.ndarray | None:
        """Détecte l'objet YOLO et retourne sa position 3D dans le repère base.

        Utilise la profondeur RealSense pour calculer la position 3D réelle.
        Repère caméra → repère base via R_base_cam / t_base_cam.

        Returns:
            np.ndarray [X, Y, Z] dans le repère base, ou None.
        """
        results = self.yolo(img, verbose=False)
        best_conf, best_box = 0.0, None
        for box in results[0].boxes:
            if box.conf[0].item() > best_conf:
                best_conf = box.conf[0].item()
                best_box = box

        if best_box is None:
            return None

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        d = depth_frame.get_distance(cx, cy)
        if d < 0.05:
            return None

        # Déprojetion pixel → point 3D dans repère caméra
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx0, cy0 = self.K[0, 2], self.K[1, 2]
        p_cam = np.array([(cx - cx0) * d / fx,
                          (cy - cy0) * d / fy,
                          d])

        # Changement de repère : caméra → base
        p_base = R_base_cam @ p_cam + t_base_cam

        # Dessin
        label = self.yolo.names[int(best_box.cls[0])]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} X:{p_base[0]:.2f} Y:{p_base[1]:.2f} Z:{p_base[2]:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return p_base


def inject_into_sim(env, pos_base: np.ndarray, env_name: str):
    """Injecte la position réelle de l'objet dans la simulation MuJoCo.

    Args:
        env: l'environnement Gymnasium (GoalEnv ou Env standard)
        pos_base: [X, Y, Z] dans le repère base (en mètres)
        env_name: nom de l'env ("reaching", "push", "push_in_hole", ...)
    """
    inner = env._inner if hasattr(env, "_inner") else env

    if env_name == "reaching":
        # Pour Reaching, on déplace le marqueur but vers l'objet détecté
        goal = np.array([pos_base[0], pos_base[1], pos_base[2]])
        if hasattr(inner.sim, "set_goal_marker"):
            inner.sim.set_goal_marker(goal)
        inner._goal = goal
    else:
        # Pour Push/PushInHole/Sliding : on force la position du cube
        # La hauteur du cube dans la simu est fixe (≈ 1.35 cm au-dessus du plan)
        z_sim = 0.0135
        inner.sim.set_cube_pose([pos_base[0], pos_base[1], z_sim])


def get_obs(env):
    """Reconstruit l'observation après injection manuelle dans la simu."""
    if hasattr(env, "_build_obs"):
        return env._build_obs()
    return env._get_obs()
