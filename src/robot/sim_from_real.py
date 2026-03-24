import argparse
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os

from stable_baselines3 import SAC, PPO, TD3
from main import make_eval_env, resolve_model_path, ALGO_CLS, ENVS

class SimFromReal:
    def __init__(self, args):
        self.args = args
        
        # --- VISION INITIALISATION ---
        print("[INIT] Chargement de YOLO...")
        self.yolo = YOLO(args.yolo_model)
        
        print("[INIT] Démarrage de la caméra RealSense...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=float)
        self.dist_coeffs = np.zeros(5)
        
        # Paramètres ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = 0.02  # 2 cm
        self.base_offset = np.array([args.offset_x, args.offset_y, args.offset_z])
        self.obj_points = np.array([
            [-self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)
        
        # --- MODELE RL ---
        model_path = args.rl_model if args.rl_model else resolve_model_path(args.env, args.algo)
        print(f"[INIT] Chargement du modèle RL ({model_path})...")
        
        # --- INITIALISATION DE L'ENVIRONNEMENT (Avec Affichage 3D Humain) ---
        print("[INIT] Lancement de l'environnement MuJoCo avec le visualiseur...")
        self.env = make_eval_env(args.env, args.algo, True)
        
        self.model = ALGO_CLS[args.algo].load(str(model_path), env=self.env)
        self.obs, _ = self.env.reset()

    def get_real_world_target(self):
        """Récupère la position X,Y de l'objet réel dans le repère de la base virtuelle."""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None
            
        img = np.asanyarray(color_frame.get_data())
        
        # 1. Détection ArUco
        corners, ids, _ = self.detector.detectMarkers(img)
        cube_pos_base = None
        
        if ids is not None and len(ids) > 0:
            _, rvec, tvec = cv2.solvePnP(self.obj_points, corners[0], self.K, self.dist_coeffs)
            R_cam_aruco, _ = cv2.Rodrigues(rvec)
            t_cam_aruco = tvec.flatten()
            
            T_cam_aruco = np.eye(4)
            T_cam_aruco[:3, :3] = R_cam_aruco
            T_cam_aruco[:3, 3] = t_cam_aruco
            
            T_aruco_base = np.eye(4)
            T_aruco_base[:3, 3] = self.base_offset
            
            T_cam_base = T_cam_aruco @ T_aruco_base
            T_base_cam = np.linalg.inv(T_cam_base)
            R_base_cam = T_base_cam[:3, :3]
            t_base_cam = T_base_cam[:3, 3]
            
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.drawFrameAxes(img, self.K, self.dist_coeffs, rvec, tvec, 0.05)

            # 2. Détection YOLO
            results = self.yolo(img, verbose=False)
            
            best_conf = 0
            best_box = None
            for box in results[0].boxes:
                if box.conf[0].item() > best_conf:
                    best_conf = box.conf[0].item()
                    best_box = box

            if best_box is not None:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Récupérer la profondeur
                d = depth_frame.get_distance(cx, cy)
                if d > 0.05:
                    p_cam = np.array([(cx - self.K[0, 2]) * d / self.K[0, 0], 
                                      (cy - self.K[1, 2]) * d / self.K[1, 1], 
                                      d])
                    # Position en repère base
                    p_base = R_base_cam @ p_cam + t_base_cam
                    cube_pos_base = p_base
                    
                    label = self.yolo.names[int(best_box.cls[0])]
                    cv2.putText(img, f"{label} X:{p_base[0]:.2f} Y:{p_base[1]:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Vue Camera (Reel)", img)
        cv2.waitKey(1)
            
        return cube_pos_base

    def run(self):
        print("\n[RUN] La simulation reproduit l'environnement réel. Bougez le vrai cylindre !")
        print("Fermez la fenêtre du simulateur MuJoCo ou appuyez sur Ctrl+C pour quitter.")
        try:
            while True:
                # 1. On capte le monde réel
                target_pos = self.get_real_world_target()
                
                if target_pos is not None:
                    # Sécuriser l'axe Z : les imprécisions de l'ArUco risquent de projeter 
                    # le point cible sous la table (Z négatif), ce qui force le robot à s'enfoncer dans le sol !
                    target_pos[2] = 0.02 # On force le but à 2 cm d'altitude (hauteur cible saine pour l'atteindre)

                    # 2. On injecte dans le monde virtuel
                    inner_env = self.env._inner if hasattr(self.env, "_inner") else self.env
                    if self.args.env == "reaching":
                        inner_env.sim.set_goal_marker(target_pos)
                        inner_env._goal = target_pos
                    else:
                        z_pos = 0.0135 # Hauteur du cube dans la simu
                        inner_env.sim.set_cube_pose([target_pos[0], target_pos[1], z_pos])
                
                # On recrée l'observation correcte depuis le simulateur mis-à-jour
                if hasattr(self.env, "_build_obs"):
                    self.obs = self.env._build_obs()
                else:
                    self.obs = self.env._get_obs()

                # Petite ligne de debug pour observer si le robot reçoit des coordonnées saines !
                if target_pos is not None:
                    # En environnement "dict" (HER), il faut checker ee_pos via _inner
                    ee_pos = self.env._inner.sim.get_end_effector_pos() if hasattr(self.env, "_inner") else self.env.sim.get_end_effector_pos()
                    print(f"Goal Reçu: X={target_pos[0]:.2f} Y={target_pos[1]:.2f} Z={target_pos[2]:.2f} | EE: X={ee_pos[0]:.2f} Y={ee_pos[1]:.2f}      ", end='\r')

                # 3. Prédiction de l'action du robot VIRTUEL face à ce monde HYBRIDE
                action, _ = self.model.predict(self.obs, deterministic=True)
                
                # 4. Step dans la simulation (le robot virtuel bouge !)
                self.obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 5. Rendu 3D MuJoCo
                self.env.render()
                
                # Si le robot a "Gagné" virtuellement, on reset sa position (sans reseter notre cube qu'on force)
                if terminated or truncated:
                    _, _ = self.env.reset()
                    
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n[STOP] Fin du programme.")
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Miroir du Réel vers la Simulation")
    parser.add_argument("--env", required=True, choices=list(ENVS.keys()), help="Environnement (reaching, push, etc)")
    parser.add_argument("--algo", required=True, choices=ALGO_CLS.keys(), help="Algorithme (her, sac, td3, ppo)")
    parser.add_argument("--rl_model", type=str, default="", help="Optionnel: Chemin direct vers le modèle RL (.zip)")
    parser.add_argument("--yolo_model", type=str, default="./best.pt", help="Chemin vers le modèle YOLO")
    parser.add_argument("--offset_x", type=float, default=0.04, help="Décalage X de la base par rapport au marqueur en mètres (ex: 4cm)")
    parser.add_argument("--offset_y", type=float, default=0.0, help="Décalage Y de la base par rapport au marqueur")
    parser.add_argument("--offset_z", type=float, default=0.0, help="Décalage Z de la base par rapport au marqueur")
    args = parser.parse_args()

    app = SimFromReal(args)
    app.run()
