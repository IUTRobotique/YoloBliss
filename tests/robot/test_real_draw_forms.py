import argparse
import time
import math
import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz

# --- Flag real robot vs meshcat ---
parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true", help="Envoyer les commandes au vrai robot Dynamixel")
parser.add_argument("--shape", choices=["circle", "rect"], default="circle", help="Forme à dessiner")
parser.add_argument("--debug", action="store_true", help="Afficher les messages de debug dans le terminal")
args = parser.parse_args()

if args.real:
    import dynamixel_sdk as dxl

# --- Configuration Dynamixel ---
ADDR_GOAL_POSITION   = 30
LEN_GOAL_POSITION    = 2
ADDR_PRESENT_POS     = 37
THRESHOLD_RAW        = 10   # écart max admissible en unités brutes (~5°)
PROTOCOL_VERSION     = 2.0
BAUDRATE           = 1000000
DEVICENAME         = '/dev/ttyACM0'
IDS = [1, 2, 3]

# --- Paramètres communs ---
DURATION = 4.0
CENTER_X, CENTER_Y, Z_HEIGHT = 0.175, 0.0, 0.04

# --- Paramètres cercle ---
RADIUS = 0.05

# --- Paramètres rectangle ---
WIDTH  = 0.17
HEIGHT = 0.13
HALF_W = WIDTH / 2
HALF_H = HEIGHT / 2
CORNERS = [
    [CENTER_X - HALF_H, CENTER_Y - HALF_W],
    [CENTER_X + HALF_H, CENTER_Y - HALF_W],
    [CENTER_X + HALF_H, CENTER_Y + HALF_W],
    [CENTER_X - HALF_H, CENTER_Y + HALF_W],
]

# --- Chargement du modèle et configuration du solver placo ---
robot = placo.RobotWrapper(
    "robot_env/robot.xml",placo.Flags.mjcf)

solver = placo.KinematicsSolver(robot)
solver.mask_fbase(True)

effector_task = solver.add_position_task("end_effector", np.array([CENTER_X, CENTER_Y, Z_HEIGHT]))
effector_task.configure("effector_pos", "soft", 1.0)

for k in range(20):
    solver.solve(True)
    robot.update_kinematics()

# --- Initialisation selon le mode ---
if args.real:
    portHandler = dxl.PortHandler(DEVICENAME)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
    groupSyncWrite = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    groupSyncRead  = dxl.GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POS, LEN_GOAL_POSITION)

    if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
        quit()

    for dxl_id in IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, 24, 1)

    for dxl_id in IDS:
        groupSyncRead.addParam(dxl_id)
else:
    viz = robot_viz(robot)

def rad_to_dxl(rad, center=512):
    return max(0, min(1023, int(center + (rad * (1024 / (300 * math.pi / 180))))))

try:
    print(f"Dessin {args.shape} en {DURATION}s ({'robot réel' if args.real else 'meshcat'}). Ctrl+C pour arrêter.")
    start_time = time.time()

    while True:
        current_time = time.time() - start_time

        if args.shape == "circle":
            angle = (2 * math.pi * current_time) / DURATION
            x = CENTER_X + RADIUS * math.cos(angle)
            y = CENTER_Y + RADIUS * math.sin(angle)
        else:
            t = (current_time % DURATION) / DURATION
            segment = int(t * 4)
            t_seg = (t * 4) - segment
            p1 = CORNERS[segment]
            p2 = CORNERS[(segment + 1) % 4]
            x = p1[0] + (p2[0] - p1[0]) * t_seg
            y = p1[1] + (p2[1] - p1[1]) * t_seg

        z = Z_HEIGHT

        # Mise à jour de la cible et résolution IK via placo
        effector_task.target_world = np.array([x, y, z])

        robot.update_kinematics()
        solver.solve(True)
        robot.update_kinematics()

        if args.debug:
            actual_pos = robot.get_T_world_frame("end_effector")[:3, 3]
            error = np.linalg.norm(actual_pos - np.array([x, y, z]))
            if error > 0.005:
                print(f"[DEBUG] Cible non atteinte — cible [{x:.3f}, {y:.3f}, {z:.3f}], erreur: {error*1000:.1f} mm")

        if args.real:
            vals = [
                rad_to_dxl(robot.get_joint("1")),
                rad_to_dxl(robot.get_joint("2")),
                rad_to_dxl(robot.get_joint("3")),
            ]

            for i, dxl_id in enumerate(IDS):
                param = [dxl.DXL_LOBYTE(vals[i]), dxl.DXL_HIBYTE(vals[i])]
                groupSyncWrite.addParam(dxl_id, param)

            groupSyncWrite.txPacket()
            groupSyncWrite.clearParam()

            groupSyncRead.txRxPacket()
            for i, dxl_id in enumerate(IDS):
                if not groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POS, LEN_GOAL_POSITION):
                    print(f"[WARN] Moteur {dxl_id}: lecture groupée échouée")
                    continue
                present = groupSyncRead.getData(dxl_id, ADDR_PRESENT_POS, LEN_GOAL_POSITION)
                if abs(present - vals[i]) > THRESHOLD_RAW:
                    print(
                        f"[WARN] Moteur {dxl_id}: envoyé={vals[i]}  lu={present}  "
                        f"écart={abs(present - vals[i])} unités"
                    )
                elif args.debug:
                    print(f"[OK]   Moteur {dxl_id}: envoyé={vals[i]}  lu={present}")
        else:
            T_target = robot.get_T_world_frame("end_effector").copy()
            T_target[:3, 3] = [x, y, z]
            frame_viz("target_effector", T_target, opacity=0.3)
            robot_frame_viz(robot, "end_effector")
            viz.display(robot.state.q)

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nArrêt...")

finally:
    if args.real:
        for dxl_id in IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, 24, 0)
        portHandler.closePort()
