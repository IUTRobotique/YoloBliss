import math
import time
import dynamixel_sdk as dxl
import numpy as np

# --- CONFIGURATION ---
DEVICENAME      = '/dev/ttyACM0'
BAUDRATE        = 1_000_000
PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED  = 32     # Registre vitesse (0 = max, 1..1023 = limité)
LEN_GOAL_POSITION  = 2

DXL_IDS = [1, 2, 3]

# Slew-rate : pas maximum autorisé entre deux commandes successives (en valeur brute 0-1023)
# 5 ≈ 1.5° par appel → très doux. Augmenter si trop lent.
MAX_STEP = 5

# Vitesse des moteurs (0 = pleine vitesse, 100 ≈ ~10% de la vitesse max)
MOTOR_SPEED = 100

portHandler   = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

# Position courante connue côté software (initialisée au centre)
_current_pos = [512, 512, 512]


def rad_to_dxl(angle_rad: float, center: int = 512) -> int:
    """Convertit un angle en radians en valeur brute Dynamixel (0-1023)."""
    raw = int(np.round(angle_rad * 1024 / (300 * math.pi / 180) + center))
    return max(250, min(850, raw))


def _clamp_step(current: int, target: int, max_step: int = MAX_STEP) -> int:
    """Avance 'current' vers 'target' d'au plus 'max_step' unités."""
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + max_step * (1 if delta > 0 else -1)


def init_real_robot():
    global _current_pos
    if not portHandler.openPort():
        raise RuntimeError(f"Impossible d'ouvrir {DEVICENAME}")
    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError(f"Impossible de régler le baudrate")

    for dxl_id in DXL_IDS:
        # Activation du couple
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        # Réglage de la vitesse (douceur)
        packetHandler.write2ByteTxRx(portHandler, dxl_id, ADDR_MOVING_SPEED, MOTOR_SPEED)

    # Lecture des positions initiales pour initialiser _current_pos
    for i, dxl_id in enumerate(DXL_IDS):
        pos, result, _ = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION)
        if result == dxl.COMM_SUCCESS:
            _current_pos[i] = pos
        else:
            _current_pos[i] = 512  # fallback au centre

    print(f"[OK] Robot initialisé sur {DEVICENAME} | Positions initiales : {_current_pos}")


def close_real_robot():
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
    print("[OK] Robot déconnecté.")


def update_real_robot_position(motor_joints):
    """Envoie les positions cibles aux moteurs avec slew-rate limiting.

    Les positions sont converties depuis les radians retournés par MuJoCo,
    puis clampées pour ne jamais faire sauter le moteur de plus de MAX_STEP
    unités brutes en un seul appel → mouvement progressif et doux.
    """
    global _current_pos

    targets = [rad_to_dxl(float(rad)) for rad in motor_joints]

    for i, dxl_id in enumerate(DXL_IDS):
        # Slew-rate : on n'avance que d'un petit pas vers la cible
        next_pos = _clamp_step(_current_pos[i], targets[i])
        _current_pos[i] = next_pos

        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
            portHandler, dxl_id, ADDR_GOAL_POSITION, next_pos
        )
        if dxl_comm_result != dxl.COMM_SUCCESS:
            pass  # on ignore silencieusement les erreurs de comm pour ne pas spammer la console
