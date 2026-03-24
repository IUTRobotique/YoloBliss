import math
import dynamixel_sdk as dxl
import numpy

ADDR_GOAL_POSITION = 30
LEN_GOAL_POSITION = 2
ADDR_PRESENT_POS = 37
THRESHOLD_RAW = 10
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
DEVICENAME = '/dev/ttyACM0'
IDS = [1, 2, 3]

# --- CONFIGURATION ---
DEVICE_NAME = '/dev/ttyACM0'  # Modifier selon votre système (ex: 'COM3' sur Windows)
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30

TORQUE_ENABLE = 1
# 512 correspond à 150° (le milieu de la course de 0 à 1023)
GOAL_POSITION_CENTER = 512
DXL_IDS = [1, 2, 3]

portHandler = dxl.PortHandler(DEVICENAME)
packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
groupSyncWrite = dxl.GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
groupSyncRead = dxl.GroupSyncRead(portHandler, packetHandler, ADDR_PRESENT_POS, LEN_GOAL_POSITION)


def rad_to_dxl(angle_rad: float, center: int = 512) -> int:
    raw = int(numpy.round(angle_rad * 1024 / (300 * math.pi / 180) + center))
    return max(250, min(850, raw))


def init_real_robot():
    if not portHandler.openPort():
        raise RuntimeError(f"Impossible d'ouvrir {DEVICENAME}")
    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError(f"Impossible de régler le baudrate")
    for dxl_id in IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, 24, 1)  # torque ON
    for dxl_id in IDS:
        groupSyncRead.addParam(dxl_id)
    print(f"[OK] Robot initialisé sur {DEVICENAME}")


def close_real_robot():
    for dxl_id in IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, 24, 0)  # torque OFF
    portHandler.closePort()


def update_real_robot_position(motor_joints):
    coords = [rad_to_dxl(rad) for rad in motor_joints]
    print(f"The coords are {coords}")
    try:
        for i, dxl_id in enumerate(DXL_IDS):
            print(f"Moteur ID {dxl_id} → position {coords[i]}")
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, dxl_id, ADDR_GOAL_POSITION, coords[i]
            )
            print(f"[WARN] Moteur {dxl_id}: {packetHandler.getTxRxResult(dxl_comm_result)}")


    except KeyboardInterrupt:
        print("\nInterruption utilisateur.")
