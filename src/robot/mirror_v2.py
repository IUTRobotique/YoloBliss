"""
Mirror the real robot into MuJoCo with torque OFF.

Move the robot by hand and watch it in the MuJoCo viewer.

Usage:
    python mirror.py
    python mirror.py --device /dev/ttyUSB0 --rate 50
"""
from __future__ import annotations

import argparse
import math
import time

import mujoco.viewer
import numpy as np

from sim_3dofs import Sim3Dofs

# ── Dynamixel constants (XL-320) ──────────────────────────────────────────────
DXL_ADDR_TORQUE_ENABLE = 24
DXL_ADDR_PRESENT_POS = 37
DXL_LEN_POSITION = 2
DXL_PROTOCOL = 2.0
DXL_BAUDRATE = 1_000_000
DXL_DEVICE = "/dev/ttyACM0"
DXL_IDS = [1, 2, 3]
DXL_CENTER = 512  # raw units for 0 rad
# Add this constant (XL-320 goal position address)
DXL_ADDR_GOAL_POSITION = 30

# Add this inverse of dxl_to_rad
def rad_to_dxl(angle_rad: float, center: int = DXL_CENTER) -> int:
    raw = int(round(angle_rad * 1024 / (300 * math.pi / 180) + center))
    return max(0, min(1023, raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror real robot into MuJoCo (torque OFF)")
    parser.add_argument("--device", type=str, default=DXL_DEVICE,
                        help=f"Serial port for Dynamixel (default: {DXL_DEVICE})")
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Read/display rate in Hz (default: 50)")
    return parser.parse_args()


def dxl_to_rad(raw: int, center: int = DXL_CENTER) -> float:
    return (raw - center) * (300 * math.pi / 180) / 1024


def main():
    args = parse_args()
    dt = 1.0 / args.rate

    import dynamixel_sdk as dxl

    port_handler = dxl.PortHandler(args.device)
    packet_handler = dxl.PacketHandler(DXL_PROTOCOL)
    sync_read = dxl.GroupSyncRead(
        port_handler, packet_handler,
        DXL_ADDR_PRESENT_POS, DXL_LEN_POSITION,
    )

    if not port_handler.openPort():
        raise RuntimeError(f"Cannot open port {args.device}")
    if not port_handler.setBaudRate(DXL_BAUDRATE):
        raise RuntimeError("Cannot set baudrate")

    # Torque OFF on all motors
    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(
            port_handler, dxl_id, DXL_ADDR_TORQUE_ENABLE, 0
        )
    print(f"Dynamixel connected on {args.device} — torque OFF, move the robot freely.")

    for dxl_id in DXL_IDS:
        sync_read.addParam(dxl_id)

    # Simulateur MuJoCo (visualisation uniquement)
    sim = Sim3Dofs()

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.6
            viewer.cam.lookat[:] = [0.175, 0.0, 0.06]

            last_ctrl = np.zeros(len(DXL_IDS))

            while viewer.is_running():
                t0 = time.monotonic()

                current_ctrl = sim.data.ctrl[:len(DXL_IDS)].copy()

                if not np.allclose(current_ctrl, last_ctrl, atol=1e-4):
                    # User moved a slider → drive real motors
                    for i, dxl_id in enumerate(DXL_IDS):
                        goal = rad_to_dxl(current_ctrl[i])
                        packet_handler.write2ByteTxRx(
                            port_handler, dxl_id, DXL_ADDR_GOAL_POSITION, goal
                        )
                    sim.set_qpos(current_ctrl)
                    last_ctrl = current_ctrl.copy()

                else:
                    # No slider interaction → mirror real robot
                    sync_read.txRxPacket()
                    q = []
                    for dxl_id in DXL_IDS:
                        raw = sync_read.getData(dxl_id, DXL_ADDR_PRESENT_POS, DXL_LEN_POSITION)
                        q.append(dxl_to_rad(raw))
                    sim.set_qpos(q)
                    sim.data.ctrl[:len(q)] = q
                    last_ctrl = np.array(q)

                sim.forward()
                viewer.sync()

                elapsed = time.monotonic() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        # Ensure torque stays OFF on exit
        for dxl_id in DXL_IDS:
            packet_handler.write1ByteTxRx(
                port_handler, dxl_id, DXL_ADDR_TORQUE_ENABLE, 0
            )
        port_handler.closePort()
        print("Port closed.")


if __name__ == "__main__":
    main()
