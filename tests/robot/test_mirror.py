#!/usr/bin/env python3
"""
Test de communication avec les servos Dynamixel XL-320.


Ce script teste la couche matérielle SANS lancer la simulation MuJoCo.

Tests effectués :
  1. Fonction dxl_to_rad() — conversion brut → radians (sans matériel)
  2. Listage des ports série disponibles
  3. (optionnel) Connexion réelle au port → lecture des positions des 3 servos

Usage :
  python src/robot/test_mirror.py                          # tests logiciels seuls
  python src/robot/test_mirror.py --port /dev/ttyACM0     # connexion réelle
  python src/robot/test_mirror.py --lister-ports           # liste les ports série
  python src/robot/test_mirror.py --port /dev/ttyACM0 --duree 5  # lit 5 secondes
"""

from __future__ import annotations

import argparse
import math
import sys
import time

# ── Constantes Dynamixel XL-320 ───────────────────────────────────────────────
DXL_CENTER       = 512       # valeur brute  = 0 rad  (150°)
DXL_MAX_RAW      = 1023      # valeur brute  max (0..1023 = 1024 pas)
DXL_MAX_DIVISOR  = 1024      # diviseur de la formule (= nombre de pas total, cohérent avec mirror.py)
DXL_RANGE_DEG    = 300       # amplitude totale en degrés
DXL_BAUDRATE     = 1_000_000
DXL_PROTOCOL     = 2.0
DXL_IDS          = [1, 2, 3]

ADDR_TORQUE_ENABLE = 24
ADDR_PRESENT_POS   = 37
LEN_PRESENT_POS    = 2


# ── Fonction de conversion (même logique que mirror.py) ───────────────────────

def dxl_to_rad(raw: int, center: int = DXL_CENTER) -> float:
    """Convertit une position brute Dynamixel XL-320 en radians.
    Même formule que mirror.py : diviseur = 1024 (nombre de pas total).
    """
    return (raw - center) * (DXL_RANGE_DEG * math.pi / 180) / DXL_MAX_DIVISOR


def rad_to_dxl(angle_rad: float, center: int = DXL_CENTER) -> int:
    """Convertit un angle en radians en valeur brute Dynamixel."""
    raw = round(angle_rad * DXL_MAX_DIVISOR / (DXL_RANGE_DEG * math.pi / 180) + center)
    return max(0, min(DXL_MAX_RAW, raw))


# ── Test 1 : conversion dxl_to_rad ────────────────────────────────────────────

def tester_conversion() -> bool:
    """Vérifie la conversion brut ↔ radians sur des cas connus."""
    print("\n[TEST 1] Conversion dxl_to_rad / rad_to_dxl")
    print("-" * 50)

    cas = [
        # (valeur brute, angle attendu en rad, description)
        (512,   0.0,                  "centre (150°)"),
        # Avec diviseur 1024 : raw=0 → (0-512)*300°/1024, raw=1023 → (1023-512)*300°/1024
        (0,    -512 * (DXL_RANGE_DEG * math.pi / 180) / DXL_MAX_DIVISOR, "butée min (0°)"),
        (1023,  511 * (DXL_RANGE_DEG * math.pi / 180) / DXL_MAX_DIVISOR, "butée max (300°)"),
        (256,  -math.pi * 150 / 180 / 2,            "quart gauche (~75°)"),
    ]

    EPSILON = 1e-3
    tous_ok = True

    for raw, expected_rad, label in cas:
        computed = dxl_to_rad(raw)
        ok = abs(computed - expected_rad) < EPSILON
        if not ok:
            tous_ok = False
        sym = "yes" if ok else "no"
        print(f"  [{sym}] {label:<25} raw={raw:>4}  "
              f"rad={computed:7.4f}  attendu={expected_rad:7.4f}")

    # Aller-retour
    for angle in [-math.pi / 4, 0.0, math.pi / 6, math.pi / 3]:
        raw   = rad_to_dxl(angle)
        back  = dxl_to_rad(raw)
        delta = abs(back - angle)
        ok_ar = delta < 0.02   # tolérance quantisation
        if not ok_ar:
            tous_ok = False
        sym = "yes" if ok_ar else "no"
        print(f"  [{sym}] aller-retour  {math.degrees(angle):+6.1f}°  "
              f"→ raw={raw}  → {math.degrees(back):+6.1f}°  delta={delta:.4f} rad")

    print(f"  → Résultat : {'OK' if tous_ok else 'ÉCHEC'}")
    return tous_ok


# ── Test 2 : listage des ports série ──────────────────────────────────────────

def lister_ports() -> list[str]:
    """Liste les ports série disponibles sur le système."""
    import glob
    ports: list[str] = []

    if sys.platform.startswith("linux"):
        ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    elif sys.platform.startswith("win"):
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/tty.usbserial*")

    print("\n[TEST 2] Ports série disponibles")
    print("-" * 50)
    if ports:
        for p in ports:
            print(f"  [i] {p}")
        print(f"  → {len(ports)} port(s) trouvé(s)")
    else:
        print("  Aucun port série détecté.")
        print("      Vérifiez que le câble USB est branché (port /dev/ttyACM0).")

    return ports


# ── Test 3 : connexion et lecture réelle ──────────────────────────────────────

def tester_connexion_reelle(port: str, duree: float) -> bool:
    """Se connecte aux servos et lit les positions pendant <duree> secondes."""
    print(f"\n[TEST 3] Connexion réelle — port {port}")
    print("-" * 50)

    try:
        import dynamixel_sdk as dxl
    except ImportError:
        print("  dynamixel_sdk non installé : pip install dynamixel_sdk")
        return False

    port_handler   = dxl.PortHandler(port)
    packet_handler = dxl.PacketHandler(DXL_PROTOCOL)

    if not port_handler.openPort():
        print(f"  Impossible d'ouvrir {port}")
        print("      → Vérifiez que le câble est branché et que l'utilisateur a accès au port.")
        print("        sudo usermod -aG dialout $USER  puis redémarrer la session.")
        return False

    if not port_handler.setBaudRate(DXL_BAUDRATE):
        print(f"  Impossible de configurer le baudrate {DXL_BAUDRATE}")
        port_handler.closePort()
        return False

    print(f"  Port {port} ouvert à {DXL_BAUDRATE} baud")

    # Désactiver le couple (mode miroir — peut bouger librement)
    for dxl_id in DXL_IDS:
        packet_handler.write1ByteTxRx(
            port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0
        )
    print(f"  Couple désactivé sur les moteurs {DXL_IDS}")

    # Sync Read — ajout des IDs
    sync_read = dxl.GroupSyncRead(
        port_handler, packet_handler,
        ADDR_PRESENT_POS, LEN_PRESENT_POS,
    )
    moteurs_ok = []
    for dxl_id in DXL_IDS:
        if sync_read.addParam(dxl_id):
            moteurs_ok.append(dxl_id)

    if not moteurs_ok:
        print("  Aucun moteur n'a répondu à l'ajout dans GroupSyncRead")
        port_handler.closePort()
        return False

    print(f"  Moteurs détectés : {moteurs_ok}")
    print(f"   Lecture des positions pendant {duree:.0f}s ")
    print()

    t_start = time.time()
    n_lectures = 0
    tous_ok = True

    try:
        while time.time() - t_start < duree:
            result = sync_read.txRxPacket()
            if result != dxl.COMM_SUCCESS:
                print(f" Erreur communication : {packet_handler.getTxRxResult(result)}")
                tous_ok = False
                break

            positions_rad = []
            for dxl_id in moteurs_ok:
                if sync_read.isAvailable(dxl_id, ADDR_PRESENT_POS, LEN_PRESENT_POS):
                    raw = sync_read.getData(dxl_id, ADDR_PRESENT_POS, LEN_PRESENT_POS)
                    positions_rad.append(dxl_to_rad(raw))
                else:
                    positions_rad.append(float("nan"))

            rad_str = "  ".join(
                f"M{mid}={math.degrees(r):+6.1f}°"
                for mid, r in zip(moteurs_ok, positions_rad)
            )
            print(f"\r  pos : {rad_str}", end="", flush=True)
            n_lectures += 1
            time.sleep(0.05)

    except KeyboardInterrupt:
        print()

    print(f"\n  {n_lectures} lectures effectuées  ({n_lectures/(time.time()-t_start):.1f} Hz)")
    port_handler.closePort()
    print(f" Port {port} fermé")
    return tous_ok


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test communication Dynamixel")
    parser.add_argument("--port",   type=str, default=None,
                        help="Port série à tester (ex: /dev/ttyACM0)")
    parser.add_argument("--duree",  type=float, default=5.0,
                        help="Durée de lecture en secondes (défaut: 5)")
    parser.add_argument("--lister-ports", action="store_true",
                        help="Liste uniquement les ports série disponibles")
    args = parser.parse_args()

    resultats: dict[str, bool] = {}

    # Test 1 : toujours exécuté
    resultats["Conversion dxl_to_rad"] = tester_conversion()

    # Test 2 : listage ports
    ports = lister_ports()
    resultats["Ports série"] = len(ports) > 0

    if args.lister_ports:
        sys.exit(0)

    # Test 3 : connexion réelle si --port fourni ou si un seul port trouvé
    port = args.port
    if port is None and len(ports) == 1:
        port = ports[0]
        print(f"\n Port unique détecté — utilisation automatique : {port}")

    if port:
        resultats[f"Connexion {port}"] = tester_connexion_reelle(port, args.duree)
    else:
        print("\n Aucun port spécifié — test matériel ignoré.")
        print("    Utilisez --port /dev/ttyACM0 pour tester la connexion réelle.")

    # Récapitulatif
    print(f"\n{'='*50}")
    print("  RÉCAPITULATIF")
    print(f"{'='*50}")
    for nom, res in resultats.items():
        sym = "OK" if res else "FAIL"
        print(f"  [{sym}] {nom}")
    print()

    # "Ports série" absent = pas de matériel connecté, ce n'est pas un échec logiciel
    tests_logiciels = {k: v for k, v in resultats.items() if k != "Ports série"}
    sys.exit(0 if all(tests_logiciels.values()) else 1)


if __name__ == "__main__":
    main()
