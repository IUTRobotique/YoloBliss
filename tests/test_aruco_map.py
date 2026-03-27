"""Tests unitaires pour src/aruco/detection_avec_repere_aruco_map.py

Couvre :
  - calibrate_camera_to_a4 : cas 4 marqueurs, 3 marqueurs, < 3 marqueurs
  - Correction du déterminant négatif (réflexions)
  - transform_camera_to_world : formule affine R*p + T
  - Erreur de reprojection nulle avec une transformation parfaite

N.B. Les dépendances matériel (pyrealsense2, YOLO, RealSense pipeline) sont
mockées avant l'import du module.

Exécution :
    python -m pytest tests/test_aruco_map.py -v
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

# ── Mocks des dépendances matériel ────────────────────────────────────────────

# pyrealsense2
_rs = types.ModuleType("pyrealsense2")
_rs.pipeline = mock.MagicMock
_rs.config = mock.MagicMock
_rs.align = mock.MagicMock
_rs.stream = mock.MagicMock()
_rs.stream.color = "color"
_rs.stream.depth = "depth"
_rs.format = mock.MagicMock()
_rs.format.bgr8 = "bgr8"
_rs.format.z16 = "z16"
_rs.rs2_deproject_pixel_to_point = mock.MagicMock(return_value=[0.0, 0.0, 1.0])
sys.modules.setdefault("pyrealsense2", _rs)

# ultralytics
_ul = types.ModuleType("ultralytics")
_ul.YOLO = mock.MagicMock
sys.modules.setdefault("ultralytics", _ul)

# cv2 — on a besoin d'un vrai numpy, mais on mock les parties ArUco
import cv2  # noqa: E402 (doit être disponible dans l'env)

# ── Chemin vers src/aruco ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARUCO_SRC = os.path.abspath(os.path.join(_HERE, "..", "src", "aruco"))
sys.path.insert(0, _ARUCO_SRC)

# ── Import du module sous test (après les mocks) ───────────────────────────────
# On patch le __init__ de DetectionAvecRepereA4 pour ne pas démarrer le
# pipeline RealSense; on instancie la classe manuellement après.
with mock.patch("detection_avec_repere_aruco_map.DetectionAvecRepereA4.__init__",
                return_value=None):
    from detection_avec_repere_aruco_map import DetectionAvecRepereA4  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detector(
    marker_width=0.225,
    marker_height=0.150,
    aruco_size=0.06,
) -> DetectionAvecRepereA4:
    """Crée une instance sans pipeline, avec les attributs minimaux."""
    d = DetectionAvecRepereA4.__new__(DetectionAvecRepereA4)
    d.aruco_marker_size = aruco_size

    half_w = marker_width / 2
    half_h = marker_height / 2
    d.a4_marker_positions = {
        3: np.array([-half_w,  half_h, 0.0]),
        4: np.array([ half_w,  half_h, 0.0]),
        5: np.array([-half_w, -half_h, 0.0]),
        6: np.array([ half_w, -half_h, 0.0]),
    }
    d.R_cam_to_world = None
    d.T_cam_to_world = None
    d.calibration_error = None
    return d


def _poses_from_R_T(
    R: np.ndarray,
    T: np.ndarray,
    marker_ids: list[int],
    a4_positions: dict,
) -> dict:
    """
    Génère des poses (tvec) parfaites à partir d'une transformation R, T telle que
        p_a4 = R @ p_cam + T   =>   p_cam = R.T @ (p_a4 - T)
    """
    poses = {}
    for mid in marker_ids:
        p_a4 = a4_positions[mid]
        p_cam = R.T @ (p_a4 - T)
        poses[mid] = {"tvec": p_cam.reshape(3, 1)}
    return poses


# ══════════════════════════════════════════════════════════════════════════════
#  calibrate_camera_to_a4
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrateCamera(unittest.TestCase):

    def setUp(self) -> None:
        self.det = _make_detector()
        # Transformation de référence : rotation de 45° autour de Z + translation
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        self.R_ref = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]], dtype=float)
        self.T_ref = np.array([0.1, -0.05, 0.5])

    def _perfect_poses(self, ids):
        return _poses_from_R_T(self.R_ref, self.T_ref, ids, self.det.a4_marker_positions)

    # -- Cas nominal : 4 marqueurs ─────────────────────────────────────────────

    def test_4_marqueurs_retourne_true(self):
        ok = self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        self.assertTrue(ok)

    def test_4_marqueurs_R_proche(self):
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        np.testing.assert_allclose(self.det.R_cam_to_world, self.R_ref, atol=1e-5)

    def test_4_marqueurs_T_proche(self):
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        np.testing.assert_allclose(
            self.det.T_cam_to_world.flatten(), self.T_ref, atol=1e-5
        )

    def test_4_marqueurs_erreur_reprojection_nulle(self):
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        self.assertIsNotNone(self.det.calibration_error)
        self.assertAlmostEqual(self.det.calibration_error, 0.0, places=4)

    # -- Cas limite : exactement 3 marqueurs ───────────────────────────────────

    def test_3_marqueurs_retourne_true(self):
        ok = self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5]))
        self.assertTrue(ok)

    def test_3_marqueurs_R_proche(self):
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5]))
        np.testing.assert_allclose(self.det.R_cam_to_world, self.R_ref, atol=1e-5)

    def test_3_marqueurs_erreur_reprojection_nulle(self):
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5]))
        self.assertAlmostEqual(self.det.calibration_error, 0.0, places=4)

    # -- Cas insuffisant : < 3 marqueurs ───────────────────────────────────────

    def test_2_marqueurs_retourne_false(self):
        ok = self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4]))
        self.assertFalse(ok)

    def test_0_marqueurs_retourne_false(self):
        self.assertFalse(self.det.calibrate_camera_to_a4({}))

    def test_moins_3_marqueurs_efface_R_T(self):
        # Pré-rempli sur un appel valide
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        self.assertIsNotNone(self.det.R_cam_to_world)
        # Puis appel invalide
        self.det.calibrate_camera_to_a4(self._perfect_poses([3]))
        self.assertIsNone(self.det.R_cam_to_world)
        self.assertIsNone(self.det.T_cam_to_world)
        self.assertIsNone(self.det.calibration_error)

    # -- Marqueurs hors A4 ignorés ─────────────────────────────────────────────

    def test_marqueurs_hors_a4_ignores(self):
        """IDs 1, 2, 7 ne font pas partie de la feuille A4 et doivent être filtrés."""
        poses = self._perfect_poses([3, 4, 5])
        poses[99] = {"tvec": np.array([0.5, 0.5, 0.5]).reshape(3, 1)}
        ok = self.det.calibrate_camera_to_a4(poses)
        self.assertTrue(ok)  # 3 marqueurs valides suffisent

    # -- Correction réflexion (déterminant négatif) ────────────────────────────

    def test_rotation_propre(self):
        """Après calibration, det(R) == +1 (pas une réflexion)."""
        self.det.calibrate_camera_to_a4(self._perfect_poses([3, 4, 5, 6]))
        det = np.linalg.det(self.det.R_cam_to_world)
        self.assertAlmostEqual(det, 1.0, places=6)

    def test_rotation_propre_avec_donnees_bruitees(self):
        """Même avec du bruit, la matrice de rotation reste propre."""
        rng = np.random.default_rng(42)
        poses = self._perfect_poses([3, 4, 5, 6])
        for mid in poses:
            poses[mid]["tvec"] += rng.normal(0, 0.001, (3, 1))
        self.det.calibrate_camera_to_a4(poses)
        det = np.linalg.det(self.det.R_cam_to_world)
        self.assertAlmostEqual(det, 1.0, places=5)


# ══════════════════════════════════════════════════════════════════════════════
#  transform_camera_to_world
# ══════════════════════════════════════════════════════════════════════════════

class TestTransformCameraToWorld(unittest.TestCase):

    def setUp(self) -> None:
        self.det = _make_detector()
        angle = np.pi / 6
        c, s = np.cos(angle), np.sin(angle)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        self.T = np.array([0.2, 0.1, 0.0])
        self.det.R_cam_to_world = self.R
        self.det.T_cam_to_world = self.T.reshape(3, 1)

    def test_transformation_identite(self):
        """R=I, T=0 → p_world == p_cam."""
        self.det.R_cam_to_world = np.eye(3)
        self.det.T_cam_to_world = np.zeros((3, 1))
        p = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(self.det.transform_camera_to_world(p), p)

    def test_transformation_translation_pure(self):
        """R=I, T=[1,2,3] → p_world = p_cam + T."""
        self.det.R_cam_to_world = np.eye(3)
        self.det.T_cam_to_world = np.array([[1.0], [2.0], [3.0]])
        p = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            self.det.transform_camera_to_world(p), [1.0, 2.0, 3.0]
        )

    def test_transformation_rotation_connue(self):
        """Vérifie R @ p + T avec valeurs attendues."""
        p_cam = np.array([0.1, 0.0, 0.5])
        expected = self.R @ p_cam + self.T
        result = self.det.transform_camera_to_world(p_cam)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sans_calibration_retourne_none(self):
        self.det.R_cam_to_world = None
        result = self.det.transform_camera_to_world(np.array([0.0, 0.0, 1.0]))
        self.assertIsNone(result)

    def test_aller_retour(self):
        """transform → inverse doit redonner le point original."""
        p_cam = np.array([0.05, -0.03, 0.6])
        p_world = self.det.transform_camera_to_world(p_cam)
        # Inverse : p_cam = R.T @ (p_world - T)
        p_cam_back = self.R.T @ (p_world - self.T)
        np.testing.assert_allclose(p_cam_back, p_cam, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
#  generate_aruco_marker (generer_marqueurs_aruco.py)
# ══════════════════════════════════════════════════════════════════════════════

_GEN_SRC = os.path.abspath(os.path.join(_HERE, "..", "src", "aruco"))
sys.path.insert(0, _GEN_SRC)

from generer_marqueurs_aruco import generate_aruco_marker, create_a4_marker_sheet  # noqa: E402


class TestGenerateArucoMarker(unittest.TestCase):

    def test_forme_image_attendue(self):
        """L'image générée doit avoir la bonne taille (size + 2*margin, avec texte)."""
        size = 200
        margin = 20
        img = generate_aruco_marker(3, size)
        # Hauteur : marker + 2*margin + espace texte (cv2.getTextSize dépend de la police)
        # On vérifie au minimum que la largeur == size + 2*margin
        self.assertEqual(img.shape[1], size + 2 * margin)
        # et que c'est une image 2D (niveaux de gris)
        self.assertEqual(img.ndim, 2)

    def test_format_uint8(self):
        img = generate_aruco_marker(4, 100)
        self.assertEqual(img.dtype, np.uint8)

    def test_differents_ids_differentes_images(self):
        img3 = generate_aruco_marker(3, 200)
        img6 = generate_aruco_marker(6, 200)
        self.assertFalse(np.array_equal(img3, img6))

    def test_ids_valides_3_a_6(self):
        """Aucune exception pour les IDs 3, 4, 5, 6."""
        for mid in [3, 4, 5, 6]:
            with self.subTest(marker_id=mid):
                img = generate_aruco_marker(mid, 200)
                self.assertIsNotNone(img)


class TestCreateA4MarkerSheet(unittest.TestCase):
    """create_a4_marker_sheet() sauvegarde un fichier PNG; on vérifie ce fichier."""

    _OUTPUT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "aruco_markers_a4.png"
    )

    @classmethod
    def setUpClass(cls):
        """Appelle la fonction une seule fois pour générer le fichier."""
        # On la lance depuis le dossier racine pour ne pas polluer ailleurs
        import os as _os
        _orig = _os.getcwd()
        _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
        _os.chdir(_root)
        try:
            create_a4_marker_sheet()
        finally:
            _os.chdir(_orig)
        cls._sheet = cv2.imread(
            _os.path.join(_root, "aruco_markers_a4.png"),
            cv2.IMREAD_GRAYSCALE,
        )

    def test_fichier_cree(self):
        self.assertIsNotNone(self._sheet)

    def test_dimensions_a4_300dpi(self):
        expected_w = int(297 * 11.8)
        expected_h = int(210 * 11.8)
        self.assertEqual(self._sheet.shape[1], expected_w)
        self.assertEqual(self._sheet.shape[0], expected_h)

    def test_image_2d_niveaux_gris(self):
        self.assertEqual(self._sheet.ndim, 2)

    def test_4_marqueurs_non_blancs(self):
        """La feuille ne doit pas être entièrement blanche (les marqueurs sont placés)."""
        self.assertLess(int(self._sheet.min()), 255)


if __name__ == "__main__":
    unittest.main()
