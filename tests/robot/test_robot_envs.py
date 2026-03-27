"""Tests unitaires pour les environnements du robot 3-DDL.

Couvre (sans MuJoCo ni matériel) :
  - PushInHoleEnv : curriculum, _compute_reward, _sample_cube_pos
  - SlidingEnv    : _compute_reward (phases approche / contact / grace)
  - SortingEnv    : target-lock (_choose_target, _compute_reward)
  - sim_to_real   : rad_to_dxl (conversions radians → Dynamixel)

Les classes Sim3Dofs et dynamixel_sdk sont mockées avant tout import.

Exécution :
    python -m pytest tests/robot/test_robot_envs.py -v
"""

from __future__ import annotations

import math
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

# ── Chemin vers src/robot ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROBOT_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src", "robot"))
sys.path.insert(0, _ROBOT_SRC)

# ── Mock de sim_3dofs (évite MuJoCo) ─────────────────────────────────────────
_mock_sim_module = types.ModuleType("sim_3dofs")


class _FakeSim:
    """Remplaçant léger de Sim3Dofs — toutes les méthodes retournent des valeurs
    contrôlables via des attributs publics."""

    n_actuators = 3

    def __init__(self, render_mode=None, scene_xml=None):
        self._qpos = np.zeros(3, dtype=float)
        self._ee_pos = np.zeros(3, dtype=float)
        self._cube_pos = np.array([0.1, 0.0, 0.0135], dtype=float)
        self._cylinder_pos = np.array([0.2, 0.0, 0.0135], dtype=float)

    # -- Getters contrôlables --
    def get_qpos(self): return self._qpos.copy()
    def get_end_effector_pos(self): return self._ee_pos.copy()
    def get_cube_pos(self): return self._cube_pos.copy()
    def get_cylinder_pos(self): return self._cylinder_pos.copy()
    def get_cube_yaw_cossin(self): return np.array([1.0, 0.0])

    # -- Actions ignorées dans les tests unitaires --
    def reset(self): pass
    def step(self, ctrl): pass
    def forward(self): pass
    def set_cube_pose(self, pos=None, quat=None): pass
    def set_cylinder_pose(self, pos=None, quat=None): pass
    def set_named_marker(self, name, pos): pass
    def render(self): return None
    def close(self): pass


_mock_sim_module.Sim3Dofs = _FakeSim  # type: ignore[attr-defined]
sys.modules["sim_3dofs"] = _mock_sim_module

# ── Mock de dynamixel_sdk (évite le port série) ───────────────────────────────
# On utilise des instances de MagicMock (pas la classe) pour que l'appel
# dxl.PortHandler(arg) soit un __call__ sur un mock existant — sans que les
# arguments soient interprétés comme un `spec` par MagicMock.__init__.
_mock_dxl = types.ModuleType("dynamixel_sdk")
for _cls in ["PortHandler", "PacketHandler", "GroupSyncWrite", "GroupSyncRead"]:
    setattr(_mock_dxl, _cls, mock.MagicMock())
sys.modules["dynamixel_sdk"] = _mock_dxl

# ── Imports des modules sous test ─────────────────────────────────────────────
# robot_env/ est un sous-dossier de src/robot — on l'ajoute au path aussi.
# On importe directement (sans le préfixe `robot_env.`) pour ne pas dépendre
# de la présence/absence du mock `robot_env` injecté par d'autres test_*.
_ROBOT_ENV_SRC = os.path.join(_ROBOT_SRC, "robot_env")
sys.path.insert(0, _ROBOT_ENV_SRC)

from push_in_hole_env import (  # noqa: E402
    PushInHoleEnv,
    CURRICULUM_MIN_DIST_START,
    CURRICULUM_MIN_DIST_END,
    CURRICULUM_EPISODES,
    HOLE_POS,
    CUBE_Z,
    SUCCESS_Z_THRESHOLD,
    APPROACH_SATURATION_DIST,
    STEP_TIME_PENALTY,
    ACTION_RATE_COEFF,
)
from sliding_env import (  # noqa: E402
    SlidingEnv,
    CONTACT_DISPLACEMENT,
    GRACE_STEPS,
    SUCCESS_DIST,
    SUCCESS_EE_DIST,
)
from sorting_env import (  # noqa: E402
    SortingEnv,
    SUCCESS_THRESHOLD,
    GOAL_CUBE_POS,
    GOAL_CYLINDER_POS,
)
import sim_to_real  # noqa: E402


# ── Helpers d'instanciation (sans reset MuJoCo) ───────────────────────────────

def _make_push_in_hole(episode: int = 100) -> PushInHoleEnv:
    env = PushInHoleEnv.__new__(PushInHoleEnv)
    env.render_mode = None
    env.sim = _FakeSim()
    env._hole_pos = HOLE_POS.copy()
    env._prev_action = np.zeros(3)
    env._step_count = 0
    env._episode_count = episode
    env.np_random = np.random.default_rng(0)
    return env


def _make_sliding() -> SlidingEnv:
    env = SlidingEnv.__new__(SlidingEnv)
    env.render_mode = None
    env.sim = _FakeSim()
    env._cube_init = np.array([0.1, 0.0, 0.0135])
    env._prev_action = np.zeros(3)
    env._step_count = 0
    env._contact_step = -1
    env.np_random = np.random.default_rng(0)
    return env


def _make_sorting() -> SortingEnv:
    env = SortingEnv.__new__(SortingEnv)
    env.render_mode = None
    env.sim = _FakeSim()
    env._goal_cube = GOAL_CUBE_POS.copy()
    env._goal_cylinder = GOAL_CYLINDER_POS.copy()
    env._prev_action = np.zeros(3)
    env._step_count = 0
    env._current_target = "cube"
    env.np_random = np.random.default_rng(0)
    return env


# ══════════════════════════════════════════════════════════════════════════════
#  PushInHoleEnv — Curriculum
# ══════════════════════════════════════════════════════════════════════════════

class TestPushInHoleCurriculum(unittest.TestCase):

    def test_dist_min_premiere_episode(self):
        """Au début (épisode 0) la distance min est celle de départ du curriculum."""
        env = _make_push_in_hole(episode=0)
        self.assertAlmostEqual(env._current_min_cube_hole_dist(),
                               CURRICULUM_MIN_DIST_START, places=6)

    def test_dist_min_apres_curriculum(self):
        """Après CURRICULUM_EPISODES épisodes, on atteint la valeur maximale."""
        env = _make_push_in_hole(episode=CURRICULUM_EPISODES)
        self.assertAlmostEqual(env._current_min_cube_hole_dist(),
                               CURRICULUM_MIN_DIST_END, places=6)

    def test_dist_min_progression_monotone(self):
        """La distance min augmente de façon monotone."""
        vals = [
            _make_push_in_hole(ep)._current_min_cube_hole_dist()
            for ep in range(0, CURRICULUM_EPISODES + 1, 200)
        ]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a, b)

    def test_dist_min_sature_apres_curriculum(self):
        """Au-delà de CURRICULUM_EPISODES, la valeur reste plafonnée."""
        env_a = _make_push_in_hole(episode=CURRICULUM_EPISODES)
        env_b = _make_push_in_hole(episode=CURRICULUM_EPISODES * 10)
        self.assertAlmostEqual(
            env_a._current_min_cube_hole_dist(),
            env_b._current_min_cube_hole_dist(),
            places=6,
        )

    def test_50_premiers_episodes_position_fixe(self):
        """Pour les 50 premiers épisodes, _sample_cube_pos retourne la position facilitée."""
        # On recree reset() manuellement (sans MuJoCo) pour verifier le spawn
        env = _make_push_in_hole(episode=0)
        for _ in range(50):
            if env._episode_count < 50:
                cube_pos = np.array([0.10, 0.05, CUBE_Z])
            else:
                cube_pos = env._sample_cube_pos()
            np.testing.assert_array_equal(cube_pos, [0.10, 0.05, CUBE_Z])
            env._episode_count += 1

    def test_apres_50_episodes_sample_aleatoire(self):
        """Après 50 épisodes, _sample_cube_pos génère des positions variées."""
        env = _make_push_in_hole(episode=100)
        positions = [env._sample_cube_pos() for _ in range(20)]
        # Au moins deux positions différentes (variabilité)
        unique = {tuple(p) for p in positions}
        self.assertGreater(len(unique), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  PushInHoleEnv — _compute_reward
# ══════════════════════════════════════════════════════════════════════════════

class TestPushInHoleReward(unittest.TestCase):

    def setUp(self):
        self.env = _make_push_in_hole(episode=100)

    def _set_positions(self, ee, cube):
        self.env.sim._ee_pos = np.array(ee)
        self.env.sim._cube_pos = np.array(cube)

    def test_reward_decrease_avec_distance_ee_cube(self):
        """Plus l'EE est loin du cube, plus le reward est faible."""
        self._set_positions([0.0, 0.0, 0.1], [0.1, 0.0, 0.0])
        r_proche, _ = self.env._compute_reward(np.zeros(3))
        self._set_positions([0.0, 0.0, 0.5], [0.1, 0.0, 0.0])
        r_loin, _ = self.env._compute_reward(np.zeros(3))
        self.assertLess(r_loin, r_proche)

    def test_saturation_approche_sous_threshold(self):
        """Sous APPROACH_SATURATION_DIST, le terme d'approche vaut 0."""
        # EE au même endroit que le cube → distance < seuil de saturation
        self._set_positions([0.1, 0.0, 0.0], [0.1, 0.0, 0.0])
        # Reward attendu : -5 * dist_cube_hole_xy - STEP_PENALTY (approche saturée)
        cube_pos = np.array([0.1, 0.0, 0.0])
        dist_xy = np.linalg.norm(cube_pos[:2] - HOLE_POS[:2])
        expected_base = -5.0 * dist_xy - STEP_TIME_PENALTY
        r, _ = self.env._compute_reward(np.zeros(3))
        self.assertAlmostEqual(r, expected_base, places=6)

    def test_reward_succes(self):
        """Le cube sous SUCCESS_Z_THRESHOLD déclenche is_success et bonus +100."""
        # Cube sous le sol
        self._set_positions([0.0, 0.0, 0.1], [HOLE_POS[0], HOLE_POS[1], -0.05])
        r, is_success = self.env._compute_reward(np.zeros(3))
        self.assertTrue(is_success)
        self.assertGreater(r, 50.0)  # bonus 100 largement dominant

    def test_pas_succes_cube_au_sol(self):
        """Le cube au niveau du sol (z ≥ threshold) ne déclenche pas le succès."""
        self._set_positions([0.0, 0.0, 0.1], [HOLE_POS[0], HOLE_POS[1], 0.0])
        _, is_success = self.env._compute_reward(np.zeros(3))
        self.assertFalse(is_success)

    def test_penalite_lissage_action(self):
        """Une action différente de prev_action ajoute une pénalité."""
        self._set_positions([0.1, 0.0, 0.0], [0.1, 0.0, 0.0])
        self.env._prev_action = np.zeros(3)
        r_nul, _ = self.env._compute_reward(np.zeros(3))
        action_brusque = np.ones(3)
        r_brusque, _ = self.env._compute_reward(action_brusque)
        self.assertLess(r_brusque, r_nul)

    def test_penalite_temporelle_constante(self):
        """Chaque step coûte STEP_TIME_PENALTY."""
        # EE collé au cube (approche saturée), pas d'action → seul le lissage manque
        self._set_positions([0.1, 0.0, 0.0], [0.1, 0.0, 0.0])
        self.env._prev_action = np.zeros(3)
        r, _ = self.env._compute_reward(np.zeros(3))
        # r = -5 * dist_xy - STEP_TIME_PENALTY
        cube_pos = np.array([0.1, 0.0, 0.0])
        dist_xy = np.linalg.norm(cube_pos[:2] - HOLE_POS[:2])
        expected = -5.0 * dist_xy - STEP_TIME_PENALTY
        self.assertAlmostEqual(r, expected, places=6)


# ══════════════════════════════════════════════════════════════════════════════
#  SlidingEnv — _compute_reward (3 phases)
# ══════════════════════════════════════════════════════════════════════════════

class TestSlidingReward(unittest.TestCase):

    def setUp(self):
        self.env = _make_sliding()

    def _set(self, ee, cube):
        self.env.sim._ee_pos = np.array(ee)
        self.env.sim._cube_pos = np.array(cube)

    # Phase 1 : pas encore de contact ─────────────────────────────────────────

    def test_phase_approche_reward_negatif_proportionnel(self):
        """Sans contact, reward == -dist(ee, cube)."""
        # On aligne _cube_init et cube_pos pour que displacement == 0 (pas de contact)
        self.env._cube_init = np.array([0.1, 0.0, 0.0])
        self._set([0.0, 0.0, 0.0], [0.1, 0.0, 0.0])
        r, _ = self.env._compute_reward(np.zeros(3))
        expected = -np.linalg.norm(np.array([0.1, 0.0, 0.0]))
        self.assertAlmostEqual(r, expected, places=5)

    def test_phase_approche_pas_succes(self):
        """Sans contact, is_success est False."""
        # On aligne _cube_init et cube_pos pour que displacement == 0 (pas de contact)
        self.env._cube_init = np.array([0.15, 0.0, 0.0])
        self._set([0.0, 0.0, 0.0], [0.15, 0.0, 0.0])
        _, is_success = self.env._compute_reward(np.zeros(3))
        self.assertFalse(is_success)

    # Phase 2 : après contact ─────────────────────────────────────────────────

    def test_phase_contact_reward_positif(self):
        """Après un déplacement > CONTACT_DISPLACEMENT, reward > 0."""
        # Cube déplacé de 3 cm
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        self._set([0.0, 0.0, 0.0], [0.03, 0.0, 0.0135])
        r, _ = self.env._compute_reward(np.zeros(3))
        self.assertGreater(r, 0.0)

    def test_premier_contact_enregistre(self):
        """Le step du premier contact est mémorisé dans _contact_step."""
        self.env._step_count = 5
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        self._set([0.0, 0.0, 0.0], [0.03, 0.0, 0.0135])
        self.env._compute_reward(np.zeros(3))
        self.assertEqual(self.env._contact_step, 5)

    def test_deuxieme_contact_ne_modifie_pas_contact_step(self):
        """Une fois _contact_step fixé, un deuxième appel ne le change pas."""
        self.env._step_count = 3
        self.env._contact_step = 3
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        self._set([0.0, 0.0, 0.0], [0.05, 0.0, 0.0135])
        self.env._step_count = 10
        self.env._compute_reward(np.zeros(3))
        self.assertEqual(self.env._contact_step, 3)

    # Succès ──────────────────────────────────────────────────────────────────

    def test_succes_cube_deplace_assez_loin(self):
        """is_success == True si déplacement > SUCCESS_DIST et EE loin du cube."""
        displacement = SUCCESS_DIST + 0.01
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        self._set([5.0, 0.0, 0.0], [displacement, 0.0, 0.0135])
        _, is_success = self.env._compute_reward(np.zeros(3))
        self.assertTrue(is_success)

    def test_pas_succes_si_ee_trop_proche(self):
        """Pas de succès si EE encore trop proche même si le cube a glissé."""
        displacement = SUCCESS_DIST + 0.01
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        # EE collé au cube
        self._set([displacement, 0.0, 0.0135], [displacement, 0.0, 0.0135])
        _, is_success = self.env._compute_reward(np.zeros(3))
        self.assertFalse(is_success)

    def test_penalite_colle_apres_grace(self):
        """Après GRACE_STEPS depuis le contact, si l'EE est encore très proche du cube → pénalité."""
        self.env._cube_init = np.array([0.0, 0.0, 0.0135])
        self.env._contact_step = 0
        self.env._step_count = GRACE_STEPS + 1
        # EE collé au cube (dist < 0.03)
        self._set([0.02, 0.0, 0.0135], [0.02 + CONTACT_DISPLACEMENT + 0.001, 0.0, 0.0135])
        r_colle, _ = self.env._compute_reward(np.zeros(3))
        # Sans pénalité (EE loin)
        self._set([5.0, 0.0, 0.0], [0.02 + CONTACT_DISPLACEMENT + 0.001, 0.0, 0.0135])
        r_loin, _ = self.env._compute_reward(np.zeros(3))
        self.assertLess(r_colle, r_loin)


# ══════════════════════════════════════════════════════════════════════════════
#  SortingEnv — target-lock (_choose_target) et _compute_reward
# ══════════════════════════════════════════════════════════════════════════════

class TestSortingTargetLock(unittest.TestCase):

    def setUp(self):
        self.env = _make_sorting()

    def _place_cube_at_goal(self):
        self.env.sim._cube_pos = GOAL_CUBE_POS.copy()

    def _place_cylinder_at_goal(self):
        self.env.sim._cylinder_pos = GOAL_CYLINDER_POS.copy()

    def _place_both_far(self):
        self.env.sim._cube_pos = np.array([0.0, 0.0, 0.0135])
        self.env.sim._cylinder_pos = np.array([0.0, 0.05, 0.0135])

    def test_cible_initiale_cube(self):
        self._place_both_far()
        self.assertEqual(self.env._current_target, "cube")

    def test_bascule_vers_cylindre_quand_cube_trie(self):
        """Quand le cube est trié, _choose_target bascule vers cylinder."""
        self._place_cube_at_goal()
        self.env.sim._cylinder_pos = np.array([0.0, 0.0, 0.0135])  # pas trié
        target = self.env._choose_target()
        self.assertEqual(target, "cylinder")

    def test_bascule_vers_cube_quand_cylindre_trie(self):
        """Quand le cylindre est trié, _choose_target bascule vers cube."""
        self._place_cylinder_at_goal()
        self.env.sim._cube_pos = np.array([0.0, 0.0, 0.0135])  # pas trié
        target = self.env._choose_target()
        self.assertEqual(target, "cube")

    def test_cible_maintenue_si_aucun_trie(self):
        """Si aucun objet n'est trié, la cible courante est conservée."""
        self._place_both_far()
        self.env._current_target = "cylinder"
        target = self.env._choose_target()
        self.assertEqual(target, "cylinder")

    def test_cible_maintenue_si_les_deux_tries(self):
        """Si les deux objets sont triés, la cible courante est conservée."""
        self._place_cube_at_goal()
        self._place_cylinder_at_goal()
        self.env._current_target = "cube"
        target = self.env._choose_target()
        self.assertEqual(target, "cube")


class TestSortingReward(unittest.TestCase):

    def setUp(self):
        self.env = _make_sorting()
        self.env.sim._ee_pos = np.zeros(3)

    def test_reward_diminue_avec_distance_objet_cible(self):
        """Plus l'objet cible est loin de son goal, plus le reward est faible."""
        self.env.sim._cube_pos = np.array([0.05, 0.0, 0.0])  # loin du goal
        r_loin, *_ = self.env._compute_reward(np.zeros(3))
        self.env.sim._cube_pos = GOAL_CUBE_POS.copy()  # sur le goal
        r_pres, *_ = self.env._compute_reward(np.zeros(3))
        self.assertLess(r_loin, r_pres)

    def test_bonus_succes_200_quand_tout_trie(self):
        """Bonus +200 uniquement quand cube ET cylindre sont triés."""
        self.env.sim._cube_pos = GOAL_CUBE_POS.copy()
        self.env.sim._cylinder_pos = GOAL_CYLINDER_POS.copy()
        r, *_, both = self.env._compute_reward(np.zeros(3))
        self.assertTrue(both)
        self.assertGreater(r, 100.0)  # +200 dominant

    def test_pas_bonus_si_seulement_cube_trie(self):
        """Pas de bonus 200 si seulement le cube est au goal."""
        self.env.sim._cube_pos = GOAL_CUBE_POS.copy()
        self.env.sim._cylinder_pos = np.array([0.0, 0.0, 0.0135])
        _, _, _, both = self.env._compute_reward(np.zeros(3))
        self.assertFalse(both)

    def test_penalite_lissage(self):
        """Action brusque → reward plus faible (pénalité lissage)."""
        self.env.sim._cube_pos = np.array([0.1, 0.0, 0.0])
        self.env._prev_action = np.zeros(3)
        r_lisse, *_ = self.env._compute_reward(np.zeros(3))
        action_brusque = np.ones(3) * 2.0
        r_brusque, *_ = self.env._compute_reward(action_brusque)
        self.assertLess(r_brusque, r_lisse)

    def test_get_target_obj_pos_cube(self):
        self.env._current_target = "cube"
        pos = self.env._get_target_obj_pos()
        np.testing.assert_array_equal(pos, self.env.sim.get_cube_pos())

    def test_get_target_obj_pos_cylinder(self):
        self.env._current_target = "cylinder"
        pos = self.env._get_target_obj_pos()
        np.testing.assert_array_equal(pos, self.env.sim.get_cylinder_pos())


# ══════════════════════════════════════════════════════════════════════════════
#  sim_to_real — rad_to_dxl
# ══════════════════════════════════════════════════════════════════════════════

class TestRadToDxl(unittest.TestCase):

    def test_zero_rad_donne_centre(self):
        """0 rad = position centrale = 512."""
        self.assertEqual(sim_to_real.rad_to_dxl(0.0), 512)

    def test_valeur_positive_superieure_centre(self):
        """Un angle positif > 0 donne une valeur > 512."""
        self.assertGreater(sim_to_real.rad_to_dxl(0.5), 512)

    def test_valeur_negative_inferieure_centre(self):
        """Un angle négatif < 0 donne une valeur < 512."""
        self.assertLess(sim_to_real.rad_to_dxl(-0.5), 512)

    def test_saturation_max(self):
        """Grande valeur positive → plafonnée à 850."""
        self.assertEqual(sim_to_real.rad_to_dxl(10.0), 850)

    def test_saturation_min(self):
        """Grande valeur négative → plafonnée à 250."""
        self.assertEqual(sim_to_real.rad_to_dxl(-10.0), 250)

    def test_formule_connue(self):
        """Vérifie la formule : raw = angle * 1024 / (300° en rad) + 512."""
        angle = math.pi / 6  # 30°
        expected_raw = int(round(angle * 1024 / (300 * math.pi / 180) + 512))
        expected_clipped = max(250, min(850, expected_raw))
        self.assertEqual(sim_to_real.rad_to_dxl(angle), expected_clipped)

    def test_centre_custom(self):
        """Le paramètre `center` décale la sortie."""
        val_512 = sim_to_real.rad_to_dxl(0.0, center=512)
        val_600 = sim_to_real.rad_to_dxl(0.0, center=600)
        self.assertEqual(val_512, 512)
        self.assertEqual(val_600, 600)

    def test_retourne_entier(self):
        self.assertIsInstance(sim_to_real.rad_to_dxl(0.3), int)

    def test_symetrie(self):
        """rad_to_dxl(+a) et rad_to_dxl(-a) sont symétriques par rapport à 512."""
        for angle in [0.1, 0.5, 1.0]:
            v_pos = sim_to_real.rad_to_dxl(angle)
            v_neg = sim_to_real.rad_to_dxl(-angle)
            self.assertEqual(v_pos + v_neg, 1024,
                             msg=f"Symétrie rompue pour angle={angle}")


if __name__ == "__main__":
    unittest.main()
