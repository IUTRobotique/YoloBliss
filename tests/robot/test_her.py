"""Tests unitaires pour her.py (SortingGoalEnv + HER-SAC).

Couvre :
  - SortingGoalEnv : espaces d'observation/action, reset, step
  - compute_reward : cas scalaire, batch, bonus de tri, dtype
  - Constantes globales HER + SAC
  - make_her_sac : type de retour, HerReplayBuffer, hyperparamètres

L'env interne SortingEnv (qui charge MuJoCo) est remplacé par un mock
léger injecté dans sys.modules avant l'import de her.py.

Exécution :
  python -m pytest tests/robot/test_her.py -v
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# ── Chemin vers src/robot ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROBOT_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src", "robot"))
sys.path.insert(0, _ROBOT_SRC)

# ══════════════════════════════════════════════════════════════════════════════
#  Mock de robot_env.sorting_env (évite le chargement de MuJoCo)
# ══════════════════════════════════════════════════════════════════════════════

# Valeur identique à sorting_env.SUCCESS_THRESHOLD
_SUCCESS_THRESHOLD = 0.05


class _MockSortingEnv:
    """Remplaçant léger de SortingEnv sans simulation MuJoCo."""

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        # Objectifs par défaut
        self._goal_cube = np.array([0.15, 0.0, 0.0], dtype=np.float32)
        self._goal_cylinder = np.array([0.25, 0.0, 0.0], dtype=np.float32)

        # sim factice
        self.sim = mock.MagicMock()
        self.sim.get_qpos.return_value = np.zeros(3, dtype=np.float32)
        self.sim.get_end_effector_pos.return_value = np.zeros(3, dtype=np.float32)
        self.sim.get_cube_pos.return_value = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        self.sim.get_cylinder_pos.return_value = np.array([0.2, 0.0, 0.0], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(12, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(12, dtype=np.float32)
        info = {
            "is_success": False,
            "cube_sorted": False,
            "cylinder_sorted": False,
            "dist_cube_goal": 0.05,
            "dist_cylinder_goal": 0.05,
        }
        return obs, 0.0, False, False, info

    def render(self):
        pass

    def close(self) -> None:
        pass


# Injection du module factice AVANT d'importer her
_mock_sorting_module = types.ModuleType("robot_env.sorting_env")
_mock_sorting_module.SortingEnv = _MockSortingEnv  # type: ignore[attr-defined]
_mock_sorting_module.SUCCESS_THRESHOLD = _SUCCESS_THRESHOLD  # type: ignore[attr-defined]

# her.py importe aussi robot_env.push_env et robot_env.sliding_env —
# on injecte des modules factices pour éviter le chargement de MuJoCo.
_mock_push_module = types.ModuleType("robot_env.push_env")
_mock_push_module.PushEnv = mock.MagicMock(name="PushEnv")  # type: ignore[attr-defined]
_mock_sliding_module = types.ModuleType("robot_env.sliding_env")
_mock_sliding_module.SlidingEnv = mock.MagicMock(name="SlidingEnv")  # type: ignore[attr-defined]

_mock_robot_env = types.ModuleType("robot_env")
sys.modules.setdefault("robot_env", _mock_robot_env)
sys.modules["robot_env.sorting_env"] = _mock_sorting_module
sys.modules["robot_env.push_env"] = _mock_push_module
sys.modules["robot_env.sliding_env"] = _mock_sliding_module

# ── Import du module sous test ─────────────────────────────────────────────────
# her_sorting.py contient SortingGoalEnv, make_her_sac et les constantes HER.
import her_sorting as her  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  GoalEnv minimal pour tester make_her_sac sans SortingGoalEnv réel
# ══════════════════════════════════════════════════════════════════════════════

class _MockGoalEnv(gym.Env):
    """GoalEnv factice avec Dict observation space à dim 6, compatible HerReplayBuffer."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def _zeros_obs(self) -> dict[str, np.ndarray]:
        return {k: np.zeros(s.shape, dtype=np.float32)
                for k, s in self.observation_space.spaces.items()}

    def reset(self, **kwargs):
        return self._zeros_obs(), {}

    def step(self, action):
        return self._zeros_obs(), 0.0, False, False, {}

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
    ) -> np.ndarray:
        n = achieved_goal.shape[0] if achieved_goal.ndim > 1 else 1
        return np.zeros(n, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – Constantes
# ══════════════════════════════════════════════════════════════════════════════

class TestHERConstantes(unittest.TestCase):
    """Vérifie les hyperparamètres définis en tête de her.py."""

    def test_n_sampled_goal_4(self) -> None:
        """4 buts virtuels relabellisés par transition réelle."""
        self.assertEqual(her.N_SAMPLED_GOAL, 4)

    def test_batch_size_256(self) -> None:
        self.assertEqual(her.BATCH_SIZE, 256)

    def test_gamma_099(self) -> None:
        self.assertAlmostEqual(her.GAMMA, 0.99)

    def test_learning_rate(self) -> None:
        self.assertAlmostEqual(her.LEARNING_RATE, 3e-4)

    def test_gradient_steps_100(self) -> None:
        self.assertEqual(her.GRADIENT_STEPS, 100)

    def test_buffer_size_positif(self) -> None:
        self.assertGreater(her.BUFFER_SIZE, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – SortingGoalEnv : espaces d'observation et d'action
# ══════════════════════════════════════════════════════════════════════════════

class TestSortingGoalEnvEspaces(unittest.TestCase):
    """Espaces d'observation (Dict GoalEnv) et d'action de SortingGoalEnv."""

    def setUp(self) -> None:
        self.env = her.SortingGoalEnv(render_mode=None)

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_space_est_dict(self) -> None:
        self.assertIsInstance(self.env.observation_space, spaces.Dict)

    def test_cles_observation_space(self) -> None:
        keys = set(self.env.observation_space.spaces.keys())
        self.assertEqual(keys, {"observation", "achieved_goal", "desired_goal"})

    def test_dim_observation_6(self) -> None:
        """observation = [qpos(3) | ee_pos(3)]."""
        self.assertEqual(
            self.env.observation_space.spaces["observation"].shape, (6,)
        )

    def test_dim_achieved_goal_6(self) -> None:
        """achieved_goal = [cube_pos(3) | cylinder_pos(3)]."""
        self.assertEqual(
            self.env.observation_space.spaces["achieved_goal"].shape, (6,)
        )

    def test_dim_desired_goal_6(self) -> None:
        """desired_goal = [goal_cube(3) | goal_cylinder(3)]."""
        self.assertEqual(
            self.env.observation_space.spaces["desired_goal"].shape, (6,)
        )

    def test_action_space_delegue_a_inner(self) -> None:
        """L'action space de SortingGoalEnv doit être celui de l'env interne."""
        self.assertIsInstance(self.env.action_space, spaces.Box)
        self.assertEqual(self.env.action_space.shape, (3,))


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – compute_reward
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeReward(unittest.TestCase):
    """Récompense relabellisable : cas scalaire, batch, bonus de tri, dtype."""

    def setUp(self) -> None:
        self.env = her.SortingGoalEnv(render_mode=None)

    def tearDown(self) -> None:
        self.env.close()

    def test_reward_max_quand_les_deux_tries(self) -> None:
        """Les deux objets au but → reward = -0 -0 + 20 + 20 + 50 = 90."""
        ag = np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
        dg = np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
        r = self.env.compute_reward(ag, dg, {})
        self.assertAlmostEqual(float(r), 90.0, places=4)

    def test_reward_negatif_si_les_deux_loin(self) -> None:
        """Les deux objets loin du but → récompense strictement négative."""
        ag = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dg = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        r = self.env.compute_reward(ag, dg, {})
        self.assertLess(float(r), 0.0)

    def test_bonus_cube_seulement(self) -> None:
        """Cube trié, cylindre loin → reward positif mais < 20."""
        ag = np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dg = np.array([0.15, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        r = float(self.env.compute_reward(ag, dg, {}))
        # Cube trié → +20, cyl loin (0.5) → -0.5 : r = 19.5
        self.assertGreater(r, 0.0)
        self.assertLess(r, 20.0)

    def test_bonus_cylindre_seulement(self) -> None:
        """Cylindre trié, cube loin → reward positif mais < 90."""
        ag = np.array([0.5, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
        dg = np.array([0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
        r = float(self.env.compute_reward(ag, dg, {}))
        self.assertGreater(r, 0.0)
        self.assertLess(r, 90.0)

    def test_compute_reward_accepte_batch(self) -> None:
        """compute_reward doit accepter des tableaux (B, 6)."""
        ag = np.array([
            [0.15, 0.0, 0.0, 0.25, 0.0, 0.0],   # les deux au but
            [0.00, 0.0, 0.0, 0.00, 0.0, 0.0],   # les deux loin
            [0.15, 0.0, 0.0, 0.00, 0.0, 0.0],   # cube trié, cyl loin
        ], dtype=np.float32)
        dg = np.array([
            [0.15, 0.0, 0.0, 0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0, 0.50, 0.0, 0.0],
            [0.15, 0.0, 0.0, 0.50, 0.0, 0.0],
        ], dtype=np.float32)
        r = self.env.compute_reward(ag, dg, {})
        self.assertEqual(r.shape, (3,))
        self.assertAlmostEqual(float(r[0]), 90.0, places=4)  # max
        self.assertLess(float(r[1]), 0.0)                    # négatif
        self.assertGreater(float(r[2]), 0.0)                 # reward partiel

    def test_dtype_float32(self) -> None:
        ag = np.array([0.10, 0.0, 0.0, 0.20, 0.0, 0.0], dtype=np.float32)
        dg = np.array([0.15, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
        r = self.env.compute_reward(ag, dg, {})
        self.assertEqual(r.dtype, np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – SortingGoalEnv : reset
# ══════════════════════════════════════════════════════════════════════════════

class TestSortingGoalEnvReset(unittest.TestCase):
    """reset() doit renvoyer un dict GoalEnv valide."""

    def setUp(self) -> None:
        self.env = her.SortingGoalEnv(render_mode=None)

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_retourne_dict(self) -> None:
        obs, info = self.env.reset()
        self.assertIsInstance(obs, dict)

    def test_reset_cles_observation(self) -> None:
        obs, _ = self.env.reset()
        self.assertSetEqual(
            set(obs.keys()), {"observation", "achieved_goal", "desired_goal"}
        )

    def test_reset_formes_correctes(self) -> None:
        obs, _ = self.env.reset()
        self.assertEqual(obs["observation"].shape, (6,))
        self.assertEqual(obs["achieved_goal"].shape, (6,))
        self.assertEqual(obs["desired_goal"].shape, (6,))

    def test_reset_info_contient_goal_cube(self) -> None:
        _, info = self.env.reset()
        self.assertIn("goal_cube", info)
        self.assertEqual(info["goal_cube"].shape, (3,))

    def test_reset_info_contient_goal_cylinder(self) -> None:
        _, info = self.env.reset()
        self.assertIn("goal_cylinder", info)
        self.assertEqual(info["goal_cylinder"].shape, (3,))


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – SortingGoalEnv : step
# ══════════════════════════════════════════════════════════════════════════════

class TestSortingGoalEnvStep(unittest.TestCase):
    """step() doit renvoyer les 5-tuple GoalEnv avec les bons types."""

    def setUp(self) -> None:
        self.env = her.SortingGoalEnv(render_mode=None)
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_step_retourne_dict_obs(self) -> None:
        obs, _, _, _, _ = self.env.step(self.env.action_space.sample())
        self.assertIsInstance(obs, dict)

    def test_step_cles_obs(self) -> None:
        obs, _, _, _, _ = self.env.step(self.env.action_space.sample())
        self.assertSetEqual(
            set(obs.keys()), {"observation", "achieved_goal", "desired_goal"}
        )

    def test_step_reward_est_float(self) -> None:
        _, reward, _, _, _ = self.env.step(self.env.action_space.sample())
        self.assertIsInstance(reward, float)

    def test_step_info_contient_is_success(self) -> None:
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        self.assertIn("is_success", info)

    def test_step_info_contient_cube_sorted(self) -> None:
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        self.assertIn("cube_sorted", info)

    def test_step_info_contient_cylinder_sorted(self) -> None:
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        self.assertIn("cylinder_sorted", info)

    def test_step_info_contient_dist_cube_goal(self) -> None:
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        self.assertIn("dist_cube_goal", info)

    def test_step_info_contient_dist_cylinder_goal(self) -> None:
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        self.assertIn("dist_cylinder_goal", info)


# ══════════════════════════════════════════════════════════════════════════════
#  Tests – make_her_sac
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeHerSac(unittest.TestCase):
    """make_her_sac : type de retour, replay buffer HER, hyperparamètres."""

    def setUp(self) -> None:
        self.env = _MockGoalEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_retourne_instance_sac(self) -> None:
        from stable_baselines3 import SAC
        model = her.make_her_sac(self.env)
        self.assertIsInstance(model, SAC)

    def test_replay_buffer_est_her(self) -> None:
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
        model = her.make_her_sac(self.env)
        self.assertIsInstance(model.replay_buffer, HerReplayBuffer)

    def test_n_sampled_goal_conforme(self) -> None:
        model = her.make_her_sac(self.env)
        self.assertEqual(model.replay_buffer.n_sampled_goal, her.N_SAMPLED_GOAL)

    def test_goal_selection_strategy_future(self) -> None:
        from stable_baselines3.her.her_replay_buffer import GoalSelectionStrategy
        model = her.make_her_sac(self.env)
        strategy = model.replay_buffer.goal_selection_strategy
        self.assertEqual(strategy, GoalSelectionStrategy.FUTURE)

    def test_batch_size_conforme(self) -> None:
        model = her.make_her_sac(self.env)
        self.assertEqual(model.batch_size, her.BATCH_SIZE)

    def test_activation_relu_dans_policy_kwargs(self) -> None:
        """ReLU est utilisé dans HER SAC."""
        model = her.make_her_sac(self.env)
        act_fn = model.policy_kwargs.get("activation_fn")
        self.assertIs(act_fn, torch.nn.ReLU)


if __name__ == "__main__":
    unittest.main(verbosity=2)
