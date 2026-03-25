"""Tests unitaires pour cross_q.py.

Couvre :
  - BatchNormCritic : architecture, forward pass, compatibilité CrossQ
  - Constantes globales (hyperparamètres)
  - make_crossq_sac : type de retour et configuration SAC

Exécution :
  python -m pytest tests/robot/test_cross_q.py -v
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

# ── Mock des dépendances MuJoCo avant l'import de cross_q ─────────────────────
# cross_q.py importe `from robot_env.reaching_env import ReachingEnv`
# On injecte un module factice pour éviter de charger MuJoCo / la simulation.
_mock_robot_env = types.ModuleType("robot_env")
_mock_reaching_env = types.ModuleType("robot_env.reaching_env")
_mock_reaching_env.ReachingEnv = mock.MagicMock(name="ReachingEnv")
_mock_robot_env.reaching_env = _mock_reaching_env

sys.modules.setdefault("robot_env", _mock_robot_env)
sys.modules.setdefault("robot_env.reaching_env", _mock_reaching_env)

# ── Import du module sous test ─────────────────────────────────────────────────
import cross_q  # noqa: E402


# ── Environnement minimal Box→Box compatible SAC MlpPolicy ────────────────────
class _FlatEnv(gym.Env):
    """Env factice Box observation / Box action pour instancier SAC sans MuJoCo."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(6, dtype=np.float32), 0.0, False, False, {}


# ══════════════════════════════════════════════════════════════════════════════
#  BatchNormCritic – Architecture
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchNormCriticArchitecture(unittest.TestCase):
    """Architecture de BatchNormCritic : couches, BN, dimensions."""

    def setUp(self) -> None:
        self.model = cross_q.BatchNormCritic(input_dim=8, output_dim=1)

    def test_herite_de_nn_module(self) -> None:
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_architecture_defaut_deux_couches_256(self) -> None:
        """net_arch=[256,256] doit produire 3 Linear et 2 BatchNorm1d."""
        modules = list(self.model.net.children())
        linears = [m for m in modules if isinstance(m, torch.nn.Linear)]
        bns = [m for m in modules if isinstance(m, torch.nn.BatchNorm1d)]
        # 2 couches cachées + 1 couche de sortie = 3 Linear
        self.assertEqual(len(linears), 3)
        # 1 BN par couche cachée = 2 BN
        self.assertEqual(len(bns), 2)

    def test_architecture_personnalisee(self) -> None:
        """net_arch=[64,128,64] doit produire 4 Linear et 3 BatchNorm1d."""
        model = cross_q.BatchNormCritic(4, 1, net_arch=[64, 128, 64])
        modules = list(model.net.children())
        linears = [m for m in modules if isinstance(m, torch.nn.Linear)]
        bns = [m for m in modules if isinstance(m, torch.nn.BatchNorm1d)]
        self.assertEqual(len(linears), 4)
        self.assertEqual(len(bns), 3)

    def test_dim_entree_et_sortie(self) -> None:
        all_linear = [m for m in self.model.net.children()
                      if isinstance(m, torch.nn.Linear)]
        self.assertEqual(all_linear[0].in_features, 8)
        self.assertEqual(all_linear[-1].out_features, 1)

    def test_activation_relu_dans_couches_cachees(self) -> None:
        modules = list(self.model.net.children())
        relus = [m for m in modules if isinstance(m, torch.nn.ReLU)]
        # 1 ReLU par couche cachée pour net_arch=[256,256]
        self.assertEqual(len(relus), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  BatchNormCritic – Forward pass
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchNormCriticForward(unittest.TestCase):
    """Forward pass : formes, gradients, modes train/eval."""

    def setUp(self) -> None:
        self.model = cross_q.BatchNormCritic(input_dim=8, output_dim=1)
        self.model.train()

    def test_forme_sortie_standard(self) -> None:
        """La sortie doit avoir la forme (batch_size, output_dim)."""
        x = torch.randn(16, 8)
        y = self.model(x)
        self.assertEqual(y.shape, (16, 1))

    def test_batch_minimal_mode_train(self) -> None:
        """BatchNorm1d requiert au moins 2 échantillons en mode train."""
        x = torch.randn(2, 8)
        y = self.model(x)
        self.assertEqual(y.shape, (2, 1))

    def test_batch_size_un_mode_eval(self) -> None:
        """En mode eval, BN utilise les stats mobiles → batch_size=1 accepté."""
        # Warm-up pour initialiser running_mean / running_var
        self.model.train()
        self.model(torch.randn(8, 8))
        self.model.eval()
        y = self.model(torch.randn(1, 8))
        self.assertEqual(y.shape, (1, 1))

    def test_gradient_calculable(self) -> None:
        """Le backward ne doit pas lever d'exception."""
        x = torch.randn(8, 8)
        y = self.model(x)
        y.sum().backward()  # aucune exception attendue

    def test_concat_batch_crossq_style(self) -> None:
        """Simule l'usage CrossQ : concat([s_t, s_{t+1}]) dans un seul forward."""
        batch_s_t = torch.randn(32, 8)
        batch_s_tp1 = torch.randn(32, 8)
        x = torch.cat([batch_s_t, batch_s_tp1], dim=0)  # (64, 8)
        y = self.model(x)
        self.assertEqual(y.shape, (64, 1))

    def test_pas_de_nan_en_sortie(self) -> None:
        self.model.eval()
        self.model.train()
        self.model(torch.randn(8, 8))  # warm-up
        self.model.eval()
        y = self.model(torch.randn(8, 8))
        self.assertFalse(torch.isnan(y).any())


# ══════════════════════════════════════════════════════════════════════════════
#  Constantes globales
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossQConstantes(unittest.TestCase):
    """Vérifie les hyperparamètres clés définis en tête de module."""

    def test_utd_ratio_20(self) -> None:
        """Le ratio UTD doit être 20 (cœur de la contribution CrossQ)."""
        self.assertEqual(cross_q.UTD_RATIO, 20)

    def test_batch_size_256(self) -> None:
        self.assertEqual(cross_q.BATCH_SIZE, 256)

    def test_gamma_099(self) -> None:
        self.assertAlmostEqual(cross_q.GAMMA, 0.99)

    def test_tau_0005(self) -> None:
        self.assertAlmostEqual(cross_q.TAU, 0.005)

    def test_learning_rate_reduit(self) -> None:
        """LR réduit à 1e-4 pour stabiliser l'entraînement sous haute pression gradient."""
        self.assertAlmostEqual(cross_q.LEARNING_RATE, 1e-4)

    def test_ent_coef_auto(self) -> None:
        self.assertEqual(cross_q.ENT_COEF, "auto")

    def test_buffer_size_positif(self) -> None:
        self.assertGreater(cross_q.BUFFER_SIZE, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  make_crossq_sac
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeCrossqSac(unittest.TestCase):
    """Tests de make_crossq_sac : type de retour et hyperparamètres SAC."""

    def setUp(self) -> None:
        self.env = _FlatEnv()

    def tearDown(self) -> None:
        self.env.close()

    def test_retourne_instance_sac(self) -> None:
        from stable_baselines3 import SAC
        model = cross_q.make_crossq_sac(self.env, utd_ratio=1)
        self.assertIsInstance(model, SAC)

    def test_gradient_steps_correspond_utd_ratio(self) -> None:
        """gradient_steps doit correspondre au utd_ratio passé en argument."""
        for utd in (1, 5, 20):
            with self.subTest(utd=utd):
                model = cross_q.make_crossq_sac(self.env, utd_ratio=utd)
                self.assertEqual(model.gradient_steps, utd)

    def test_batch_size_conforme(self) -> None:
        model = cross_q.make_crossq_sac(self.env, utd_ratio=1)
        self.assertEqual(model.batch_size, cross_q.BATCH_SIZE)

    def test_gamma_conforme(self) -> None:
        model = cross_q.make_crossq_sac(self.env, utd_ratio=1)
        self.assertAlmostEqual(model.gamma, cross_q.GAMMA)

    def test_activation_relu_dans_policy_kwargs(self) -> None:
        """ReLU est préféré à Tanh avec BN pour éviter la saturation de gradient."""
        model = cross_q.make_crossq_sac(self.env, utd_ratio=1)
        act_fn = model.policy_kwargs.get("activation_fn")
        self.assertIs(act_fn, torch.nn.ReLU)


if __name__ == "__main__":
    unittest.main(verbosity=2)
