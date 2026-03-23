"""Couche simulation : encapsule l'accès à MuJoCo pour le robot 3-DDL.

Cette classe est indépendante de Gymnasium.  Elle expose uniquement les
primitives physiques (chargement du modèle, intégration numérique, lecture
des capteurs, envoi des commandes) ainsi que la visualisation.
"""

from __future__ import annotations

import os
from collections import deque

import mujoco
import numpy as np


# Chemin vers la scène MuJoCo
SCENE_XML = os.path.join(os.path.dirname(__file__), "/robot_env/scene_push.xml")

# Pas de simulation MuJoCo par défaut (s)
DEFAULT_SIM_DT: float = 0.005   # 5 ms => 200 Hz

# Pas de contrôle (action) par défaut (s)
DEFAULT_CTRL_DT: float = 0.04   # 40 ms => 25 Hz

# Plage de délai par défaut pour la lecture des capteurs et l'application des actions (s)
DEFAULT_MIN_DELAY_S: float = 0.005  # 5 ms
DEFAULT_MAX_DELAY_S: float = 0.015  # 15 ms


class Sim3Dofs:
    """Wrapper MuJoCo pour le robot 3-DDL.

    Parameters
    ----------
    render_mode:
        ``"human"``    → viewer interactif (mujoco.viewer)
        ``"rgb_array"`` → rendu hors-écran (mujoco.Renderer)
        ``None``        → pas de rendu
    """

    def __init__(
        self,
        render_mode: str | None = None,
        sim_dt: float = DEFAULT_SIM_DT,
        ctrl_dt: float = DEFAULT_CTRL_DT,
        min_delay_s: float = DEFAULT_MIN_DELAY_S,
        max_delay_s: float = DEFAULT_MAX_DELAY_S,
        scene_xml: str = SCENE_XML,
    ) -> None:
        self.render_mode = render_mode
        self.sim_dt: float = sim_dt
        self.ctrl_dt: float = ctrl_dt
        self.min_delay_s: float = min_delay_s
        self.max_delay_s: float = max_delay_s
        self._n_substeps: int = max(1, round(ctrl_dt / sim_dt))
        # Conversion des délais (s) en nombre de sim-steps
        _min_delay_steps = max(0, round(min_delay_s / sim_dt))
        _max_delay_steps = max(_min_delay_steps, round(max_delay_s / sim_dt))
        self._obs_delay_range = (_min_delay_steps, _max_delay_steps)
        self._act_delay_range = (_min_delay_steps, _max_delay_steps)
        self._rng = np.random.default_rng()

        # ── Modèle / données ────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = sim_dt

        # Ids utiles : "capteur" et actionneurs
        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )
        self._actuator_ids: list[int] = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("1", "2", "3")
        ]
        self.n_actuators: int = len(self._actuator_ids)

        # Id du goal_marker (−1 si absent du modèle)
        _marker_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker"
        )
        self._goal_mocap_id: int = (
            int(self.model.body_mocapid[_marker_body])
            if _marker_body >= 0
            else -1
        )

        # Id du geom workspace_box (−1 si absent)
        self._workspace_box_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "workspace_box"
        )

        # Id du body cube (−1 si absent du modèle)
        # Le joint libre ajoute 7 coordonnées dans qpos (pos + quat) après les actionneurs.
        _cube_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )
        self._has_cube: bool = _cube_body >= 0

        # Id du body cylinder (−1 si absent du modèle)
        _cylinder_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cylinder"
        )
        self._has_cylinder: bool = _cylinder_body >= 0

        # Pour la détection de contact EE–cube : body de l'end-effector et geom du cube
        self._ee_body_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector_2"
        )
        _cube_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"
        )
        self._cube_geom_id: int = _cube_geom
        _cylinder_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_geom"
        )
        self._cylinder_geom_id: int = _cylinder_geom

        # ── Buffers de délais ─────────────────────────────────────
        self._qpos_buffer: deque[np.ndarray] = deque(
            maxlen=self._obs_delay_range[1] + 1
        )
        self._ctrl_buffer: deque[np.ndarray] = deque(
            maxlen=self._act_delay_range[1] + 1
        )
        # Délais courants (tirés une fois par épisode dans reset())
        self._current_obs_delay: int = 0
        self._current_act_delay: int = 0

        # ── Rendu ────────────────────────────────────────────────
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None

    # ── Primitives de simulation ─────────────────────────────────

    def reset(
        self,
        qpos: np.ndarray | None = None,
        cube_pos: np.ndarray | None = None,
    ) -> None:
        """Réinitialise les données MuJoCo, applique optionnellement ``qpos``
        et ``cube_pos``, puis exécute ``mj_forward``.

        Parameters
        ----------
        qpos:
            Positions articulaires du robot (n_actuators,). Si None → pose neutre.
        cube_pos:
            Position (x, y, z) du cube. Ignoré si le modèle ne contient pas de
            body ``cube``. Si None, la position définie dans le XML est conservée.
        """
        mujoco.mj_resetData(self.model, self.data)
        if qpos is not None:
            self.data.qpos[: self.n_actuators] = qpos
        self.data.qvel[: self.n_actuators] = 0.0
        for aid in self._actuator_ids:
            self.data.ctrl[aid] = 0.0
        if cube_pos is not None and self._has_cube:
            self.set_cube_pose(cube_pos)
        mujoco.mj_forward(self.model, self.data)

        # Tirage d'un délai fixe par épisode
        self._current_obs_delay = int(
            self._rng.integers(self._obs_delay_range[0], self._obs_delay_range[1] + 1)
        )
        self._current_act_delay = int(
            self._rng.integers(self._act_delay_range[0], self._act_delay_range[1] + 1)
        )

        # Initialisation des buffers avec l'état courant
        qpos0 = self.data.qpos[: self.n_actuators].copy()
        ctrl0 = np.zeros(self.n_actuators)
        self._qpos_buffer.clear()
        self._ctrl_buffer.clear()
        for _ in range(self._obs_delay_range[1] + 1):
            self._qpos_buffer.append(qpos0.copy())
        for _ in range(self._act_delay_range[1] + 1):
            self._ctrl_buffer.append(ctrl0.copy())

    def forward(self) -> None:
        """Calcule la dynamique directe (cinématique, forces, contacts) sans intégration temporelle."""
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl: np.ndarray) -> None:
        """Applique les commandes ``ctrl`` avec le délai fixe de l'épisode et
        intègre ``_n_substeps`` fois. Les buffers sont mis à jour à chaque
        sim-step pour une granularité au niveau de ``sim_dt``."""
        for _ in range(self._n_substeps):
            # Enregistrement de la commande au niveau sim-step
            self._ctrl_buffer.append(ctrl.copy())
            # Délai fixe pour cet épisode (en sim-steps)
            idx = max(0, len(self._ctrl_buffer) - 1 - self._current_act_delay)
            delayed_ctrl = self._ctrl_buffer[idx]
            for i, aid in enumerate(self._actuator_ids):
                self.data.ctrl[aid] = delayed_ctrl[i]
            mujoco.mj_step(self.model, self.data)
            # Enregistrement de la position après ce sim-step
            self._qpos_buffer.append(self.data.qpos[: self.n_actuators].copy())

    # ── Lecture des capteurs ─────────────────────────────────────

    def get_qpos(self) -> np.ndarray:
        """Retourne les positions articulaires avec le délai fixe de l'épisode courant."""
        idx = max(0, len(self._qpos_buffer) - 1 - self._current_obs_delay)
        return self._qpos_buffer[idx].copy()

    def set_qpos(self, qpos: np.ndarray) -> None:
        """Écrit directement les positions articulaires."""
        self.data.qpos[: self.n_actuators] = qpos

    def get_cube_pos(self) -> np.ndarray:
        """Retourne la position (x, y, z) du cube dans le repère monde.

        Lève ``RuntimeError`` si le modèle ne contient pas de body ``cube``.
        """
        if not self._has_cube:
            raise RuntimeError("Ce modèle ne contient pas de body 'cube'.")
        n = self.n_actuators
        return self.data.qpos[n : n + 3].copy()

    def get_cube_yaw_cossin(self) -> np.ndarray:
        """Retourne l'orientation du cube autour de Z sous forme (cos θ, sin θ).

        Le quaternion MuJoCo est (w, x, y, z). Le yaw est extrait par :
            θ = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
        On retourne (cos θ, sin θ) pour éviter la discontinuité en ±π.

        Lève ``RuntimeError`` si le modèle ne contient pas de body ``cube``.
        """
        if not self._has_cube:
            raise RuntimeError("Ce modèle ne contient pas de body 'cube'.")
        n = self.n_actuators
        w, x, y, z = self.data.qpos[n + 3 : n + 7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([np.cos(yaw), np.sin(yaw)])

    def set_cube_pose(
        self, pos: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        """Positionne le cube ; orientation identité si ``quat`` est None.

        Lève ``RuntimeError`` si le modèle ne contient pas de body ``cube``.
        """
        if not self._has_cube:
            raise RuntimeError("Ce modèle ne contient pas de body 'cube'.")
        n = self.n_actuators
        self.data.qpos[n : n + 3] = pos
        self.data.qpos[n + 3 : n + 7] = quat if quat is not None else [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[n : n + 6] = 0.0

    # ── Cylindre ──────────────────────────────────────────────

    def get_cylinder_pos(self) -> np.ndarray:
        """Retourne la position (x, y, z) du cylindre dans le repère monde.

        Lève ``RuntimeError`` si le modèle ne contient pas de body ``cylinder``
        
        Pour l'instant devrait n'être appelé que par sorting env.
        """
        if not self._has_cylinder:
            raise RuntimeError("Ce modèle ne contient pas de body 'cylinder'.")
        n = self.n_actuators + 7  # après le cube (pos+quat = 7)
        return self.data.qpos[n : n + 3].copy()

    def set_cylinder_pose(
        self, pos: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        """Positionne le cylindre ; orientation identité si ``quat`` est None.

        Lève ``RuntimeError`` si le modèle ne contient pas de body ``cylinder``.
        """
        if not self._has_cylinder:
            raise RuntimeError("Ce modèle ne contient pas de body 'cylinder'.")
        n = self.n_actuators + 7
        self.data.qpos[n : n + 3] = pos
        self.data.qpos[n + 3 : n + 7] = quat if quat is not None else [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[n : n + 6] = 0.0

    def ee_touches_cylinder(self) -> bool:
        """Retourne True si un geom du body end_effector_2 est en contact avec cylinder_geom."""
        if not self._has_cylinder or self._cylinder_geom_id < 0 or self._ee_body_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self._cylinder_geom_id:
                if self.model.geom_bodyid[g2] == self._ee_body_id:
                    return True
            elif g2 == self._cylinder_geom_id:
                if self.model.geom_bodyid[g1] == self._ee_body_id:
                    return True
        return False

    # ── End-effector ─────────────────────────────────────────

    def get_end_effector_pos(self) -> np.ndarray:
        """Retourne la position cartésienne de l'end-effector (copie)."""
        return self.data.site_xpos[self._site_id].copy()

    def ee_touches_cube(self) -> bool:
        """Retourne True si un geom du body end_effector_2 est en contact avec cube_geom.

        Utilise la liste de contacts MuJoCo (data.contact[:data.ncon]).
        Retourne toujours False si le modèle n'a pas de cube.
        """
        if not self._has_cube or self._cube_geom_id < 0 or self._ee_body_id < 0:
            return False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            # Un des deux geoms doit être cube_geom,
            # l'autre doit appartenir au body end_effector_2.
            if g1 == self._cube_geom_id:
                if self.model.geom_bodyid[g2] == self._ee_body_id:
                    return True
            elif g2 == self._cube_geom_id:
                if self.model.geom_bodyid[g1] == self._ee_body_id:
                    return True
        return False

    # ── Visualisation du but ─────────────────────────────────────

    def set_goal_marker(self, pos: np.ndarray) -> None:
        """Déplace la sphère du goal marker vers la position ``pos``."""
        if self._goal_mocap_id >= 0:
            self.data.mocap_pos[self._goal_mocap_id] = pos

    def set_named_marker(self, body_name: str, pos: np.ndarray) -> None:
        """Déplace un marqueur mocap identifié par le nom de son body."""
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id >= 0:
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = pos

    # ── Espace de travail ────────────────────────────────────────

    def sync_workspace_box(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> None:
        """Redimensionne le geom ``workspace_box`` pour qu'il coïncide avec
        les plages de tirage de but passées en argument."""
        if self._workspace_box_id < 0:
            return
        cx = (x_range[0] + x_range[1]) / 2
        cy = (y_range[0] + y_range[1]) / 2
        cz = (z_range[0] + z_range[1]) / 2
        self.model.geom_pos[self._workspace_box_id] = [cx, cy, cz]
        self.model.geom_size[self._workspace_box_id] = [
            (x_range[1] - x_range[0]) / 2,
            (y_range[1] - y_range[0]) / 2,
            (z_range[1] - z_range[0]) / 2,
        ]

    # ── Rendu ────────────────────────────────────────────────────

    def render(self):
        """Effectue le rendu selon ``self.render_mode``."""
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            mujoco.mj_forward(self.model, self.data)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def close(self) -> None:
        """Libère les ressources de rendu."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
