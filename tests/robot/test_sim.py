"""Test rapide : lance le viewer MuJoCo avec le robot et permet de le manipuler."""

import os
import mujoco
import mujoco.viewer

SCENE_XML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "robot", "robot_env", "scene_push.xml"))

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
