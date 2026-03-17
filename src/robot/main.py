"""Lance une simulation visuelle avec le modele SAC entraine."""

from stable_baselines3 import SAC
from push_in_hole_env import PushInHoleEnv

# Charger le modele entraine
model = SAC.load("push_in_hole_sac")

env = PushInHoleEnv(render_mode="human")

for ep in range(10):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep+1}: reward={total_reward:.2f}, "
          f"success={info['is_success']}, "
          f"dist_cube_hole={info['dist_cube_hole']:.4f}")

env.close()
