"""NE MARCHE PAS"""


"""Training basique avec Stable-Baselines3 sur le PushInHoleEnv."""

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from push_in_hole_env import PushInHoleEnv

# Cree l'env (sans rendu pendant le training)
env = make_vec_env(PushInHoleEnv, n_envs=1)

# SAC : bon algo off-policy pour actions continues
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=100_000,
    learning_starts=1000,
    gamma=0.99,
    tau=0.005,
)

# Training
TOTAL_TIMESTEPS = 50_000
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Sauvegarde
model.save("push_in_hole_sac")
print("Modele sauvegarde dans push_in_hole_sac.zip")

# --- Test du modele entraine ---
print("\n--- Test sur 10 episodes ---")
env_test = PushInHoleEnv(render_mode="human")
successes = 0
n_episodes = 10

for ep in range(n_episodes):
    obs, info = env_test.reset()
    done = False
    total_reward = 0.0

    while not done:
        # Greedy : deterministic=True -> pas d'exploration, action la plus probable
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        env_test.render()
        total_reward += reward
        done = terminated or truncated

    if info.get("is_success", False):
        successes += 1
    print(f"Episode {ep+1}: reward={total_reward:.2f}, success={info.get('is_success', False)}")

print(f"\nTaux de succes: {successes}/{n_episodes}")
env_test.close()
