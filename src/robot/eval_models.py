import os
import argparse
import numpy as np
from stable_baselines3 import PPO, TD3, SAC

from robot_env.push_in_hole_env import PushInHoleEnv

def evaluate_model(model_path, algo_class, env_class, n_episodes=50):
    env = env_class(render_mode=None)
    model = algo_class.load(model_path, env=env)
    
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            successes += 1
            
    success_rate = successes / n_episodes
    print(f"Model: {model_path} | Success Rate: {success_rate * 100:.2f}% ({successes}/{n_episodes})")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--td3", action="store_true")
    parser.add_argument("--her", action="store_true")
    args = parser.parse_args()
    
    if args.ppo:
        path = "src/robot/models/ppo/ppo_final.zip"
        if os.path.exists(path):
            evaluate_model(path, PPO, PushInHoleEnv)
        else:
            print(f"PPO model not found at {path}")
            
    if args.td3:
        path = "src/robot/models/td3/td3_final.zip"
        if os.path.exists(path):
            from td3_algo import make_env
            evaluate_model(path, TD3, make_env)
        else:
            print(f"TD3 model not found at {path}")
            
    if args.her:
        path = "src/robot/models/her_sac/her_sac_final.zip"
        if os.path.exists(path):
            from her_push_in_hole import make_env
            evaluate_model(path, SAC, make_env)
        else:
            print(f"HER-SAC model not found at {path}")
