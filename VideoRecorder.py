import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

"""
Class for recording trained agents
"""
env_id = "LunarLander-v3"
video_folder = "logs/videos/"
os.makedirs(video_folder, exist_ok=True)
video_length = 5000

vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
model = DQN.load("./max_ep_len_runs/max_ep_len_350/DQN_max_ep_len_350_2.zip")
obs = vec_env.reset()

# Record the video starting at the first step
vec_env = VecVideoRecorder(vec_env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_id}")

vec_env.reset()
for _ in range(video_length + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, _, _, _ = vec_env.step(action)
# Save the video
vec_env.close()