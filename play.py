import gymnasium as gym
import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from battle_city_env import BattleCityEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import config

# --- RESNET ARCHITECTURE (MUST MATCH TRAIN.PY) ---
class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class CustomTacticalCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] 
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            x = self.initial_conv(sample)
            x = self.res_blocks(x)
            n_flatten = self.final_conv(x).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(observations)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return self.linear(x)

def play():
    final_path = f"{config.MODEL_DIR}/battle_city_resnet_v1.zip"
    interrupted_path = f"{config.MODEL_DIR}/battle_city_resnet_v1_interrupted.zip"
    
    model_path = None
    if os.path.exists(interrupted_path):
        model_path = interrupted_path
    elif os.path.exists(final_path):
        model_path = final_path
        
    if model_path is None:
        print(f"Model not found! Looked for:\n1. {interrupted_path}\n2. {final_path}")
        print("Run train.py first!")
        return

    print(f"Loading model: {model_path}")
    
    # Create REAL Environment
    env = BattleCityEnv(render_mode='human', stack_size=config.STACK_SIZE)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Starting Game...")
    obs = env.reset()
    
    total_reward = 0
    while True:
        # DETERMINISTIC MODE ON: Best actions only
        action, _states = model.predict(obs, deterministic=True) 
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        env.render()
        time.sleep(0.02)
        
        if done:
            print(f"Game Over. Total Reward: {total_reward}")
            obs = env.reset()
            total_reward = 0

if __name__ == "__main__":
    play()
