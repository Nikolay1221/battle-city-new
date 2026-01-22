import gymnasium as gym
import os
import time
import cv2
import sys

# --- NUMPY 2.0 COMPATIBILITY PATCH (ROBUST) ---
# Maps all submodules from numpy.core to numpy._core to satisfy unpickling of NumPy 2.0 arrays
try:
    import numpy.core
    # 1. Map the main package
    sys.modules["numpy._core"] = numpy.core
    
    # 2. Map all submodules dynamically
    for attr_name in dir(numpy.core):
        attr = getattr(numpy.core, attr_name)
        if isinstance(attr, type(sys)): # Check if it's a module
            sys.modules[f"numpy._core.{attr_name}"] = attr
            
    # 3. Explicitly ensure critical ones exist (just in case they aren't in dir() for some reason)
    if hasattr(numpy.core, 'numeric'):
        sys.modules['numpy._core.numeric'] = numpy.core.numeric
    if hasattr(numpy.core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    
    print(" Applied NumPy 2.0 -> 1.x compatibility patch.")
except Exception as e:
    print(f" Warning: Failed to apply NumPy patch: {e}")

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

from virtual_env import VirtualBattleCityEnv

def play(mode="sim"):
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
    print(f"Mode: {mode.upper()}")

    # Environment Setup
    if mode == "real":
        env = BattleCityEnv(render_mode='human', stack_size=config.STACK_SIZE)
    else:
        env = VirtualBattleCityEnv(render_mode='rgb_array', stack_size=config.STACK_SIZE)

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, training=False)

    # --- ROBUST LOADING (BYPASS NUMPY INCOMPATIBILITY) ---
    print("Creating fresh model...")
    # Initialize model with same architecture
    policy_kwargs = dict(
        features_extractor_class=CustomTacticalCNN,
        features_extractor_kwargs=dict(features_dim=512), 
        net_arch=dict(pi=[512, 512], vf=[512, 512]),   
        activation_fn=torch.nn.ReLU,
    )
    
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("Extracting weights from zip...")
    import zipfile
    import io
    
    try:
        with zipfile.ZipFile(model_path, "r") as archive:
            # Load PyTorch Policy Weights directly
            with archive.open("policy.pth") as f:
                policy_state_dict = torch.load(f, map_location="cpu")
                model.policy.load_state_dict(policy_state_dict)
                print("Weights loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR loading weights: {e}")
        return

    print("Starting Game...")
    obs = env.reset()
    
    total_reward = 0
    while True:
        # DETERMINISTIC MODE ON: Best actions only
        action, _states = model.predict(obs, deterministic=True) 
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if mode == "real":
            env.render()
            
            # Show Agent View (Neural Network Input)
            # Obs shape is (1, C, H, W) -> (1, 4, 52, 52)
            try:
                # Extract the most recent frame (last channel)
                raw_view = obs[0, -1, :, :] 
                
                # Resize for visibility (520x520)
                debug_view = cv2.resize(raw_view, (520, 520), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Agent Perception (Input)", debug_view)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Vis Error: {e}")
        else:
            # Manual Render for Virtual Env
            img = env.envs[0].get_tactical_rgb()
            img = cv2.resize(img, (520, 520), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Virtual Battle City AI", img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        time.sleep(0.02)
        
        if done:
            print(f"Game Over. Total Reward: {total_reward}")
            obs = env.reset()
            total_reward = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "sim"], default="sim", help="Choose 'real' (NES) or 'sim' (VirtualEnv)")
    args = parser.parse_args()
    play(mode=args.mode)
