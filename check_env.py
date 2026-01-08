import cv2
import numpy as np
import ctypes
import sys
from battle_city_env import BattleCityEnv

# --- CONTROLS ---
VK_W, VK_A, VK_S, VK_D = 0x57, 0x41, 0x53, 0x44
VK_F = 0x46 # Fire
VK_RETURN = 0x0D # Start

def get_key_state(vk):
    return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0

def get_action():
    # Mapping to NES Actions (Simple Discrete or MultiDiscrete?)
    # BattleCityEnv uses Discrete(256) from nes_py.
    # We need to constructing the byte.
    # Bit map: A B Select Start Up Down Left Right
    #          0 1 2      3     4  5    6    7
    # Values:  1 2 4      8     16 32   64   128
    
    action = 0
    if get_key_state(VK_F):      action |= 1   # A (Fire)
    # B (2) unused
    # Select (4) unused
    if get_key_state(VK_RETURN): action |= 8   # Start
    if get_key_state(VK_W):      action |= 16  # Up
    if get_key_state(VK_S):      action |= 32  # Down
    if get_key_state(VK_A):      action |= 64  # Left
    if get_key_state(VK_D):      action |= 128 # Right
    
    return action

print("--- MANUAL ENV CHECK ---")
print("Controls: WASD + F (Fire) + Enter (Start)")
print("Watch Console for REWARDS!")

env = BattleCityEnv()
obs = env.reset()

try:
    total_reward = 0
    while True:
        action = get_action()
        obs, reward, done, info = env.step(action)
        
        if reward != 0:
            print(f"💰 REWARD: {reward}")
            total_reward += reward
            
        if done:
            print(f"☠️ DONE triggered! Total Score: {total_reward}")
            obs = env.reset()
            total_reward = 0
            
        # Render
        env.render()
        
        # Checking window close is handled by env.render() usually, 
        # but nes_py might rely on cv2 externally.
        # Actually NESEnv render() uses pyglet or cv2? 
        # nes_py uses simple_image_viewer (pyglet) or human mode.
        # Let's just trust render().
        
except KeyboardInterrupt:
    print("\nStopped.")

env.close()
