import numpy as np
import time
import os
from battle_city_env import BattleCityEnv
from collections import Counter

def capture_clean_map():
    print("Initializing Environment for Map Capture...")
    
    env = BattleCityEnv(render_mode=None, use_vision=False)
    env.reset()
    
    print("Waiting for game to stabilize (120 frames)...")
    # Increase wait time to ensure the "curtain" opening animation is finished
    for _ in range(120):
        env.step(0) 
        
    print("Analyzing Screen Colors...")
    frame = env.raw_env.screen
    
    # Crop play area
    screen_area = frame[16:224, 16:224]
    
    # Reshape to list of pixels
    pixels = screen_area.reshape(-1, 3)
    
    # Count most common colors
    pixel_tuples = [tuple(p) for p in pixels]
    counts = Counter(pixel_tuples)
    
    print("\nTOP 10 COLORS ON SCREEN:")
    for color, count in counts.most_common(10):
        print(f"RGB: {color} | Count: {count}")
        
    print("\nCURRENT REFERENCE COLORS:")
    for name, val in env.TILE_COLORS.items():
        print(f"{name}: {val}")

    print("\nCapturing Tactical Map...")
    clean_grid = env._get_tactical_map()
    
    # Remove dynamic entities
    clean_grid[clean_grid == 150] = 0 
    clean_grid[clean_grid == 200] = 0 
    clean_grid[clean_grid == 255] = 0 
    
    filename = "level1_map.npy"
    np.save(filename, clean_grid)
    print(f"Map saved to {filename}")
    print("Unique values in map:", np.unique(clean_grid))
    
    env.close()

if __name__ == "__main__":
    capture_clean_map()