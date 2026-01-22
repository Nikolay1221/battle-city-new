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

    print("ID MAP:", env.ID_MAP)
    
    # --- CUSTOM STATIC MAP CAPTURE (Ignores Dynamic Entities) ---
    # We implement a custom version of _get_tactical_map that only looks for static terrain
    # to avoid enemies/players overwriting valid terrain blocks.
    
    grid = np.zeros((env.GRID_SIZE, env.GRID_SIZE), dtype=np.uint8)
    frame_rgb = env.raw_env.screen
    
    # 1. SCAN TERRAIN (Visual)
    # Crop to 208x208 (16:224)
    screen_area = frame_rgb[16:224, 16:224]
    
    # Reshape to cells (52x52 grid -> 4x4 pixels per cell)
    # (52, 4, 52, 4, 3) -> (52, 52, 4, 4, 3) -> (52, 52, 16, 3)
    tiles = screen_area.reshape(env.GRID_SIZE, 4, env.GRID_SIZE, 4, 3).transpose(0, 2, 1, 3, 4)
    tiles_flat = tiles.reshape(env.GRID_SIZE, env.GRID_SIZE, 16, 3).astype(np.float32)
    
    # Calculate Distances to Known Colors
    # env.known_colors is (N, 3)
    # tiles_flat is (52, 52, 16, 3)
    # diffs -> (52, 52, 16, N, 3) -- optimize by broadcasting
    
    # Use direct comparison for critical colors to avoid broadcasting overhead/complexity
    # or just use the env logic if accessible. 
    # Since we can't easily access 'idx_brick' without parsing labels, we'll re-implement simplified counting.
    
    # Get indices for static types
    idx_brick = env.known_labels.index("brick")
    idx_brick_dark = env.known_labels.index("brick_dark")
    idx_steel = env.known_labels.index("steel")
    idx_eagle = env.known_labels.index("eagle")
    
    # Compute Nearest Neighbors manually
    diffs = tiles_flat[..., np.newaxis, :] - env.known_colors
    dists = np.sum(diffs**2, axis=-1)
    labels = np.argmin(dists, axis=-1)
    mask_bg = np.max(tiles_flat, axis=-1) > 40
    
    # Count pixels per cell
    count_brick = np.sum(((labels == idx_brick) | (labels == idx_brick_dark)) & mask_bg, axis=2)
    count_steel = np.sum((labels == idx_steel) & mask_bg, axis=2)
    count_eagle = np.sum((labels == idx_eagle) & mask_bg, axis=2)
    
    # Apply to Grid (ONLY STATIC)
    grid[count_brick > 1] = env.ID_MAP["brick"]
    grid[count_steel > 2] = env.ID_MAP["steel"]
    grid[count_eagle > 1] = env.ID_MAP["eagle"]
    
    clean_grid = grid
    # -----------------------------------------------------------

    print("Raw Grid Unique Values:", np.unique(clean_grid))
    
    # No need to remove dynamic entities as we didn't include them 
    
    filename = "level1_map.npy"
    np.save(filename, clean_grid)
    print(f"Map saved to {filename}")
    print("Unique values in map:", np.unique(clean_grid))
    
    env.close()

if __name__ == "__main__":
    capture_clean_map()