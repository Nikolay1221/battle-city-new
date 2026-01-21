import numpy as np
import time
import os
from battle_city_env import BattleCityEnv

# Map Editor IDs
ID_EMPTY = 0
ID_BRICK = 25
ID_STEEL = 50
ID_EAGLE = 75
ID_WATER = 100
ID_FOREST = 125

def get_high_fidelity_map(env):
    """
    Scans the screen and creates a 26x26 map compatible with map_editor.py
    """
    TARGET_GRID = 26
    grid = np.zeros((TARGET_GRID, TARGET_GRID), dtype=np.uint8)
    frame_rgb = env.raw_env.screen
    
    # 1. SCAN TERRAIN (Visual)
    # 16:224 crop based on env standard. 208x208 resolution.
    screen_area = frame_rgb[16:224, 16:224]
    
    # Reshape to 26x26 grid (each cell is 8x8 pixels)
    tiles = screen_area.reshape(TARGET_GRID, 8, TARGET_GRID, 8, 3).transpose(0, 2, 1, 3, 4)
    tiles_flat = tiles.reshape(TARGET_GRID, TARGET_GRID, 64, 3).astype(np.float32)
    
    # Calculate difference from known colors
    diffs = tiles_flat[..., np.newaxis, :] - env.known_colors
    dists = np.sum(diffs**2, axis=-1)
    labels = np.argmin(dists, axis=-1)
    mask_bg = np.max(tiles_flat, axis=-1) > 40
    
    idx_brick = env.known_labels.index("brick")
    idx_brick_dark = env.known_labels.index("brick_dark")
    idx_steel = env.known_labels.index("steel")
    idx_eagle = env.known_labels.index("eagle")
    
    # Count pixels per cell
    count_brick = np.sum(((labels == idx_brick) | (labels == idx_brick_dark)) & mask_bg, axis=2)
    count_steel = np.sum((labels == idx_steel) & mask_bg, axis=2)
    count_eagle = np.sum((labels == idx_eagle) & mask_bg, axis=2)
    
    # Fill Grid
    # Priority: LOWEST to HIGHEST (Last one wins)
    
    # 1. Steel (Lowest priority because Mortar looks like Steel)
    grid[count_steel > 8] = ID_STEEL
    
    # 2. Brick (Overwrites Steel if mixed)
    grid[count_brick > 3] = ID_BRICK
    
    # 3. Eagle (Highest priority, rare)
    grid[count_eagle > 3] = ID_EAGLE
    
    return grid

def capture_clean_map():
    print("Initializing Environment for Map Capture...")
    
    # Sandbox mode removes enemies
    env = BattleCityEnv(render_mode=None, use_vision=False, sandbox_mode=True)
    env.reset(seed=42) # FIXED SEED
    
    print("Waiting for game to stabilize (60 frames)...")
    for i in range(60):
        env.step(0) 
        
    print("Capturing Map...")
    editor_grid = get_high_fidelity_map(env)
    
    # Save
    filename = "level1_map.npy"
    if os.path.exists(filename):
        os.remove(filename)
        
    np.save(filename, editor_grid)
    print(f"Map saved to {filename}")
    
    # Stats
    unique, counts = np.unique(editor_grid, return_counts=True)
    print("Map Stats:")
    for v, c in zip(unique, counts):
        name = "Unknown"
        if v == ID_EMPTY: name = "Empty"
        elif v == ID_BRICK: name = "Brick"
        elif v == ID_STEEL: name = "Steel"
        elif v == ID_EAGLE: name = "Eagle"
        print(f"  {name} ({v}): {c} cells")
        
    env.close()

if __name__ == "__main__":
    capture_clean_map()