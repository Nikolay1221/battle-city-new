import numpy as np
import cv2
import json
import os

class MapManager:
    def __init__(self, map_file="level1_map.npy", destruction_pattern_file="destruction_pattern.json"):
        self.GRID_SIZE = 52
        self.ID_MAP = {
            "empty": 0,
            "brick": 200,   
            "steel": 255,   
            "eagle": 254,
            "BaseWall": 200 # Treated as brick
        }
        
        self.current_map = self._load_map(map_file)
        self.pattern_data = self._load_patterns(destruction_pattern_file)
        
    def _load_map(self, map_file):
        try:
            print(f"Attempting to load {map_file}...")
            loaded_map = np.load(map_file)
            print(f"Map loaded! Shape: {loaded_map.shape}")
            
            if loaded_map.shape != (52, 52):
                loaded_map = cv2.resize(loaded_map, (52, 52), interpolation=cv2.INTER_NEAREST)
                
            temp_map = loaded_map.astype(np.uint8)
            
            # Draw Base Protection (Standard Battle City Layout) - REMOVED
            # We now rely on level1_map.npy to contain the base layout.
            # This allows user to edit/move the base in map_editor.py.
            
            # Fallback: If map is completely empty (all 0), MAYBE we should add it?
            # But let's assume the map file is correct.
            # If we want to ensure Eagle exists:
            if np.count_nonzero(temp_map == self.ID_MAP["eagle"]) == 0:
                 print("WARNING: No Eagle found in map! Putting default at (24, 48)")
                 self._fill_rect(temp_map, 24, 48, 2, 2, self.ID_MAP["eagle"])
                 # Walls around eagle
                 self._fill_rect(temp_map, 22, 48, 2, 2, self.ID_MAP["brick"]) # Left
                 self._fill_rect(temp_map, 22, 46, 2, 2, self.ID_MAP["brick"]) # Left-Top
                 self._fill_rect(temp_map, 24, 46, 2, 2, self.ID_MAP["brick"]) # Top
                 self._fill_rect(temp_map, 26, 48, 2, 2, self.ID_MAP["brick"]) # Right
                 self._fill_rect(temp_map, 26, 46, 2, 2, self.ID_MAP["brick"]) # Right-Top
            
            return temp_map
        except Exception as e:
            print(f"FAILED TO LOAD MAP: {e}")
            return np.zeros((52, 52), dtype=np.uint8)

    def _fill_rect(self, grid, x, y, w, h, val):
        grid[y:y+h, x:x+w] = val

    def _load_patterns(self, pattern_file):
        data = {"UP": [], "DOWN": [], "LEFT": [], "RIGHT": []}
        try:
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                     loaded = json.load(f)
                     if "patterns" in loaded:
                         raw = loaded.get("patterns", {})
                         for k in data.keys():
                             data[k] = raw.get(k, [])
        except Exception as e:
            print(f"Failed to load patterns: {e}")
            
        # Fallback
        if not data["UP"]:
            default = [[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            for k in data.keys(): data[k] = default
            
        return data

    def get_map(self):
        return self.current_map

    def is_solid(self, x, y):
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
            return True # Out of bounds is solid
        tile = self.current_map[y, x]
        return tile != 0 # 0 is empty

    def get_tile(self, x, y):
        if not (0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE):
            return -1
        return self.current_map[y, x]

    def destroy(self, cx, cy, direction):
        """
        Apply destruction pattern at center cx, cy based on direction.
        Returns: True if something was destroyed/hit.
        """
        dir_names = ["UP", "RIGHT", "DOWN", "LEFT"]
        dname = dir_names[direction]
        offsets = self.pattern_data[dname]
        
        hit_something = False
        hit_eagle = False
        
        for off in offsets:
            tx, ty = cx + off[0], cy + off[1]
            if not (0 <= tx < self.GRID_SIZE and 0 <= ty < self.GRID_SIZE):
                continue
            
            tile = self.current_map[ty, tx]
            
            if tile == self.ID_MAP["brick"]:
                self.current_map[ty, tx] = 0
                hit_something = True
            elif tile == self.ID_MAP["eagle"]:
                hit_eagle = True
                hit_something = True
            elif tile == self.ID_MAP["steel"]:
                hit_something = True # Hit but don't destroy
                
        return hit_something, hit_eagle
