import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

class VirtualBattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, stack_size=4):
        super(VirtualBattleCityEnv, self).__init__()
        self.render_mode = render_mode
        self.STACK_SIZE = stack_size
        
        # Load Map (Expect 26x26 grid)
        try:
            loaded_map = np.load("level1_map.npy")
            if loaded_map.shape != (26, 26):
                # Try to resize if shape is wrong
                print(f"Warning: Map shape is {loaded_map.shape}, resizing to 26x26")
                loaded_map = cv2.resize(loaded_map, (26, 26), interpolation=cv2.INTER_NEAREST)
            self.initial_map = loaded_map.astype(np.uint8)
        except:
            print("Map not found or invalid! Using empty map.")
            self.initial_map = np.zeros((26, 26), dtype=np.uint8)
            
        # High Res Grid: 52x52 (Each cell is 4x4 pixels)
        self.GRID_SIZE = 52

        # Actions:
        self.action_space = spaces.Discrete(10)

        # Observation: The Map + Player Pos
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        self.current_map = None
        self.player_pos = [16, 48]
        self.player_dir = 0
        
        self.max_steps = 1000
        self.steps = 0
        self.frames = []
        self.sandbox_mode = True
        self.visited_sectors = set()
        
        # Rewards
        self.rew_explore = 0.1
        self.rew_stuck = -0.01
        self.rew_brick = 0.0
        self.rew_base = -20.0

        # ID Mapping - GRAYSCALE (Matches Real Env)
        self.ID_MAP = {
            "empty": 0,
            "brick": 200,   # Walls are bright
            "steel": 255,   # Indestructible is brightest
            "eagle": 255,   # Protect at all costs
            "player": 150,  # Distinct grey
            "enemy": 80,    # Darker grey
            "bullet": 255   # Bullets should be bright
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Scale up map 26x26 -> 52x52
        # First, map the loaded values (which might be old IDs) to new Grayscale IDs
        # Assuming old map had: 25=Brick, 50=Steel, 75=Eagle
        temp_map = self.initial_map.copy()
        
        # Create a display map with new IDs
        display_map = np.zeros_like(temp_map)
        display_map[temp_map == 25] = self.ID_MAP["brick"]
        display_map[temp_map == 50] = self.ID_MAP["steel"]
        display_map[temp_map == 75] = self.ID_MAP["eagle"]
        
        self.current_map = np.kron(display_map, np.ones((2,2), dtype=np.uint8))

        # Ensure shape is exactly (52, 52)
        if self.current_map.shape != (52, 52):
             self.current_map = cv2.resize(self.current_map, (52, 52), interpolation=cv2.INTER_NEAREST)

        self.visited_sectors = set()

        # Try default spawn (16, 48)
        self.player_pos = [16, 48]
        if not self._is_free(16, 48):
            for r in range(48, 0, -2):
                for c in range(4, 48, 2):
                    if self._is_free(c, r):
                        self.player_pos = [c, r]
                        break
                if self._is_free(self.player_pos[0], self.player_pos[1]): break

        self.player_dir = 0
        self.steps = 0

        # Fill stack
        frame = self._get_frame()
        self.frames = [frame for _ in range(self.STACK_SIZE)]
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False
        
        move_action = 0
        fire_action = False

        if action == 0: pass
        elif action >= 1 and action <= 4:
            move_action = action
        elif action == 5:
            fire_action = True
        elif action >= 6 and action <= 9:
            move_action = action - 5
            fire_action = True
            
        if getattr(self, 'sandbox_mode', False):
            fire_action = False

        dx, dy = 0, 0
        if move_action == 1: dy = -1; self.player_dir = 0 
        elif move_action == 2: dy = 1; self.player_dir = 2 
        elif move_action == 3: dx = -1; self.player_dir = 3 
        elif move_action == 4: dx = 1; self.player_dir = 1 
        
        if move_action > 0:
            nx, ny = self.player_pos[0] + dx, self.player_pos[1] + dy
            if self._is_free(nx, ny):
                self.player_pos = [nx, ny]
                sec_x, sec_y = nx // 4, ny // 4
                if (sec_x, sec_y) not in self.visited_sectors:
                    reward += self.rew_explore
                    self.visited_sectors.add((sec_x, sec_y))
            else:
                reward += self.rew_stuck 
                
        if fire_action:
            bx, by = self.player_pos[0], self.player_pos[1]
            bdx, bdy = 0, 0
            if self.player_dir == 0: bdy = -1
            elif self.player_dir == 1: bdx = 1
            elif self.player_dir == 2: bdy = 1
            elif self.player_dir == 3: bdx = -1
            
            for _ in range(52):
                bx += bdx
                by += bdy
                if not (0 <= bx < 52 and 0 <= by < 52): break
                tile = self.current_map[by, bx]
                
                # Check against new IDs
                if tile == self.ID_MAP["brick"]:
                    self.current_map[by, bx] = 0
                    reward += self.rew_brick
                    break
                elif tile == self.ID_MAP["steel"]: break
                elif tile == self.ID_MAP["eagle"]:
                    terminated = True
                    reward += self.rew_base
                    break

        frame = self._get_frame()
        self.frames.append(frame)
        if len(self.frames) > self.STACK_SIZE: self.frames.pop(0)
        
        if self.steps >= self.max_steps:
            truncated = True
            
        info = {
            "exploration_pct": (len(self.visited_sectors) / (13*13)) * 100,
            "kills": 0,
            "env_type": 0 # MARK AS VIRTUAL ENV
        }
            
        return self._get_obs(), reward, terminated, truncated, info

    def _is_free(self, x, y):
        if x < 0 or y < 0 or x > 51 or y > 51: return False
        for r in range(y, y+4):
            for c in range(x, x+4):
                if r >= 52 or c >= 52: return False
                if self.current_map[r, c] != 0: return False
        return True

    def _get_frame(self):
        grid = self.current_map.copy()
        px, py = self.player_pos
        max_r = min(52, py+4)
        max_c = min(52, px+4)
        grid[py:max_r, px:max_c] = self.ID_MAP["player"]
        return grid

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)
    
    def get_tactical_rgb(self):
        """
        Returns the EXACT visual representation the network sees (Grayscale).
        """
        grid = self._get_frame()
        # Convert Gray -> RGB for display consistency
        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return img

    def get_debug_info(self):
        return "VIRTUAL"

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.get_tactical_rgb()
        return None
