import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

# Import new modular Game Core
from game_modules.game_core import BattleCityGame

class VirtualBattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, stack_size=4):
        super(VirtualBattleCityEnv, self).__init__()
        self.render_mode = render_mode
        self.STACK_SIZE = stack_size
        
        self.GRID_SIZE = 52

        # Actions: 0-NOOP, 1-4 Move, 5 Fire, 6-9 Move+Fire
        self.action_space = spaces.Discrete(10)

        # Observation
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        # Instantiate Game Core
        self.game = BattleCityGame()
        
        self.max_steps = 3000 
        self.steps = 0
        self.frames = []
        
        # Statistics
        self.visited_sectors = set()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game.reset()
        self.steps = 0
        self.visited_sectors = set()
        
        # Fill stack
        frame = self.game.get_frame()
        self.frames = [frame for _ in range(self.STACK_SIZE)]
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # Delegate logic to Game Core
        reward, terminated = self.game.step(action)
        
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True
        
        # Update Exploration
        px, py = self.game.player.x, self.game.player.y
        sec_x, sec_y = int(px) // 4, int(py) // 4
        if (sec_x, sec_y) not in self.visited_sectors:
             self.visited_sectors.add((sec_x, sec_y))
             reward += 0.005 # Exploration reward
             
        # Update observation stack
        frame = self.game.get_frame()
        self.frames.append(frame)
        if len(self.frames) > self.STACK_SIZE: self.frames.pop(0)
            
        info = {
            "kills": self.game.episode_kills,
            "exploration_pct": (len(self.visited_sectors) / (13*13)) * 100,
            "env_type": 1
        }
            
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)
        
    def _get_frame(self):
        return self.game.get_frame()
    
    def get_tactical_rgb(self):
        grid = self.game.get_frame()
        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return img

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.get_tactical_rgb()
        return None
