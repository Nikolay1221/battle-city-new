import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
import numpy as np
from collections import deque
import os
import cv2 
import config 

class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, rom_path=config.ROM_PATH, render_mode=None, use_vision=False, stack_size=4, target_stage=None, enemy_count=20, no_shooting=False):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.USE_VISION = use_vision
        self.STACK_SIZE = stack_size
        self.target_stage = target_stage
        
        self.MAX_STEPS = 100_000_000 
        self.steps_in_episode = 0
        
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"ROM file not found at: {self.rom_path}")

        self.raw_env = NESEnv(self.rom_path)
        
        # Define Actions
        self.actions_list = [
            ['NOOP'],
            ['up'], ['down'], ['left'], ['right'],
            ['A'], # Fire
            ['up', 'A'], ['down', 'A'], ['left', 'A'], ['right', 'A']
        ]
        
        self.env = JoypadSpace(self.raw_env, self.actions_list)
        self.action_space = spaces.Discrete(len(self.actions_list))

        # Tactical Grid Settings (High Res: 52x52)
        self.GRID_SIZE = 52 
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        # Color definitions (RGB) for visual scanning
        self.TILE_COLORS = {
            "brick": np.array([228, 92, 16]), 
            "brick_dark": np.array([168, 16, 0]), 
            "empty": np.array([0, 0, 0]),
            "eagle": np.array([60, 90, 60]), 
            "steel": np.array([124, 124, 124]),
            "player": np.array([232, 208, 32]), 
            "enemy_silver": np.array([180, 180, 180]),
            "enemy_red": np.array([168, 0, 32]),
            "bullet": np.array([255, 255, 255]) 
        }
        self.known_labels = list(self.TILE_COLORS.keys())
        self.known_colors = np.array(list(self.TILE_COLORS.values()), dtype=np.float32)

        # ID Mapping - GRAYSCALE INTENSITY
        self.ID_MAP = {
            "empty": 0,
            "brick": 200,   
            "steel": 255,   
            "eagle": 255,   
            "player": 150,  
            "enemy": 80,    
            "bullet": 255   
        }
        
        self.ram_stack = deque(maxlen=self.STACK_SIZE)
        self.frames = deque(maxlen=self.STACK_SIZE)

        # Rewards Configuration
        # Rewards Configuration (From Config)
        self.enemy_count = enemy_count
        self.no_shooting = no_shooting
        
        self.rew_kill = getattr(config, 'REW_KILL', 1.0)
        self.rew_death = getattr(config, 'REW_DEATH', -1.0)
        self.rew_base = getattr(config, 'REW_BASE', -20.0)
        self.rew_explore = getattr(config, 'REW_EXPLORE', 0.01)
        
        # CONDITIONAL WIN REWARD
        # Only grant win reward if this is a "Full" game (20 enemies)
        if self.enemy_count >= 20:
             self.rew_win = getattr(config, 'REW_WIN', 50.0)
        else:
             self.rew_win = 0.0 # No reward for clearing a partial/empty level
        
        self.rew_stuck = -0.001 
        self.rew_brick = 0.0 
        
        self.rew_dist = 0.0 
        self.prev_min_dist = 999.0

        # RAM Addresses
        self.ADDR_LIVES = 0x51
        self.ADDR_STATE = 0x92
        self.ADDR_BASE_STATUS = 0x68 # NEW: Base Latch Address
        self.ADDR_ENEMIES_LEFT = 0x80 # Enemies remaining to spawn
        self.ADDR_ENEMIES_ON_SCREEN = 0xA0 # Enemies currently active
        self.ADDR_KILLS = [0x73, 0x74, 0x75, 0x76] 
        self.ADDR_SCORE = [0x70, 0x71, 0x72] 
        self.ADDR_BONUS = 0x62
        self.ADDR_STAGE = 0x85
        self.ADDR_MAP = 0x0731
        self.ADDR_BASE_TILE = 0x07D3
        self.ADDR_X_ARR = 0x90 
        self.ADDR_Y_ARR = 0x98
        
        self.prev_lives = 3
        self.prev_kill_sum = 0
        self.prev_score_sum = 0
        self.prev_x = 0
        self.prev_y = 0
        self.idle_steps = 0
        self.visited_sectors = set()
        
        self.episode_kills = 0 
        self.level_cleared = False
        self.base_active_latch = False # NEW: Latch for base status

    def _get_tactical_map(self):
        """Creates a 52x52 GRAYSCALE matrix representing the game state (VISUAL ONLY)."""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        frame_rgb = self.raw_env.screen
        
        # 1. SCAN TERRAIN (Visual)
        screen_area = frame_rgb[16:224, 16:224]
        tiles = screen_area.reshape(self.GRID_SIZE, 4, self.GRID_SIZE, 4, 3).transpose(0, 2, 1, 3, 4)
        tiles_flat = tiles.reshape(self.GRID_SIZE, self.GRID_SIZE, 16, 3).astype(np.float32)
        
        diffs = tiles_flat[..., np.newaxis, :] - self.known_colors
        dists = np.sum(diffs**2, axis=-1)
        labels = np.argmin(dists, axis=-1)
        mask_bg = np.max(tiles_flat, axis=-1) > 40
        
        # Indices
        idx_brick = self.known_labels.index("brick")
        idx_brick_dark = self.known_labels.index("brick_dark")
        idx_steel = self.known_labels.index("steel")
        idx_eagle = self.known_labels.index("eagle")
        idx_player = self.known_labels.index("player")
        idx_enemy_s = self.known_labels.index("enemy_silver")
        idx_enemy_r = self.known_labels.index("enemy_red")
        idx_bullet = self.known_labels.index("bullet")
        
        # Count pixels per cell
        count_brick = np.sum(((labels == idx_brick) | (labels == idx_brick_dark)) & mask_bg, axis=2)
        count_steel = np.sum((labels == idx_steel) & mask_bg, axis=2)
        count_eagle = np.sum((labels == idx_eagle) & mask_bg, axis=2)
        count_player = np.sum((labels == idx_player) & mask_bg, axis=2)
        count_enemy = np.sum(((labels == idx_enemy_s) | (labels == idx_enemy_r)) & mask_bg, axis=2)
        count_bullet = np.sum((labels == idx_bullet) & mask_bg, axis=2)
        
        # Apply Grayscale Values
        grid[count_brick > 1] = self.ID_MAP["brick"]
        grid[count_steel > 2] = self.ID_MAP["steel"]
        grid[count_eagle > 1] = self.ID_MAP["eagle"]
        
        # Dynamic objects (lower threshold)
        grid[count_player > 0] = self.ID_MAP["player"]
        grid[count_enemy > 0] = self.ID_MAP["enemy"]
        grid[count_bullet > 0] = self.ID_MAP["bullet"]
        
        return grid

    def _get_obs(self):
        current_map = self._get_tactical_map()
        while len(self.frames) < self.STACK_SIZE:
            self.frames.append(current_map)
        self.frames.append(current_map)
        obs_stack = np.stack(self.frames, axis=-1)
        return obs_stack

    def get_tactical_rgb(self):
        if not self.frames: return np.zeros((52, 52, 3), dtype=np.uint8)
        grid = self.frames[-1]
        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.raw_env.reset()
        self.steps_in_episode = 0
        self.episode_score = 0.0
        self.episode_kills = 0
        self.level_cleared = False
        self.base_active_latch = False # Reset Latch
        self.visited_sectors = set()
        self.frames.clear()
        
        # Start Sequence
        for _ in range(80): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(60): self.raw_env.step(0)

        # --- CUSTOM ENEMY COUNT RAM HACK ---
        if 0 < self.enemy_count < 20:
            # Note: 0x80 is "Enemies Remaining to Spawn". 
            # The game starts with 20. If we set it to (N-2), it will spawn roughly N.
            # (Because 2-3 are usually already on screen or pending).
            # This is an approximation.
            target = max(0, self.enemy_count - 3) 
            self.raw_env.ram[self.ADDR_ENEMIES_LEFT] = target
            # Also clear any on screen if we want very few? No, let them spawn.

        # Init state
        self.prev_lives = int(self.raw_env.ram[self.ADDR_LIVES])
        self.prev_x = int(self.raw_env.ram[self.ADDR_X_ARR])
        self.prev_y = int(self.raw_env.ram[self.ADDR_Y_ARR])
        self.prev_stage = int(self.raw_env.ram[self.ADDR_STAGE]) # Track Stage
        
        self.prev_kill_sum = sum([int(self.raw_env.ram[addr]) for addr in self.ADDR_KILLS])
        self.prev_score_sum = sum([int(self.raw_env.ram[addr]) for addr in self.ADDR_SCORE])

        self.idle_steps = 0
        self.frames.clear()
        return self._get_obs(), {}

    def step(self, action):
        nes_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        repeat = getattr(config, 'FRAME_SKIP', 4)
        
        ram = self.raw_env.ram
        old_x, old_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])

        # --- MODE: No Shooting ---
        if self.no_shooting:
            # Remap fire actions to movement only
            if action == 5:
                action = 0
            elif action >= 6 and action <= 9:
                action = action - 5

        for _ in range(repeat):
            obs, r, d, i = self.env.step(action)
            nes_reward += r 
            if d:
                terminated = True
                break
        
        # --- LIMIT ENEMIES ON SCREEN (HYBRID TELEPORT) ---
        # If enemy_count is small (e.g. 0, 2, 5), we force excess slots to (0,0).
        # We assume standard Battle City uses slots 1-6 max.
        if self.enemy_count < 6:
             for i in range(self.enemy_count + 1, 7):
                 # Addresses: X = 0x90+i, Y = 0x98+i
                 if 0x90 + i < 0x100: 
                      self.raw_env.ram[0x90 + i] = 0
                      self.raw_env.ram[0x98 + i] = 0
             
             # Special Case: If 0 enemies, we also prevent idle timeouts
             if self.enemy_count == 0:
                 self.idle_steps = 0 
        # -------------------------------------------------
        
        self.steps_in_episode += 1 
        ram = self.raw_env.ram
        reward = 0 
        
        # 1. Kill Rewards
        curr_kill_sum = sum([int(ram[addr]) for addr in self.ADDR_KILLS])
        diff = curr_kill_sum - self.prev_kill_sum
        
        if diff > 0:
            reward += self.rew_kill * diff
            self.episode_kills += diff
            
        self.prev_kill_sum = curr_kill_sum
        
        # CHECK FOR VICTORY
        curr_stage = int(ram[self.ADDR_STAGE])
        
        # Condition 1: Kills Limit (Only if full game)
        # If enemy_count < 20, we don't grant win for killing all (because there aren't 20)
        # unless we explicitly want to logic that out. 
        # Currently requested: "REW_WIN = 20.0 начислять не будем" if < 20.
        if self.enemy_count >= 20: 
            if self.episode_kills >= 20 and not self.level_cleared:
                reward += self.rew_win
                self.level_cleared = True
                terminated = True 
                info['is_success'] = True
                info['win_reason'] = 'kills_limit'

        # Condition 2: Stage Changed
        # This means the internal game logic is happy (all enemies dead).
        if curr_stage != self.prev_stage:
             # Only grant the BIG reward if it was a full game
             if not self.level_cleared and self.enemy_count >= 20:
                 reward += self.rew_win
                 self.level_cleared = True
                 info['is_success'] = True
                 info['win_reason'] = 'stage_cleared'
             
             # Terminate regardless
             terminated = True
             
        self.prev_stage = curr_stage
        
        # 2. Death
        curr_lives = int(ram[self.ADDR_LIVES])
        if curr_lives < 10 and self.prev_lives < 10:
             if curr_lives < self.prev_lives:
                reward += self.rew_death
        self.prev_lives = curr_lives
        
        # 3. Game Over Checks
        if curr_lives == 0:
            terminated = True
            info['game_over_reason'] = 'no_lives'

        base_status = int(ram[self.ADDR_BASE_STATUS])
        if base_status != 0: self.base_active_latch = True
        if self.base_active_latch and base_status == 0:
             terminated = True
             reward += self.rew_base
             info['game_over_reason'] = 'base_destroyed'
        
        # 4. EXPLORATION REWARD
        curr_x, curr_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])
        
        if abs(curr_x - old_x) > 2 or abs(curr_y - old_y) > 2:
             sec_x, sec_y = curr_x // 16, curr_y // 16
             if (sec_x, sec_y) not in self.visited_sectors:
                 reward += self.rew_explore
                 self.visited_sectors.add((sec_x, sec_y))
             self.idle_steps = 0
        else:
             if action != 0: 
                 reward += self.rew_stuck
             self.idle_steps += 1
        
        info['exploration_pct'] = len(self.visited_sectors) / 169.0 * 100
        info['nes_reward'] = nes_reward
        info['kills'] = self.episode_kills 
        # Env Type: 0 = Peaceful/Sim/Simplified, 1 = Full Standard Combat
        info['env_type'] = 1 if (self.enemy_count >= 20 and not self.no_shooting) else 0


        self.prev_x, self.prev_y = curr_x, curr_y
        
        if self.steps_in_episode >= self.MAX_STEPS:
            truncated = True 

        self.episode_score += reward
        info['score'] = self.episode_score
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def cheat_clear_enemies(self):
        """Debug tool: Clear all enemies by manipulating RAM."""
        # 1. Reset spawn counter
        self.raw_env.ram[self.ADDR_ENEMIES_LEFT] = 0
        # 2. Reset on-screen counter
        self.raw_env.ram[self.ADDR_ENEMIES_ON_SCREEN] = 0
        # 3. Optional: Zero out enemy HPs 
        for i in range(1, 5):
            self.raw_env.ram[0x60 + i] = 0
            
    def close(self):
        self.env.close()
