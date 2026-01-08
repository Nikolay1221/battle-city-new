import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from collections import deque

class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(BattleCityEnv, self).__init__()
        
        ROM_PATH = 'BattleCity_fixed.nes' 
        self.render_mode = render_mode
        
        # Raw Env
        self.raw_env = NESEnv(ROM_PATH)
        
        # Define Restricted Actions: Move, Fire, Move+Fire
        # NO START/SELECT!
        actions = [
            ['NOOP'],
            ['up'],
            ['down'],
            ['left'],
            ['right'],
            ['A'], # Fire
            ['up', 'A'],
            ['down', 'A'],
            ['left', 'A'],
            ['right', 'A']
        ]
        
        # Wrap in JoypadSpace
        self.env = JoypadSpace(self.raw_env, actions)
        
        # FIX: JoypadSpace uses old 'gym' spaces. We need 'gymnasium' spaces.
        # We manually redefine the action space to match.
        self.action_space = spaces.Discrete(len(actions))

        # Observation Space: FLAT RAM (MLP Friendly)
        # No Dictionary. Just pure 2048 numbers. This is fastest.
        # "Blind Mode" - Agent sees only the Matrix code.
        self.observation_space = spaces.Box(low=0, high=255, shape=(2048,), dtype=np.uint8)
        
        # Load Templates
        self.game_over_tmpl = cv2.imread('templates/game_over.png', cv2.IMREAD_GRAYSCALE)
        if self.game_over_tmpl is None:
            raise ValueError("Template templates/game_over.png not found!")

        # RAM Addresses
        self.ADDR_LIVES = 0x51
        self.ADDR_STATE = 0x92
        self.ADDR_KILLS = [0x73, 0x74, 0x75, 0x76] 
        self.ADDR_BONUS = 0x62
        self.ADDR_STAGE = 0x85
        
        # Scanner & Helper Vars
        self.valid_actions = actions
        
        # Initialize Previous State
        self.prev_lives = 3
        self.prev_kills = 0
        self.prev_score = 0
        self.prev_bonus = 0
        self.prev_stage = 0
        
        # Idle Penalty Vars
        # FOUND BY SCANNER & CONFIRMED BY USER:
        # X: 0x90 
        # Y: 0x98 (User reports 0x98 for Up/Down)
        self.ADDR_X = 0x0090
        self.ADDR_Y = 0x0098
        self.prev_x = 0
        self.prev_y = 0
        self.idle_steps = 0
        self.IDLE_THRESHOLD = 30 # 30 steps * 4 frames = 120 frames (~2 sec)
        
        # Step Logic
        self.steps_in_episode = 0
        self.MAX_STEPS = 5000 
        
        # Frame Stack Buffer
        # self.frames = deque(maxlen=4) # Removed for RAM-only observation

    # Removed _process_frame as screen is no longer observed

    def _get_obs(self):
        """
        Get RAM State (2KB).
        Optimization: No screen capture, no OpenCV, no resizing.
        Pure memory read (Microseconds).
        """
        raw_ram = self.raw_env.ram
        # Take first 2048 bytes
        return np.array(raw_ram[:2048], dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
             super().reset(seed=seed)
        
        # Frame Stacking Buffer
        self.frames = deque(maxlen=4)
        
        # self.raw_env is accessible.
        
        obs = self.raw_env.reset() # Reset Raw Env
        
        self.steps_in_episode = 0
        self.episode_score = 0.0 # Reset score
        self.visited_sectors = set() # Track visited 16x16 zones
        self.frames.clear()
        
        # Auto-Skip Menu using RAW ENV actions (byte)
        # Start button is 0x08 (bit 3) -> 8
        # print("DEBUG: Resetting... Looping to Start Game (Optimized)")
        
        # Removed 60-frame warm-up to reduce "pause"
        
        # 2. Hardcoded Start Sequence (3 Presses for Level Select)
        # Sequence: Title -> [Start] -> Mode -> [Start] -> Level Select? -> [Start] -> Game
        # print("DEBUG: Resetting... Executing Hardcoded Start Sequence (3 Presses)")
        
        # 1. Wait for Title (Robust Buffer)
        for _ in range(80): self.raw_env.step(0)
            
        # 2. Press Start (Title -> Mode)
        for _ in range(10): self.raw_env.step(8) # Hold longer
        for _ in range(30): self.raw_env.step(0) # Wait for fade
        
        # 3. Press Start (Mode -> Stage)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)

        # 4. Press Start (Stage -> Game)
        for _ in range(10): self.raw_env.step(8)
        
        # 5. Wait for Curtain (Game Start)
        for _ in range(60): self.raw_env.step(0)

        # Check debug
        state = int(self.raw_env.ram[self.ADDR_STATE])
        lives = int(self.raw_env.ram[self.ADDR_LIVES])
        # print(f"DEBUG: Sequence Complete. State: {state:02X}, Lives: {lives}")
            
        self.prev_lives = int(self.raw_env.ram[self.ADDR_LIVES])
        self.prev_kills = [int(self.raw_env.ram[addr]) for addr in self.ADDR_KILLS]
        self.prev_bonus = int(self.raw_env.ram[self.ADDR_BONUS])
        self.prev_bonus = int(self.raw_env.ram[self.ADDR_BONUS])
        self.prev_stage = int(self.raw_env.ram[self.ADDR_STAGE])
        self.prev_x = int(self.raw_env.ram[self.ADDR_X])
        self.prev_y = int(self.raw_env.ram[self.ADDR_Y])
        self.idle_steps = 0
        
        # Initial Frame Processing - SKIPPED (RAM Mode)
        # processed = self._process_frame(obs)
        # for _ in range(4):
        #     self.frames.append(processed)
            
        return self._get_obs(), {} # Return (Obs, Info) for Gymnasium
        
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        # Frame Skip x4
        for _ in range(4):
            # Use Wrapped Env step (maps action index to button press)
            obs, r, d, i = self.env.step(action)
            total_reward += r 
            if d:
                done = True
                break
        
        self.steps_in_episode += 1 
        
        # Process Frame (SKIPPED in RAM Mode)
        # processed = self._process_frame(obs)
        # self.frames.append(processed)
        
        ram = self.raw_env.ram # Access RAM from raw env
        reward = 0 
        
         # 0. Time Penalty
        if self.steps_in_episode >= self.MAX_STEPS:
            truncated = True # Time Limit = Truncated
            info["TimeLimit.truncated"] = True

        # 1. Kill Rewards
        curr_kills = [int(ram[addr]) for addr in self.ADDR_KILLS]
        for i in range(4):
            diff = curr_kills[i] - self.prev_kills[i]
            if diff > 0 and diff < 10:
                # OLD: pts = (i + 1) * 100 (100..400)
                # NORMALIZED TO GOLDEN ZONE (Points / 1000)
                # Tank 1 (100pts) -> 0.1
                # Tank 2 (200pts) -> 0.2
                # Tank 3 (300pts) -> 0.3
                # Tank 4 (400pts) -> 0.4
                base_scores = [0.1, 0.2, 0.3, 0.4] 
                pts = base_scores[i]
                reward += pts * diff 
        self.prev_kills = curr_kills
        
        # 2. Bonus
        # RAM[0x62] behaves as a timer (0 -> 49 ... -> 0).
        # We reward on INCREASE (0->49 or restart).
        curr_bonus = int(ram[self.ADDR_BONUS])
        
        if curr_bonus > self.prev_bonus:
             reward += 0.5 # Normalized (+5.0 -> +0.5)
             
        # Trigger on Stage Reset (0->1 etc) - usually level num increases
        curr_stage = int(ram[self.ADDR_STAGE])
        if curr_stage > self.prev_stage:
             reward += 2.0 # Normalized (+20.0 -> +2.0)
        self.prev_stage = curr_stage
        
        # 3. Death
        curr_lives = int(ram[self.ADDR_LIVES])
        if curr_lives < 10 and self.prev_lives < 10:
             # Check if we didn't just reset (sometimes state creates ghost lives)
             if curr_lives < self.prev_lives:
                reward -= 0.1 # Normalized (-0.5 -> -0.1)
        self.prev_lives = curr_lives
        
        # 4. Game Over (Vision - Restored & Tuned)
        # RAM check failed for user. Vision worked but missed Stage 2. 
        # Lowering threshold 0.8 -> 0.7 to be more robust against background changes.
        gray_full = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(gray_full, self.game_over_tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > 0.7: 
             terminated = True # Game Over = Terminated
             reward -= 1.0 # Normalized (-10.0 -> -1.0)
             # print(f"DEBUG: Game Over Screen Detected! (Score: {max_val:.2f})")
             
        # RAM Check Disabled (User reported non-functional)
        # if int(ram[self.ADDR_STATE]) == 0xE0: ... 
            
        # 5. Idle Penalty (Coordinate Based)
        
        curr_x = int(ram[self.ADDR_X])
        curr_y = int(ram[self.ADDR_Y])
        
        # 6. MOVEMENT REWARD (Normalized)
        # Small incentive to move, but NOT enough to outweigh killing.
        # Kill (+5.0) >> Move (+0.05). 
        if curr_lives > 0:
            # 6. MOVEMENT REWARD (Normalized)
            if curr_x != self.prev_x or curr_y != self.prev_y:
                 reward += 0.03 # Movement reward
                 self.idle_steps = 0
                 
                 # 7. GRID EXPLORATION (New!)
                 # Map is approx 240x240. Sectors of 16x16 (Standard Tile Size)
                 # X: 0..255, Y: 0..240
                 sec_x = curr_x // 16
                 sec_y = curr_y // 16
                 sector = (sec_x, sec_y)
                 
                 if sector not in self.visited_sectors:
                     reward += 0.1 # Discovery Bonus! (Like finding a coin)
                     self.visited_sectors.add(sector)
                     # print(f"DEBUG: New Sector Discovered {sector}!")
            else:
                 self.idle_steps += 1
                 
            # Threshold: 10 steps
            if self.idle_steps > 10:
                reward -= 0.02 # Standard normalized penalty


        else:
            self.idle_steps = 0 # Reset if dead
            
        self.prev_x = curr_x
        self.prev_y = curr_y
        
        # Debug info
        curr_state = int(ram[self.ADDR_STATE])
        info['idle_steps'] = self.idle_steps
        info['ram_state'] = curr_state
        info['x'] = curr_x
        info['y'] = curr_y
        info['kills'] = sum(curr_kills)
        
        self.episode_score += reward
        info['score'] = self.episode_score

        # DEBUG: Verify Agent "Vision" and "Action"
        # Print every 60 steps (1 sec)
        # if self.steps_in_episode % 60 == 0:
        #    pass

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        # Robust Render: Try with mode, then without
        try:
            self.env.render(mode=mode)
        except TypeError:
            self.env.render() # Try no-args (New Gym/NesPy updates)

    def close(self):
        self.env.close()
