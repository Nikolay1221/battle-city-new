import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
import cv2
import numpy as np
from collections import deque

class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, use_vision=False, stack_size=4):
        super(BattleCityEnv, self).__init__()
        
        ROM_PATH = 'BattleCity_fixed.nes' 
        self.render_mode = render_mode
        self.USE_VISION = use_vision
        self.STACK_SIZE = stack_size
        
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

        # Dynamic Observation Space
        ram_size = 2048 * self.STACK_SIZE
        
        if self.USE_VISION:
            # DICT: Screen + RAM
            self.observation_space = spaces.Dict({
                "screen": spaces.Box(low=0, high=255, shape=(84, 84, self.STACK_SIZE), dtype=np.uint8),
                "ram": spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32)
            })
        else:
            # BOX: RAM Only
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(ram_size,), dtype=np.float32)
        
        # RAM Stack Buffer
        self.ram_stack = deque(maxlen=self.STACK_SIZE)

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
        
        self.prev_lives = 3
        self.prev_kills = [0, 0, 0, 0]
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
        self.frames = deque(maxlen=self.STACK_SIZE)

    def _process_frame(self, obs):
        """Resize to 84x84 and Grayscale"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # (84, 84)

    def _get_obs(self):
        # ... (Image processing handled by frames stack, assumed done in step/reset) ...
        # 2. RAM Stack
        raw_ram = self.raw_env.ram
        # Normalize IMMEDIATELY (0-255 -> 0.0-1.0)
        current_ram_snap = np.array(raw_ram[:2048], dtype=np.float32) / 255.0
        
        # Add to stack
        self.ram_stack.append(current_ram_snap)
        
        # Fill if empty (first frame)
        while len(self.ram_stack) < self.STACK_SIZE:
             self.ram_stack.append(current_ram_snap)
             
        # Flatten Stack: [Frame1, Frame2, Frame3, Frame4] -> Vector
        ram_obs = np.concatenate(self.ram_stack) 
        
        if not self.USE_VISION:
            return ram_obs # Return ONLY vector (Box Space)

        # 1. Screen (Only if USE_VISION)
        if len(self.frames) < self.STACK_SIZE:
            screen_obs = np.zeros((self.STACK_SIZE, 84, 84), dtype=np.uint8)
        else:
            screen_obs = np.array(self.frames, dtype=np.uint8)
            
        # Transpose to (H, W, C) for Stable Baselines CnnPolicy
        screen_obs = np.moveaxis(screen_obs, 0, -1)
            
        return {
            "screen": screen_obs,
            "ram": ram_obs
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # self.raw_env is accessible.
        self.raw_env.reset()
        
        self.steps_in_episode = 0
        self.episode_score = 0.0 # Reset score
        self.visited_sectors = set() # Track visited 16x16 zones
        
        # Clear Buffers
        self.frames.clear()
        self.ram_stack.clear()
        
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
        
        # Initial Frame Processing
        obs = self.raw_env.screen # Define obs
        processed = self._process_frame(obs)
        for _ in range(self.STACK_SIZE):
            self.frames.append(processed)
            
        # Fill RAM Stack (Initialize with current state)
        initial_ram = np.array(self.raw_env.ram[:2048], dtype=np.float32) / 255.0
        for _ in range(self.STACK_SIZE):
            self.ram_stack.append(initial_ram)
            
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
        
        # Process and Push new frame
        processed = self._process_frame(obs)
        self.frames.append(processed)
        
        info = {}
        info['render'] = processed # FOR VISUALIZATION WINDOW (Bypass Agent)
        
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
             
             # HEAVY PENALTY FOR BASE DESTRUCTION
             # If game ended but we didn't lose a life -> Base was destroyed!
             if curr_lives >= self.prev_lives:
                 reward -= 5.0 # Massive punishment for losing base
                 # print("DEBUG: BASE DESTROYED! Punishment: -5.0")
             else:
                 reward -= 1.0 # Standard Game Over (Ran out of lives)

    # (Fixing reset RAM type below in separate chunk if needed, but I'll try to do it here if close enough? No, 276 is far from 193. Need MultiReplace)
             
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



        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        # Robust Render: Try with mode, then without
        try:
            self.env.render(mode=mode)
        except TypeError:
            self.env.render() # Try no-args (New Gym/NesPy updates)

    def close(self):
        self.env.close()
