import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random

class VirtualBattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, stack_size=4):
        super(VirtualBattleCityEnv, self).__init__()
        self.render_mode = render_mode
        self.STACK_SIZE = stack_size
        
        # Load Map
        try:
            loaded_map = np.load("level1_map.npy")
            if loaded_map.shape != (26, 26):
                loaded_map = cv2.resize(loaded_map, (26, 26), interpolation=cv2.INTER_NEAREST)
            self.initial_map = loaded_map.astype(np.uint8)
        except:
            self.initial_map = np.zeros((26, 26), dtype=np.uint8)
            
        self.GRID_SIZE = 52

        # Actions: 0-NOOP, 1-4 Move, 5 Fire, 6-9 Move+Fire
        self.action_space = spaces.Discrete(10)

        # Observation
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        self.current_map = None
        self.player_pos = [16, 48]
        self.player_dir = 0 
        
        self.enemies = [] 
        self.bullets = [] 
        
        self.max_steps = 3000 
        self.steps = 0
        self.frames = []
        self.visited_sectors = set()
        self.episode_kills = 0
        self.total_enemies_spawned = 0 
        
        # Rewards
        self.rew_explore = 0.005 
        self.rew_stuck = -0.01
        self.rew_brick = 0.02 
        self.rew_base = -20.0
        self.rew_kill = 10.0  
        self.rew_death = -5.0 
        self.rew_win = 100.0  

        # ID Mapping - GRAYSCALE
        self.ID_MAP = {
            "empty": 0,
            "brick": 200,   
            "steel": 255,   
            "eagle": 254,   
            "player": 150,  
            "enemy": 80,    
            "bullet": 255   
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Prepare Map
        temp_map = self.initial_map.copy()
        
        # FORCE EAGLE AND BASE PROTECTION (Standard Battle City Layout)
        # Eagle at (12, 24) in 26x26 grid
        temp_map[24, 12] = 75 # Eagle
        # Brick Wall around Eagle
        temp_map[24, 11] = 25
        temp_map[24, 13] = 25
        temp_map[23, 11] = 25
        temp_map[23, 12] = 25
        temp_map[23, 13] = 25
        
        display_map = np.zeros_like(temp_map)
        display_map[temp_map == 25] = self.ID_MAP["brick"]
        display_map[temp_map == 50] = self.ID_MAP["steel"]
        display_map[temp_map == 75] = self.ID_MAP["eagle"]
        
        self.current_map = np.kron(display_map, np.ones((2,2), dtype=np.uint8))
        if self.current_map.shape != (52, 52):
             self.current_map = cv2.resize(self.current_map, (52, 52), interpolation=cv2.INTER_NEAREST)

        self.visited_sectors = set()
        self.episode_kills = 0
        self.total_enemies_spawned = 0
        self.steps = 0
        
        # Spawn Player (Left of Base)
        self.player_pos = [16, 48]
        self.player_dir = 0
        
        # Spawn Initial Enemies
        self.enemies = []
        for _ in range(4):
            self.spawn_enemy()
        
        self.bullets = []

        # Fill stack
        frame = self._get_frame()
        self.frames = [frame for _ in range(self.STACK_SIZE)]
        
        return self._get_obs(), {}

    def spawn_enemy(self):
        if len(self.enemies) >= 4: return
        if self.total_enemies_spawned >= 20: return

        spawns = [[0, 0], [24, 0], [48, 0]]
        random.shuffle(spawns) 
        
        for pos in spawns:
            overlap = False
            for e in self.enemies:
                if abs(e[0] - pos[0]) < 6 and abs(e[1] - pos[1]) < 6: 
                    overlap = True
                    break
            
            if abs(self.player_pos[0] - pos[0]) < 6 and abs(self.player_pos[1] - pos[1]) < 6:
                overlap = True

            if not overlap:
                self.enemies.append([pos[0], pos[1], 2]) 
                self.total_enemies_spawned += 1
                return 

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False
        truncated = False
        
        # 1. Player Action
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
            player_bullets = sum(1 for b in self.bullets if b[3] == 0)
            if player_bullets < 1:
                bx, by = self.player_pos[0] + 2, self.player_pos[1] + 2 
                self.bullets.append([bx, by, self.player_dir, 0]) 

        # 2. Enemy Logic
        for e in self.enemies:
            if random.random() < 0.05:
                e[2] = random.choice([0, 1, 2, 3])
            
            edx, edy = 0, 0
            if e[2] == 0: edy = -1
            elif e[2] == 1: edx = 1
            elif e[2] == 2: edy = 1
            elif e[2] == 3: edx = -1
            
            nex, ney = e[0] + edx, e[1] + edy
            if self._is_free(nex, ney, ignore_enemies=True):
                e[0], e[1] = nex, ney
            else:
                e[2] = random.choice([0, 1, 2, 3]) 
                
            if random.random() < 0.03: 
                 self.bullets.append([e[0]+2, e[1]+2, e[2], 1]) 

        # 3. Bullet Physics
        for b in self.bullets:
            bx, by, bdir, owner = b
            bdx, bdy = 0, 0
            if bdir == 0: bdy = -2 
            elif bdir == 1: bdx = 2
            elif bdir == 2: bdy = 2
            elif bdir == 3: bdx = -2
            b[0] += bdx
            b[1] += bdy

        bullets_to_remove = set()
        for i in range(len(self.bullets)):
            for j in range(i + 1, len(self.bullets)):
                b1 = self.bullets[i]
                b2 = self.bullets[j]
                if b1[3] != b2[3]:
                    if abs(b1[0] - b2[0]) < 4 and abs(b1[1] - b2[1]) < 4:
                        bullets_to_remove.add(i)
                        bullets_to_remove.add(j)

        active_bullets = []
        for i, b in enumerate(self.bullets):
            if i in bullets_to_remove: continue
            
            bx, by, bdir, owner = b
            
            if not (0 <= bx < 52 and 0 <= by < 52): continue
            
            # IMPROVED COLLISION DETECTION (Area Check)
            hit_map = False
            for cy in range(int(by)-1, int(by)+2):
                for cx in range(int(bx)-1, int(bx)+2):
                    if not (0 <= cx < 52 and 0 <= cy < 52): continue
                    
                    tile = self.current_map[cy, cx]
                    
                    if tile == self.ID_MAP["eagle"]:
                        terminated = True
                        reward += self.rew_base
                        hit_map = True
                        break 
                    
                    elif tile == self.ID_MAP["brick"]:
                        self.current_map[cy, cx] = 0
                        reward += self.rew_brick
                        hit_map = True
                        
                    elif tile == self.ID_MAP["steel"]:
                        hit_map = True
                        
                if terminated: break
            
            if terminated: break 
            if hit_map: continue 
                
            # Tank Collision
            hit = False
            if owner == 0: # Player bullet
                for e in self.enemies:
                    if abs(e[0] - bx) < 4 and abs(e[1] - by) < 4:
                        self.enemies.remove(e)
                        reward += self.rew_kill
                        self.episode_kills += 1
                        self.spawn_enemy() 
                        hit = True
                        break
            else: # Enemy bullet
                if abs(self.player_pos[0] - bx) < 4 and abs(self.player_pos[1] - by) < 4:
                    reward += self.rew_death
                    terminated = True 
                    hit = True
            
            if not hit:
                active_bullets.append(b)
                
        self.bullets = active_bullets

        # Win Condition
        if self.episode_kills >= 20:
            reward += self.rew_win
            terminated = True

        frame = self._get_frame()
        self.frames.append(frame)
        if len(self.frames) > self.STACK_SIZE: self.frames.pop(0)
        
        if self.steps >= self.max_steps:
            truncated = True
            
        info = {
            "exploration_pct": (len(self.visited_sectors) / (13*13)) * 100,
            "kills": self.episode_kills,
            "env_type": 1
        }
            
        return self._get_obs(), reward, terminated, truncated, info

    def _is_free(self, x, y, ignore_enemies=False):
        if x < 0 or y < 0 or x > 48 or y > 48: return False
        # Check map
        for r in range(y, y+4):
            for c in range(x, x+4):
                if r >= 52 or c >= 52: return False
                if self.current_map[r, c] != 0: return False
        
        # Check player collision (for enemies)
        if ignore_enemies:
             if abs(self.player_pos[0] - x) < 4 and abs(self.player_pos[1] - y) < 4:
                 return False
                 
        return True

    def _get_frame(self):
        grid = self.current_map.copy()
        
        # Draw Player
        px, py = self.player_pos
        grid[py:py+4, px:px+4] = self.ID_MAP["player"]
        
        # Draw Enemies
        for e in self.enemies:
            ex, ey = e[0], e[1]
            grid[ey:ey+4, ex:ex+4] = self.ID_MAP["enemy"]
            
        # Draw Bullets
        for b in self.bullets:
            bx, by = int(b[0]), int(b[1])
            if 0 <= bx < 52 and 0 <= by < 52:
                grid[by, bx] = self.ID_MAP["bullet"]
                
        return grid

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)
    
    def get_tactical_rgb(self):
        grid = self._get_frame()
        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return img

    def get_debug_info(self):
        return "VIRTUAL_COMBAT"

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.get_tactical_rgb()
        return None
