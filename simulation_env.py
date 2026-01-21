import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random

# Import our new Object System
from simulation_objects import (
    Entity, Tank, Enemy, Bullet, 
    GRID_SIZE, ID_EMPTY, ID_BRICK, ID_STEEL, ID_EAGLE, 
    ID_PLAYER, ID_ENEMY, ID_BULLET,
    check_entity_collision
)

class SimulationBattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, stack_size=4):
        super(SimulationBattleCityEnv, self).__init__()
        self.render_mode = render_mode
        self.STACK_SIZE = stack_size
        
        # Load Map (Expect 26x26 grid)
        try:
            loaded_map = np.load("level1_map.npy")
            if loaded_map.shape != (26, 26):
                # Resize if needed
                loaded_map = cv2.resize(loaded_map, (26, 26), interpolation=cv2.INTER_NEAREST)
            self.initial_map = loaded_map.astype(np.uint8)
        except:
            print("Map not found! Using empty map.")
            self.initial_map = np.zeros((26, 26), dtype=np.uint8)
            
        self.GRID_SIZE = GRID_SIZE # 52
        
        # Actions: 0..9 (Noop, Up, Down, Left, Right, Fire, Up+Fire...)
        self.action_space = spaces.Discrete(10)

        # Observation: Map + Player Pos
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        self.current_map = None
        
        # ENTITIES
        self.player = None
        self.enemies = []
        self.bullets = []
        
        self.max_steps = 2000
        self.steps = 0
        self.frames = []
        self.sandbox_mode = False
        self.visited_sectors = set()
        
        # Rewards
        self.rew_explore = 0.1
        self.rew_stuck = -0.01
        self.rew_brick = 0.0
        self.rew_base = -20.0
        self.rew_kill = 1.0

        # Colors for ID (Legacy dict for renderer if needed, but we use constants now)
        self.ID_MAP = {
            "empty": ID_EMPTY,
            "brick": ID_BRICK,
            "steel": ID_STEEL,
            "eagle": ID_EAGLE,
            "player": ID_PLAYER,
            "enemy": ID_ENEMY,
            "bullet": ID_BULLET
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Prepare Map (26x26 -> 52x52)
        temp_map = self.initial_map.copy()
        display_map = np.zeros_like(temp_map)
        display_map[temp_map == 25] = ID_BRICK
        display_map[temp_map == 50] = ID_STEEL
        display_map[temp_map == 75] = ID_EAGLE
        
        self.current_map = np.kron(display_map, np.ones((2,2), dtype=np.uint8))
        if self.current_map.shape != (52, 52):
             self.current_map = cv2.resize(self.current_map, (52, 52), interpolation=cv2.INTER_NEAREST)

        # 2. Reset Entities
        self.player = Tank(16, 48, is_player=True)
        # Check if player spawn is blocked by static map?
        # If so, find free spot (Legacy logic simplified)
        # We assume 16,48 is valid for now.
        
        self.enemies = []
        self.bullets = []
        
        self.visited_sectors = set()
        self.spawn_timer = 0
        self.total_enemies_spawned = 0
        self.MAX_ENEMIES_ON_SCREEN = 4
        self.MAX_ENEMIES_TOTAL = 20
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
        
        # --- 1. PLAYER INPUT ---
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
            
        if self.sandbox_mode: fire_action = False

        # --- 2. UPDATE PLAYER ---
        # Move
        if move_action > 0:
            dx, dy = 0, 0
            # 1: Up, 2: Down, 3: Left, 4: Right
            # Tank Dir: 0:Up, 1:Right, 2:Down, 3:Left
            
            # Map Action to Dir & Delta
            target_dir = 0
            if move_action == 1: dy = -1; target_dir = 0 # UP
            elif move_action == 2: dy = 1; target_dir = 2 # DOWN
            elif move_action == 3: dx = -1; target_dir = 3 # LEFT
            elif move_action == 4: dx = 1; target_dir = 1 # RIGHT
            
            self.player.direction = target_dir
            
            # Try Move
            # Obstacles for player = Enemies
            moved = self.player.try_move(dx * self.player.speed, dy * self.player.speed, self.current_map, self.enemies)
            
            if moved:
                # Explore Reward
                sec_x, sec_y = int(self.player.x) // 4, int(self.player.y) // 4
                if (sec_x, sec_y) not in self.visited_sectors:
                    reward += self.rew_explore
                    self.visited_sectors.add((sec_x, sec_y))
            else:
                reward += self.rew_stuck
        
        # Fire
        if fire_action and self.player.cooldown <= 0:
            # Limit bullets?
            player_bullets = len([b for b in self.bullets if b.owner == self.player])
            if player_bullets < 1: # Max 1 on screen
                new_bullet = self.player.fire()
                if new_bullet: self.bullets.append(new_bullet)

        if self.player.cooldown > 0: self.player.cooldown -= 1

        # --- 3. SPAWN ENEMIES ---
        self.spawn_timer += 1
        if (self.spawn_timer > 60 and 
            len(self.enemies) < self.MAX_ENEMIES_ON_SCREEN and 
            self.total_enemies_spawned < self.MAX_ENEMIES_TOTAL):
            
            spawn_x = [2, 24, 48][self.total_enemies_spawned % 3] 
            spawn_y = 2
            
            # STRICT SPAWN CHECK
            # Create a temp Enemy rect at spawn point
            spawn_rect = (spawn_x, spawn_y, 4, 4) # 4x4 Tank
            
            # Check against ALL entities (Player + Enemies)
            # We can use check_entity_collision helper
            # Construct a dummy entity or list
            all_tanks = [self.player] + self.enemies
            
            if not check_entity_collision(spawn_rect, all_tanks):
                # Spawn!
                new_enemy = Enemy(spawn_x, spawn_y)
                self.enemies.append(new_enemy)
                self.total_enemies_spawned += 1
                self.spawn_timer = 0

        # --- 4. UPDATE ENEMIES ---
        for e in self.enemies:
            e.tick(self.current_map, self.player, self.enemies)
            
            # Enemy Fire?
            if random.random() < 0.03: # 3% chance per frame
                 # Limit enemy bullets
                enemy_bullets = len([b for b in self.bullets if b.owner == e])
                if enemy_bullets < 1:
                    new_bullet = e.fire()
                    if new_bullet: self.bullets.append(new_bullet)
            
            if e.cooldown > 0: e.cooldown -= 1

        # --- 5. UPDATE BULLETS ---
        active_bullets = []
        for b in self.bullets:
            entities_to_check = [self.player] + self.enemies
            hit_entity = b.tick(self.current_map, entities_to_check)
            
            if b.active:
                active_bullets.append(b)
            elif hit_entity:
                # Handle kill logic
                if hit_entity == self.player:
                     reward += -1.0 # Hit player
                     # Reset player? Or End game?
                     # For now, just penalty or terminate
                     terminated = True 
                elif isinstance(hit_entity, Enemy):
                    if hit_entity in self.enemies:
                        self.enemies.remove(hit_entity)
                        reward += self.rew_kill
                        
        self.bullets = active_bullets

        # --- 6. RENDER & OBSERVATION ---
        frame = self._get_frame()
        self.frames.append(frame)
        if len(self.frames) > self.STACK_SIZE: self.frames.pop(0)
        
        if self.steps >= self.max_steps:
            truncated = True
            
        info = {
            "exploration_pct": (len(self.visited_sectors) / (13*13)) * 100,
            "kills": 0, # TODO: Track kills
            "env_type": 1 
        }
            
        return self._get_obs(), reward, terminated, truncated, info

    def _get_frame(self):
        grid = self.current_map.copy()
        
        # 1. Bullets
        for b in self.bullets:
            bx, by = int(b.x), int(b.y)
            if 0 <= bx < self.GRID_SIZE and 0 <= by < self.GRID_SIZE:
                grid[by, bx] = ID_BULLET

        # 2. Enemies
        for e in self.enemies:
            ex, ey = int(e.x), int(e.y)
            # Draw 4x4
            er, ec = min(self.GRID_SIZE, ey+4), min(self.GRID_SIZE, ex+4)
            grid[ey:er, ex:ec] = ID_ENEMY
            self._draw_gun(grid, e.x, e.y, e.direction)

        # 3. Player
        px, py = int(self.player.x), int(self.player.y)
        pr, pc = min(self.GRID_SIZE, py+4), min(self.GRID_SIZE, px+4)
        grid[py:pr, px:pc] = ID_PLAYER
        self._draw_gun(grid, self.player.x, self.player.y, self.player.direction)
        
        return grid

    def _draw_gun(self, grid, x, y, direction):
        x, y = int(x), int(y)
        gun_color = 255
        
        coords = []
        if direction == 0: coords = [(x+1, y), (x+2, y)] # Up
        elif direction == 1: coords = [(x+3, y+1), (x+3, y+2)] # Right
        elif direction == 2: coords = [(x+1, y+3), (x+2, y+3)] # Down
        elif direction == 3: coords = [(x, y+1), (x, y+2)] # Left
            
        for (gx, gy) in coords:
            if 0 <= gx < self.GRID_SIZE and 0 <= gy < self.GRID_SIZE:
                grid[gy, gx] = gun_color

    def _get_obs(self):
        return np.stack(self.frames, axis=-1)
    
    def get_tactical_rgb(self):
        grid = self._get_frame()
        h, w = grid.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Colors
        img[grid == ID_BRICK] = [160, 80, 40]
        img[grid == ID_STEEL] = [200, 200, 200]
        img[grid == ID_PLAYER] = [220, 200, 0]
        img[grid == ID_ENEMY] = [200, 40, 40]
        img[grid == ID_BULLET] = [255, 255, 255]
        
        return img

    def render(self):
        if self.render_mode == 'rgb_array':
            return self.get_tactical_rgb()
        return None
    
    def toggle_sandbox_mode(self):
        self.sandbox_mode = not self.sandbox_mode
        print(f"SANDBOX MODE: {self.sandbox_mode}")
