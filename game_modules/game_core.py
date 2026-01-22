from .map_manager import MapManager
from .tank import Player, Enemy
from .bullet import Bullet
import numpy as np

class BattleCityGame:
    def __init__(self):
        self.map_manager = MapManager()
        self.player = None
        self.enemies = []
        self.bullets = []
        
        self.episode_kills = 0
        self.total_enemies_spawned = 0
        self.max_enemies = 4
        
        self.reset()
        
    def reset(self):
        self.map_manager = MapManager() # Reload map
        self.player = Player(16, 48)
        self.enemies = []
        self.bullets = []
        self.episode_kills = 0
        self.total_enemies_spawned = 0
        
        # Initial spawn
        for _ in range(self.max_enemies):
            self.spawn_enemy()
            
    def spawn_enemy(self):
        if len(self.enemies) >= 4: return
        if self.total_enemies_spawned >= 20: return
        
        spawns = [[0, 0], [24, 0], [48, 0]]
        # Simple selection or random
        import random
        random.shuffle(spawns)
        
        for pos in spawns:
            # Check overlap
            overlap = False
            # Check vs existing
            for e in self.enemies:
                if abs(e.x - pos[0]) < 6 and abs(e.y - pos[1]) < 6:
                    overlap = True; break
            # Check vs player
            if abs(self.player.x - pos[0]) < 6 and abs(self.player.y - pos[1]) < 6:
                 overlap = True
                 
            if not overlap:
                self.enemies.append(Enemy(pos[0], pos[1]))
                self.total_enemies_spawned += 1
                return

    def step(self, action):
        reward = 0
        terminated = False
        
        # 1. Update Bullets (Physics First)
        # We iterate over copy
        for b in list(self.bullets):
            b.update()
            
        # 2. Bullet Collisions (Bullet vs Unit/Map)
        # Also Bullet vs Bullet?
        # Let's keep it simple: Bullet vs Bullet destroys both
        to_remove = set()
        for i in range(len(self.bullets)):
            for j in range(i+1, len(self.bullets)):
                 b1 = self.bullets[i]
                 b2 = self.bullets[j]
                 if b1.owner_id != b2.owner_id:
                     if abs(b1.x - b2.x) < 4 and abs(b1.y - b2.y) < 4:
                         to_remove.add(b1)
                         to_remove.add(b2)
                         
        for b in to_remove:
            if b in self.bullets: self.bullets.remove(b)
            
        # Check collision for remaining
        active_bullets = []
        for b in self.bullets:
            hit, game_over = b.check_collision(self.map_manager, self.enemies, self.player)
            
            if game_over:
                terminated = True
                reward += -5.0 # Death penalty
            
            if hit:
                # If hit enemy, did we kill it?
                # The bullet checks marks enemy.alive = False
                pass
            else:
                active_bullets.append(b)
                
        self.bullets = active_bullets
        
        # Cleanup Dead Enemies
        alive_enemies = []
        for e in self.enemies:
            if e.alive:
                alive_enemies.append(e)
            else:
                reward += 10.0 # Kill reward
                self.episode_kills += 1
                self.spawn_enemy()
        self.enemies = alive_enemies
        
        if self.episode_kills >= 20:
            reward += 100.0 # Win
            terminated = True
            
        if terminated: return reward, True
            
        # 3. Player Update
        moved, new_bullet = self.player.update(action, self.map_manager, self.enemies)
        if moved:
            # Exploration reward?
            pass
        if new_bullet:
            # Limit bullets
            p_bullets = sum(1 for b in self.bullets if b.owner_id == 0)
            if p_bullets < 1:
                # IMMEDIATE MUZZLE CHECK (Point-Blank Fix)
                # Check collision before adding to list.
                # If it hits a wall/enemy instantly at spawn, destroy it and don't spawn bullet.
                hit, game_over = new_bullet.check_collision(self.map_manager, self.enemies, self.player)
                
                if hit:
                    # Bullet hit something immediately.
                    # If it hit map, map_manager.destroy already modified the map.
                    # If it hit enemy (owner=0), enemy is marked dead.
                    # We just don't append the bullet.
                    pass
                else:
                    self.bullets.append(new_bullet)
                
        # 4. Enemy Update
        for e in self.enemies:
            bullet = e.update(self.map_manager, self.player, self.enemies)
            if bullet:
                # IMMEDIATE MUZZLE CHECK for Enemy
                # Note: Enemy firing rate is controlled by random, but we still check limits?
                # Original code didn't check limits for enemies, just spawn.
                # We check collision.
                hit, game_over = bullet.check_collision(self.map_manager, self.enemies, self.player)
                
                if game_over and bullet.owner_id != 0: # Enemy hit player
                    terminated = True
                    reward += -5.0
                    
                if not hit:
                    self.bullets.append(bullet)
                
        return reward, terminated

    def get_frame(self):
        grid = self.map_manager.get_map().copy()
        
        # Draw Player
        px, py = self.player.x, self.player.y
        grid[py:py+4, px:px+4] = 150 # ID_MAP["player"]
        
        # Draw Enemies
        for e in self.enemies:
            ex, ey = e.x, e.y
            grid[ey:ey+4, ex:ex+4] = 80 # ID_MAP["enemy"]
            
        # Draw Bullets
        for b in self.bullets:
            rect = b.get_render_rect() # x, y, w, h
            bx, by, bw, bh = rect
            # Clip
            if 0 <= bx < 52 and 0 <= by < 52:
                 grid[by:by+bh, bx:bx+bw] = 255
                 
        return grid
