from .bullet import Bullet
import random

class Tank(object):
    def __init__(self, x, y, direction, speed=1, tank_id=0):
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.tank_id = tank_id # 0: Player, 1: Enemy? 
        self.alive = True
        self.cooldown = 0
        
    def move(self, dx, dy, map_manager, other_tanks):
        # Proposed new position
        nx, ny = self.x + dx, self.y + dy
        
        # Check Map Collision (4x4 body)
        if not self._is_free(nx, ny, map_manager):
            return False
            
        # Check Unit Collision
        for t in other_tanks:
            if t is self: continue
            if abs(t.x - nx) < 4 and abs(t.y - ny) < 4:
                return False
                
        self.x, self.y = nx, ny
        return True

    def _is_free(self, x, y, map_manager):
        # 4x4 body check
        for r in range(y, y+4):
            for c in range(x, x+4):
                if map_manager.is_solid(c, r):
                    return False
        return True

    def fire(self):
        if self.cooldown > 0: return None
        
        # True Muzzle Spawning
        px, py = self.x, self.y
        bx, by = 0, 0
        
        if self.direction == 0:   bx, by = px + 1, py
        elif self.direction == 1: bx, by = px + 3, py + 1
        elif self.direction == 2: bx, by = px + 1, py + 3
        elif self.direction == 3: bx, by = px, py + 1
        
        self.cooldown = 20 # 20 frames delay (approx 3 shots per sec)
        return Bullet(bx, by, self.direction, self.tank_id)

class Player(Tank):
    def __init__(self, x, y):
        super().__init__(x, y, 0, speed=1, tank_id=0) # Dir 0 (UP)
        
    def update(self, action, map_manager, enemies):
        if self.cooldown > 0: self.cooldown -= 1
        
        # action: 0-NOOP, 1-4 Move, 5 Fire ...
        dx, dy = 0, 0
        fire = False
        
        move_act = 0
        if 1 <= action <= 4: move_act = action
        elif 6 <= action <= 9: 
            move_act = action - 5
            fire = True
        elif action == 5: fire = True
        
        if move_act == 1: dy = -1; self.direction = 0
        elif move_act == 2: dy = 1; self.direction = 2
        elif move_act == 3: dx = -1; self.direction = 3
        elif move_act == 4: dx = 1; self.direction = 1
        
        moved = False
        if move_act > 0:
            moved = self.move(dx, dy, map_manager, enemies)
            
        bullet = None
        if fire:
            bullet = self.fire()
            
        return moved, bullet

class Enemy(Tank):
    def __init__(self, x, y):
        super().__init__(x, y, 2, speed=1, tank_id=1) # Dir 2 (DOWN)
        self.change_dir_timer = 0
        
    def update(self, map_manager, player, enemies):
        if self.cooldown > 0: self.cooldown -= 1
        
        # AI LOGIC
        # 1. Check if we are blocked in current direction
        dx, dy = 0, 0
        if self.direction == 0: dy = -1
        elif self.direction == 1: dx = 1
        elif self.direction == 2: dy = 1
        elif self.direction == 3: dx = -1
        
        # Provisional check
        # We try to move. If move fails, we pick new direction immediately.
        # Check collision against Player AND other Enemies
        all_tanks = [player] + enemies
        
        moved = self.move(dx, dy, map_manager, all_tanks)
        
        if not moved:
            # Blocked! Pick new direction.
            # Simple AI: Pick random direction that is DIFFERENT from current.
            possible = [0, 1, 2, 3]
            if self.direction in possible: possible.remove(self.direction)
            self.direction = random.choice(possible)
            
        else:
            # Not blocked, but maybe turn?
            # Battle City tanks turn at intersections (grid multiples of 16 usually, here 4 is grid size for logic?)
            # Virtual grid is pixel based.
            # Let's just have small random chance to turn, biased towards Player/Eagle (Bottom Center).
            if random.random() < 0.05:
                # Weighted choice:
                # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
                # Eagle is at roughly (24, 48).
                # Calculate bias
                weights = [1.0, 1.0, 1.0, 1.0]
                
                # Bias towards Player Base (24, 48)
                target_x, target_y = 24, 48
                if self.x < target_x: weights[1] += 2.0 # Prefer Right
                if self.x > target_x: weights[3] += 2.0 # Prefer Left
                if self.y < target_y: weights[2] += 2.0 # Prefer Down
                if self.y > target_y: weights[0] += 0.5 # Prefer Up (less likely)
                
                # Normalize? Random.choices handles weights.
                self.direction = random.choices([0, 1, 2, 3], weights=weights, k=1)[0]

        if random.random() < 0.03:
            return self.fire()
        return None
