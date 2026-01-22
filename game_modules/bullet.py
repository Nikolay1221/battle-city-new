class Entity:
    def __init__(self, x, y, direction, speed):
        self.x = x
        self.y = y
        self.direction = direction # 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT
        self.speed = speed
        self.alive = True

    @property
    def rect(self):
        # Default size, overridden by subclasses
        return [self.x, self.y, 1, 1]

class Bullet(Entity):
    def __init__(self, x, y, direction, owner_id):
        super().__init__(x, y, direction, 2) # Speed 2
        self.owner_id = owner_id # 0: Player, 1: Enemy
        
    def update(self):
        dx, dy = 0, 0
        if self.direction == 0: dy = -self.speed
        elif self.direction == 1: dx = self.speed
        elif self.direction == 2: dy = self.speed
        elif self.direction == 3: dx = -self.speed
        self.x += dx
        self.y += dy

    def get_render_rect(self):
        # 2px width perpendicular to direction, 1px length
        bx, by = int(self.x), int(self.y)
        if self.direction == 0 or self.direction == 2: # Horizontal
            return (bx, by, 2, 1)
        else: # Vertical
            return (bx, by, 1, 2)
            
    def check_collision(self, map_manager, enemies, player):
        """
        Check collision with Map, Player, Enemies.
        Returns: (hit, terminated)
        """
        bx, by = int(self.x), int(self.y)
        
        # 1. Map Interaction (using MapManager)
        # We need to check collision at the current position.
        # But wait, logic calls for an "Area Check" using patterns.
        # So we pass the bullet center to map_manager.destroy
        
        hit_map, hit_eagle = map_manager.destroy(bx, by, self.direction)
        if hit_eagle:
             return True, True # Hit, Game Over
             
        if hit_map:
             return True, False
             
        # 2. Check bounds
        if not (0 <= bx < 52 and 0 <= by < 52):
            return True, False # Destroy bullet, no game over
            
        # 3. Unit Collision
        hit_unit = False
        
        # Helper collision rect check (4x4)
        def is_hit(tx, ty):
            return abs(tx - bx) < 4 and abs(ty - by) < 4
            
        if self.owner_id == 0: # Player Bullet -> Hits Enemies
            for e in enemies:
                if is_hit(e.x, e.y):
                    e.alive = False # Mark for removal
                    hit_unit = True
                    return True, False # Hit, Not Game Over (unless last enemy?)
                    
        else: # Enemy Bullet -> Hits Player
             if is_hit(player.x, player.y):
                 return True, True # Hit, Game Over
                 
        return False, False
