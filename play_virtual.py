
import pygame
import numpy as np
from virtual_env import VirtualBattleCityEnv

def main():
    env = VirtualBattleCityEnv(render_mode='human')
    obs, info = env.reset()
    
    pygame.init()
    # 52 * 10 = 520 size window
    SCALE = 10
    screen = pygame.display.set_mode((52 * SCALE, 52 * SCALE))
    pygame.display.set_caption("Virtual Battle City - Test Mode")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = 0 # NOOP
        
        # Input Handling
        keys = pygame.key.get_pressed()
        
        move_action = 0
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        fire = keys[pygame.K_SPACE]
        
        if move_action > 0:
            if fire: action = move_action + 5 # 6,7,8,9
            else: action = move_action
        else:
            if fire: action = 5
            else: action = 0
        
        # Check explicit Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Step Env
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        # Use _get_frame() to get the full visual state (Map + Player + Enemies + Bullets)
        grid = env._get_frame()
        
        # Draw
        screen.fill((0,0,0))
        
        # Colors (approximate)
        # 0=Black, 200=Brick(Brown), 255=Steel(Gray)/Bullet, 254=Eagle(Magenta), 150=Player(Yellow), 80=Enemy(Red)
        
        # Fast pixel array access or simple rects? 52x52 is small enough for rects
        for r in range(52):
            for c in range(52):
                val = grid[r, c]
                if val == 0: continue
                
                color = (255, 255, 255)
                if val == 200: color = (160, 60, 0)
                elif val == 255: color = (180, 180, 180)
                elif val == 254: color = (255, 0, 255)
                elif val == 150: color = (255, 200, 0)
                elif val == 80: color = (200, 0, 0)
                
                pygame.draw.rect(screen, color, (c*SCALE, r*SCALE, SCALE, SCALE))
                
        pygame.display.flip()
        
        if terminated or truncated:
            obs, info = env.reset()
            
        clock.tick(30) # 30 FPS
        
    pygame.quit()

if __name__ == "__main__":
    main()
