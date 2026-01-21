import gymnasium as gym
import pygame
import numpy as np
import sys
import os
import pandas as pd
from collections import deque

# Ensure we can import battle_city_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from battle_city_env import BattleCityEnv

def main():
    print("Initializing Battle City Visualizer...")
    
    pygame.init()
    pygame.font.init()
    font_mono = pygame.font.SysFont("Courier New", 14, bold=True)
    font_ui = pygame.font.SysFont("Arial", 16, bold=True)
    
    # Init Env
    # DEBUG: Force enemy_count=2 to verify logic
    env = BattleCityEnv(render_mode='rgb_array', use_vision=False, enemy_count=2)
    obs, info = env.reset()
    
    frame = env.raw_env.screen.copy()
    h, w, c = frame.shape
    
    SCALE = 3
    SIDE_PANEL = 500
    screen = pygame.display.set_mode((w * SCALE + SIDE_PANEL, h * SCALE))
    pygame.display.set_caption("Battle City - Clean Tactical View")
    
    clock = pygame.time.Clock()
    running = True

    msg_log = deque(maxlen=10)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_k: 
                    print("CHEAT: Clearing Enemies!")
                    env.cheat_clear_enemies()
                elif event.key == pygame.K_j:
                    # No longer exists, reused for something else?
                    pass
        
        # --- EXPERIMENTAL: FORCE 2 ENEMIES LOGIC ---
        # 1. Stop spawning new enemies from reserve
        # 0x80 = Enemies Lines Count (Values like 20, 19, 18...)
        # If we set it to 0, the game stops generating NEW ones after current ones die.
        # But we want to KEEP getting them, just max 2 on screen? 
        # Actually, let's just keep clamping the on-screen slots.
        
        # 2. Teleport Excess Enemies (Slots 3, 4, 5, 6)
        TARGET_COUNT = 2
        ram = env.raw_env.ram
        
        # Clamp reserve so we don't play forever? 
        # Let's try keeping reserve high (so they keep spawning) but only allow 2 slots.
        # If we teleport slots 3 & 4 to 0,0, they might get "stuck" occupying the slot?
        # Let's test.
        
        for i in range(TARGET_COUNT + 1, 7): # Slots 3 to 6
             if 0x90 + i < 0x100:
                 env.raw_env.ram[0x90 + i] = 0 # X
                 env.raw_env.ram[0x98 + i] = 0 # Y
                 # Also maybe stun them?
                 
        # -------------------------------------------
        
        # Input
        keys = pygame.key.get_pressed()
        action = 0
        
        # Menu
        raw_action = 0
        if keys[pygame.K_RETURN]: raw_action |= 0x08
        if keys[pygame.K_TAB]:    raw_action |= 0x04
        
        if raw_action > 0:
            obs, reward, terminated, truncated, info = env.raw_env.step(raw_action)
        else:
            up, down = keys[pygame.K_UP], keys[pygame.K_DOWN]
            left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
            fire = keys[pygame.K_z]
            
            if up:
                if fire: action = 6
                else:    action = 1
            elif down:
                if fire: action = 7
                else:    action = 2
            elif left:
                if fire: action = 8
                else:    action = 3
            elif right:
                if fire: action = 9
                else:    action = 4
            elif fire:
                action = 5
                
            obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            msg_log.append(f"EPISODE END (Score: {info.get('score', 0):.1f})")
            env.reset()

        # Render Game
        frame = env.raw_env.screen
        surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
        screen.blit(surf, (0, 0))
        
        # Render Side Panel Background
        pygame.draw.rect(screen, (30, 30, 30), (w*SCALE, 0, SIDE_PANEL, h*SCALE))
        x_start = w*SCALE + 20
        y_pos = 20
        
        # --- TACTICAL MAP ---
        screen.blit(font_ui.render("TACTICAL MAP (26x26)", True, (255, 255, 255)), (x_start, y_pos))
        y_pos += 30
        
        tactical_rgb = env.get_tactical_rgb()
        cell_size = 12
        map_surf = pygame.surfarray.make_surface(tactical_rgb.swapaxes(0,1))
        map_surf = pygame.transform.scale(map_surf, (26 * cell_size, 26 * cell_size))
        
        screen.blit(map_surf, (x_start, y_pos))
        
        # Grid Lines
        for i in range(27):
            pygame.draw.line(screen, (50, 50, 50),
                             (x_start, y_pos + i * cell_size),
                             (x_start + 26 * cell_size, y_pos + i * cell_size))
            pygame.draw.line(screen, (50, 50, 50),
                             (x_start + i * cell_size, y_pos),
                             (x_start + i * cell_size, y_pos + 26 * cell_size))

        y_pos += 26 * cell_size + 20
        
        # --- RAM INSPECTOR ---
        ram = env.raw_env.ram
        
        screen.blit(font_ui.render("RAM INSPECTOR:", True, (255, 255, 0)), (x_start, y_pos))
        y_pos += 25
        
        # DEBUG: Enemies Control
        enemies_left = ram[0x80]
        enemies_on_screen = ram[0xA0]
        screen.blit(font_mono.render(f"ENEMIES LEFT (0x80): {enemies_left}", True, (255, 100, 255)), (x_start, y_pos))
        y_pos += 20
        screen.blit(font_mono.render(f"ON SCREEN (0xA0):    {enemies_on_screen}", True, (255, 100, 255)), (x_start, y_pos))
        y_pos += 20

        # 1. Player
        px, py = ram[0x90], ram[0x98]
        p_dir = ram[0x99] # Direction
        screen.blit(font_mono.render(f"PLAYER: XY({px:03d},{py:03d}) DIR({p_dir})", True, (200, 255, 200)), (x_start, y_pos))
        y_pos += 20
        
        # 2. Base
        # 2. Base
        base_status = ram[0x68]
        latch = getattr(env, 'base_active_latch', False)
        
        if latch:
             status_txt = f"LATCHED (ACTIVE)"
             status_col = (255, 255, 0)
        else:
             status_txt = "WAITING..."
             status_col = (100, 100, 100)
             
        if base_status == 0 and latch:
             base_txt = f"BASE: DESTROYED (0x{base_status:02X})"
             base_col = (255, 0, 0)
        elif base_status != 0:
             base_txt = f"BASE: ALIVE (0x{base_status:02X})"
             base_col = (0, 255, 0)
        else:
             base_txt = f"BASE: INIT (0x{base_status:02X})"
             base_col = (100, 100, 255)

        screen.blit(font_mono.render(base_txt, True, base_col), (x_start, y_pos))
        y_pos += 15
        screen.blit(font_mono.render(status_txt, True, status_col), (x_start, y_pos))
        y_pos += 20
        
        # 3. Enemies
        screen.blit(font_mono.render("ENEMIES (HP | X, Y):", True, (200, 200, 200)), (x_start, y_pos))
        y_pos += 20
        
        active_enemies = 0
        for i in range(1, 5):
            hp = ram[0x60 + i]
            ex, ey = ram[0x90 + i], ram[0x98 + i]
            
            if hp > 0:
                active_enemies += 1
                txt = f"#{i}: HP={hp} | {ex:03d}, {ey:03d}"
                col = (255, 100, 100)
            else:
                txt = f"#{i}: DEAD"
                col = (80, 80, 80)
                
            screen.blit(font_mono.render(txt, True, col), (x_start, y_pos))
            y_pos += 18
            
        y_pos += 10
        
        # --- STATS ---
        lives = ram[0x51]
        kills = sum([ram[0x73+i] for i in range(4)])
        stage = ram[0x85]
        
        screen.blit(font_ui.render(f"LIVES: {lives} | KILLS: {kills} | STAGE: {stage}", True, (255, 255, 255)), (x_start, y_pos))
        y_pos += 30
        
        # --- LOG ---
        for msg in list(msg_log)[-5:]:
            screen.blit(font_mono.render(msg, True, (150, 150, 150)), (x_start, y_pos))
            y_pos += 20

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()