import gymnasium as gym
import pygame
import numpy as np
import sys
import os
import pandas as pd
from collections import deque

# Ensure we can import battle_city_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from battle_city_env import BattleCityEnv

def main():
    print("Initializing Battle City Visualizer...")
    
    pygame.init()
    pygame.font.init()
    font_mono = pygame.font.SysFont("Courier New", 14, bold=True)
    font_ui = pygame.font.SysFont("Arial", 16, bold=True)
    
    # Init Env with Modes
    modes = list(config.ENV_VARIANTS.keys())
    # Filter out VIRTUAL for play.py
    modes = [m for m in modes if m != "VIRTUAL"]
    current_mode_idx = 0
    
    def make_env_for_mode(mode_name):
        variant = config.ENV_VARIANTS[mode_name]
        reward_profile = variant.get("reward_profile", "DEFAULT")
        reward_config = config.REWARD_VARIANTS.get(reward_profile, None)
        
        print(f"Switching to Mode: {mode_name}")
        print(f" - Enemies: {variant.get('enemy_count')}")
        print(f" - Profile: {reward_profile}")
        
        return BattleCityEnv(
            render_mode='rgb_array', 
            use_vision=False, 
            enemy_count=variant.get("enemy_count", 20),
            no_shooting=variant.get("no_shooting", False),
            reward_config=reward_config,
            exploration_trigger=variant.get("exploration_trigger", None)
        )

    # Initial Start (Default to PROFILE_EXPLORER if available, else STANDARD)
    start_mode = "PROFILE_EXPLORER" if "PROFILE_EXPLORER" in modes else "STANDARD"
    current_mode_idx = modes.index(start_mode)
    env = make_env_for_mode(start_mode)
    obs, info = env.reset()
    
    frame = env.raw_env.screen.copy()
    h, w, c = frame.shape
    
    SCALE = 3
    SIDE_PANEL = 500
    screen = pygame.display.set_mode((w * SCALE + SIDE_PANEL, h * SCALE))
    pygame.display.set_caption("Battle City - Clean Tactical View")
    
    clock = pygame.time.Clock()
    running = True

    msg_log = deque(maxlen=50) # Increased history
    popups = [] # List of [text, timer, color]
    scroll_offset = 0 # For log scrolling

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL:
                scroll_offset += event.y # Scroll up/down
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_m:
                    # Switch Mode
                    current_mode_idx = (current_mode_idx + 1) % len(modes)
                    new_mode = modes[current_mode_idx]
                    env.close()
                    env = make_env_for_mode(new_mode)
                    obs, info = env.reset()
                    msg_log.append(f"SWITCHED MODE: {new_mode}")
                elif event.key == pygame.K_k: 
                    print("CHEAT: Clearing Enemies!")
                    # env.cheat_clear_enemies() # Not implemented in base env yet
                    pass 
                elif event.key == pygame.K_j:
                    pass
        
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
        
        # --- REWARD POPUPS MANAGER ---
        events = info.get('reward_events', [])
        if events:
            for e in events:
                msg_log.append(f">>> {e} <<<")
                # Add to visual popups: [Text, Timer(frames), Color]
                col = (255, 255, 0) # Default Yellow
                if "DEFENDER" in e: col = (100, 255, 100) # Green for Defense
                if "MILESTONE" in e: col = (255, 100, 255) # Purple for Milestones
                popups.append([e, 120, col]) # 2 Seconds (60fps * 2)

        # Draw Popups (Kill Feed Style - Top Left, Stacking Down)
        # Filter dead ones
        popups = [[txt, t-1, c] for txt, t, c in popups if t > 0]
        
        # Show max 5 recent popups
        visible_popups = popups[-5:] 
        
        for i, (txt, t, c) in enumerate(visible_popups):
            # i=0 is the oldest valid, i=4 is newest
            # Let's stack them downwards: Oldest at top? Or Newest at top?
            # Standard kill feed: Newest at bottom usually, or Newest at top.
            # Let's put Newest at Bottom of the list (which is index -1).
            
            label = font_ui.render(txt, True, c)
            # Text Outline
            outline = font_ui.render(txt, True, (0,0,0))
            
            # Position: Top Left
            p_x = 10
            p_y = 10 + (i * 25) # Stack downwards
            
            # Simple background box for readability
            bg_rect = pygame.Rect(p_x - 2, p_y - 2, label.get_width() + 4, label.get_height() + 4)
            pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect) # Semi-transparent black? Pygame rect doesn't support alpha directly without surface. Just black for now.
            
            screen.blit(label, (p_x, p_y))

        
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
        for i in range(1, 7):
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
        
        screen.blit(font_ui.render(f"LIVES: {lives} | KILLS: {kills} | STAGE: {stage} | MODE: {modes[current_mode_idx]}", True, (255, 255, 255)), (x_start, y_pos))
        y_pos += 30
        
        # --- EXPLORATION PROGRESS ---
        explore_pct = info.get('exploration_pct', 0.0)
        trigger_pct = info.get('trigger_pct', 0.0)
        
        # Draw Bar
        bar_w = 200
        bar_h = 15
        pygame.draw.rect(screen, (50, 50, 50), (x_start, y_pos, bar_w, bar_h))
        pygame.draw.rect(screen, (0, 100, 255), (x_start, y_pos, int(bar_w * explore_pct), bar_h))
        
        # Draw Trigger Marker
        if trigger_pct > 0:
             trig_x = x_start + int(bar_w * trigger_pct)
             pygame.draw.line(screen, (255, 0, 0), (trig_x, y_pos - 5), (trig_x, y_pos + bar_h + 5), 2)
             msg = f"EXPLORE: {explore_pct*100:.1f}% (Trig: {trigger_pct*100:.0f}%)"
        else:
             msg = f"EXPLORE: {explore_pct*100:.1f}%"
             
        screen.blit(font_mono.render(msg, True, (200, 200, 255)), (x_start + bar_w + 10, y_pos))
        y_pos += 30
        
        # --- SCROLLABLE LOG ---
        # Show last 10 messages with scroll offset
        log_list = list(msg_log)
        # Apply scroll
        # Scroll 0 = Show newest at bottom. 
        # Scroll > 0 = Look back in history.
        
        max_visible = 10
        total_logs = len(log_list)
        
        # Clamp scroll
        max_scroll = max(0, total_logs - max_visible)
        if scroll_offset > max_scroll: scroll_offset = max_scroll
        if scroll_offset < 0: scroll_offset = 0
        
        # Determine slice
        # If scroll is 0, we want indices [-10:]
        # If scroll is 1, we want indices [-11:-1]
        start_idx = total_logs - max_visible - scroll_offset
        if start_idx < 0: start_idx = 0
        end_idx = start_idx + max_visible
        
        visible_logs = log_list[start_idx:end_idx]
        
        screen.blit(font_ui.render(f"EVENT LOG ({scroll_offset}/{max_scroll}) [Wheel to Scroll]", True, (200, 200, 0)), (x_start, y_pos))
        y_pos += 20

        for msg in visible_logs:
            screen.blit(font_mono.render(msg, True, (150, 150, 150)), (x_start, y_pos))
            y_pos += 20

        pygame.display.flip()
        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()