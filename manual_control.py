import pygame
import sys
import numpy as np
import cv2
from battle_city_env import BattleCityEnv

def main():
    print("Initializing Environment...")
    env = BattleCityEnv()
    obs = env.reset()

    pygame.init()
    SCALE = 6
    W, H = 84, 84
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption("Battle City Manual AI Test")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    print("Controls: Arrow Keys to Move, Z to Fire, ESC to Quit")
    print("====================================================")
    
    frame_count = 0
    while running:
        # FPS Limit: 15 FPS (since env skips 4 frames per step, 60/4 = 15 updates/sec)
        clock.tick(15)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN: # Manual Start/Pause
                    print(">>> MANUAL START/PAUSE PRESSED (Single)")
                    env.raw_env.step(8) # Send Start directly

        # Input Mapping
        keys = pygame.key.get_pressed()
        action = 0
        
        is_fire = keys[pygame.K_z]
        
        if keys[pygame.K_UP]:
            action = 6 if is_fire else 1
        elif keys[pygame.K_DOWN]:
            action = 7 if is_fire else 2
        elif keys[pygame.K_LEFT]:
            action = 8 if is_fire else 3
        elif keys[pygame.K_RIGHT]:
            action = 9 if is_fire else 4
        elif is_fire:
            action = 5
        else:
            action = 0

        prev_state = env.raw_env.ram[0x92]
        # frame_count = 0 logic removed

        obs, reward, done, info = env.step(action)
        total_reward += reward
        frame_count += 1

        # DEBUG: Monitor Game State (0x92) logic removed for clean output
        # curr_state = env.raw_env.ram[0x92]

        # Print Reward logic (Filter out small time penalty ~ -0.0112)
        # Print Reward logic (Filter out small time penalty ~ -0.0112)
        if abs(reward) > 0.02:
             print(f"[REWARD] Step: {reward:+.2f} | Total: {total_reward:+.2f}")
             
             if reward <= -5000:
                 print(" >>> GAME OVER SCREEN DETECTED")
             elif reward <= -1000: 
                 print(" >>> DEATH / LIFE LOST")
             elif reward >= 2000:
                 print(" >>> STAGE COMPLETED (LEVEL UP!)")
             elif abs(reward + 0.1) < 0.001: # Check for -0.1 (Idle)
                 # print(" >>> IDLE PENALTY")
                 # Show Coords instead as requested
                 if 'x' in info and 'y' in info:
                      print(f" >>> IDLE! Pos: ({info['x']}, {info['y']})")
                 else:
                      print(f" >>> IDLE PENALTY")
             elif reward == 500:
                 print(" >>> BONUS ITEM COLLECTED")
             elif reward >= 100:
                 print(" >>> ENEMY KILLED")

        # DEBUG: Periodic State Check (every ~2 sec) to debug "Idle Penalty Stops"
        if frame_count % 60 == 0:
             if 'ram_state' in info:
                  x_info = info.get('x', '?')
                  y_info = info.get('y', '?')
                  print(f"DEBUG: State={info['ram_state']:02X} | Pos=({x_info},{y_info}) | IdleSteps={info.get('idle_steps', 0)}")
        
        # Render
        # obs is (84, 84, 4), stack of 4 frames. We view the latest (last channel).
        frame = obs[:, :, -1] 
        
        # Convert to RGB for visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Resize to make it visible
        frame_big = cv2.resize(frame_rgb, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
        
        # Pygame expects (Width, Height, Channels) where Width is X (columns) and Height is Y (rows).
        # Valid CV2 image is (Rows, Cols, Channels) i.e. (Y, X, C).
        # We need to transpose to (X, Y, C) for make_surface.
        frame_big = np.transpose(frame_big, (1, 0, 2))

        surf = pygame.surfarray.make_surface(frame_big)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if done:
            print(f"!!! GAME ENDED !!! Final Score: {total_reward}")
            print("Resetting...")
            obs = env.reset()
            total_reward = 0
            
    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
