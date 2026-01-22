import pygame
import numpy as np
import os
import json

# Constants
GRID_SIZE = 52
CELL_SIZE = 16 # Pixels per cell (Reduced for 52x52)
SIDE_PANEL = 200
WIDTH = GRID_SIZE * CELL_SIZE + SIDE_PANEL
HEIGHT = GRID_SIZE * CELL_SIZE

# Pattern Editor Constants
PATTERN_GRID_SIZE = 7 # 7x7 grid for editing pattern
PATTERN_CELL_SIZE = 40 # Larger cells for pattern editor
PATTERN_FILE = "destruction_pattern.json"
CELL_SIZE = 16 # Pixels per cell (Reduced for 52x52)
SIDE_PANEL = 200
WIDTH = GRID_SIZE * CELL_SIZE + SIDE_PANEL
HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_TEXT = (255, 255, 255)

# Tile Types & Colors
# Tile Types & Colors (Synced with battle_city_env.py)
TILES = {
    0:  {"name": "Empty", "color": (0, 0, 0)},
    200: {"name": "Brick", "color": (160, 60, 0)},
    255: {"name": "Steel", "color": (180, 180, 180)},
    254: {"name": "Eagle", "color": (255, 0, 255)}, 
    # 100/125 were unused in env, but keeping them just in case, though they won't render in env unless added there
    100: {"name": "Water", "color": (0, 0, 255)}, 
    125: {"name": "Forest", "color": (0, 100, 0)},
}

# Load or Create Map
MAP_FILE = "level1_map.npy"

def load_map():
    if os.path.exists(MAP_FILE):
        try:
            arr = np.load(MAP_FILE)
            if arr.shape == (26, 26):
                print("Upscaling 26x26 map to 52x52...")
                arr = np.kron(arr, np.ones((2,2), dtype=np.uint8))
            return arr
        except:
            print("Error loading map, creating new.")
    return np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

def save_map(grid):
    np.save(MAP_FILE, grid)
    print(f"Map saved to {MAP_FILE}")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Battle City Map Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    grid = load_map()
    
    # Editor State
    selected_tile = 200 # Default to Brick
    tool_mode = "PAINT" # PAINT or DESTROY
    blast_radius = 2
    
    running = True
    mouse_down = False
    
    # --- PATTERN EDITOR STATE ---
    pattern_mode = False 
    current_direction = "UP"
    # Structure: {"UP": [], "DOWN": [], "LEFT": [], "RIGHT": []}
    pattern_data = {
        "UP": [(0,0)], "DOWN": [(0,0)], "LEFT": [(0,0)], "RIGHT": [(0,0)]
    }
    
    # Load initial pattern
    if os.path.exists(PATTERN_FILE):
        try:
            with open(PATTERN_FILE, 'r') as f:
                loaded = json.load(f)
                # Migration check: if old format (list), convert to dict
                if "offsets" in loaded:
                    defaults = [tuple(o) for o in loaded["offsets"]]
                    pattern_data = {k: list(defaults) for k in ["UP", "DOWN", "LEFT", "RIGHT"]}
                else:
                    patterns = loaded.get("patterns", {})
                    for k in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        pattern_data[k] = [tuple(o) for o in patterns.get(k, [])]
        except:
             pass

    def save_pattern():
        try:
            # Save all directions
            output = {
                "active": True,
                "patterns": {k: list(v) for k, v in pattern_data.items()}
            }
            with open(PATTERN_FILE, 'w') as f:
                json.dump(output, f)
            print("Directional Patterns saved!")
        except Exception as e:
            print(f"Error saving pattern: {e}")

    # UI Buttons for Main Editor
    buttons = []
    y_off = 50
    for tile_id, info in TILES.items():
        rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, y_off, 30, 30)
        buttons.append({"id": tile_id, "rect": rect, "info": info})
        y_off += 50
        
    # Extra UI Controls
    # Mode Toggle (Paint/Destroy)
    btn_mode_rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, y_off + 20, 160, 30)
    
    # Radius Controls
    btn_rad_minus = pygame.Rect(WIDTH - SIDE_PANEL + 20, y_off + 70, 30, 30)
    btn_rad_plus = pygame.Rect(WIDTH - SIDE_PANEL + 100, y_off + 70, 30, 30)
    
    # PATTERN EDITOR TOGGLE
    btn_pattern_edit_rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, y_off + 120, 160, 40)
    
    # DIRECTION BUTTONS (Pattern Mode)
    # Centered bottom of pattern view roughly
    btn_dir_up = pygame.Rect(WIDTH//2 - 20, HEIGHT//2 + 160, 40, 30)
    btn_dir_left = pygame.Rect(WIDTH//2 - 70, HEIGHT//2 + 200, 40, 30)
    btn_dir_down = pygame.Rect(WIDTH//2 - 20, HEIGHT//2 + 200, 40, 30)
    btn_dir_right = pygame.Rect(WIDTH//2 + 30, HEIGHT//2 + 200, 40, 30)
    
    save_btn_rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, HEIGHT - 60, 160, 40)

    try:
        while running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left Click
                        mouse_down = True
                        mx, my = event.pos
                        
                        if pattern_mode:
                            # --- PATTERN EDITOR LOGIC ---
                            # Center grid
                            cx, cy = WIDTH // 2, HEIGHT // 2
                            grid_w = PATTERN_GRID_SIZE * PATTERN_CELL_SIZE
                            grid_h = PATTERN_GRID_SIZE * PATTERN_CELL_SIZE
                            start_x = cx - grid_w // 2
                            start_y = cy - grid_h // 2
                            
                            if start_x <= mx < start_x + grid_w and start_y <= my < start_y + grid_h:
                                # Clicked on Pattern Grid
                                pc = (mx - start_x) // PATTERN_CELL_SIZE
                                pr = (my - start_y) // PATTERN_CELL_SIZE
                                
                                center_idx = PATTERN_GRID_SIZE // 2
                                offset = (pc - center_idx, pr - center_idx)
                                
                                # Edit ONLY current direction list
                                current_list = pattern_data[current_direction]
                                if offset in current_list:
                                    current_list.remove(offset)
                                else:
                                    current_list.append(offset)
                                    
                            # Save & Exit
                            if btn_pattern_edit_rect.collidepoint(mx, my):
                                save_pattern()
                                pattern_mode = False 
                                
                            # Direction Switching
                            if btn_dir_up.collidepoint(mx, my): current_direction = "UP"
                            if btn_dir_down.collidepoint(mx, my): current_direction = "DOWN"
                            if btn_dir_left.collidepoint(mx, my): current_direction = "LEFT"
                            if btn_dir_right.collidepoint(mx, my): current_direction = "RIGHT"
                                
                        else:
                            # --- MAIN EDITOR LOGIC ---
                            ui_clicked = False
                            
                            # Tile Palette
                            for btn in buttons:
                                if btn["rect"].collidepoint(mx, my):
                                    selected_tile = btn["id"]
                                    tool_mode = "PAINT"
                                    ui_clicked = True
                                    break
                            
                            # Mode Toggle
                            if btn_mode_rect.collidepoint(mx, my):
                                if tool_mode == "PAINT": tool_mode = "DESTROY"
                                else: tool_mode = "PAINT"
                                ui_clicked = True
                                
                            # Radius
                            if btn_rad_minus.collidepoint(mx, my):
                                blast_radius = max(0, blast_radius - 1)
                                ui_clicked = True
                            if btn_rad_plus.collidepoint(mx, my):
                                blast_radius = min(10, blast_radius + 1)
                                ui_clicked = True
                            
                            # Pattern Editor Toggle
                            if btn_pattern_edit_rect.collidepoint(mx, my):
                                pattern_mode = True
                                ui_clicked = True

                            if save_btn_rect.collidepoint(mx, my):
                                save_map(grid)
                                ui_clicked = True
                                
                            # Check Grid Click / Paint
                            if not ui_clicked and mx < GRID_SIZE * CELL_SIZE:
                                c = mx // CELL_SIZE
                                r = my // CELL_SIZE
                                if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                                    if tool_mode == "PAINT":
                                        grid[r, c] = selected_tile
                                    elif tool_mode == "DESTROY":
                                        # Use Pattern for Destruction if simpler "Hole Size" isn't preferred
                                        # But user asked to "mark where bullet hits". 
                                        # Let's keep the Radius logic for now as a "Simple Tool"
                                        # The "Pattern" is for the GAME ENGINE logic mostly, but let's allow painting it here too?
                                        # Stick to Radius logic for Main Editor as per previous request.
                                        
                                        for dr in range(-blast_radius, blast_radius + 1):
                                            for dc in range(-blast_radius, blast_radius + 1):
                                                nr, nc = r + dr, c + dc
                                                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                                    grid[nr, nc] = 0

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: mouse_down = False
                    
                elif event.type == pygame.MOUSEMOTION:
                     if mouse_down and not pattern_mode:
                        mx, my = event.pos
                        if mx < GRID_SIZE * CELL_SIZE:
                            c = mx // CELL_SIZE
                            r = my // CELL_SIZE
                            if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                                if tool_mode == "PAINT":
                                    grid[r, c] = selected_tile
                                elif tool_mode == "DESTROY":
                                    for dr in range(-blast_radius, blast_radius + 1):
                                        for dc in range(-blast_radius, blast_radius + 1):
                                            nr, nc = r + dr, c + dc
                                            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                                                grid[nr, nc] = 0

                elif event.type == pygame.KEYDOWN:
                    if not pattern_mode:
                        if event.key == pygame.K_s:
                            save_map(grid)
                        elif event.key == pygame.K_e: selected_tile = 254; tool_mode = "PAINT"
                        elif event.key == pygame.K_b: selected_tile = 200; tool_mode = "PAINT"
                        elif event.key == pygame.K_x: selected_tile = 255; tool_mode = "PAINT"
                        elif event.key == pygame.K_SPACE: selected_tile = 0; tool_mode = "PAINT"
                        elif event.key == pygame.K_d: tool_mode = "DESTROY"

            # Drawing
            screen.fill(COLOR_BG)
            
            if pattern_mode:
                # --- DRAW PATTERN EDITOR ---
                cx, cy = WIDTH // 2, HEIGHT // 2
                grid_w = PATTERN_GRID_SIZE * PATTERN_CELL_SIZE
                grid_h = PATTERN_GRID_SIZE * PATTERN_CELL_SIZE
                start_x = cx - grid_w // 2
                start_y = cy - grid_h // 2
                
                # Draw Backdrop
                overlay = pygame.Surface((WIDTH, HEIGHT))
                overlay.set_alpha(200)
                overlay.fill((0,0,0))
                screen.blit(overlay, (0,0))
                
                # Draw Grid
                center_idx = PATTERN_GRID_SIZE // 2
                
                # Get current direction pattern
                current_pattern = pattern_data.get(current_direction, [])
                
                for pr in range(PATTERN_GRID_SIZE):
                    for pc in range(PATTERN_GRID_SIZE):
                        rect = (start_x + pc * PATTERN_CELL_SIZE, start_y + pr * PATTERN_CELL_SIZE, PATTERN_CELL_SIZE, PATTERN_CELL_SIZE)
                        
                        offset = (pc - center_idx, pr - center_idx)
                        
                        # Color
                        if offset == (0,0): color = (0, 0, 255) # Center (Bullet)
                        elif offset in current_pattern: color = (255, 0, 0) # Destruction
                        else: color = (50, 50, 50) # Empty
                        
                        pygame.draw.rect(screen, color, rect)
                        pygame.draw.rect(screen, (200, 200, 200), rect, 1) # Border
                        
                # UI Button to Exit
                pygame.draw.rect(screen, (0, 150, 0), btn_pattern_edit_rect)
                lbl = font.render("SAVE & EXIT", True, COLOR_TEXT)
                screen.blit(lbl, (btn_pattern_edit_rect.x + 30, btn_pattern_edit_rect.y + 10))
                
                # Instruction
                lbl_inst = font.render(f"Direction: {current_direction}", True, (255, 255, 0))
                screen.blit(lbl_inst, (cx - 50, start_y - 40))
                
                # DIRECTION BUTTONS
                # UP
                col = (0, 200, 0) if current_direction == "UP" else (100, 100, 100)
                pygame.draw.rect(screen, col, btn_dir_up)
                screen.blit(font.render("^", True, COLOR_TEXT), (btn_dir_up.x + 15, btn_dir_up.y + 5))
                
                # DOWN
                col = (0, 200, 0) if current_direction == "DOWN" else (100, 100, 100)
                pygame.draw.rect(screen, col, btn_dir_down)
                screen.blit(font.render("v", True, COLOR_TEXT), (btn_dir_down.x + 15, btn_dir_down.y + 5))
                
                # LEFT
                col = (0, 200, 0) if current_direction == "LEFT" else (100, 100, 100)
                pygame.draw.rect(screen, col, btn_dir_left)
                screen.blit(font.render("<", True, COLOR_TEXT), (btn_dir_left.x + 15, btn_dir_left.y + 5))
                
                # RIGHT
                col = (0, 200, 0) if current_direction == "RIGHT" else (100, 100, 100)
                pygame.draw.rect(screen, col, btn_dir_right)
                screen.blit(font.render(">", True, COLOR_TEXT), (btn_dir_right.x + 15, btn_dir_right.y + 5))
                
            else:
                # --- DRAW MAIN EDITOR ---
                # Draw Grid
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        tile_id = grid[r, c]
                        color = TILES.get(tile_id, TILES[0])["color"]
                        
                        rect = (c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, color, rect)
                        pygame.draw.rect(screen, COLOR_GRID, rect, 1) # Border

                # Draw UI
                # Selected Label
                sel_name = TILES[selected_tile]["name"]
                lbl = font.render(f"P: {sel_name}", True, COLOR_TEXT)
                screen.blit(lbl, (WIDTH - SIDE_PANEL + 20, 10))
                
                # Palette Buttons
                for btn in buttons:
                    if btn["id"] == selected_tile and tool_mode == "PAINT":
                        pygame.draw.rect(screen, (255, 255, 0), (btn["rect"].x-2, btn["rect"].y-2, 34, 34), 2)
                    pygame.draw.rect(screen, btn["info"]["color"], btn["rect"])
                    name_lbl = font.render(btn["info"]["name"], True, COLOR_TEXT)
                    screen.blit(name_lbl, (btn["rect"].x + 40, btn["rect"].y + 5))
                    
                # UI: Mode Toggle
                mode_color = (200, 50, 50) if tool_mode == "DESTROY" else (50, 100, 200)
                pygame.draw.rect(screen, mode_color, btn_mode_rect)
                mode_lbl = font.render(f"MODE: {tool_mode}", True, COLOR_TEXT)
                screen.blit(mode_lbl, (btn_mode_rect.x + 20, btn_mode_rect.y + 5))
                
                # UI: Radius controls
                if tool_mode == "DESTROY":
                    pygame.draw.rect(screen, (100, 100, 100), btn_rad_minus)
                    screen.blit(font.render("-", True, COLOR_TEXT), (btn_rad_minus.x + 10, btn_rad_minus.y + 2))
                    
                    pygame.draw.rect(screen, (100, 100, 100), btn_rad_plus)
                    screen.blit(font.render("+", True, COLOR_TEXT), (btn_rad_plus.x + 10, btn_rad_plus.y + 2))
                    
                    # Dynamic Label based on size
                    size_desc = "Single" if blast_radius == 0 else f"{blast_radius*2+1}x{blast_radius*2+1}"
                    rad_lbl = font.render(f"Hole: {size_desc}", True, COLOR_TEXT)
                    screen.blit(rad_lbl, (btn_rad_minus.x + 40, btn_rad_minus.y + 5))
                
                # Pattern Editor Button
                pygame.draw.rect(screen, (100, 100, 200), btn_pattern_edit_rect)
                pat_lbl = font.render("EDIT PATTERN", True, COLOR_TEXT)
                screen.blit(pat_lbl, (btn_pattern_edit_rect.x + 20, btn_pattern_edit_rect.y + 10))
                    
                # Save Button
                pygame.draw.rect(screen, (0, 150, 0), save_btn_rect)
                save_lbl = font.render("SAVE MAP (S)", True, COLOR_TEXT)
                screen.blit(save_lbl, (save_btn_rect.x + 30, save_btn_rect.y + 10))
                
                # Cursor Preview
                mx, my = pygame.mouse.get_pos()
                if mx < GRID_SIZE * CELL_SIZE:
                    if tool_mode == "PAINT":
                        preview_color = TILES[selected_tile]["color"]
                        pygame.draw.circle(screen, preview_color, (mx, my), 5)
                    elif tool_mode == "DESTROY":
                        # Draw blast area
                        c = mx // CELL_SIZE
                        r = my // CELL_SIZE
                        
                        area_size = (blast_radius * 2 + 1) * CELL_SIZE
                        area_x = (c - blast_radius) * CELL_SIZE
                        area_y = (r - blast_radius) * CELL_SIZE
                        
                        surf = pygame.Surface(((blast_radius * 2 + 1) * CELL_SIZE, (blast_radius * 2 + 1) * CELL_SIZE), pygame.SRCALPHA)
                        surf.fill((255, 0, 0, 100)) # Semi-transparent red
                        screen.blit(surf, (area_x, area_y))

            pygame.display.flip()
            clock.tick(60)

    except KeyboardInterrupt:
        print("\nExiting Editor...")
    
    pygame.quit()

if __name__ == "__main__":
    main()