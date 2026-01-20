import pygame
import numpy as np
import os

# Constants
GRID_SIZE = 26
CELL_SIZE = 24 # Pixels per cell
SIDE_PANEL = 200
WIDTH = GRID_SIZE * CELL_SIZE + SIDE_PANEL
HEIGHT = GRID_SIZE * CELL_SIZE

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_TEXT = (255, 255, 255)

# Tile Types & Colors
TILES = {
    0:  {"name": "Empty", "color": (0, 0, 0)},
    25: {"name": "Brick", "color": (160, 60, 0)},
    50: {"name": "Steel", "color": (180, 180, 180)},
    75: {"name": "Eagle (Base)", "color": (255, 0, 255)}, # Magenta for visibility
    100: {"name": "Water", "color": (0, 0, 255)}, 
    125: {"name": "Forest", "color": (0, 100, 0)},
}

# Load or Create Map
MAP_FILE = "level1_map.npy"

def load_map():
    if os.path.exists(MAP_FILE):
        try:
            return np.load(MAP_FILE)
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
    selected_tile = 25 # Default to Brick
    running = True
    mouse_down = False
    
    # UI Buttons
    buttons = []
    y_off = 50
    for tile_id, info in TILES.items():
        rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, y_off, 30, 30)
        buttons.append({"id": tile_id, "rect": rect, "info": info})
        y_off += 50
        
    save_btn_rect = pygame.Rect(WIDTH - SIDE_PANEL + 20, HEIGHT - 60, 160, 40)

    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    mouse_down = True
                    mx, my = event.pos
                    
                    # Check UI Click
                    ui_clicked = False
                    for btn in buttons:
                        if btn["rect"].collidepoint(mx, my):
                            selected_tile = btn["id"]
                            ui_clicked = True
                            break
                    
                    if save_btn_rect.collidepoint(mx, my):
                        save_map(grid)
                        ui_clicked = True
                        
                    # Check Grid Click
                    if not ui_clicked and mx < GRID_SIZE * CELL_SIZE:
                        c = mx // CELL_SIZE
                        r = my // CELL_SIZE
                        if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                            grid[r, c] = selected_tile
                            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: mouse_down = False
                
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down:
                    mx, my = event.pos
                    if mx < GRID_SIZE * CELL_SIZE:
                        c = mx // CELL_SIZE
                        r = my // CELL_SIZE
                        if 0 <= c < GRID_SIZE and 0 <= r < GRID_SIZE:
                            grid[r, c] = selected_tile
                            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_map(grid)
                elif event.key == pygame.K_e: # Hotkey for Eagle
                    selected_tile = 75
                elif event.key == pygame.K_b: # Hotkey for Brick
                    selected_tile = 25
                elif event.key == pygame.K_x: # Hotkey for Steel
                    selected_tile = 50
                elif event.key == pygame.K_SPACE: # Hotkey for Eraser
                    selected_tile = 0

        # Drawing
        screen.fill(COLOR_BG)
        
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
        lbl = font.render(f"Selected: {sel_name}", True, COLOR_TEXT)
        screen.blit(lbl, (WIDTH - SIDE_PANEL + 20, 10))
        
        # Palette Buttons
        for btn in buttons:
            # Highlight selected
            if btn["id"] == selected_tile:
                pygame.draw.rect(screen, (255, 255, 0), (btn["rect"].x-2, btn["rect"].y-2, 34, 34), 2)
                
            pygame.draw.rect(screen, btn["info"]["color"], btn["rect"])
            
            # Label
            name_lbl = font.render(btn["info"]["name"], True, COLOR_TEXT)
            screen.blit(name_lbl, (btn["rect"].x + 40, btn["rect"].y + 5))
            
        # Save Button
        pygame.draw.rect(screen, (0, 150, 0), save_btn_rect)
        save_lbl = font.render("SAVE MAP (S)", True, COLOR_TEXT)
        screen.blit(save_lbl, (save_btn_rect.x + 30, save_btn_rect.y + 10))
        
        # Cursor Preview
        mx, my = pygame.mouse.get_pos()
        if mx < GRID_SIZE * CELL_SIZE:
            preview_color = TILES[selected_tile]["color"]
            pygame.draw.circle(screen, preview_color, (mx, my), 5)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()