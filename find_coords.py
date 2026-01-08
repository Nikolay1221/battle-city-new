import time
import numpy as np
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace

def scan_for_coords():
    print("INITIALIZING RAM SCANNER (IMPROVED)...")
    # Setup Env
    rom_path = 'BattleCity_fixed.nes'
    raw_env = NESEnv(rom_path)
    actions = [['NOOP'], ['right'], ['left'], ['down'], ['up']]
    env = JoypadSpace(raw_env, actions)
    env.reset()

    print("\n>>> STARTING GAME (Waiting 10 seconds to ensure gameplay)...")
    # Optimized start sequence
    for _ in range(60): env.step(0)
    env.env.step(8) # Press Start
    for _ in range(60): env.step(0)
    env.env.step(8) # Press Start (Select Players)
    for _ in range(60): env.step(0)
    env.env.step(8) # Press Start (Start Stage)
    
    # Wait for curtain to rise and game to start (~3 seconds)
    for _ in range(180): env.step(0) 

    print(">>> GAME READY. SCANNING...")

    # Helper to get changing addresses
    def get_changing_addrs(action_idx, steps=120):
        prev_ram = env.env.ram.copy()
        counts = {}
        ram_history = []
        
        for _ in range(steps):
            env.step(action_idx)
            curr_ram = env.env.ram
            diff = curr_ram != prev_ram # Boolean array
            indices = np.where(diff)[0]
            
            for idx in indices:
                counts[idx] = counts.get(idx, 0) + 1
            
            prev_ram = curr_ram.copy()
            ram_history.append(curr_ram.copy())
            
        return counts, ram_history

    # 1. Scan Right
    print("\n[PHASE 1] Moving RIGHT...")
    x_counts, x_hist = get_changing_addrs(1, steps=100)
    
    # 2. Scan Down
    print("\n[PHASE 2] Moving DOWN...")
    y_counts, y_hist = get_changing_addrs(3, steps=100)
    
    # 3. Filter
    # True X should change on Right, but NOT on Down (or much less)
    # True Y should change on Down, but NOT on Right
    
    possible_x = []
    for addr, count in x_counts.items():
        if count > 5: # Changed at least 5 times
            # If it also changed significantly during Y move, it's a timer
            y_count = y_counts.get(addr, 0)
            if y_count < 10: # Allow small changes (noise) but not consistent
                 possible_x.append(addr)

    possible_y = []
    for addr, count in y_counts.items():
        if count > 5:
            x_count = x_counts.get(addr, 0)
            if x_count < 10:
                 possible_y.append(addr)

    print("\n" + "="*40)
    print("RESULTS (Likely Coordinates)")
    print("="*40)
    
    print("\nPOSSIBLE X ADDRESSES:")
    for addr in possible_x:
        val_start = x_hist[0][addr]
        val_end = x_hist[-1][addr]
        print(f"Addr 0x{addr:02X} | Changes: {x_counts[addr]} | Value: {val_start} -> {val_end}")

    print("\nPOSSIBLE Y ADDRESSES:")
    for addr in possible_y:
        val_start = y_hist[0][addr]
        val_end = y_hist[-1][addr]
        print(f"Addr 0x{addr:02X} | Changes: {y_counts[addr]} | Value: {val_start} -> {val_end}")
        
    env.close()

if __name__ == "__main__":
    scan_for_coords()
