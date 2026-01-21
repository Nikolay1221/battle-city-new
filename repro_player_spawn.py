import sys
import os
import numpy as np

sys.path.append(os.getcwd())
try:
    from simulation_env import SimulationBattleCityEnv
    from simulation_objects import Enemy
except ImportError:
    print("Error importing simulation_env")
    sys.exit(1)

def test_player_spawn_explicit():
    print("Testing Player Spawn Blocking (OOP)...")
    env = SimulationBattleCityEnv()
    env.reset()
    
    # 1. Spawn Trigger Setup
    env.total_enemies_spawned = 1
    # Next Spawn: Index 1 -> 24 (Center)
    spawn_x = 24
    
    # 2. Place Player EXACTLY at Spawn Point (24, 2)
    env.player.x = 24.0
    env.player.y = 2.0
    env.enemies = []
    
    env.spawn_timer = 61
    
    env.step(0)
    
    print(f"Enemies count: {len(env.enemies)}")
    
    if len(env.enemies) == 0:
        print("SUCCESS: Player BLOCKED the spawn.")
    else:
        print("FAILURE: Enemy spawned ON TOP of Player!")
        e = env.enemies[0]
        print(f"Enemy at {e.x}, {e.y}. Player at {env.player.x}, {env.player.y}")

if __name__ == "__main__":
    test_player_spawn_explicit()
