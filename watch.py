import gym
import time
import os
import glob
from stable_baselines3 import PPO
from battle_city_env import BattleCityEnv

MODEL_DIR = "models/"

def get_latest_model():
    if not os.path.exists(MODEL_DIR):
        return None
    list_of_files = glob.glob(f'{MODEL_DIR}/*.zip') 
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def watch():
    print("--- BATTLE CITY SPECTATOR ---")
    print("Searching for brains in models/...")
    
    # Wait for first model
    while True:
        model_path = get_latest_model()
        if model_path:
            break
        print("Waiting for training to produce first checkpoint...", end='\r')
        time.sleep(2)
        
    current_model_path = model_path
    print(f"\nLoaded Initial Brain: {current_model_path}")
    
    # Load Model
    model = PPO.load(current_model_path)
    
    # CRITICAL: Match Training Wrappers (None for DummyVecEnv)
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_render_env():
        e = BattleCityEnv()
        e.render_mode = 'human' # FORCE SB3 to allow rendering
        e.metadata = {'render_modes': ['human'], 'render_fps': 60}
        return e
    
    env = DummyVecEnv([make_render_env]) 
    
    # Loop Games
    while True:
        # Check for newer model between games
        latest = get_latest_model()
        if latest and latest != current_model_path:
            print(f"Update found! Loading new brain: {latest}")
            current_model_path = latest
            model = PPO.load(current_model_path)
            
        print(f"Playing Episode with {os.path.basename(current_model_path)}")
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            try:
                # Force Render Logic...
                if hasattr(env.envs[0], 'env'):
                     if hasattr(env.envs[0].env, 'render'):
                         env.envs[0].env.render()
                     else:
                        env.envs[0].unwrapped.render() 
                else:
                    env.render()
            except Exception:
                pass
                
            # Print Score
            print(f"Score: {total_reward:.2f}", end='\r')
                
            time.sleep(0.016) 
            
        print(f"\nGame Finished. Final Score: {total_reward:.2f}")
        time.sleep(1)

if __name__ == "__main__":
    watch()
