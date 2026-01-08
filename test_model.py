import time
import os
import glob
import numpy as np
from stable_baselines3 import PPO
from battle_city_env import BattleCityEnv
import config
# from gymnasium.wrappers import FrameStack # Not needed

def find_best_model():
    # 1. Search for numbered checkpoints
    files = glob.glob(f"{config.MODEL_DIR}/battle_city_ppo_*.zip")
    if files:
        # Sort by step count
        try:
            files.sort(key=lambda x: int(x.split('_')[-2])) # ...ppo_123_steps.zip
            return files[-1]
        except:
            pass
    
    # 2. Check for manual saves
    if os.path.exists(f"{config.MODEL_DIR}/battle_city_final.zip"):
        return f"{config.MODEL_DIR}/battle_city_final.zip"
    
    # 3. Last resort
    if os.path.exists(f"{config.MODEL_DIR}/battle_city_interrupted.zip"):
        return f"{config.MODEL_DIR}/battle_city_interrupted.zip"
        
    return None

def main():
    print("--- 📺 BATTLE CITY AI: SPECTATOR MODE 📺 ---")
    
    # 1. Init Environment with VISUALS
    # render_mode='human' forces nes-py to open a window
    env = BattleCityEnv(render_mode='human') 
    
    # 2. Load Model
    model_path = find_best_model()
    if not model_path:
        print("❌ No model found in models/ directory!")
        return

    print(f"🔥 Loading Brain: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("🚀 Starting Game! (Press Ctrl+C to stop)")
    
    # 3. Game Loop
    obs, info = env.reset()
    try:
        while True:
            # AI Decision
            action, _states = model.predict(obs, deterministic=False) # True = Less wobbly, False = More creative
            
            # Execute
            # model.predict returns a numpy array (e.g. array(3)), but nes_py wants int
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            # Render is automatic in 'human' mode for nes-py
            # But we limit FPS to 60 to make it watchable
            env.render()
            time.sleep(1/1000.0) # 1000 FPS Limit (Very Fast) 
            
            if terminated or truncated:
                print(f"🏁 Game Over. Score: {info.get('score', 0)}")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n👋 Exiting Spectator Mode")
    finally:
        env.close()

if __name__ == "__main__":
    main()
