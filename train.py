import gymnasium as gym
import os
import time
import cv2            # Added for rendering
import numpy as np    # Added for rendering
import pickle         # Added for Graph Persistence
from collections import deque # Added for Score History
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from battle_city_env import BattleCityEnv
import torch as th # Added for Architecture Config

# --- CONFIG ---
NUM_CPU = 48 # DOUBLE SPEED (Blind Mode allows massive concurrency)
TOTAL_TIMESTEPS = 5000000 
CHECKPOINT_FREQ = 10000
MODEL_DIR = "models"
LOG_DIR = "logs"

class ConsoleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ConsoleLoggerCallback, self).__init__(verbose)
        self.last_time_steps = 0

    def _on_step(self) -> bool:
        # Print every 1000 steps
        if self.num_timesteps % 1000 == 0:
            mean_rew = "N/A"
            if len(self.model.ep_info_buffer) > 0:
                mean_rew = f"{sum([ep['r'] for ep in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer):.2f}"
            print(f"[{self.num_timesteps} steps] Mean Reward: {mean_rew} (Playing...)", end='\r')
        return True

import traceback

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.windows_initialized = False # Flag for window setup
        self.history_path = f"{MODEL_DIR}/score_history.pkl"
        
        # Load History if exists
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'rb') as f:
                    self.score_history = pickle.load(f)
                print(f"[Graph] Loaded history: {len(self.score_history)} games.")
            except Exception as e:
                print(f"[Graph] Load failed: {e}")
                self.score_history = deque() # Unlimited
        else:
            self.score_history = deque() # Unlimited
        
    def _on_step(self) -> bool:
        try:
            # We are in BLIND MODE. No 'screen' observation.
            # But we can still visualize the SCORE HISTORY GRAPH.
            
            # Detect End of Episode to Record Score
            dones = self.locals.get('dones', [False])
            infos = self.locals.get('infos', [{}])
            
            # Check ALL environments for finished games
            any_finished = False
            for i, done in enumerate(dones):
                if done:
                    final_score = infos[i].get('score', 0)
                    self.score_history.append(final_score)
                    any_finished = True
            
            if any_finished:
                # Save History (If any game finished)
                try:
                    with open(self.history_path, 'wb') as f:
                        pickle.dump(self.score_history, f)
                except Exception as e:
                    pass

            # --- RENDER GRAPH WINDOW ---
            # Update Graph window every step (it's fast)
            
            if not self.windows_initialized:
                 cv2.namedWindow("Score History", cv2.WINDOW_AUTOSIZE)
                 cv2.moveWindow("Score History", 100, 100)
                 self.windows_initialized = True

            # Create black canvas
            g_w, g_h = 600, 300 
            graph_frame = np.zeros((g_h, g_w, 3), dtype=np.uint8)

            if len(self.score_history) > 1:
                scores = list(self.score_history)
                # Auto-scale
                min_s, max_s = min(scores), max(scores)
                if max_s == min_s: max_s += 1 
                
                total_points = len(scores)
                # Downsample for drawing if too long (take last 500 or step)
                draw_step = max(1, total_points // 600)
                display_scores = scores[::draw_step]
                total_display = len(display_scores)

                for i in range(1, total_display):
                    p1 = display_scores[i-1]
                    p2 = display_scores[i]
                    
                    # Normalize X
                    x1 = int((i-1) * (g_w / (total_display - 1)))
                    x2 = int(i * (g_w / (total_display - 1)))
                    
                    # Normalize Y
                    y1 = int((g_h - 20) - ((p1 - min_s) / (max_s - min_s)) * (g_h - 40))
                    y2 = int((g_h - 20) - ((p2 - min_s) / (max_s - min_s)) * (g_h - 40))
                    
                    cv2.line(graph_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # Stats Text
                avg_score = sum(scores) / len(scores)
                cv2.putText(graph_frame, f"Max: {max_s:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(graph_frame, f"Avg: {avg_score:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(graph_frame, f"Last: {scores[-1]:.1f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(graph_frame, f"Games: {total_points}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            else:
                 cv2.putText(graph_frame, "Waiting for games...", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

            cv2.imshow("Score History", graph_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            pass # Don't crash training loop on render error
            
        return True

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"--- BATTLE CITY AI TRAINING (VISUAL MODE) ---")
    
    # Use DummyVecEnv (Single Process Logic)
    # Pass render_mode=None to Envs (Visualization is handled by Main Process Callback)
    # Using SubprocVecEnv for Multicore Speed
    # Windows Note: SubprocVecEnv requires non-lambda functions usually. 
    # We use a list comprehension of factory functions.
    if NUM_CPU > 1:
        env = SubprocVecEnv([lambda: BattleCityEnv(render_mode=None) for _ in range(NUM_CPU)])
    else:
        env = DummyVecEnv([lambda: BattleCityEnv(render_mode=None)])
    
    # Auto-Resume Logic
    latest_model_path = f"{MODEL_DIR}/battle_city_interrupted.zip"
    final_model_path = f"{MODEL_DIR}/battle_city_final.zip"
    
    if os.path.exists(latest_model_path):
        print(f"Loading saved model: {latest_model_path}")
        model = PPO.load(latest_model_path, env=env)
        reset_timesteps = False
    elif os.path.exists(final_model_path):
        print(f"Loading saved model: {final_model_path}")
        model = PPO.load(final_model_path, env=env)
        reset_timesteps = False
    else:
        print("Creating BLIND GOD MODEL (2048x1024) - RAM Only...")
        # One Neuron per Memory Cell Architecture
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[2048, 1024], vf=[2048, 1024])
        )
        
        # Using MlpPolicy for Flat RAM Observation
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.00001, 
            n_steps=512,          
            batch_size=256,       
            ent_coef=0.2,         
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cuda", # Ideally use GPU if installed, otherwise auto-fallbacks
            policy_kwargs=policy_kwargs
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="battle_city_ppo"
    )
    
    logger_callback = ConsoleLoggerCallback()
    render_callback = RenderCallback()

    print("Start VISUAL Learning... (Press Ctrl+C to stop)")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, logger_callback, render_callback],
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
        model.save(f"{MODEL_DIR}/battle_city_final")
        print("Training Finished!")
        
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n[User] Stopped by Keyboard (Ctrl+C).")
        else:
            print(f"\n[CRITICAL] Training Interrupted/Crashed: {e}")
            
        print("Saving EMERGENCY model...")
        model.save(f"{MODEL_DIR}/battle_city_interrupted")
        print("Saved.")
        
    finally:
        print("Closing environment...")
        try:
             env.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):
             # These are normal during forced shutdown of subprocesses
             pass
        except Exception as e:
             print(f"Cleanup Warnings: {e}")

if __name__ == "__main__":
    train()
