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
from stable_baselines3.common.env_util import make_vec_env # Needed for dynamic kwargs
from battle_city_env import BattleCityEnv
import torch as th # Added for Architecture Config

import torch as th # Added for Architecture Config
import config # <--- IMPORT CONFIG

# Network Architecture calculation (Dependent on Config)
# 2048 bytes * STACK_SIZE
first_layer_size = 1024 * config.STACK_SIZE 
if first_layer_size < 512: first_layer_size = 512

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
        self.history_path = f"{config.MODEL_DIR}/score_history.pkl"
        
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
            # ... (Existing logic for collecting scores) ...
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
                    print(f"[Graph] Save failed: {e}")

            current_obs = self.locals.get('new_obs')
            
            # --- SCORE GRAPH (SEPARATE WINDOW) ---
            # Update Graph window every step (it's fast)
            try:
                # Create black canvas
                g_w, g_h = 800, 400 
                graph_frame = np.zeros((g_h, g_w, 3), dtype=np.uint8)

                if len(self.score_history) > 1:
                    scores = list(self.score_history)
                    min_s, max_s = min(scores), max(scores)
                    if max_s == min_s: max_s += 1 
                    
                    # Dynamic X-Scale
                    total_points = len(scores)
                    
                    for i in range(1, total_points):
                        p1_val = scores[i-1]
                        p2_val = scores[i]
                        
                        # Scales
                        x1 = int((i-1) * (g_w / (total_points - 1)))
                        x2 = int(i * (g_w / (total_points - 1)))
                        
                        # Invert Y (0 is top)
                        y1 = int((g_h - 20) - ((p1_val - min_s) / (max_s - min_s)) * (g_h - 40))
                        y2 = int((g_h - 20) - ((p2_val - min_s) / (max_s - min_s)) * (g_h - 40))
                        
                        cv2.line(graph_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        
                        # Only draw dots if not too crowded
                        if total_points < 100:
                            cv2.circle(graph_frame, (x2, y2), 3, (0, 0, 255), -1)

                    # Stats Text
                    cv2.putText(graph_frame, f"Max Score: {max_s:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(graph_frame, f"Avg (All {total_points}): {np.mean(scores):.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(graph_frame, f"Last: {scores[-1]:.2f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                else:
                     cv2.putText(graph_frame, "Waiting for games...", (100, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.imshow("Score History Graph", graph_frame)
            except: pass
            
            # One-time Window Setup
            if not self.windows_initialized:
                try:
                     cv2.namedWindow("Battle City AI Training", cv2.WINDOW_AUTOSIZE)
                     cv2.namedWindow("Score History Graph", cv2.WINDOW_AUTOSIZE)
                     cv2.moveWindow("Battle City AI Training", 50, 50)
                     cv2.moveWindow("Score History Graph", 800, 50)
                     self.windows_initialized = True
                except Exception as e:
                    print(f"\n[WARNING] Could not open display: {e}. Switching to HEADLESS MODE (Console only).")
                    self.windows_initialized = "HEADLESS"

            if self.windows_initialized == "HEADLESS":
                return True # Skip rendering
                
            # render() logic handled via INFO to save bandwidth
            # We only render the first environment's view
            try:
                # Get the frame from INFO (bypassing Agent's blindfold)
                frame = infos[0].get('render')
                
                if frame is not None:
                    # Resize for Window (84x84 -> 672x672)
                    frame_img = frame.astype('uint8')
                    frame_big = cv2.resize(frame_img, (672, 672), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to BGR for Colored Text
                    display_frame = cv2.cvtColor(frame_big, cv2.COLOR_GRAY2BGR)
                    
                    # Draw HUD
                    kills = infos[0].get('kills', 0)
                    score = infos[0].get('score', 0.0)
                    total_steps = self.num_timesteps
                    
                    # Top HUD
                    cv2.putText(display_frame, f"KILLS: {kills}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) # Red
                    cv2.putText(display_frame, f"SCORE: {score:.1f}", (400, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) # Green
                    cv2.putText(display_frame, f"Steps: {total_steps}", (20, 650), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) 
                    
                    cv2.imshow("Battle City AI Training", display_frame)
                    cv2.waitKey(1) # 1ms delay
                    
            except Exception as render_err:
                 # print(f"Render Error: {render_err}")
                 pass
        except Exception as e:
            if self.num_timesteps % 1000 == 0:
                print(f"\n[DEBUG] CV2 Render Error: {e}")
            pass
        return True

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"--- BATTLE CITY AI TRAINING (VISUAL MODE) ---")
    
    # Use SubprocVecEnv for Multicore Speed
    # Windows Note: SubprocVecEnv requires non-lambda functions usually. 
    # We use a list comprehension of factory functions.
    
    # Pass Config to Environment
    env_kwargs = {'use_vision': config.USE_VISION, 'stack_size': config.STACK_SIZE}
    
    if config.NUM_CPU > 1:
        # We need to import stable_baselines3.common.env_util to use make_vec_env with SubprocVecEnv correctly if not done automatically
        env = make_vec_env(BattleCityEnv, n_envs=config.NUM_CPU, seed=42, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    else:
        env = make_vec_env(BattleCityEnv, n_envs=1, seed=42, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

    # Load or Create Model
    # Choose Policy Type based on Config
    policy_type = "MultiInputPolicy" if config.USE_VISION else "MlpPolicy"
    
    print(f"Stats: Vision={config.USE_VISION}, Stack={config.STACK_SIZE}, Policy={policy_type}")
    # Auto-Resume Logic
    latest_model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
    final_model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    
    if os.path.exists(latest_model_path):
        print(f"Loading interrupted model from {latest_model_path}...")
        model = PPO.load(latest_model_path, env=env)
        reset_timesteps = False
    elif os.path.exists(final_model_path):
        print(f"Loading existing model from {final_model_path}...")
        model = PPO.load(final_model_path, env=env)
        reset_timesteps = False
    else:
        print(f"Creating NEW MODEL ({first_layer_size}x{first_layer_size//2})...")
        
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[first_layer_size, first_layer_size//2], vf=[first_layer_size, first_layer_size//2])
        )
        
        model = PPO(
            policy_type, 
            env, 
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            learning_rate=config.LEARNING_RATE,
            n_steps=config.N_STEPS,          # Frequent updates
            batch_size=config.BATCH_SIZE,       # Standard batch size
            ent_coef=config.ENTROPY_COEF,         # EXTREME CURIOSITY (Prevent boredom)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            # ent_coef=0.1, (Removed duplicate)
            device="cuda",
            policy_kwargs=policy_kwargs
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ,
        save_path=config.MODEL_DIR,
        name_prefix="battle_city_ppo"
    )
    
    logger_callback = ConsoleLoggerCallback()
    render_callback = RenderCallback()

    print("Start VISUAL Learning... (Press Ctrl+C to stop)")
    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, logger_callback, render_callback],
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
        model.save(f"{config.MODEL_DIR}/battle_city_final")
        print("Training Finished!")
        
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            print("\n[User] Stopped by Keyboard (Ctrl+C).")
        else:
            print(f"\n[CRITICAL] Training Interrupted/Crashed: {e}")
            
        print("Saving EMERGENCY model...")
        print("Saving EMERGENCY model...")
        model.save(f"{config.MODEL_DIR}/battle_city_interrupted")
        print("Saved.")
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
