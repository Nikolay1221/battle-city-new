import gymnasium as gym
import os
import time
import cv2            # Added for rendering
import numpy as np    # Added for rendering
import pickle         # Added for Graph Persistence
from collections import deque # Added for Score History
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor # Keep just in case
from battle_city_env import BattleCityEnv
import torch as th # Added for Architecture Config

import config # Load Configuration

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
        # --- 1. DATA COLLECTION (Safe) ---
        try:
            # Detect End of Episode
            dones = self.locals.get('dones', [False])
            infos = self.locals.get('infos', [{}])
            
            any_finished = False
            for i, done in enumerate(dones):
                if done:
                    final_score = infos[i].get('score', 0)
                    self.score_history.append(final_score)
                    any_finished = True
            
            if any_finished:
                # Save History
                try:
                    with open(self.history_path, 'wb') as f:
                        pickle.dump(self.score_history, f)
                except:
                    pass
        except Exception as e:
            print(f"[Stats Error] {e}")

        # --- 2. LOGGING TO CONSOLE (Safe) ---
        # Record custom metrics to the main table
        if len(self.score_history) > 0:
            self.logger.record("time/total_games", len(self.score_history))
            # Calculate mean of last 100 games for the log
            recent_scores = list(self.score_history)[-100:]
            self.logger.record("rollout/history_mean_reward", sum(recent_scores) / len(recent_scores))

        # --- 3. VISUALIZATION (Unsafe on Colab/Headless) ---
        # Optimization: Redraw graph every 1000 steps, but process events (waitKey) more often
        try:
            # 1. Frequent Event Processing (Keep Window Responsive)
            if self.windows_initialized and self.num_timesteps % 100 == 0:
                 cv2.waitKey(1)

            # 2. Infrequent Heavy Rendering (Save FPS)
            if self.num_timesteps % 1000 == 0:
                # Update Graph window
                if not self.windows_initialized:
                     # Check if we have a display (primitive check)
                     if os.environ.get('DISPLAY') is None and os.name != 'nt':
                         return True # Skip render on headless Linux without X11
                     
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
                    # Downsample for drawing
                    draw_step = max(1, total_points // 600)
                    display_scores = scores[::draw_step]
                    total_display = len(display_scores)

                    for i in range(1, total_display):
                        p1 = display_scores[i-1]
                        p2 = display_scores[i]
                        
                        x1 = int((i-1) * (g_w / (total_display - 1)))
                        x2 = int(i * (g_w / (total_display - 1)))
                        
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
            
        except Exception:
            pass 
            
        return True

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"--- BATTLE CITY AI TRAINING (VISUAL MODE) ---")
    
    # Use DummyVecEnv (Single Process Logic)
    # Pass render_mode=None to Envs (Visualization is handled by Main Process Callback)
    # Using SubprocVecEnv for Multicore Speed
    # Windows Note: SubprocVecEnv requires non-lambda functions usually. 
    # We use a list comprehension of factory functions.
    if config.NUM_CPU > 1:
        env = SubprocVecEnv([lambda: BattleCityEnv(render_mode=None) for _ in range(config.NUM_CPU)])
    else:
        env = DummyVecEnv([lambda: BattleCityEnv(render_mode=None)])
    
    env = VecMonitor(env) # Standard Metrics (ep_rew_mean, etc)
    
    # Auto-Resume Logic
    latest_model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
    final_model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    
    # Smart Load Logic: Find best checkpoint
    checkpoints = [f for f in os.listdir(config.MODEL_DIR) if f.startswith("battle_city_ppo_") and f.endswith(".zip")]
    best_checkpoint = None
    if checkpoints:
        # Sort by step count (battle_city_ppo_123_steps.zip)
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[3]))
            best_checkpoint = f"{config.MODEL_DIR}/{checkpoints[-1]}"
        except:
            pass
            
    latest_model_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
    final_model_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    
    # Priority: Best Numbered Checkpoint > Interrupted > Final
    # (Because Interrupted might be a crashed 0-step model)
    
    model_to_load = None
    if best_checkpoint and os.path.exists(best_checkpoint):
        model_to_load = best_checkpoint
    elif os.path.exists(latest_model_path):
        model_to_load = latest_model_path
    elif os.path.exists(final_model_path):
        model_to_load = final_model_path
        
    if model_to_load:
        print(f"Loading saved model: {model_to_load}")
        model = PPO.load(model_to_load, env=env)
        
        # Explicitly update hyperparameters
        model.ent_coef = config.ENT_COEF
        model.learning_rate = config.LEARNING_RATE
        model.n_steps = config.N_STEPS
        model.batch_size = config.BATCH_SIZE
        model.gamma = config.GAMMA
        
        reset_timesteps = False
    else:
        # SAFETY CHECK: Do not overwrite existing models accidentally
        existing_models = [f for f in os.listdir(config.MODEL_DIR) if f.endswith(".zip")]
        if existing_models:
             raise RuntimeError(f"❌ SAFETY STOP: Found {len(existing_models)} existing models in '{config.MODEL_DIR}' but failed to load the specific one.\n"
                                "I will NOT start a fresh training to protect your files.\n"
                                "Please check file names or delete 'models/' content manually if you want a fresh start.")

        print("Creating NEW BLIND GOD MODEL (2048x1024) - RAM Only...")
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
            tensorboard_log=config.LOG_DIR,
            learning_rate=config.LEARNING_RATE, 
            n_steps=config.N_STEPS,          
            batch_size=config.BATCH_SIZE,       
            ent_coef=config.ENT_COEF,         
            gamma=config.GAMMA,
            gae_lambda=0.95,
            clip_range=0.2,
            device="cuda", # Ideally use GPU if installed, otherwise auto-fallbacks
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
        model.save(f"{config.MODEL_DIR}/battle_city_interrupted")
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
