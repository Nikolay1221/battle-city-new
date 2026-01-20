# --- SYSTEM & HARDWARE ---
import multiprocessing
import os
# Reduced from 24 to 16 to save VRAM
NUM_CPU = 16 

# --- HYBRID CONFIGURATION ---
NUM_VIRTUAL = 16 # ALL VIRTUAL

HEADLESS_MODE = False 

# --- TRAINING DURATION ---
TOTAL_TIMESTEPS = 10_000_000 

# --- OBSERVATION ---
STACK_SIZE = 4 
FRAME_SKIP = 4 
ROM_PATH = "BattleCity_fixed.nes"
TARGET_STAGE = 0 

USE_RECURRENT = False 
USE_TRANSFORMER = False

# --- VIRTUAL TRAINING ---
USE_VIRTUAL = True  
USE_HYBRID  = False 

# --- SAVING ---
CHECKPOINT_FREQ = 100_000 
MODEL_DIR = "models"
LOG_DIR = "logs"

# --- PPO HYPERPARAMETERS ---
LR_START      = 0.0005  # Increased to learn shooting faster
LR_MIN        = 0.00001
LR_MAX        = 0.001 
TARGET_KL     = 0.03 

# Adjusted for Stability
N_STEPS       = 1024    
BATCH_SIZE    = 512     
N_EPOCHS      = 4       
ENT_COEF      = 0.05    # Increased entropy to encourage shooting (random actions)
GAMMA         = 0.99     
CLIP_RANGE    = 0.2
ALLOW_NEW_MODEL = True
