import os

# --- HARDWARE ---
NUM_CPU = 16
NUM_VIRTUAL = 8 # For hybrid mode

# --- RENDERING ---
# Set to False to see the windows (RenderCallback)
HEADLESS_MODE = False 

# --- TRAINING DURATION ---
TOTAL_TIMESTEPS = 10_000_000 

# --- HYPERPARAMETERS ---
LR_START = 3e-4  
LR_MIN = 1e-5
LR_MAX = 5e-4
TARGET_KL = 0.015 

BATCH_SIZE = 2048 
N_STEPS = 1024   
N_EPOCHS = 4
GAMMA = 0.99
CLIP_RANGE = 0.2
ENT_COEF = 0.01

# --- ENVIRONMENT ---
STACK_SIZE = 4
FRAME_SKIP = 4
TARGET_STAGE = 1 

# --- MODEL ARCHITECTURE ---
USE_RECURRENT = False 
USE_TRANSFORMER = False

# --- MODE ---
USE_VIRTUAL = True   # Enable Virtual Environment
USE_HYBRID  = False  # Disable Hybrid

# --- PATHS ---
MODEL_DIR = "models/"
LOG_DIR = "logs/"
CHECKPOINT_FREQ = 100000
