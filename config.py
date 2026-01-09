# BATTLE CITY AI - CONFIGURATION FILE

# --- ENVIRONMENT SETTINGS ---
USE_VISION = True   # Set True to enable Vision (Pixels) + RAM. False = RAM Only (Faster).
STACK_SIZE = 4       # Number of Frames/RAM-dumps to stack (Temporal Context). 
                     # Higher = More "Memory" of movement, but bigger inputs.

# --- TRAINING SETTINGS ---
NUM_CPU = 24         # Number of parallel environments (Processes).
                     # Recommended: Count of Logical Processors on your CPU.
                     
TOTAL_TIMESTEPS = 5_000_000  # Total frames to train for.
CHECKPOINT_FREQ = 10_000     # Save model every N steps.

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.0003  # 3e-4: Standard stable PPO rate.
ENTROPY_COEF = 0.05     # 0.05: Balanced exploration. Tries new things but doesn't ignore rewards.
CLIP_RANGE = 0.2        # 0.2: Standard PPO clipping. Allows healthy updates.
BATCH_SIZE = 256
N_STEPS = 512           # Steps per update per env.

# --- PATHS ---
MODEL_DIR = "models"
LOG_DIR = "logs"
ROM_PATH = 'BattleCity_fixed.nes'
