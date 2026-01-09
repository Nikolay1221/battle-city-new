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
LEARNING_RATE = 0.00001 # 1e-5: Very slow and stable. Good for preventing collapse.
ENTROPY_COEF = 0.2      # 0.2: High curiosity. Forces agent to try new things.
BATCH_SIZE = 256
N_STEPS = 512           # Steps per update per env.

# --- PATHS ---
MODEL_DIR = "models"
LOG_DIR = "logs"
ROM_PATH = 'BattleCity_fixed.nes'
